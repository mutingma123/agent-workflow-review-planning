"""Blocks X1 to X7 of the formulation deck's Mac Studio protocol: the job
list, and one job's plans, warm starts and scores.

A job is one cell of one replication; a set is all jobs of one block and
replication, the unit the monitor commits. Seeds follow the manuscript:
layered workflow seed = replication, training sample 1000 + replication,
test sample 900000 + replication, so cells of one replication share the
workflow and common random numbers, and X1 replications 1 to 8 rebuild the
manuscript's E1 workflows.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

import kp_cuts as kc
import kp_model as km

DEFAULTS = dict(
    lam=0.5, alpha=0.9, gamma=0.3, q_same=0.9, q_diff=0.5, fan=2.0, budget_share=0.6,
    tau=0.15, rho=0.3, L=8, W=5, p_scale=1.0, e_sys_share=0.25,
    train_S=200, test_S=50000, ev_time_limit=60.0, ls_seconds=30.0,
    # Stopping: a solve ends at a proven gap of mip_gap (5 percent; 1 percent in
    # the scaling study X7) or at time_limit, whichever comes first. time_limit
    # None lets kp_campaign compute it from the campaign budget.
    mip_gap=0.05, x7_mip_gap=0.01, time_limit=None, time_limit_max=1800.0, time_limit_min=120.0,
    # Budget: the whole campaign ends within campaign_hours of its first launch,
    # with reserve_hours kept for the last jobs and the final records.
    campaign_hours=48.0, reserve_hours=2.0, efficiency=0.85, job_overhead=40.0,
    threads=2, soft_mem_gb=None, nodefile_start=None, mip_focus=None,
    reps=dict(X1=41, X2=10, X3=10, X4=10, X5=10, X6=10, X7=3),
    blocks=["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    x2_q_same=0.70, x2_q_diff=[0.45, 0.70, 0.95], x2_copulas=["gaussian", "t"], t_nu=4.0,
    x3_dbar=[0.6, 0.8, 1.0], x3_tie=0.5,
    x5_beta=[0.02, 0.01, 0.005],
    x6_kinds=["tree", "random"], x6_rho=[0.0, 0.3, 0.6],
    x7_sizes=[[8, 5], [10, 6], [10, 8], [12, 10]], x7_S=[200, 400, 800, 1600, 3200],
    x7_methods=["EF", "BD"], x7_pi_S=[200, 400, 800], x7_pi_size=[8, 5], x7_chain=False,
)


def config(overrides: dict | None = None) -> dict:
    c = copy.deepcopy(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k == "reps" and isinstance(v, dict):
            c["reps"].update(v)
        else:
            c[k] = v
    if c["time_limit"] is None:
        c["time_limit"] = c["time_limit_max"]
    return c


def preflight_config() -> dict:
    """Tiny settings that run every block end to end in minutes."""
    return config(dict(L=4, W=3, p_scale=3.0, train_S=40, test_S=2000, time_limit=20.0, ev_time_limit=10.0,
                       ls_seconds=2.0, reps={b: 1 for b in DEFAULTS["reps"]}, campaign_hours=1.0,
                       reserve_hours=0.05, time_limit_min=5.0,
                       x7_sizes=[[4, 3], [5, 3]], x7_S=[30, 60], x7_pi_S=[30], x7_pi_size=[4, 3]))


def job_seconds(job: dict, c: dict, time_limit: float | None = None) -> float:
    """Worst-case wall time of a job: every solve at its time limit."""
    T = c["time_limit"] if time_limit is None else time_limit
    ls = min(c["ls_seconds"], 0.1 * T)
    return c["job_overhead"] + c["ev_time_limit"] * job["n_ev"] + job["n_mip"] * (T + ls)


def _job(block, rep, cell, n_mip, n_ev=1, **kw):
    j = dict(block=block, rep=int(rep), cell=cell, n_mip=int(n_mip), n_ev=int(n_ev), **kw)
    j["id"] = f"{block}/rep_{rep:03d}/{cell}"
    j["set"] = f"{block}/rep_{rep:03d}"
    return j


def jobs_for(block: str, rep: int, c: dict) -> list:
    out = []
    if block == "X1":
        for rho in (0.0, 0.3, 0.6):
            for tau in (0.05, 0.15, 0.30):
                out.append(_job("X1", rep, f"rho{rho:.1f}_tau{tau:.2f}", 2 if rho == 0 else 3, rho=rho, tau=tau))
    elif block == "X2":
        for qd in c["x2_q_diff"]:
            for cop in c["x2_copulas"]:
                out.append(_job("X2", rep, f"qdiff{qd:.2f}_{cop}", 2, q_diff=qd, copula=cop))
    elif block == "X3":
        for d in c["x3_dbar"]:
            for tie in (["independent", "G0"] if d < 1 else ["independent"]):
                out.append(_job("X3", rep, f"dbar{d:.1f}_{tie}", 2, dbar=d, tie=tie))
    elif block == "X4":
        for mode in ("all", "deliverables"):
            out.append(_job("X4", rep, f"loss_{mode}", 1 if mode == "all" else 2, ell_mode=mode))
    elif block == "X5":
        out.append(_job("X5", rep, "caps", 1 + len(c["x5_beta"])))
    elif block == "X6":
        for kind in c["x6_kinds"]:
            for rho in c["x6_rho"]:
                out.append(_job("X6", rep, f"{kind}_rho{rho:.1f}", 1, kind=kind, rho=rho))
    elif block == "X7":
        for (L, W) in c["x7_sizes"]:
            for S in c["x7_S"]:
                for meth in c["x7_methods"]:
                    out.append(_job("X7", rep, f"n{L * W:03d}_S{S:04d}_{meth}", 1, n_ev=0, L=L, W=W, S=S, method=meth))
        L, W = c["x7_pi_size"]
        for S in c["x7_pi_S"]:
            out.append(_job("X7", rep, f"n{L * W:03d}_S{S:04d}_PI", 1, n_ev=0, L=L, W=W, S=S, method="PI"))
    else:
        raise ValueError(block)
    for j in out:
        j["wc"] = job_seconds(j, c)
    return out


def budget(c: dict, workers: int, blocks=None, hours: float | None = None) -> dict:
    """Time limit per solve that fits the campaign in its wall-clock budget.

    Capacity is workers x (hours - reserve) x efficiency job-seconds. Every
    solve gets the same limit T, from capacity = sum over jobs of
    (overhead + EV + n_mip (T + local search)). When T falls below
    time_limit_min, replications are cut, X1 first down to 8, then the other
    blocks down to 5 and X7 down to 2, until T reaches the minimum."""
    c = copy.deepcopy(c)
    hours = c["campaign_hours"] if hours is None else hours
    cap = max(0.0, (hours - c["reserve_hours"]) * 3600.0 * workers * c["efficiency"])
    floors = dict(X1=8, X7=2)
    cuts = []

    def solve_T(cc):
        q = build_queue(cc, blocks)
        fixed = sum(cc["job_overhead"] + cc["ev_time_limit"] * j["n_ev"] for j in q)
        mips = sum(j["n_mip"] for j in q)
        # T + min(ls, 0.1 T) per solve: solve for T with ls at its cap, then check
        T = (cap - fixed) / max(1, mips) - cc["ls_seconds"]
        if T < 10 * cc["ls_seconds"]:
            T = (cap - fixed) / max(1, mips) / 1.1
        return T, q

    T, q = solve_T(c)
    order = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    while T < c["time_limit_min"]:
        cand = [b for b in order if b in (blocks or c["blocks"]) and c["reps"][b] > floors.get(b, 5)]
        if not cand:
            break
        b = cand[0] if "X1" in cand and c["reps"]["X1"] > 8 else max(cand, key=lambda x: c["reps"][x])
        c["reps"][b] -= 1
        cuts.append(b)
        T, q = solve_T(c)
    T = float(min(c["time_limit_max"], max(c["time_limit_min"], T)))
    c["time_limit"] = T
    q = build_queue(c, blocks)
    worst = sum(j["wc"] for j in q)
    return dict(time_limit=T, reps=c["reps"], cuts=cuts, jobs=len(q), solves=sum(j["n_mip"] for j in q),
                worst_job_hours=worst / 3600.0, worst_wall_hours=worst / 3600.0 / workers / c["efficiency"],
                capacity_job_hours=cap / 3600.0, config=c)


def build_queue(c: dict, blocks=None) -> list:
    """Replication-major order: replication 1 of every block, then
    replication 2, so early sets cover every block."""
    blocks = blocks or c["blocks"]
    top = max(c["reps"][b] for b in blocks)
    q = []
    for rep in range(1, top + 1):
        for b in blocks:
            if rep <= c["reps"][b]:
                q += jobs_for(b, rep, c)
    return q


# ----------------------------------------------------------------- one job
def _q(c, same=None, diff=None):
    return km.q_matrix(c["q_same"] if same is None else same, c["q_diff"] if diff is None else diff)


def _opts(c, log_file=None):
    return dict(threads=c["threads"], mip_gap=c["mip_gap"], soft_mem_gb=c.get("soft_mem_gb"),
                nodefile_start=c.get("nodefile_start"), mip_focus=c.get("mip_focus"), log_file=log_file)


class Planner:
    """Plans of one job on one instance with a shared warm-start pool."""

    def __init__(self, inst, train, test, c, env, log_file=None, progress=None, job=None):
        import kp_solve as ks
        self.ks, self.inst, self.train, self.test, self.c, self.env = ks, inst, train, test, c, env
        self.opts = _opts(c, log_file)
        self.progress = progress
        job = job or {}
        self.deadline = job.get("deadline")           # epoch seconds the job must end by
        self.mips_left = max(1, int(job.get("n_mip", 1)))
        if job.get("block") == "X7":
            self.opts["mip_gap"] = c["x7_mip_gap"]
        self.pool = []                       # (name, a, y) candidates for warm starts
        self.plans = {}

    def score(self, name, a, y, solver=None):
        """Scores on the test sample and the job's training sample, both with
        the true distribution and loss weights."""
        d = km.metrics(self.inst, a, y, self.test, self.c["lam"], self.c["alpha"])
        d["in_sample_risk_obj"] = km.objective(self.inst, a, y, self.train, self.c["lam"], self.c["alpha"])
        d["a"], d["y"] = [int(v) for v in a], [int(v) for v in y]
        if solver:
            d.update({f"solver_{k}": v for k, v in solver.items() if k not in ("a", "y")})
        self.plans[name] = d
        self.pool.append((name, np.asarray(a, int), np.asarray(y, bool)))
        return d

    def best_start(self, sc, lam, cap_beta=None, ls=True, inst=None):
        """Warm start: the pool candidate with the best objective on the sample
        being solved, improved by local search; with a cap, a candidate that
        meets it on that sample is preferred."""
        inst = inst or self.inst
        alpha = self.c["alpha"]

        def val(a, y):
            e = km.propagate(inst, a, y, sc)
            v = km.objective(inst, a, y, sc, lam, alpha, e=e)
            if cap_beta is not None:
                v += 1e3 * max(0.0, float(np.mean(e.sum(1) >= inst.e_sys)) - cap_beta)
            return v

        same_use = [(n, a, y) for n, a, y in self.pool]
        name, a, y = min(same_use, key=lambda t: val(t[1], t[2]))
        info = dict(start_from=name, start_value=val(a, y))
        if ls and self.c["ls_seconds"] > 0:
            a, y, v, st = kc.local_search(inst, sc, a, y, lam, alpha, self.c["ls_seconds"], cap_beta=cap_beta)
            info.update(st)
        return (a, y), info

    def route(self):
        a, y = self.ks.solve_route(self.inst, self.env, threads=min(2, self.c["threads"]))
        return self.score("ROUTE", a, y)

    def ev(self, q):
        name, a0, y0 = self.pool[0]
        a, y, st = self.ks.solve_ev(self.inst, q, self.env, self.opts, self.c["ev_time_limit"], start=(a0, y0))
        return self.score("EV", a, y, st)

    def ef(self, name, sc, lam, cap_beta=None, per_scenario_reviews=False, method="EF", plan_inst=None):
        """Solve on sample sc (and plan_inst, when the plan assumes other loss
        weights), then score the plan under the true instance."""
        pin = plan_inst or self.inst
        T = float(self.c["time_limit"])
        if self.deadline:                            # share the job's remaining time
            T = min(T, (self.deadline - time.time() - 20.0) / self.mips_left)
        T = max(10.0, T)
        self.mips_left = max(1, self.mips_left - 1)
        ls_keep = self.c["ls_seconds"]
        self.c = dict(self.c, ls_seconds=min(ls_keep, 0.1 * T))
        start, info = self.best_start(sc, lam, cap_beta, inst=pin)
        self.c = dict(self.c, ls_seconds=ls_keep)
        T = T - info.get("ls_seconds", 0.0)
        if method == "BD":
            res = self.ks.solve_benders(pin, sc, lam, self.c["alpha"], self.env, self.opts, T, start=start,
                                        label=name, progress_path=self.progress)
        else:
            res = self.ks.solve_ef(pin, sc, lam, self.c["alpha"], self.env, self.opts, T, start=start,
                                   cap_beta=cap_beta, per_scenario_reviews=per_scenario_reviews, label=name,
                                   progress_path=self.progress)
        res.update(info)
        if "a" not in res:
            # the PI bound has no plan by design; otherwise no incumbent (infeasible cap, time or
            # memory limit). Either way keep the solver record.
            self.plans[name] = dict(no_solution=not per_scenario_reviews, bound_only=bool(per_scenario_reviews),
                                    **{f"solver_{k}": v for k, v in res.items()})
            return res
        res["solve_sample_obj"] = km.objective(pin, res["a"], res["y"], sc, lam, self.c["alpha"])
        return self.score(name, res["a"], res["y"], res)


def _layered(c, rep, **kw):
    return km.make_instance(c["L"], c["W"], c["fan"], c["budget_share"], kw.pop("tau", c["tau"]), seed=rep,
                            p_scale=c["p_scale"], e_sys_share=c["e_sys_share"], **kw)


def run_job(job: dict, c: dict, env, log_file=None, progress=None) -> dict:
    t0 = time.time()
    b, rep = job["block"], job["rep"]
    lam, gam = c["lam"], c["gamma"]
    tr_seed, te_seed = 1000 + rep, 900000 + rep
    S, St = c["train_S"], c["test_S"]
    out = dict(cell=job["cell"])
    if b == "X1":
        rho = job["rho"]
        inst = _layered(c, rep, tau=job["tau"])
        q = _q(c)
        train = km.sample_scenarios(inst, S, rho, gam, q, tr_seed)
        test = km.sample_scenarios(inst, St, rho, gam, q, te_seed)
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(q)
        P.ef("RN", train, 0.0)
        if rho > 0:
            ind = km.sample_scenarios(inst, S, 0.0, gam, q, tr_seed)
            P.ef("IND", ind, lam)
        P.ef("RA", train, lam)
    elif b == "X2":
        inst = _layered(c, rep)
        q_true = _q(c, c["x2_q_same"], job["q_diff"])
        cop = job["copula"]
        train = km.sample_scenarios(inst, S, c["rho"], gam, q_true, tr_seed, copula=cop, nu=c["t_nu"])
        test = km.sample_scenarios(inst, St, c["rho"], gam, q_true, te_seed, copula=cop, nu=c["t_nu"])
        assumed = km.sample_scenarios(inst, S, c["rho"], gam, _q(c), tr_seed)     # manuscript's assumption
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(_q(c))
        P.ef("RA", assumed, lam)
        P.ef("RA-PAIR", train, lam)
    elif b == "X3":
        inst = _layered(c, rep, dbar=job["dbar"])
        q = _q(c)
        tie = None if job["tie"] == "independent" else c["x3_tie"]
        train = km.sample_scenarios(inst, S, c["rho"], gam, q, tr_seed, review_tie=tie)
        test = km.sample_scenarios(inst, St, c["rho"], gam, q, te_seed, review_tie=tie)
        perfect = _layered(c, rep, dbar=1.0)
        assumed = km.sample_scenarios(perfect, S, c["rho"], gam, q, tr_seed)       # reviews assumed perfect
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(q)
        P.ef("RA", assumed, lam)
        P.ef("RA-PAIR", train, lam)
    elif b == "X4":
        inst = _layered(c, rep, ell_mode=job["ell_mode"])
        q = _q(c)
        train = km.sample_scenarios(inst, S, c["rho"], gam, q, tr_seed)
        test = km.sample_scenarios(inst, St, c["rho"], gam, q, te_seed)
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(q)
        if job["ell_mode"] == "deliverables":
            P.ef("RA", train, lam, plan_inst=_layered(c, rep))   # plans with every output weighted 1
        P.ef("RA-PAIR", train, lam)
    elif b == "X5":
        inst = _layered(c, rep)
        q = _q(c)
        train = km.sample_scenarios(inst, S, c["rho"], gam, q, tr_seed)
        test = km.sample_scenarios(inst, St, c["rho"], gam, q, te_seed)
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(q)
        P.ef("RA-PAIR", train, lam)
        for beta in sorted(c["x5_beta"], reverse=True):        # looser cap first: its plan starts the next
            P.ef(f"RA-CAP{beta:g}", train, lam, cap_beta=beta)
        out["betas"] = sorted(c["x5_beta"], reverse=True)
    elif b == "X6":
        inst = km.make_template(job["kind"], rep, c["budget_share"], c["tau"], c["p_scale"],
                                e_sys_share=c["e_sys_share"])
        q = _q(c)
        train = km.sample_scenarios(inst, S, job["rho"], gam, q, tr_seed)
        test = km.sample_scenarios(inst, St, job["rho"], gam, q, te_seed)
        P = Planner(inst, train, test, c, env, log_file, progress, job)
        P.route()
        P.ev(q)
        P.ef("RA-PAIR", train, lam)
        out["template"] = dict(n=inst.n, nA=inst.nA, and_groups=len(inst.and_groups),
                               or_groups=len(inst.or_groups), e_sys=inst.e_sys, B=inst.B, H=inst.H)
    elif b == "X7":
        inst = km.make_instance(job["L"], job["W"], c["fan"], c["budget_share"], c["tau"], seed=100 + rep,
                                p_scale=c["p_scale"], e_sys_share=c["e_sys_share"])
        q = _q(c)
        sc = km.sample_scenarios(inst, job["S"], c["rho"], gam, q, 2000 + rep)
        test = km.sample_scenarios(inst, min(St, 20000), c["rho"], gam, q, te_seed)
        P = Planner(inst, sc, test, c, env, log_file, progress, job)
        P.route()
        if job["method"] == "PI":
            res = P.ef("PI", sc, lam, per_scenario_reviews=True)
            out["pi"] = {k: v for k, v in res.items() if k not in ("a", "y", "trace")}
            out["pi"]["trace"] = res.get("trace", [])
        else:
            P.ef(job["method"], sc, lam, method=job["method"])
        out["censored"] = {k: (v.get("solver_time_to_target") is None) for k, v in P.plans.items() if k != "ROUTE"}
    else:
        raise ValueError(b)
    out.update(n=P.inst.n, nA=P.inst.nA, B=P.inst.B, H=P.inst.H, e_sys=P.inst.e_sys, plans=P.plans,
               wall=time.time() - t0)
    return out


def fake_run_job(job: dict, c: dict, env=None, log_file=None, progress=None) -> dict:
    """Stand-in for tests of the scheduler and monitors: no solver."""
    time.sleep(float(c.get("fake_seconds", 0.05)))
    if job.get("cell", "").endswith(c.get("fake_fail_suffix", "\0")):
        raise RuntimeError("fake failure")
    return dict(cell=job["cell"], plans=dict(ROUTE=dict(p_systemic=0.1), RA=dict(p_systemic=0.02)), wall=0.0)
