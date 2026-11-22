"""Gurobi solvers for the extended formulation: routing with spot checks
(ROUTE), the mean-value plan (EV), the extensive form with optional
systemic-failure cap and per-scenario reviews (the PI bound), and
branch-and-Benders-cut with closed-form cuts for fixed workflows.

Every solver takes one Gurobi environment, so a worker process checks out
its license once and reuses it for all of its jobs.
"""
from __future__ import annotations

import json
import math
import os
import time

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB

import kp_cuts as kc
import kp_model as km


def make_env(threads: int | None = None) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if threads:
        env.setParam("Threads", int(threads))
    env.start()
    return env


def apply_params(m: gp.Model, opts: dict, time_limit: float):
    """opts: threads, mip_gap, soft_mem_gb, nodefile_start, nodefile_dir,
    seed, mip_focus, log_file."""
    m.Params.Threads = int(opts.get("threads", 4))
    m.Params.MIPGap = float(opts.get("mip_gap", 0.01))
    m.Params.TimeLimit = max(1.0, float(time_limit))
    if opts.get("soft_mem_gb"):
        m.Params.SoftMemLimit = float(opts["soft_mem_gb"])
    if opts.get("nodefile_start"):
        m.Params.NodefileStart = float(opts["nodefile_start"])
    if opts.get("nodefile_dir"):
        m.Params.NodefileDir = str(opts["nodefile_dir"])
    if opts.get("seed") is not None:
        m.Params.Seed = int(opts["seed"])
    if opts.get("mip_focus") is not None:
        m.Params.MIPFocus = int(opts["mip_focus"])
    if opts.get("log_file"):
        m.Params.LogFile = str(opts["log_file"])
        m.Params.OutputFlag = 1
        m.Params.LogToConsole = 0


GAP_LEVELS = (0.20, 0.10, 0.05, 0.02, 0.01)


class Tracker:
    """Callback state: the first time the gap reaches the target and each of
    GAP_LEVELS, a sparse trace of (seconds, incumbent, bound), and a progress
    file the monitor reads."""

    def __init__(self, target: float, label: str = "", progress_path=None, every: float = 10.0):
        self.target, self.label, self.path, self.every = target, label, progress_path, every
        self.t_target = None
        self.t_gap = {f"{g:g}": None for g in GAP_LEVELS}
        self.trace = []
        self.last = -1e9
        self.last_write = -1e9

    def note(self, t, best, bound, force=False):
        gap = abs(best - bound) / max(abs(best), 1e-10) if best < GRB.INFINITY else float("inf")
        if self.t_target is None and gap <= self.target + 1e-12:
            self.t_target = float(t)
        for g in GAP_LEVELS:
            if self.t_gap[f"{g:g}"] is None and gap <= g + 1e-12:
                self.t_gap[f"{g:g}"] = float(t)
        if force or t - self.last >= self.every:
            self.trace.append((round(float(t), 2), float(best) if best < GRB.INFINITY else None, float(bound)))
            self.last = t
        if self.path and t - self.last_write >= 30.0:
            self.last_write = t
            try:
                tmp = str(self.path) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(dict(plan=self.label, seconds=round(float(t), 1),
                                   incumbent=float(best) if best < GRB.INFINITY else None,
                                   bound=float(bound), gap=gap if math.isfinite(gap) else None,
                                   written=time.strftime("%Y-%m-%dT%H:%M:%S")), f)
                os.replace(tmp, self.path)
            except OSError:
                pass

    def __call__(self, model, where):
        if where == GRB.Callback.MIP:
            self.note(model.cbGet(GRB.Callback.RUNTIME), model.cbGet(GRB.Callback.MIP_OBJBST),
                      model.cbGet(GRB.Callback.MIP_OBJBND))
        elif where == GRB.Callback.MIPSOL:
            self.note(model.cbGet(GRB.Callback.RUNTIME), model.cbGet(GRB.Callback.MIPSOL_OBJBST),
                      model.cbGet(GRB.Callback.MIPSOL_OBJBND), force=True)


def pair_mode_for(q) -> str:
    return "z" if km.same_diff(np.asarray(q)) is not None else "w"


# ---------------------------------------------------------------- first stage
def first_stage(m: gp.Model, inst: km.Instance, mode: str, reviews: bool = True, use_fixed=None):
    """x (n, M), y (n), u (n or None), pair variables z (nA) or w (nA, M, M).

    use_fixed fixes the set of used tasks of a template (ROUTE's default
    decomposition); None lets a template choose its decomposition."""
    n, M = inst.n, inst.M
    x = m.addVars(n, M, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, vtype=GRB.BINARY, name="y") if reviews else None
    u = None
    if inst.fixed:
        m.addConstrs((x.sum(i, "*") == 1 for i in range(n)), name="assign")
    elif use_fixed is not None:
        uf = np.asarray(use_fixed).astype(bool)
        m.addConstrs((x.sum(i, "*") == int(uf[i]) for i in range(n)), name="assign")
        if reviews:
            for i in range(n):
                if not uf[i]:
                    y[i].UB = 0.0
    else:
        u = m.addVars(n, vtype=GRB.BINARY, name="u")
        for r in inst.sinks:
            u[int(r)].LB = 1.0
        for i, kids in inst.and_groups.items():
            for j in kids:
                m.addConstr(u[j] == u[i], name=f"and[{i},{j}]")
        for i, kids in inst.or_groups.items():
            m.addConstr(gp.quicksum(u[j] for j in kids) == u[i], name=f"or[{i}]")
        m.addConstrs((x.sum(i, "*") == u[i] for i in range(n)), name="assign")
        if reviews:
            m.addConstrs((y[i] <= u[i] for i in range(n)), name="review_used")
    cost = gp.quicksum(inst.c[i, k] * x[i, k] for i in range(n) for k in range(M))
    m.addConstr(cost <= inst.B, name="budget")
    if reviews:
        m.addConstr(gp.quicksum(inst.h[i] * y[i] for i in range(n)) <= inst.H, name="review")
    for k in range(len(inst.Q)):
        if np.isfinite(inst.Q[k]):
            mods = [mm for mm in range(M) if km.MODEL_PROVIDER[mm] == k]
            m.addConstr(gp.quicksum(inst.c[i, mm] * x[i, mm] for i in range(n) for mm in mods) <= inst.Q[k],
                        name=f"quota[{k}]")
    if mode == "z":
        z = m.addVars(inst.nA, lb=0.0, ub=1.0, name="z")
        for a, (j, i) in enumerate(inst.arcs):
            for k in range(M):
                m.addConstr(z[a] >= x[j, k] + x[i, k] - 1, name=f"same_lo[{a},{k}]")
                m.addConstr(z[a] <= 1 - x[j, k] + x[i, k], name=f"same_hi[{a},{k}]")
        pair = z
    else:
        w = m.addVars(inst.nA, M, M, lb=0.0, ub=1.0, name="w")
        for a, (j, i) in enumerate(inst.arcs):
            for k in range(M):
                m.addConstr(w.sum(a, k, "*") <= x[j, k], name=f"w_row[{a},{k}]")
                m.addConstr(w.sum(a, "*", k) <= x[i, k], name=f"w_col[{a},{k}]")
            uj = 1 if u is None else u[j]
            ui = 1 if u is None else u[i]
            if use_fixed is not None:
                uj, ui = int(use_fixed[j]), int(use_fixed[i])
            m.addConstr(w.sum(a, "*", "*") >= uj + ui - 1, name=f"w_cover[{a}]")
        pair = w
    return x, y, u, pair


def set_start(inst, x, y, u, pair, mode, a, yv):
    a = np.asarray(a, dtype=int)
    for i in range(inst.n):
        for k in range(inst.M):
            x[i, k].Start = 1.0 if a[i] == k else 0.0
        if y is not None:
            y[i].Start = float(bool(yv[i]))
        if u is not None:
            u[i].Start = float(a[i] >= 0)
    pv = kc.pair_values(inst, a, mode)
    if mode == "z":
        for k in range(inst.nA):
            pair[k].Start = float(pv[k])
    else:
        for k in range(inst.nA):
            for p_ in range(inst.M):
                for q_ in range(inst.M):
                    pair[k, p_, q_].Start = float(pv[k, p_, q_])


def extract(inst, x, y, u=None):
    a = np.full(inst.n, -1, dtype=int)
    for i in range(inst.n):
        vals = [x[i, k].X for k in range(inst.M)]
        if max(vals) > 0.5:
            a[i] = int(np.argmax(vals))
    yv = np.array([y[i].X > 0.5 for i in range(inst.n)]) if y is not None else np.zeros(inst.n, bool)
    return a, yv


def _possible(inst: km.Instance, sc: km.Scenarios) -> np.ndarray:
    """(S, n): output i can be released wrong in scenario s under some plan."""
    poss = np.zeros((sc.S, inst.n), dtype=bool)
    own_any = sc.xi.any(axis=2)
    live = sc.U < sc.q.max()
    for i in range(inst.n):
        val = own_any[:, i].copy()
        for k in inst.in_arcs[i]:
            val |= poss[:, inst.arcs[k, 0]] & live[:, k]
        poss[:, i] = val
    return poss


def _result(m, method, t0, build, tracker, inst, x, y, u, extra=None):
    out = dict(method=method, status=int(m.Status), build_time=build, time=time.time() - t0,
               rows=int(m.NumConstrs), cols=int(m.NumVars), binaries=int(m.NumBinVars),
               nodes=float(m.NodeCount), time_to_target=tracker.t_target if tracker else None,
               time_to_gap=tracker.t_gap if tracker else {}, trace=tracker.trace if tracker else [],
               time_limit=float(m.Params.TimeLimit))
    out["bound"] = float(m.ObjBound) if (m.SolCount or m.Status == GRB.OPTIMAL) else float("nan")
    if m.SolCount:
        out["obj"], out["gap"] = float(m.ObjVal), float(m.MIPGap)
        if x is not None:
            out["a"], out["y"] = extract(inst, x, y, u)
    else:
        out["obj"], out["gap"] = float("nan"), float("nan")
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------ extensive form
def solve_ef(inst: km.Instance, sc: km.Scenarios, lam: float, alpha: float, env: gp.Env, opts: dict,
             time_limit: float, start=None, cap_beta: float | None = None, per_scenario_reviews: bool = False,
             label: str = "EF", progress_path=None) -> dict:
    """Extensive form of the extended program on sample sc.

    cap_beta adds binary f_s with sum_i e_si <= E_sys - 1 + (n - E_sys + 1) f_s
    and sum_s f_s <= floor(beta |Omega|). per_scenario_reviews replaces y by
    y_s (S, n) under the same hours in every scenario: the perfect-information
    bound on review policies (its objective is a bound, not a plan)."""
    t0 = time.time()
    S, n, M = sc.S, inst.n, inst.M
    mode = pair_mode_for(sc.q)
    m = gp.Model(label, env=env)
    x, y, u, pair = first_stage(m, inst, mode, reviews=not per_scenario_reviews)
    if per_scenario_reviews:
        ys = m.addVars(S, n, vtype=GRB.BINARY, name="ys")
        for s in range(S):
            m.addConstr(gp.quicksum(inst.h[i] * ys[s, i] for i in range(n)) <= inst.H)
            if u is not None:
                for i in range(n):
                    m.addConstr(ys[s, i] <= u[i])
    m.update()
    xcol = np.array([[x[i, k].index for k in range(M)] for i in range(n)])
    if per_scenario_reviews:
        ycol_s = np.array([[ys[s, i].index for i in range(n)] for s in range(S)])
    else:
        ycol = np.array([y[i].index for i in range(n)])
    poss = _possible(inst, sc)
    ecol = -np.ones((S, n), dtype=np.int64)
    ne = int(poss.sum())
    base = m.NumVars
    ecol[poss] = base + np.arange(ne)
    obj_e = np.broadcast_to(inst.ell[None, :], (S, n))[poss]
    m.addMVar(ne, lb=0.0, ub=1.0, obj=(1 - lam) / S * obj_e, name="e")
    use_cvar = lam > 0
    if use_cvar:
        eta = m.addVar(lb=0.0, ub=float(inst.ell.sum()), obj=lam, name="eta")
        m.addMVar(S, lb=0.0, obj=lam / ((1 - alpha) * S), name="r")
    m.update()
    rows, cols, vals, rhs, sense = [], [], [], [], []
    r = 0

    def ycols(s_idx, i_idx):
        return ycol_s[s_idx, i_idx] if per_scenario_reviews else ycol[i_idx]

    # own-error rows: e_si + delta_si y_i - sum_m xi_sim x_im >= 0
    ss, ii, mm = np.nonzero(sc.xi)
    key = ss * n + ii
    uniq, inv = np.unique(key, return_inverse=True)
    us, ui = uniq // n, uniq % n
    nr = len(uniq)
    dl = sc.delta[us, ui]
    rows += [r + np.arange(nr), r + np.nonzero(dl)[0], r + inv]
    cols += [ecol[us, ui], ycols(us[dl], ui[dl]), xcol[ii, mm]]
    vals += [np.ones(nr), np.ones(int(dl.sum())), -np.ones(len(inv))]
    rhs.append(np.zeros(nr))
    r += nr
    always = sc.U < sc.q.min()
    live = sc.U < sc.q.max()
    sd = km.same_diff(sc.q)
    for k, (j, i) in enumerate(inst.arcs):
        valid = poss[:, j] & live[:, k]
        sa = np.nonzero(valid & always[:, k])[0]              # e_i - e_j + delta y_i >= 0
        na = len(sa)
        da = sc.delta[sa, i]
        rows += [r + np.arange(na), r + np.arange(na), r + np.nonzero(da)[0]]
        cols += [ecol[sa, i], ecol[sa, j], ycols(sa[da], np.full(int(da.sum()), i))]
        vals += [np.ones(na), -np.ones(na), np.ones(int(da.sum()))]
        rhs.append(np.zeros(na))
        r += na
        sb = np.nonzero(valid & ~always[:, k])[0]
        nb = len(sb)
        if nb == 0:
            continue
        db = sc.delta[sb, i]
        rows += [r + np.arange(nb), r + np.arange(nb), r + np.nonzero(db)[0]]
        cols += [ecol[sb, i], ecol[sb, j], ycols(sb[db], np.full(int(db.sum()), i))]
        vals += [np.ones(nb), -np.ones(nb), np.ones(int(db.sum()))]
        if mode == "z":
            qs, qd = sd
            ks = (sc.U[sb, k] < qs).astype(float)
            kd = (sc.U[sb, k] < qd).astype(float)
            # e_i - e_j + delta y_i - (ks - kd) z >= kd - 1
            rows.append(r + np.arange(nb))
            cols.append(np.full(nb, pair[k].index))
            vals.append(-(ks - kd))
            rhs.append(kd - 1.0)
        else:
            # e_i - e_j + delta y_i - sum_{passing pairs} w >= -1
            kap = sc.U[sb, k][:, None, None] < sc.q[None, :, :]
            s_i, p_i, q_i = np.nonzero(kap)
            rows.append(r + s_i)
            cols.append(np.array([pair[k, int(p_), int(q_)].index for p_, q_ in zip(p_i, q_i)], dtype=np.int64))
            vals.append(-np.ones(len(s_i)))
            rhs.append(-np.ones(nb))
        r += nb
    ge_rows = r
    if use_cvar:   # r_s + eta - sum_i ell_i e_si >= 0
        rcol = base + ne + 1 + np.arange(S)
        etacol = base + ne
        ps, pi = np.nonzero(poss)
        wsel = inst.ell[pi] != 0
        rows += [r + np.arange(S), r + np.arange(S), r + ps[wsel]]
        cols += [rcol, np.full(S, etacol), ecol[ps[wsel], pi[wsel]]]
        vals += [np.ones(S), np.ones(S), -inst.ell[pi[wsel]]]
        rhs.append(np.zeros(S))
        r += S
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(r, m.NumVars))
    m.addMConstr(A, None, GRB.GREATER_EQUAL, np.concatenate(rhs))
    f = None
    if cap_beta is not None:
        cnt = poss.sum(axis=1)
        risky = np.nonzero(cnt >= inst.e_sys)[0]
        big = inst.n - inst.e_sys + 1
        f = m.addVars(len(risky), vtype=GRB.BINARY, name="f")
        m.update()
        vlist = m.getVars()
        for t_, s in enumerate(risky):
            cols_s = ecol[s, poss[s]]
            m.addConstr(gp.LinExpr([1.0] * len(cols_s), [vlist[c] for c in cols_s]) - big * f[t_]
                        <= inst.e_sys - 1, name=f"sys[{s}]")
        m.addConstr(f.sum() <= math.floor(cap_beta * S + 1e-9), name="cap")
    build = time.time() - t0
    if start is not None:
        a0, y0 = start
        set_start(inst, x, y, u, pair, mode, a0, y0)
        e0 = km.propagate(inst, a0, y0, sc)
        evars = m.getVars()
        for (s, i) in zip(*np.nonzero(poss)):
            evars[ecol[s, i]].Start = float(e0[s, i])
        if per_scenario_reviews:
            for s in range(S):
                for i in range(n):
                    ys[s, i].Start = float(bool(y0[i]))
        if f is not None:
            cnt0 = e0.sum(axis=1)
            for t_, s in enumerate(risky):
                f[t_].Start = float(cnt0[s] >= inst.e_sys)
    apply_params(m, opts, time_limit - build)
    tracker = Tracker(float(opts.get("mip_gap", 0.01)), label, progress_path)
    m.optimize(tracker)
    extra = dict(pair_mode=mode, ge_rows=ge_rows)
    if per_scenario_reviews:
        return _result(m, label, t0, build, tracker, inst, None, None, None, extra)
    return _result(m, label, t0, build, tracker, inst, x, y, u, extra)


# ------------------------------------------------------- Benders, fixed workflow
def solve_benders(inst: km.Instance, sc: km.Scenarios, lam: float, alpha: float, env: gp.Env, opts: dict,
                  time_limit: float, start=None, warm_rounds: int = 60, warm_share: float = 0.25,
                  node_rounds: int = 20, node_every: int = 25, label: str = "BD", progress_path=None) -> dict:
    """Branch-and-Benders-cut with the closed-form cuts of kp_cuts.CutEngine:
    cutting planes on the master's linear relaxation, lazy cuts at integer
    solutions, user cuts at fractional nodes, and incumbents completed with
    their true scenario losses. Fixed workflows only; no cap."""
    if not inst.fixed:
        raise ValueError("Benders is implemented for fixed workflows")
    t0 = time.time()
    S, n, M = sc.S, inst.n, inst.M
    mode = pair_mode_for(sc.q)
    eng = kc.CutEngine(inst, sc, mode=mode)
    m = gp.Model(label, env=env)
    x, y, u, pair = first_stage(m, inst, mode)
    use_cvar = lam > 0
    th = m.addVars(S, lb=0.0, obj=(1 - lam) / S, name="theta")
    if use_cvar:
        eta = m.addVar(lb=0.0, ub=float(inst.ell.sum()), obj=lam, name="eta")
        tv = m.addVars(S, lb=0.0, obj=lam / ((1 - alpha) * S), name="r")
        m.addConstrs((tv[s] >= th[s] - eta for s in range(S)), name="cvar")
    m.update()
    xl = [[x[i, k] for k in range(M)] for i in range(n)]
    yl = [y[i] for i in range(n)]
    pkeys = list(range(inst.nA)) if mode == "z" else [(a, p_, q_) for a in range(inst.nA)
                                                      for p_ in range(M) for q_ in range(M)]
    pl = [pair[k] for k in pkeys]
    thl = [th[s] for s in range(S)]
    stats = dict(cuts=0, callbacks=0, warm_rounds=0, warm_cuts=0)

    def pvals(getter):
        v = np.array(getter(pl))
        return dict(zv=v) if mode == "z" else dict(wv=v.reshape(inst.nA, M, M))

    def expr(s, parent, Wt):
        xt, yt, pt, const = eng.cut(s, parent, Wt)
        ex = gp.LinExpr(const)
        if xt:
            ex.addTerms([c for (_, _, c) in xt], [xl[i][k] for (i, k, _) in xt])
        if yt:
            ex.addTerms([c for (_, c) in yt], [yl[i] for (i, _) in yt])
        if pt:
            if mode == "z":
                ex.addTerms([c for (_, c) in pt], [pair[k] for (k, _) in pt])
            else:
                ex.addTerms([c for (*_, c) in pt], [pair[k, a_, b_] for (k, a_, b_, _) in pt])
        return ex

    for i in range(n):
        yl[i].VType = GRB.CONTINUOUS
        for k in range(M):
            xl[i][k].VType = GRB.CONTINUOUS
    m.Params.Method = 1
    apply_params(m, opts, time_limit)
    lp_bound = float("nan")
    for rnd in range(warm_rounds):
        if time.time() - t0 > warm_share * time_limit:
            break
        m.Params.TimeLimit = max(1.0, warm_share * time_limit - (time.time() - t0))
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            break
        lp_bound = float(m.ObjVal)
        xv = np.array([[xl[i][k].X for k in range(M)] for i in range(n)])
        yv = np.array([yl[i].X for i in range(n)])
        tvv = np.array([thl[s].X for s in range(S)])
        _, loss, parent, Wt = eng.evaluate(xv, yv, **pvals(lambda vs: [v.X for v in vs]))
        viol = np.nonzero(loss > tvv + 1e-6 * (1.0 + np.abs(loss)))[0]
        stats["warm_rounds"] = rnd + 1
        if len(viol) == 0:
            break
        for s in viol:
            m.addConstr(thl[s] >= expr(int(s), parent, Wt))
        stats["warm_cuts"] += len(viol)
    for i in range(n):
        yl[i].VType = GRB.BINARY
        for k in range(M):
            xl[i][k].VType = GRB.BINARY
    m.Params.Method = -1
    m.Params.LazyConstraints = 1
    m.Params.PreCrush = 1
    m.update()
    kq = min(S - 1, int(np.floor((1.0 - alpha) * S + 1e-9)))
    pend = dict(sol=None, best=float("inf"), root_rounds=0, repairs=0, node_cuts=0)
    xflat = [xl[i][k] for i in range(n) for k in range(M)]
    tl = [tv[s] for s in range(S)] if use_cvar else []

    def value(loss):
        return (1 - lam) * float(loss.mean()) + (lam * km.cvar(loss, alpha) if use_cvar else 0.0)

    def stash(a, yv, loss):
        val = value(loss)
        if val < pend["best"] - 1e-9:
            pend["best"] = val
            pv = kc.pair_values(inst, a, mode)
            vars_ = xflat + yl + pl + thl
            vals_ = list(np.eye(M)[a].ravel()) + [float(v) for v in yv] + list(np.ravel(pv)) + list(loss)
            if use_cvar:
                eta_v = float(np.sort(loss)[::-1][kq])
                vars_ = vars_ + [eta] + tl
                vals_ = vals_ + [eta_v] + list(np.maximum(loss - eta_v, 0.0))
            pend["sol"] = (vars_, vals_)

    if start is not None:
        a0, y0 = start
        set_start(inst, x, y, u, pair, mode, a0, y0)
        l0 = km.propagate(inst, a0, y0, sc) @ inst.ell
        for s in range(S):
            thl[s].Start = float(l0[s])
        pend["best"] = value(l0)
    tracker = Tracker(float(opts.get("mip_gap", 0.01)), label, progress_path)

    def cb(model, where):
        tracker(model, where)
        if where == GRB.Callback.MIPSOL:
            stats["callbacks"] += 1
            xv = np.round(np.array(model.cbGetSolution(xflat)).reshape(n, M))
            yv = np.round(np.array(model.cbGetSolution(yl)))
            tvv = np.array(model.cbGetSolution(thl))
            a = xv.argmax(axis=1)
            pv = kc.pair_values(inst, a, mode)
            _, loss, parent, Wt = eng.evaluate(xv, yv, **(dict(zv=pv) if mode == "z" else dict(wv=pv)))
            viol = np.nonzero(loss > tvv + 1e-6 * (1.0 + np.abs(loss)))[0]
            for s in viol:
                model.cbLazy(thl[s] >= expr(int(s), parent, Wt))
            stats["cuts"] += len(viol)
            if len(viol):
                stash(a, yv, loss)
        elif where == GRB.Callback.MIPNODE:
            if pend["sol"] is not None:
                vars_, vals_ = pend["sol"]
                model.cbSetSolution(vars_, vals_)
                model.cbUseSolution()
                pend["sol"] = None
                pend["repairs"] += 1
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            nd = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
            if nd == 0:
                if pend["root_rounds"] >= node_rounds:
                    return
                pend["root_rounds"] += 1
            elif nd % node_every != 0:
                return
            xv = np.array(model.cbGetNodeRel(xflat)).reshape(n, M)
            yv = np.array(model.cbGetNodeRel(yl))
            tvv = np.array(model.cbGetNodeRel(thl))
            _, loss, parent, Wt = eng.evaluate(xv, yv, **pvals(model.cbGetNodeRel))
            viol = np.nonzero(loss > tvv + 1e-3 * (1.0 + np.abs(loss)))[0]
            for s in viol:
                model.cbCut(thl[s] >= expr(int(s), parent, Wt))
            pend["node_cuts"] += len(viol)

    m.Params.TimeLimit = max(1.0, time_limit - (time.time() - t0))
    m.optimize(cb)
    extra = dict(pair_mode=mode, lp_bound=lp_bound, warm_rounds=stats["warm_rounds"], warm_cuts=stats["warm_cuts"],
                 cuts=stats["cuts"] + stats["warm_cuts"] + pend["node_cuts"], callbacks=stats["callbacks"],
                 node_cuts=pend["node_cuts"], repairs=pend["repairs"])
    return _result(m, label, t0, 0.0, tracker, inst, x, y, u, extra)


# ------------------------------------------------------------------ baselines
def solve_route(inst: km.Instance, env: gp.Env, threads: int = 2, use=None):
    """Routing with spot checks: models minimize the expected number of task
    errors under the budget and quotas; reviews then maximize the expected
    loss of own errors caught, p ell dbar, under the hours. A template keeps
    its default decomposition (first alternative of every OR group)."""
    n, M = inst.n, inst.M
    use = inst.default_use() if use is None else np.asarray(use).astype(bool)
    m = gp.Model("route", env=env)
    m.Params.Threads = threads
    x = m.addVars(n, M, vtype=GRB.BINARY)
    m.addConstrs(x.sum(i, "*") == int(use[i]) for i in range(n))
    cost = gp.quicksum(inst.c[i, k] * x[i, k] for i in range(n) for k in range(M))
    m.addConstr(cost <= inst.B)
    for k in range(len(inst.Q)):
        if np.isfinite(inst.Q[k]):
            mods = [mm for mm in range(M) if km.MODEL_PROVIDER[mm] == k]
            m.addConstr(gp.quicksum(inst.c[i, mm] * x[i, mm] for i in range(n) for mm in mods) <= inst.Q[k])
    m.setObjective(gp.quicksum(inst.p[i, k] * x[i, k] for i in range(n) for k in range(M)) + 1e-6 * cost)
    m.optimize()
    a = np.full(n, -1, dtype=int)
    for i in range(n):
        if use[i]:
            a[i] = max(range(M), key=lambda k: x[i, k].X)
    pa = np.where(use, inst.p[np.arange(n), np.maximum(a, 0)], 0.0)
    k2 = gp.Model("spot", env=env)
    k2.Params.Threads = threads
    y = k2.addVars(n, vtype=GRB.BINARY)
    for i in range(n):
        if not use[i]:
            y[i].UB = 0.0
    k2.addConstr(gp.quicksum(inst.h[i] * y[i] for i in range(n)) <= inst.H)
    k2.setObjective(gp.quicksum(pa[i] * inst.ell[i] * inst.dbar[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
    k2.optimize()
    yv = np.array([y[i].X > 0.5 for i in range(n)])
    return a, yv


def solve_ev(inst: km.Instance, q, env: gp.Env, opts: dict, time_limit: float = 60.0, start=None):
    """Mean-value plan: every random event replaced by its mean. e_i is the
    expected error level of output i, a review lowers it by dbar_i, and a
    handoff passes it on with the mean probability of its model pair. With
    q_same >= q_diff and dbar = 1 these are the manuscript's rows."""
    n, M = inst.n, inst.M
    q = np.asarray(q, float)
    mode = pair_mode_for(q)
    m = gp.Model("EV", env=env)
    x, y, u, pair = first_stage(m, inst, mode)
    e = m.addVars(n, lb=0.0, ub=1.0, obj=list(inst.ell), name="e")
    for i in range(n):
        m.addConstr(e[i] >= gp.quicksum(inst.p[i, k] * x[i, k] for k in range(M)) - inst.dbar[i] * y[i])
    for a, (j, i) in enumerate(inst.arcs):
        if mode == "z":
            qs, qd = km.same_diff(q)
            if qd <= qs:
                m.addConstr(e[i] >= qd * e[j] - inst.dbar[i] * y[i])
            else:
                m.addConstr(e[i] >= qd * e[j] - pair[a] - inst.dbar[i] * y[i])
            m.addConstr(e[i] >= qs * e[j] - (1.0 - pair[a]) - inst.dbar[i] * y[i])
        else:
            for p_ in range(M):
                for q_ in range(M):
                    m.addConstr(e[i] >= q[p_, q_] * e[j] - (1.0 - pair[a, p_, q_]) - inst.dbar[i] * y[i])
    if start is not None:
        set_start(inst, x, y, u, pair, mode, *start)
    o = dict(opts)
    o.pop("log_file", None)
    apply_params(m, o, time_limit)
    m.Params.MIPGap = 1e-4
    m.optimize()
    a_, y_ = extract(inst, x, y, u)
    return a_, y_, dict(status=int(m.Status), gap=float(m.MIPGap) if m.SolCount else float("nan"),
                        time=float(m.Runtime))
