"""Rebuild every X1 job of the Mac Studio campaign from its seeds and derive
the quantities the job records do not store.

For each of the 41 replications and nine cells of block X1, the script
rebuilds the workflow and the 50,000-scenario test sample with the campaign's
implementation, checks that every
stored plan scores exactly as recorded, and writes

  results/x1_mechanism.csv  wrong outputs split into own and inherited errors,
                            per replication, cell and plan
  results/x1_crossed.csv    crossed plans that pair one plan's assignment with
                            another plan's reviews, scored on the same sample
  results/x1_check.json     the reproduction check

Run after the campaign completes:

    python Analysis/rebuild_x1.py
"""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "Implementation"))

import kp_blocks as kb  # noqa: E402
import kp_model as km   # noqa: E402

C = kb.config()
LAM, ALPHA = C["lam"], C["alpha"]
RHOS, TAUS = (0.0, 0.3, 0.6), (0.05, 0.15, 0.30)


def decompose(inst, a, y, sc):
    """Per plan: task errors made, wrong outputs released, split into own
    errors and inherited errors, and inherited errors that arrived over a
    handoff whose two tasks run on one model."""
    a = np.asarray(a, int)
    y = np.asarray(y).astype(bool)
    S, n = sc.S, inst.n
    own = sc.xi[:, np.arange(n), a]
    e = np.zeros((S, n), dtype=bool)
    via_same_out = np.zeros((S, n), dtype=bool)
    stopped = np.zeros((S, n), dtype=bool)
    for i in range(n):
        incoming = np.zeros(S, dtype=bool)
        via_same = np.zeros(S, dtype=bool)
        for k in inst.in_arcs[i]:
            j = inst.arcs[k, 0]
            arr = e[:, j] & (sc.U[:, k] < sc.q[a[j], a[i]])
            incoming |= arr
            if a[j] == a[i]:
                via_same |= arr
        would = own[:, i] | incoming
        if y[i]:
            stopped[:, i] = would & sc.delta[:, i]
            e[:, i] = would & ~sc.delta[:, i]
        else:
            e[:, i] = would
        via_same_out[:, i] = e[:, i] & via_same & ~own[:, i]
    loss = e.sum(1)
    inh = (e & ~own).sum(1)
    sysf = loss >= inst.e_sys
    return dict(
        task_errors=float(own.sum(1).mean()),
        wrong=float(loss.mean()),
        wrong_own=float((e & own).sum(1).mean()),
        wrong_inherited=float(inh.mean()),
        wrong_inherited_same=float(via_same_out.sum(1).mean()),
        stopped=float(stopped.sum(1).mean()),
        p_systemic=float(sysf.mean()),
        p_clean=float((loss == 0).mean()),
        inherited_in_systemic=float(inh[sysf].sum()),
        wrong_in_systemic=float(loss[sysf].sum()),
    )


def spot_reviews(inst, a):
    """Routing's review rule for a given assignment: the reviews that catch the
    most expected own errors within the review hours (a 0-1 knapsack)."""
    n = inst.n
    val = inst.p[np.arange(n), a] * inst.ell * inst.dbar
    res = milp(c=-val, integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=LinearConstraint(inst.h[None, :], -np.inf, inst.H))
    return np.asarray(np.round(res.x), bool)


def one_rep(rep):
    mech, crossed, check = [], [], dict(rep=rep, plans=0, mismatches=[], route_rule_matches=0, route_rule_total=0)
    for rho in RHOS:
        test = None
        for tau in TAUS:
            cell = f"rho{rho:.1f}_tau{tau:.2f}"
            rec = json.loads((RUN / "runs" / "X1" / f"rep_{rep:03d}" / f"{cell}.json").read_text())
            P = rec["result"]["plans"]
            inst = kb._layered(C, rep, tau=tau)
            if test is None:          # the sample depends on the workflow and rho, not on tau
                test = km.sample_scenarios(inst, C["test_S"], rho, C["gamma"], kb._q(C), 900000 + rep)
            for name, p in P.items():
                m = km.metrics(inst, p["a"], p["y"], test, LAM, ALPHA)
                check["plans"] += 1
                if m["p_systemic"] != p["p_systemic"] or abs(m["risk_obj"] - p["risk_obj"]) > 1e-9:
                    check["mismatches"].append([cell, name, m["p_systemic"], p["p_systemic"]])
                d = decompose(inst, p["a"], p["y"], test)
                d.update(rep=rep, cell=cell, rho=rho, tau=tau, plan=name, eff_models=p["eff_models"],
                         same_share=p["same_model_arc_share"], n_reviews=p["n_reviews"],
                         share_A=p["share_A"], share_B=p["share_B"], share_C=p["share_C"], share_D=p["share_D"])
                mech.append(d)
            aRA, yRA = np.array(P["RA"]["a"]), np.array(P["RA"]["y"], bool)
            aRT, yRT = np.array(P["ROUTE"]["a"]), np.array(P["ROUTE"]["y"], bool)
            check["route_rule_total"] += 1
            check["route_rule_matches"] += int(np.array_equal(spot_reviews(inst, aRT), yRT))
            ySpot = spot_reviews(inst, aRA)
            assert inst.h[ySpot].sum() <= inst.H + 1e-9 and inst.h[yRA].sum() <= inst.H + 1e-9
            for name, a, y in (("A_RA+R_rule", aRA, ySpot), ("A_ROUTE+R_RA", aRT, yRA)):
                m = km.metrics(inst, a, y, test, LAM, ALPHA)
                crossed.append(dict(rep=rep, cell=cell, rho=rho, tau=tau, plan=name, p_systemic=m["p_systemic"],
                                    exp_loss=m["exp_loss"], cvar=m["cvar"], risk_obj=m["risk_obj"],
                                    n_reviews=m["n_reviews"], same_share=m["same_model_arc_share"],
                                    route=P["ROUTE"]["p_systemic"], ra=P["RA"]["p_systemic"]))
    return mech, crossed, check


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    reps = range(1, C["reps"]["X1"] + 1)
    with Pool(6) as pool:
        out = pool.map(one_rep, reps)
    mech = [r for m, _, _ in out for r in m]
    crossed = [r for _, c, _ in out for r in c]
    checks = [c for _, _, c in out]
    summary = dict(replications=len(checks), plans_scored=sum(c["plans"] for c in checks),
                   mismatches=[m for c in checks for m in c["mismatches"]],
                   route_rule_reproduced=sum(c["route_rule_matches"] for c in checks),
                   route_rule_total=sum(c["route_rule_total"] for c in checks))
    (HERE / "results").mkdir(exist_ok=True)
    write_csv(mech, HERE / "results" / "x1_mechanism.csv")
    write_csv(crossed, HERE / "results" / "x1_crossed.csv")
    (HERE / "results" / "x1_check.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
