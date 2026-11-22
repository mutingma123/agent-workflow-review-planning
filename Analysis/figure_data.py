"""Data files for the pgfplots and TikZ figures of the manuscript, computed
from the 698 job records and the X1 rebuild. Each file is a whitespace table
with a header row, read by \\addplot table or \\pgfplotstableread.

    python figure_data.py      # writes ../Manuscript/figures/data/*.dat
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
OUT = HERE.parent / "Manuscript" / "figures" / "data"
sys.path.insert(0, str(RUN / "Implementation"))
import kp_analysis as ka  # noqa: E402
import kp_blocks as kb  # noqa: E402


def tci(s):
    s = pd.Series(s).dropna()
    h = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
    return float(s.mean()), float(h)


def write(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, sep=" ", index=False, float_format="%.5g", na_rep="nan")
    print("wrote", name)


def x1(df):
    """Systemic failure (percent) by plan, one file per review share."""
    d = df[df.block == "X1"]
    for tau in (0.05, 0.15, 0.30):
        rows = []
        for j, rho in enumerate((0.0, 0.3, 0.6)):
            row = dict(x=j, rho=rho)
            for p in ("ROUTE", "EV", "RN", "IND", "RA"):
                s = d[(d.plan == p) & (d.cell == f"rho{rho:.1f}_tau{tau:.2f}")].p_systemic * 100
                row[p], row[p + "h"] = tci(s) if len(s) else (np.nan, np.nan)
            rows.append(row)
        write(pd.DataFrame(rows), f"x1_tau{int(round(tau * 100)):03d}.dat")


def plans():
    """Nodes and handoffs of the X1 workflow of replication 1 (base cell) with
    the routing and the risk-averse plans; task numbers are fixed."""
    rec = json.loads((RUN / "runs" / "X1" / "rep_001" / "rho0.3_tau0.15.json").read_text())
    c = kb.config()
    inst = kb._layered(c, 1, tau=0.15)
    W = c["W"]
    P = rec["result"]["plans"]
    nodes = pd.DataFrame(dict(id=np.arange(1, inst.n + 1), stage=inst.layer + 1,
                              row=W - 1 - (np.arange(inst.n) % W)))
    edges = pd.DataFrame(dict(tail=inst.arcs[:, 0] + 1, head=inst.arcs[:, 1] + 1))
    for name in ("ROUTE", "RA"):
        a, y = np.array(P[name]["a"]), np.array(P[name]["y"], int)
        nodes[f"m{name}"] = a + 1
        nodes[f"r{name}"] = y
        edges[f"s{name}"] = (a[inst.arcs[:, 0]] == a[inst.arcs[:, 1]]).astype(int)
    write(nodes, "plans_nodes.dat")
    write(edges, "plans_edges.dat")
    meta = {name: dict(reviews=int(np.sum(P[name]["y"])), same=round(100 * P[name]["same_model_arc_share"]),
                       systemic=round(100 * P[name]["p_systemic"], 1)) for name in ("ROUTE", "RA")}
    (OUT / "plans_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def mechanism():
    m = pd.read_csv(HERE / "results" / "x1_mechanism.csv")
    m = m[m.rho == 0.3]
    for tau in (0.05, 0.15, 0.30):
        g = m[np.isclose(m.tau, tau)].groupby("plan").mean(numeric_only=True).reindex(
            ["ROUTE", "EV", "RN", "IND", "RA"])
        out = pd.DataFrame(dict(plan=g.index, own=g.wrong_own.values,
                                across=(g.wrong_inherited - g.wrong_inherited_same).values,
                                same=g.wrong_inherited_same.values, total=g.wrong.values))
        write(out, f"mech_tau{int(round(tau * 100)):03d}.dat")


def x2(df):
    d = df[df.block == "X2"]
    for cop in ("gaussian", "t"):
        rows = []
        for q in (0.45, 0.70, 0.95):
            row = dict(q=q)
            for p in ("ROUTE", "EV", "RA", "RA-PAIR"):
                s = d[(d.plan == p) & (d.cell == f"qdiff{q:.2f}_{cop}")]
                key = p.replace("-", "")
                row[key], row[key + "h"] = tci(s.p_systemic * 100)
                row[key + "same"] = 100 * s.same_model_arc_share.mean()
            rows.append(row)
        write(pd.DataFrame(rows), f"x2_{cop}.dat")


def x7(df):
    d = df[(df.block == "X7") & df.plan.isin(["EF", "BD"])].copy()
    d["n"] = (d.L * d.W).astype(int)
    h = d[d.n == 40].groupby(["S", "plan"])[["risk_obj", "in_sample_risk_obj"]].mean().unstack("plan")
    h.columns = [f"{p}{'test' if k == 'risk_obj' else 'train'}" for k, p in h.columns]
    write(h.reset_index(), "x7_n040.dat")


def main():
    df, _ = ka.load(RUN / "runs")
    x1(df)
    plans()
    mechanism()
    x2(df)
    x7(df)


if __name__ == "__main__":
    main()
