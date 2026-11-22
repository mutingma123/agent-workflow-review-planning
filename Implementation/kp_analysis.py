"""Analysis of the campaign's saved results: tables, paired tests, the
expectations fixed before the run, the scaling fit, and figures.

    python kp_analysis.py [--runs ../runs] [--tables ../tables] [--figures ../figures] [--records ../Records]

Reads only runs/<block>/rep_<r>/<cell>.json with status ok; safe to run
while the campaign runs. Writes tables/<block>_summary.csv,
tables/paired.csv, tables/expectations.csv, tables/x7_scaling.csv,
Records/Expectations.md and figures/*.pdf.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = ["p_systemic", "exp_loss", "cvar", "risk_obj", "wrong_per_run", "p_clean", "eff_models",
           "same_model_arc_share", "review_share_last_stage", "n_reviews", "n_used", "n_used_handoffs",
           "in_sample_risk_obj"]
SOLVER = ["solver_gap", "solver_time", "solver_status", "solver_time_to_target", "solver_bound", "solver_obj",
          "solver_time_limit"]


def load(runs: Path) -> tuple[pd.DataFrame, list]:
    """One row per (block, rep, cell, plan); and the list of job records."""
    rows, recs = [], []
    for f in sorted(runs.glob("X*/rep_*/*.json")):
        if f.name.endswith(".progress.json"):
            continue
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        recs.append(r)
        if r.get("status") != "ok":
            continue
        res = r.get("result", {})
        for plan, d in res.get("plans", {}).items():
            row = dict(block=r["block"], rep=r["rep"], cell=r["cell"], plan=plan, set=r.get("set"))
            for k in ("rho", "tau", "q_diff", "copula", "dbar", "tie", "ell_mode", "kind", "L", "W", "S", "method"):
                if k in r:
                    row[k] = r[k]
            row["no_solution"] = bool(d.get("no_solution", False))
            for k in METRICS + SOLVER:
                if k in d:
                    row[k] = d[k]
            # A level first reached at the final bound update gets no callback time: a solve that
            # ended optimal (status 2) at gap g had reached every level at or above g by its end.
            done = d.get("solver_status") == 2 and isinstance(d.get("solver_gap"), (int, float))
            t_end = (d.get("solver_time") or 0.0) - (d.get("solver_build_time") or 0.0)
            if done and "solver_time" in d and row.get("solver_time_to_target") is None:
                row["solver_time_to_target"] = t_end
            ttg = d.get("solver_time_to_gap") or {}
            for g, t in ttg.items():
                if t is None and done and d["solver_gap"] <= float(g) + 1e-12:
                    t = t_end
                row[f"t_gap_{g}"] = t
            rows.append(row)
    return pd.DataFrame(rows), recs


def summary(df: pd.DataFrame) -> pd.DataFrame:
    keep = [m for m in METRICS + ["solver_gap", "solver_time"] if m in df]
    g = df[~df["no_solution"]].groupby(["block", "cell", "plan"])
    out = g[keep].mean()
    out["n"] = g.size()
    if "solver_gap" in df:
        out["share_gap_le_5pct"] = g["solver_gap"].apply(lambda s: float(np.mean(s.dropna() <= 0.05)) if s.notna().any() else np.nan)
    return out.reset_index()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def paired(df: pd.DataFrame, block: str, base: str, other: str, metric: str = "p_systemic") -> pd.DataFrame:
    """Per cell: mean of (other - base) over replications, t interval, p value."""
    d = df[(df.block == block) & df.plan.isin([base, other]) & ~df.no_solution]
    piv = d.pivot_table(index=["cell", "rep"], columns="plan", values=metric)
    if base not in piv or other not in piv:
        return pd.DataFrame()
    diff = (piv[other] - piv[base]).dropna()
    rows = []
    for cell, s in diff.groupby(level=0):
        n = len(s)
        mean = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
        p = float(stats.ttest_1samp(s, 0.0).pvalue) if n > 1 and sd > 0 else float("nan")
        rows.append(dict(block=block, cell=cell, base=base, other=other, metric=metric, n=n, mean_diff=mean,
                         ci_low=mean - half, ci_high=mean + half, half_width=half, p=p))
    return pd.DataFrame(rows)


def tost(s: pd.Series, margin: float) -> float:
    """p value of two one-sided tests of |mean| < margin (equivalence)."""
    s = s.dropna()
    n = len(s)
    if n < 2 or s.std(ddof=1) == 0:
        return float("nan")
    se = s.std(ddof=1) / math.sqrt(n)
    p_low = 1 - stats.t.cdf((s.mean() + margin) / se, n - 1)
    p_high = stats.t.cdf((s.mean() - margin) / se, n - 1)
    return float(max(p_low, p_high))


def tobit_loglinear(y: np.ndarray, X: np.ndarray, cens: np.ndarray):
    """Right-censored normal regression of y on X (y censored at its value
    where cens is True). Returns coefficients, standard errors and sigma."""
    def nll(theta):
        b, ls = theta[:-1], theta[-1]
        sg = math.exp(ls)
        mu = X @ b
        z = (y - mu) / sg
        ll = np.where(cens, stats.norm.logsf(z), stats.norm.logpdf(z) - ls)
        return -float(ll.sum())
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    th0 = np.append(b0, math.log(max(1e-3, float(np.std(y - X @ b0)))))
    res = optimize.minimize(nll, th0, method="BFGS")
    cov = res.hess_inv if hasattr(res, "hess_inv") else np.full((len(th0), len(th0)), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return res.x[:-1], se[:-1], math.exp(res.x[-1]), bool(res.success)


def x7_scaling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df[(df.block == "X7") & df.plan.isin(["EF", "BD"])].copy()
    if d.empty:
        return d, {}
    n = d["L"] * d["W"]
    T = d["solver_time_to_target"]
    limit = d.get("solver_time_limit", pd.Series(np.nan, index=d.index)).fillna(d["solver_time"])
    cens = T.isna()
    d["n"] = n
    d["log_T"] = np.log(np.where(cens, limit.clip(lower=1.0), T.clip(lower=0.1)))
    d["censored"] = cens
    fits = {}
    for meth, g in d.groupby("plan"):
        if len(g) < 4 or g["n"].nunique() < 2:
            continue
        X = np.column_stack([np.ones(len(g)), g["n"].to_numpy(float), np.log(g["S"].to_numpy(float))])
        b, se, sg, ok = tobit_loglinear(g["log_T"].to_numpy(float), X, g["censored"].to_numpy(bool))
        fits[meth] = dict(b0=b[0], b1=b[1], b2=b[2], se_b1=se[1], se_b2=se[2], sigma=sg, converged=ok,
                          n_obs=len(g), censored_share=float(g["censored"].mean()))
    return d, fits


def expectations(df: pd.DataFrame, x7fits: dict) -> pd.DataFrame:
    """The deck's expectations X1 to X7 with the rule each is judged by."""
    rows = []

    def add(block, text, rule, value, passed):
        # passed: True, False, None (the data cannot judge the rule) or "tie" (no change to judge)
        outcome = ("not enough data" if passed is None else "no change" if passed == "tie"
                   else "pass" if passed else "fail")
        rows.append(dict(block=block, expectation=text, rule=rule, value=value, outcome=outcome))

    def mean(block, plan, metric, **cond):
        if metric not in df.columns or any(k not in df.columns for k in cond):
            return float("nan")                              # a block with no records yet
        d = df[(df.block == block) & (df.plan == plan) & ~df.no_solution]
        for k, v in cond.items():
            d = d[d[k] == v]
        return float(d[metric].mean()) if len(d) else float("nan")

    pr = paired(df, "X1", "ROUTE", "RA")
    base = pr[pr.cell == "rho0.3_tau0.15"] if len(pr) else pr
    if len(base):
        hw = 100 * float(base.half_width.iloc[0])
        add("X1", "Base-cell 95% half-width of RA minus ROUTE systemic failure falls to about 0.67 points",
            "half-width at most 1.0 point", f"{hw:.2f} points (n={int(base.n.iloc[0])})",
            None if math.isnan(hw) else hw <= 1.0)
    else:
        add("X1", "Base-cell half-width", "half-width at most 1.0 point", "", None)
    x1 = df[(df.block == "X1") & df.plan.isin(["RA", "IND"]) & ~df.no_solution]
    if len(x1):
        piv = x1.pivot_table(index=["cell", "rep"], columns="plan", values="p_systemic")
        if "IND" in piv:
            ps = [(cell, tost((s["IND"] - s["RA"]).dropna(), 0.01)) for cell, s in piv.groupby(level=0)]
            ps = [(c, p) for c, p in ps if not math.isnan(p)]
            ok = all(p < 0.05 for _, p in ps) if ps else None
            add("X1", "Planning under independent errors is equivalent to planning under correlation (H2)",
                "TOST p < 0.05 at a 1-point margin in every cell with rho > 0",
                "; ".join(f"{c}: p={p:.3f}" for c, p in ps), ok)
    for cop in ("gaussian", "t"):
        lo = mean("X2", "RA-PAIR", "eff_models", copula=cop, q_diff=min(df.get("q_diff", pd.Series([0.45])).dropna(), default=0.45))
        hi = mean("X2", "RA-PAIR", "eff_models", copula=cop, q_diff=max(df.get("q_diff", pd.Series([0.95])).dropna(), default=0.95))
        add("X2", f"RA-PAIR mixes models less when different models pass more errors ({cop})",
            "mean effective models at the highest q_diff below that at the lowest", f"{hi:.2f} vs {lo:.2f}",
            None if math.isnan(lo) or math.isnan(hi) else hi < lo)
    adv = {}
    for cop in ("gaussian", "t"):
        adv[cop] = mean("X2", "ROUTE", "p_systemic", copula=cop) - mean("X2", "RA-PAIR", "p_systemic", copula=cop)
    add("X2", "RA-PAIR's advantage over ROUTE grows under the t copula",
        "mean ROUTE minus RA-PAIR systemic failure larger under t", f"t {adv['t']:.4f} vs gaussian {adv['gaussian']:.4f}",
        None if any(math.isnan(v) for v in adv.values()) else adv["t"] > adv["gaussian"])
    e_lo = mean("X3", "RA-PAIR", "eff_models", dbar=0.6)
    e_hi = mean("X3", "RA-PAIR", "eff_models", dbar=1.0)
    add("X3", "RA-PAIR uses more models as review success falls", "mean effective models at dbar 0.6 above dbar 1.0",
        f"{e_lo:.2f} vs {e_hi:.2f}", None if math.isnan(e_lo) or math.isnan(e_hi) else e_lo > e_hi)
    r_del = mean("X4", "RA-PAIR", "review_share_last_stage", ell_mode="deliverables")
    r_all = mean("X4", "RA-PAIR", "review_share_last_stage", ell_mode="all")
    add("X4", "A larger share of reviews sits in the last stage when only deliverables count",
        "mean last-stage review share higher under deliverable loss", f"{r_del:.3f} vs {r_all:.3f}",
        None if math.isnan(r_del) or math.isnan(r_all) else r_del > r_all)
    caps = sorted({p for p in df[df.block == "X5"].plan.unique() if p.startswith("RA-CAP")},
                  key=lambda p: float(p.replace("RA-CAP", "")))
    if caps:
        w = [mean("X5", p, "wrong_per_run") for p in caps]
        add("X5", "Expected wrong outputs rise as the cap tightens", "mean wrong outputs per run decrease in beta",
            "; ".join(f"{p}: {v:.3f}" for p, v in zip(caps, w)),
            None if any(math.isnan(v) for v in w) else "tie" if max(w) - min(w) < 1e-12
            else all(w[i] >= w[i + 1] - 1e-12 for i in range(len(w) - 1)))
    for kind in ("tree", "random"):
        h0 = mean("X6", "RA-PAIR", "n_used_handoffs", kind=kind, rho=0.0)
        h6 = mean("X6", "RA-PAIR", "n_used_handoffs", kind=kind, rho=0.6)
        add("X6", f"Chosen decompositions have fewer handoffs as rho rises ({kind})",
            "mean used handoffs at rho 0.6 below rho 0", f"{h6:.2f} vs {h0:.2f}",
            None if math.isnan(h0) or math.isnan(h6) else h6 < h0)
    for meth, f in x7fits.items():
        lo = f["b1"] - 1.96 * f["se_b1"]
        add("X7", f"Time to a 1% gap grows exponentially in tasks ({meth})",
            "censored-regression slope on tasks with 95% interval above 0",
            f"b1={f['b1']:.4f} (se {f['se_b1']:.4f}), censored {f['censored_share']:.0%}, "
            f"{'converged' if f['converged'] else 'not converged'}",
            # a fit that did not converge, or rests on almost only censored times, cannot judge the slope
            None if math.isnan(lo) or not f["converged"] or f["censored_share"] > 0.9 else lo > 0)
    return pd.DataFrame(rows)


def figures(df: pd.DataFrame, x7: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    d = df[(df.block == "X1") & ~df.no_solution]
    if len(d):
        cells = sorted(d.cell.unique())
        plans = [p for p in ["ROUTE", "EV", "IND", "RN", "RA"] if p in d.plan.unique()]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        w = 0.8 / max(1, len(plans))
        for k, p in enumerate(plans):
            m = d[d.plan == p].groupby("cell")["p_systemic"].mean().reindex(cells)
            ax.bar(np.arange(len(cells)) + k * w, 100 * m.values, w, label=p)
        ax.set_xticks(np.arange(len(cells)) + 0.4 - w / 2)
        ax.set_xticklabels([c.replace("_", "\n") for c in cells], fontsize=8)
        ax.set_ylabel("systemic failure (% of runs)")
        ax.legend(ncol=len(plans), fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "x1_systemic.pdf")
        plt.close(fig)
    if len(x7):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for (meth, S), g in x7.groupby(["plan", "S"]):
            m = g.groupby("n")["log_T"].mean()
            ax.plot(m.index, np.exp(m.values), marker="o", label=f"{meth}, {S} scenarios")
        ax.set_yscale("log")
        ax.set_xlabel("tasks")
        ax.set_ylabel("seconds to a 1% gap (limit if censored)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "x7_scaling.pdf")
        plt.close(fig)


def write_all(runs: Path, tables: Path, figs: Path | None, records: Path | None) -> dict:
    df, recs = load(runs)
    tables.mkdir(parents=True, exist_ok=True)
    status = pd.Series([r.get("status") for r in recs]).value_counts().to_dict()
    if df.empty:
        return dict(rows=0, status=status)
    summ = summary(df)
    for b, g in summ.groupby("block"):
        g.to_csv(tables / f"{b}_summary.csv", index=False)
    comps = [("X1", "ROUTE", "RA"), ("X1", "EV", "RA"), ("X1", "IND", "RA"), ("X1", "RN", "RA"),
             ("X2", "ROUTE", "RA-PAIR"), ("X2", "RA", "RA-PAIR"), ("X3", "ROUTE", "RA-PAIR"), ("X3", "RA", "RA-PAIR"),
             ("X4", "ROUTE", "RA-PAIR"), ("X4", "RA", "RA-PAIR"), ("X6", "ROUTE", "RA-PAIR")]
    pr = pd.concat([paired(df, *c) for c in comps], ignore_index=True) if comps else pd.DataFrame()
    if len(pr):
        pr["p_holm"] = np.nan
        for b, idx in pr.groupby("block").groups.items():
            ok = pr.loc[idx, "p"].notna()
            pr.loc[idx[ok.values], "p_holm"] = holm(pr.loc[idx[ok.values], "p"].to_numpy())
        pr.to_csv(tables / "paired.csv", index=False)
    x7, fits = x7_scaling(df)
    if len(x7):
        x7.to_csv(tables / "x7_scaling.csv", index=False)
    exp = expectations(df, fits)
    exp.to_csv(tables / "expectations.csv", index=False)
    if records is not None:
        records.mkdir(parents=True, exist_ok=True)
        lines = ["# Expectations fixed before the run, and their outcomes", "",
                 f"Computed by `Implementation/kp_analysis.py` from {len(recs)} job records ({status}).", "",
                 "| Block | Expectation | Rule | Value | Outcome |", "| --- | --- | --- | --- | --- |"]
        for _, r in exp.iterrows():
            lines.append(f"| {r.block} | {r.expectation} | {r.rule} | {r.value} | {r.outcome} |")
        (records / "Expectations.md").write_text("\n".join(lines) + "\n")
    if figs is not None:
        figures(df, x7, figs)
    return dict(rows=len(df), status=status, expectations=len(exp), x7_fits=fits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tables", default=str(ROOT / "tables"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--records", default=str(ROOT / "Records"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    out = write_all(Path(a.runs), Path(a.tables), None if a.no_figures else Path(a.figures), Path(a.records))
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
