"""Every number the manuscript quotes, computed from the 698 job records of the
Mac Studio campaign and from the X1 rebuild (rebuild_x1.py).

Writes
  results/keynumbers.json        the quoted numbers, grouped by block
  ../Manuscript/tables/*.tex     the tables of the manuscript
  ../Manuscript/tables/keynumbers.tex  macros for the numbers that appear in
                                 the abstract, the introduction and the conclusion

The paired tests, the Holm adjustment (one family per block) and the two
one-sided tests are the campaign's own functions (kp_analysis.py), so the
tables agree with the campaign summaries.

    python key_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
TAB = HERE.parent / "Manuscript" / "tables"
sys.path.insert(0, str(RUN / "Implementation"))
import kp_analysis as ka  # noqa: E402

CELLS = [f"rho{r:.1f}_tau{t:.2f}" for r in (0.0, 0.3, 0.6) for t in (0.05, 0.15, 0.30)]
BASE = "rho0.3_tau0.15"
K = {}


def pct(x, d=2):
    return round(100 * float(x), d)


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def tint(s):
    s = pd.Series(s).dropna()
    n = len(s)
    m = float(s.mean())
    h = float(stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return m, m - h, m + h


def cellname(c):
    rho, tau = c.split("_")
    return f"{rho[3:]} & {tau[3:]}"


def load():
    df, _ = ka.load(RUN / "runs")
    df = df[~df.get("no_solution", False).fillna(False).astype(bool)] if "no_solution" in df else df
    return df


def proved(s, level=0.05):
    s = s.dropna()
    return int((s <= level + 1e-12).sum()), int(len(s))


def paired_block(df, block, comps):
    pr = pd.concat([ka.paired(df, block, b, o) for b, o in comps], ignore_index=True)
    pr["p_holm"] = np.nan
    ok = pr["p"].notna()
    pr.loc[ok, "p_holm"] = ka.holm(pr.loc[ok, "p"].to_numpy())
    return pr


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fmt_ci(m, lo, hi, d=2):
    return f"${100 * m:+.{d}f}$ [${100 * lo:+.{d}f}$, ${100 * hi:+.{d}f}$]"


# ------------------------------------------------------------------ campaign
def campaign(df):
    c = json.loads((RUN / "runs" / "campaign.json").read_text())
    done = json.loads((RUN / "runs" / "CAMPAIGN_DONE.json").read_text())
    recs = [json.loads(p.read_text()) for p in (RUN / "runs").glob("X*/rep_*/*.json")]
    t0 = pd.Timestamp(c["first_started"])
    t1 = pd.Timestamp(done["finished"])
    mip = df[df.solver_gap.notna() & (df.plan != "EV") & (df.block != "X7")]
    x7 = df[(df.block == "X7") & df.plan.isin(["EF", "BD"])]
    ev = df[(df.plan == "EV")]
    K["campaign"] = dict(
        jobs=done["total"], ok=done["ok"], failed=done["failed"], skipped=done["skipped"],
        solves=c["budget"]["solves"], workers=c["workers"], threads=c["threads"],
        time_limit_s=r(c["budget"]["time_limit"], 1), mip_gap=c["config"]["mip_gap"],
        x7_mip_gap=c["config"]["x7_mip_gap"], machine=c["machine"],
        wall_hours=r((t1 - t0).total_seconds() / 3600, 1),
        job_hours=r(sum(x["wall"] for x in recs) / 3600, 1),
        mip_x1_x6=len(mip), mip_x1_x6_proved5=int((mip.solver_gap <= 0.05 + 1e-12).sum()),
        x7_solves=len(x7), ev_solves=len(ev), ev_max_gap=float(ev.solver_gap.max()),
        mem_gb_per_job=r(c["mem_gb_per_job"], 1),
    )


# ------------------------------------------------------------------------ X1
def x1(df):
    d = df[df.block == "X1"]
    piv = d.pivot_table(index=["cell", "rep"], columns="plan", values="p_systemic")
    pr = paired_block(df, "X1", [("ROUTE", "RA"), ("EV", "RA"), ("IND", "RA"), ("RN", "RA")])
    g = d.groupby(["cell", "plan"])
    out = {}
    rows = []
    for c in CELLS:
        s = piv.loc[c]
        e = dict(cell=c)
        for p in ("ROUTE", "EV", "RN", "IND", "RA"):
            if p in s and s[p].notna().any():
                e[p] = pct(s[p].mean())
        q = pr[(pr.cell == c) & (pr.base == "ROUTE")].iloc[0]
        e["ra_route"] = [pct(q.mean_diff), pct(q.ci_low), pct(q.ci_high)]
        e["ra_route_holm"] = float(q.p_holm)
        e["ratio"] = float(s["RA"].mean() / s["ROUTE"].mean())
        e["ra_lower"] = int((s["RA"] < s["ROUTE"]).sum())
        e["n"] = int(s["RA"].notna().sum())
        for b in ("EV", "RN", "IND"):
            qq = pr[(pr.cell == c) & (pr.base == b)]
            if len(qq):
                qq = qq.iloc[0]
                e[f"ra_{b.lower()}"] = [100 * v for v in (qq.mean_diff, qq.ci_low, qq.ci_high)]
                e[f"ra_{b.lower()}_holm"] = float(qq.p_holm)
        if "IND" in s and s["IND"].notna().any():
            e["tost_p"] = ka.tost((s["IND"] - s["RA"]).dropna(), 0.01)
        ra = d[(d.cell == c) & (d.plan == "RA")]
        e["ra_proved"] = proved(ra.solver_gap)
        e["ra_mean_gap"] = pct(ra.solver_gap.mean(), 1)
        out[c] = e
        rows.append(e)
    # base cell, every plan
    base = {}
    for p in ("ROUTE", "EV", "RN", "IND", "RA"):
        x = d[(d.cell == BASE) & (d.plan == p)]
        base[p] = dict(p_systemic=pct(x.p_systemic.mean()), exp_loss=r(x.exp_loss.mean(), 2),
                       cvar=r(x.cvar.mean(), 2), risk_obj=float(x.risk_obj.mean()), p_clean=pct(x.p_clean.mean(), 1),
                       eff_models=r(x.eff_models.mean(), 2), same=pct(x.same_model_arc_share.mean(), 1),
                       n_reviews=r(x.n_reviews.mean(), 1), in_sample=float(x.in_sample_risk_obj.mean()),
                       proved=proved(x.solver_gap) if p not in ("ROUTE",) else None,
                       mean_gap=pct(x.solver_gap.mean(), 1) if x.solver_gap.notna().any() else None)
    base_risk = piv.loc[BASE].mean()
    base_ratio = base_risk["RA"] / base_risk["ROUTE"]
    hw = pr[(pr.cell == BASE) & (pr.base == "ROUTE")].iloc[0].half_width
    # attribution of the ROUTE to RA drop to the modeling elements
    att = {}
    for c in CELLS:
        m = piv.loc[c].mean()
        drop = m["ROUTE"] - m["RA"]
        att[c] = dict(handoffs=pct((m["ROUTE"] - m["EV"]) / drop, 1), scenarios=pct((m["EV"] - m["RN"]) / drop, 1),
                      tail=pct((m["RN"] - m["RA"]) / drop, 1))
    cert = {p: proved(d[d.plan == p].solver_gap) for p in ("RN", "IND", "RA")}
    # risk objective of IND against RA out of sample
    rp = d.pivot_table(index=["cell", "rep"], columns="plan", values="risk_obj").groupby(level=0).mean()
    ind_better = [c for c in CELLS if c in rp.index and not math.isnan(rp.loc[c].get("IND", float("nan")))
                  and rp.loc[c, "IND"] < rp.loc[c, "RA"]]
    diffs = {k: [out[c][k][0] for c in CELLS if k in out[c]] for k in ("ra_route", "ra_ev", "ra_rn", "ra_ind")}
    K["x1"] = dict(cells=out, base=base, attribution=att, certification=cert,
                   ra_route_range=[min(diffs["ra_route"]), max(diffs["ra_route"])],
                   ra_ev_range=[min(diffs["ra_ev"]), max(diffs["ra_ev"])],
                   ra_rn_range=[min(diffs["ra_rn"]), max(diffs["ra_rn"])],
                   ra_ind_range=[min(diffs["ra_ind"]), max(diffs["ra_ind"])],
                   ratio_range=[min(out[c]["ratio"] for c in CELLS), max(out[c]["ratio"] for c in CELLS)],
                   ra_lower_total=sum(out[c]["ra_lower"] for c in CELLS), jobs_total=sum(out[c]["n"] for c in CELLS),
                   half_width_base=pct(hw), base_relative_cut_pct=pct(1 - base_ratio, 1),
                   base_factor=r(1 / base_ratio, 1),
                   ra_ev_holm_max=max(out[c]["ra_ev_holm"] for c in CELLS),
                   ra_rn_significant=[c for c in CELLS if out[c]["ra_rn_holm"] < 0.05],
                   ra_ind_excludes_zero=[c for c in CELLS if "ra_ind" in out[c]
                                         and (out[c]["ra_ind"][1] > 0 or out[c]["ra_ind"][2] < 0)],
                   tost_max=max(out[c]["tost_p"] for c in CELLS if "tost_p" in out[c]),
                   ind_lower_risk_cells=ind_better,
                   holm_route_range=[min(out[c]["ra_route_holm"] for c in CELLS),
                                     max(out[c]["ra_route_holm"] for c in CELLS)])
    # tables
    lines = []
    for e in rows:
        ind = f"{e['IND']:.2f}" if "IND" in e else "n/a"
        lines.append(f"{cellname(e['cell'])} & {e['ROUTE']:.2f} & {e['EV']:.2f} & {e['RN']:.2f} & {ind} & "
                     f"{e['RA']:.2f} & ${e['ra_route'][0]:+.2f}$ [${e['ra_route'][1]:+.2f}$, ${e['ra_route'][2]:+.2f}$] & "
                     f"{e['ratio']:.2f} & {e['ra_lower']} of {e['n']}\\\\")
    (TAB / "x1_cells.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for e in rows:
        ind = (f"${e['ra_ind'][0]:+.2f}$ [${e['ra_ind'][1]:+.2f}$, ${e['ra_ind'][2]:+.2f}$] & {fmt_p(e['tost_p'])}"
               if "ra_ind" in e else "n/a & n/a")
        lines.append(f"{cellname(e['cell'])} & ${e['ra_ev'][0]:+.2f}$ [${e['ra_ev'][1]:+.2f}$, ${e['ra_ev'][2]:+.2f}$] & "
                     f"{fmt_p(e['ra_ev_holm'])} & ${e['ra_rn'][0]:+.2f}$ [${e['ra_rn'][1]:+.2f}$, ${e['ra_rn'][2]:+.2f}$] & "
                     f"{fmt_p(e['ra_rn_holm'])} & {ind}\\\\")
    (TAB / "x1_ablation.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for p in ("ROUTE", "EV", "RN", "IND", "RA"):
        b = base[p]
        lines.append(f"{p} & {b['p_systemic']:.2f} & {b['exp_loss']:.2f} & {b['cvar']:.2f} & "
                     f"{b['risk_obj']:.2f} & {b['in_sample']:.2f} & {b['p_clean']:.1f} & {b['eff_models']:.2f} & "
                     f"{b['same']:.1f} & {b['n_reviews']:.1f}\\\\")
    (TAB / "x1_base.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- mechanism
def mechanism():
    m = pd.read_csv(HERE / "results" / "x1_mechanism.csv")
    out = {}
    for c in CELLS:
        g = m[m.cell == c].groupby("plan")
        mean = g.mean(numeric_only=True)
        tot = g.sum(numeric_only=True)
        out[c] = {p: dict(task_errors=r(mean.loc[p, "task_errors"], 3), wrong=r(mean.loc[p, "wrong"], 3),
                          own=r(mean.loc[p, "wrong_own"], 3), inherited=r(mean.loc[p, "wrong_inherited"], 3),
                          inherited_same=r(mean.loc[p, "wrong_inherited_same"], 3),
                          inherited_share=pct(mean.loc[p, "wrong_inherited"] / mean.loc[p, "wrong"], 1),
                          inherited_share_systemic=pct(tot.loc[p, "inherited_in_systemic"]
                                                       / tot.loc[p, "wrong_in_systemic"], 1),
                          shares=[pct(mean.loc[p, f"share_{x}"], 1) for x in "ABCD"],
                          eff_models=r(mean.loc[p, "eff_models"], 2), same=pct(mean.loc[p, "same_share"], 1))
                  for p in mean.index}
    base = m[m.cell == BASE].groupby("plan").mean(numeric_only=True)
    ratios = base.loc["RA"] / base.loc["ROUTE"]
    K["mechanism"] = dict(cells=out, base=out[BASE],
                          ra_more_task_errors_pct=pct(ratios["task_errors"] - 1, 1),
                          wrong_output_cut_pct=pct(1 - ratios["wrong"], 1),
                          inherited_cut_pct=pct(1 - ratios["wrong_inherited"], 1),
                          inherited_same_cut_pct=pct(1 - ratios["wrong_inherited_same"], 1),
                          route_inherited_share_range=[min(out[c]["ROUTE"]["inherited_share"] for c in CELLS),
                                                       max(out[c]["ROUTE"]["inherited_share"] for c in CELLS)],
                          route_fewest_task_errors_cells=sum(
                              out[c]["ROUTE"]["task_errors"] == min(v["task_errors"] for v in out[c].values())
                              for c in CELLS),
                          route_most_systemic_cells=None)
    # per job: ROUTE makes fewer task errors and fails systemically more often than RA
    w = m.pivot_table(index=["rep", "cell"], columns="plan", values=["task_errors", "p_systemic"])
    both = (w["task_errors"]["ROUTE"] < w["task_errors"]["RA"]) & (w["p_systemic"]["ROUTE"] > w["p_systemic"]["RA"])
    K["mechanism"]["route_fewer_errors_more_failures_jobs"] = [int(both.sum()), int(len(both))]
    sysm = m.groupby(["cell", "plan"]).p_systemic.mean().unstack()
    K["mechanism"]["route_most_systemic_cells"] = int((sysm.idxmax(axis=1) == "ROUTE").sum())


# ------------------------------------------------------------------- crossed
def crossed():
    c = pd.read_csv(HERE / "results" / "x1_crossed.csv")
    out = {}
    lines = []
    for cell in CELLS:
        g = c[c.cell == cell]
        a = g[g.plan == "A_RA+R_rule"]
        b = g[g.plan == "A_ROUTE+R_RA"]
        route, ra = a.route.mean(), a.ra.mean()
        drop = route - ra
        e = dict(route=pct(route), ra=pct(ra), assign=pct(a.p_systemic.mean()), review=pct(b.p_systemic.mean()),
                 assign_share=pct((route - a.p_systemic.mean()) / drop, 1),
                 review_share=pct((route - b.p_systemic.mean()) / drop, 1),
                 assign_beats_ra=int((a.p_systemic.to_numpy() < a.ra.to_numpy()).sum()),
                 review_beats_ra=int((b.p_systemic.to_numpy() < b.ra.to_numpy()).sum()), n=len(a))
        out[cell] = e
        lines.append(f"{cellname(cell)} & {e['route']:.2f} & {e['assign']:.2f} & {e['review']:.2f} & {e['ra']:.2f} & "
                     f"{e['assign_share']:.1f} & {e['review_share']:.1f} & {e['assign_beats_ra']} & {e['review_beats_ra']}\\\\")
    (TAB / "x1_crossed.tex").write_text("\n".join(lines) + "\n")
    bytau = {}
    for t in ("0.05", "0.15", "0.30"):
        cs = [x for x in CELLS if x.endswith(t)]
        bytau[t] = dict(assign=[min(out[x]["assign_share"] for x in cs), max(out[x]["assign_share"] for x in cs)],
                        review=[min(out[x]["review_share"] for x in cs), max(out[x]["review_share"] for x in cs)])
    K["crossed"] = dict(cells=out, by_tau=bytau,
                        max_beats=max(max(e["assign_beats_ra"], e["review_beats_ra"]) for e in out.values()))


# ------------------------------------------------------------------ X2 to X6
def extensions(df):
    out = {}
    lines = []
    for block, comps, cells in (
            ("X2", [("ROUTE", "RA-PAIR"), ("RA", "RA-PAIR")],
             [f"qdiff{q}_{c}" for c in ("gaussian", "t") for q in ("0.45", "0.70", "0.95")]),
            ("X3", [("ROUTE", "RA-PAIR"), ("RA", "RA-PAIR")],
             ["dbar0.6_independent", "dbar0.6_G0", "dbar0.8_independent", "dbar0.8_G0", "dbar1.0_independent"]),
            ("X4", [("ROUTE", "RA-PAIR"), ("RA", "RA-PAIR")], ["loss_all", "loss_deliverables"])):
        pr = paired_block(df, block, comps)
        d = df[df.block == block]
        gm = d.groupby(["cell", "plan"]).mean(numeric_only=True)
        bo = {}
        for c in cells:
            e = {p: dict(p_systemic=pct(gm.loc[(c, p), "p_systemic"]), eff=r(gm.loc[(c, p), "eff_models"], 3),
                         same=pct(gm.loc[(c, p), "same_model_arc_share"], 1),
                         last=pct(gm.loc[(c, p), "review_share_last_stage"], 1),
                         wrong=r(gm.loc[(c, p), "wrong_per_run"], 3),
                         n_reviews=r(gm.loc[(c, p), "n_reviews"], 1),
                         risk=r(gm.loc[(c, p), "risk_obj"], 3))
                 for p in d[d.cell == c].plan.unique()}
            for b, o in comps:
                q = pr[(pr.cell == c) & (pr.base == b)]
                if len(q):
                    q = q.iloc[0]
                    e[f"{o}-{b}"] = dict(diff=[100 * v for v in (q.mean_diff, q.ci_low, q.ci_high)],
                                         holm=None if math.isnan(q.p_holm) else float(q.p_holm))
            pp = d[(d.cell == c) & (d.plan == "RA-PAIR")]
            e["proved"] = proved(pp.solver_gap)
            bo[c] = e
            ra_col = e.get("RA", {}).get("p_systemic")
            if block in ("X2", "X3"):
                dd = e["RA-PAIR-RA"]
                digits = 3 if 0 < abs(dd["diff"][0]) < 0.005 else 2
                mean, low, high = dd["diff"]
                label = {"X2": lambda x: (x.split("_")[0][5:], "Gaussian" if x.endswith("gaussian") else "$t$"),
                         "X3": lambda x: (x.split("_")[0][4:], "tied to $G_0$" if x.endswith("G0") else "independent")}[block](c)
                lines.append(f"{block} & {label[0]} & {label[1]} & {e['ROUTE']['p_systemic']:.2f} & {ra_col:.2f} & "
                             f"{e['RA-PAIR']['p_systemic']:.2f} & ${mean:+.{digits}f}$ [${low:+.{digits}f}$, ${high:+.{digits}f}$] & "
                             f"{fmt_p(dd['holm'])}\\\\")
        out[block] = bo
    (TAB / "extensions.tex").write_text("\n".join(lines) + "\n")
    x2 = out["X2"]
    rates = df[df.block == "X2"].groupby(["copula", "q_diff", "plan"]).p_systemic.mean().unstack("plan")
    gain = (rates["ROUTE"] - rates["RA-PAIR"]).groupby("copula").mean()
    relative = rates["RA-PAIR"] / rates["ROUTE"]
    K["x2"] = dict(cells=x2, gain_gauss_mean=pct(gain["gaussian"]), gain_t_mean=pct(gain["t"]),
                   rel_gauss=[r(x) for x in relative.loc["gaussian"]],
                   rel_t=[r(x) for x in relative.loc["t"]])
    K["x3"] = dict(cells=out["X3"])
    deliverables = df[(df.block == "X4") & (df.cell == "loss_deliverables")]
    rates = deliverables.groupby("plan").p_systemic.mean()
    K["x4"] = dict(cells=out["X4"], deliverable_multiplier=r(rates["RA-PAIR"] / rates["RA"], 2))
    # X5: caps
    d = df[df.block == "X5"]
    piv = d.pivot_table(index="rep", columns="plan", values="p_systemic")
    wr = d.groupby("plan").wrong_per_run.mean()
    same_plan = 0
    for p in (RUN / "runs" / "X5").glob("rep_*/caps.json"):
        P = json.loads(p.read_text())["result"]["plans"]
        same_plan += all(P[k]["a"] == P["RA-PAIR"]["a"] and P[k]["y"] == P["RA-PAIR"]["y"]
                         for k in P if k.startswith("RA-CAP"))
    K["x5"] = dict(wrong={k: r(v, 3) for k, v in wr.items()},
                   p_systemic={k: pct(v) for k, v in d.groupby("plan").p_systemic.mean().items()},
                   identical_plans=same_plan, reps=len(piv),
                   test_above={b: int((piv[f"RA-CAP{b:g}"] > b).sum()) for b in (0.02, 0.01, 0.005)})
    # X6: delegation
    d = df[df.block == "X6"]
    moved, extra = 0, []
    for p in (RUN / "runs" / "X6").glob("rep_*/*.json"):
        P = json.loads(p.read_text())["result"]["plans"]
        nu, nr = P["RA-PAIR"]["n_used"], P["ROUTE"]["n_used"]
        if nu != nr:
            moved += 1
            extra.append(nu - nr)
    gm = d.groupby(["cell", "plan"]).mean(numeric_only=True)
    pp = d[d.plan == "RA-PAIR"]
    K["x6"] = dict(records=int(len(pp)), moved=moved, extra_tasks=[min(extra), max(extra)] if extra else None,
                   max_systemic=pct(d.p_systemic.max(), 3), max_cell_systemic=pct(gm.p_systemic.max(), 4),
                   proved=proved(pp.solver_gap),
                   handoffs={c: {p: r(gm.loc[(c, p), "n_used_handoffs"], 2) for p in ("ROUTE", "EV", "RA-PAIR")}
                             for c in sorted(d.cell.unique())},
                   risk={c: {p: r(gm.loc[(c, p), "risk_obj"], 3) for p in ("ROUTE", "EV", "RA-PAIR")}
                         for c in sorted(d.cell.unique())},
                   eff={c: {p: r(gm.loc[(c, p), "eff_models"], 2) for p in ("ROUTE", "EV", "RA-PAIR")}
                        for c in sorted(d.cell.unique())},
                   same={c: {p: pct(gm.loc[(c, p), "same_model_arc_share"], 1) for p in ("ROUTE", "EV", "RA-PAIR")}
                         for c in sorted(d.cell.unique())})


# ------------------------------------------------------------------------ X7
def x7(df):
    d = df[(df.block == "X7") & df.plan.isin(["EF", "BD"])].copy()
    d["n"] = (d["L"] * d["W"]).astype(int)
    d["S"] = d["S"].astype(int)
    route = df[(df.block == "X7") & (df.plan == "ROUTE")]
    g = d.groupby(["n", "S", "plan"])
    # Count the actual extensive-form rows and columns for each replicated size.
    dimensions = []
    for path in sorted((RUN / "runs/X7").glob("rep_*/*_EF.json")):
        job = json.loads(path.read_text())
        if job["status"] == "ok":
            ef = job["result"]["plans"]["EF"]
            dimensions.append(dict(n=job["L"] * job["W"], S=job["S"], rep=job["rep"],
                                   variables=ef["solver_cols"], constraints=ef["solver_rows"]))
    sizes = pd.DataFrame(dimensions).set_index(["n", "S", "rep"], verify_integrity=True).sort_index()
    rows = {}
    lines = []
    for (n, S), _ in d.groupby(["n", "S"]):
        e = {}
        for p in ("EF", "BD"):
            x = g.get_group((n, S, p))
            e[p] = dict(gap=pct(x.solver_gap.mean(), 1), risk=float(x.risk_obj.mean()),
                        in_sample=float(x.in_sample_risk_obj.mean()), p_systemic=pct(x.p_systemic.mean()),
                        proved5=proved(x.solver_gap, 0.05), proved1=proved(x.solver_gap, 0.01))
        form = sizes.loc[(n, S)]
        ef_reps = g.get_group((n, S, "EF")).rep
        if set(form.index) != set(ef_reps):
            raise ValueError(f"Missing extensive-form sizes for {n} tasks and {S} scenarios")
        variables = round(form.variables.mean())
        constraints = round(form.constraints.mean())
        e["EF"].update(variables_mean=variables, constraints_mean=constraints)
        rows[f"{n}_{S}"] = e
        lines.append(f"{n} & {S:,} & {variables:,} & {constraints:,} & {e['EF']['risk']:.2f} & "
                     f"{e['BD']['risk']:.2f} & {e['EF']['in_sample']:.2f} & {e['BD']['in_sample']:.2f}\\\\".replace(",", "{,}"))
    (TAB / "x7.tex").write_text("\n".join(lines) + "\n")
    gaps = g.solver_gap.mean().unstack("plan")
    risks = g.risk_obj.mean().unstack("plan")
    bd_above = int((gaps.BD > gaps.EF).sum())
    ef_lower = int((risks.EF < risks.BD).sum())
    ties = int((risks.EF == risks.BD).sum())
    pi = df[(df.block == "X7") & (df.plan == "PI")]
    pi_zero = (pi.solver_status == 2) & (pi.solver_obj == 0) & (pi.solver_bound == 0)
    bd40 = risks.BD.loc[40]
    K["x7"] = dict(sizes=rows, bd_gap_above_ef=bd_above, combos=len(rows), ef_risk_lower=ef_lower, risk_ties=ties,
                   ef_gap_range=[min(v["EF"]["gap"] for v in rows.values()), max(v["EF"]["gap"] for v in rows.values())],
                   bd_gap_range=[min(v["BD"]["gap"] for v in rows.values()), max(v["BD"]["gap"] for v in rows.values())],
                   ef_proved1=int((d[d.plan == "EF"].solver_gap <= 0.01 + 1e-12).sum()),
                   ef_proved5=int((d[d.plan == "EF"].solver_gap <= 0.05 + 1e-12).sum()),
                   bd_proved5=int((d[d.plan == "BD"].solver_gap <= 0.05 + 1e-12).sum()),
                   per_method=int((d.plan == "EF").sum()),
                   plan_systemic=[min(d.p_systemic) * 100, max(d.p_systemic) * 100],
                   route_systemic=[pct(route.p_systemic.min()), pct(route.p_systemic.max())],
                   cell_mean_systemic=[pct(g.p_systemic.mean().min(), 4), pct(g.p_systemic.mean().max(), 4)],
                   bd40_excess_at_200_pct={str(S): pct(bd40[200] / bd40[S] - 1, 1) for S in (400, 1600)},
                   pi_records=len(pi), pi_zero_certified=int(pi_zero.sum()),
                   pi_task_sizes=sorted((pi.L * pi.W).astype(int).unique().tolist()),
                   pi_training_sizes=sorted(pi.S.astype(int).unique().tolist()))


def macros():
    b = K["x1"]["base"]
    c = K["x1"]["cells"][BASE]
    mac = dict(
        RouteSys=f"{b['ROUTE']['p_systemic']:.2f}", RaSys=f"{b['RA']['p_systemic']:.2f}",
        BaseDiff=f"{-c['ra_route'][0]:.2f}", BaseLo=f"{-c['ra_route'][2]:.2f}", BaseHi=f"{-c['ra_route'][1]:.2f}",
        BaseRel=f"{K['x1']['base_relative_cut_pct']:.1f}",
        BaseFactor=f"{K['x1']['base_factor']:.1f}",
        RaLower=str(K["x1"]["ra_lower_total"]), JobsTotal=str(K["x1"]["jobs_total"]),
        RouteTaskErr=f"{K['mechanism']['base']['ROUTE']['task_errors']:.2f}",
        RaTaskErr=f"{K['mechanism']['base']['RA']['task_errors']:.2f}",
        InhShare=f"{K['mechanism']['base']['ROUTE']['inherited_share']:.1f}",
        Jobs=f"{K['campaign']['jobs']}", Solves=f"{K['campaign']['solves']:,}".replace(",", "{,}"),
    )
    (TAB / "keynumbers.tex").write_text("".join(f"\\newcommand{{\\kn{k}}}{{{v}}}\n" for k, v in mac.items()))
    K["macros"] = mac


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    df = load()
    campaign(df)
    x1(df)
    mechanism()
    crossed()
    extensions(df)
    x7(df)
    macros()
    (HERE / "results" / "keynumbers.json").write_text(json.dumps(K, indent=1, default=float) + "\n")
    print(json.dumps(K["macros"], indent=1))


if __name__ == "__main__":
    main()
