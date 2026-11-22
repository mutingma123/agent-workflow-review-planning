"""Code-only checks of the formulation runs. No Gurobi model is built.

    python -m unittest discover -s Verification -p test_code_only.py -v     (from the run folder)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Implementation"))

import kp_blocks as kb      # noqa: E402
import kp_campaign as kc_   # noqa: E402
import kp_cuts as kc        # noqa: E402
import kp_model as km       # noqa: E402

MANUSCRIPT_RESULTS = Path(os.environ["PILOT_RESULTS"]) if os.environ.get("PILOT_RESULTS") else None


class Reproduction(unittest.TestCase):
    @unittest.skipUnless(MANUSCRIPT_RESULTS and MANUSCRIPT_RESULTS.is_file(), "manuscript results not in this checkout")
    def test_manuscript_plans_score_identically(self):
        rec = None
        with MANUSCRIPT_RESULTS.open() as records:
            for line in records:
                r = json.loads(line)
                if r["id"] == "block=E1|qd=0.5|rep=1|rho=0.3|tau=0.15":
                    rec = r
                    break
        self.assertIsNotNone(rec, "the pilot file must contain the E1 base setting of replication 1")
        inst = km.make_instance(8, 5, 2.0, 0.6, 0.15, seed=1)
        self.assertAlmostEqual(inst.B, rec["result"]["B"], places=9)
        test = km.sample_scenarios(inst, 20000, 0.3, 0.3, km.q_matrix(0.9, 0.5), seed=900001)
        for pol in ("ROUTE", "EV", "RN", "IND", "RA"):
            p = rec["result"]["policies"][pol]
            m = km.metrics(inst, p["a"], p["y"], test, 0.5, 0.9)
            self.assertEqual(m["p_systemic"], p["p_systemic"], pol)
            self.assertAlmostEqual(m["risk_obj"], p["risk_obj"], places=9)


class Model(unittest.TestCase):
    def setUp(self):
        self.inst = km.make_instance(4, 3, 2.0, 0.6, 0.15, seed=3, p_scale=4.0, dbar=0.7)
        self.rng = np.random.default_rng(1)

    def test_kappa_reproduces_manuscript_classes(self):
        sc = km.sample_scenarios(self.inst, 500, 0.3, 0.3, km.q_matrix(0.9, 0.5), seed=5)
        any_ = sc.U < 0.5
        same = (sc.U >= 0.5) & (sc.U < 0.9)
        for k in range(self.inst.nA):
            self.assertTrue(np.array_equal(sc.kappa(k, 0, 1), any_[:, k]))
            self.assertTrue(np.array_equal(sc.kappa(k, 2, 2), any_[:, k] | same[:, k]))

    def test_propagation_matches_brute_force(self):
        q = self.rng.random((4, 4))
        sc = km.sample_scenarios(self.inst, 80, 0.4, 0.3, q, seed=9, review_tie=0.5)
        for _ in range(10):
            a = self.rng.integers(0, 4, self.inst.n)
            y = self.rng.random(self.inst.n) < 0.4
            e = km.propagate(self.inst, a, y, sc)
            for s in range(sc.S):
                ref = np.zeros(self.inst.n, bool)
                for i in range(self.inst.n):
                    v = bool(sc.xi[s, i, a[i]])
                    for k in self.inst.in_arcs[i]:
                        j = self.inst.arcs[k, 0]
                        v |= ref[j] and sc.U[s, k] < q[a[j], a[i]]
                    if y[i] and sc.delta[s, i]:
                        v = False
                    ref[i] = v
                self.assertTrue(np.array_equal(ref, e[s]))

    def test_review_tie_keeps_marginal(self):
        inst = km.make_instance(8, 5, 2.0, 0.6, 0.15, seed=2, dbar=0.8)
        sc = km.sample_scenarios(inst, 20000, 0.3, 0.3, km.q_matrix(0.9, 0.5), seed=3, review_tie=0.5)
        self.assertAlmostEqual(float(sc.delta.mean()), 0.8, delta=0.01)

    def test_templates_default_plan_is_feasible(self):
        for kind in ("tree", "random"):
            for s in range(1, 6):
                T = km.make_template(kind, s, 0.6, 0.15)
                u = T.default_use()
                a = np.where(u, 3, -1)
                self.assertTrue(km.feasible(T, a, np.zeros(T.n, bool)), (kind, s))
                self.assertTrue(np.all(T.arcs[:, 0] < T.arcs[:, 1]))     # topological numbering


class Cuts(unittest.TestCase):
    def check(self, mode, q):
        rng = np.random.default_rng(4)
        inst = km.make_instance(4, 3, 2.0, 0.6, 0.15, seed=6, p_scale=4.0, dbar=0.7)
        sc = km.sample_scenarios(inst, 120, 0.4, 0.3, q, seed=8, review_tie=0.5)
        eng = kc.CutEngine(inst, sc, mode=mode)
        for _ in range(10):
            a = rng.integers(0, 4, inst.n)
            y = rng.random(inst.n) < 0.3
            pv = kc.pair_values(inst, a, mode)
            kw = dict(zv=pv) if mode == "z" else dict(wv=pv)
            e, loss, par, Wt = eng.evaluate(np.eye(4)[a], y.astype(float), **kw)
            self.assertTrue(np.array_equal(e > 0.5, km.propagate(inst, a, y, sc)))
            x2, y2 = rng.dirichlet(np.ones(4), inst.n), rng.random(inst.n)
            kw2 = dict(zv=rng.random(inst.nA)) if mode == "z" else dict(wv=rng.random((inst.nA, 4, 4)) / 4)
            _, l2, _, _ = eng.evaluate(x2, y2, **kw2)
            for s in range(sc.S):
                self.assertAlmostEqual(eng.cut_value(s, par, Wt, np.eye(4)[a], y.astype(float), **kw), loss[s], places=9)
                self.assertLessEqual(eng.cut_value(s, par, Wt, x2, y2, **kw2), l2[s] + 1e-9)

    def test_z_mode(self):
        self.check("z", km.q_matrix(0.9, 0.5))

    def test_z_mode_reversed(self):
        self.check("z", km.q_matrix(0.4, 0.8))

    def test_w_mode(self):
        self.check("w", np.random.default_rng(2).random((4, 4)))

    def test_local_search_feasible_and_not_worse(self):
        inst = km.make_instance(8, 5, 2.0, 0.6, 0.15, seed=1)
        sc = km.sample_scenarios(inst, 100, 0.3, 0.3, km.q_matrix(0.9, 0.5), seed=1001)
        a0, y0 = np.full(inst.n, 3), np.zeros(inst.n, bool)
        a, y, v, st = kc.local_search(inst, sc, a0, y0, 0.5, 0.9, seconds=3)
        self.assertTrue(km.feasible(inst, a, y))
        self.assertLessEqual(v, st["ls_start"])


class Plan(unittest.TestCase):
    def test_queue(self):
        c = kb.config()
        q = kb.build_queue(c)
        self.assertEqual(len({j["id"] for j in q}), len(q))
        self.assertEqual(sum(1 for j in q if j["block"] == "X1"), 9 * c["reps"]["X1"])
        self.assertTrue(all(j["set"] == j["id"].rsplit("/", 1)[0] for j in q))

    def test_budget_fits_48_hours_on_eleven_workers(self):
        b = kb.budget(kb.config(), 11)
        self.assertGreaterEqual(b["time_limit"], kb.DEFAULTS["time_limit_min"])
        self.assertLessEqual(b["worst_wall_hours"], 48 - kb.DEFAULTS["reserve_hours"] + 1e-6)
        self.assertEqual(b["cuts"], [])

    def test_budget_cuts_replications_when_short(self):
        b = kb.budget(kb.config(), 1)
        self.assertGreaterEqual(b["time_limit"], kb.DEFAULTS["time_limit_min"] - 1e-9)
        self.assertTrue(b["cuts"])
        self.assertGreaterEqual(b["reps"]["X1"], 8)

    def test_notebook_defaults_hold_solves(self):
        nb = ROOT / "Notebooks" / "formulation_runs.ipynb"
        if not nb.exists():
            self.skipTest("notebook not found")
        text = json.loads(nb.read_text())
        src = "".join("".join(c["source"]) for c in text["cells"] if c["cell_type"] == "code")
        for flag in ("PREFLIGHT = False", "LAUNCH = False"):
            self.assertIn(flag, src)


class Scheduler(unittest.TestCase):
    def test_fake_campaign_runs_resumes_and_skips(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            code = kc_.main(["launch", "--fake", "--workers", "2", "--threads", "1", "--blocks", "X4,X5",
                             "--runs", str(runs), "--new-campaign"])
            self.assertEqual(code, 0)
            recs = [json.loads(p.read_text()) for p in runs.glob("X*/rep_*/*.json")]
            self.assertEqual(len(recs), 30)
            self.assertTrue(all(r["status"] == "ok" for r in recs))
            self.assertTrue((runs / "CAMPAIGN_DONE.json").exists())
            self.assertEqual(kc_.main(["launch", "--fake", "--workers", "2", "--threads", "1", "--blocks", "X4,X5",
                                       "--runs", str(runs)]), 0)
            runs2 = Path(d) / "runs2"
            kc_.main(["launch", "--fake", "--workers", "1", "--threads", "1", "--blocks", "X5",
                      "--runs", str(runs2), "--hours", "0.0001", "--new-campaign"])
            st = [json.loads(p.read_text())["status"] for p in runs2.glob("X*/rep_*/*.json")]
            self.assertTrue(st and all(s == "skipped" for s in st))

    def test_relaunch_after_deadline_closes_the_campaign_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            base = ["launch", "--fake", "--workers", "1", "--threads", "1", "--blocks", "X5", "--runs", str(runs)]
            kc_.main(base + ["--hours", "0.0001", "--new-campaign"])
            first = json.loads((runs / "campaign.json").read_text())["deadline"]
            (runs / "CAMPAIGN_DONE.json").unlink()
            rec = sorted(runs.glob("X5/rep_*/*.json"))[0]
            rec.write_text(json.dumps(dict(json.loads(rec.read_text()), status="failed", error_kind="error")))
            self.assertEqual(kc_.main(base + ["--retry-failed"]), 0)
            self.assertEqual(json.loads((runs / "campaign.json").read_text())["deadline"], first)
            self.assertTrue((runs / "CAMPAIGN_DONE.json").exists())
            self.assertEqual(json.loads(rec.read_text())["status"], "failed")
            kc_.main(["stop", "--runs", str(runs)])
            self.assertTrue((runs / "STOPPED.json").exists())


if __name__ == "__main__":
    unittest.main()
