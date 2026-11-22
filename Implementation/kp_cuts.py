"""Closed-form recourse, Benders cuts and local search for the extended
formulation. NumPy only; no solver.

The solvers use the lower-bound form of the recourse. For scenario s it is

    min sum_i ell_i e_i
    s.t. e_i >= sum_m xi_im x_im - delta_i y_i                 (own error)
         e_i >= e_j + theta_ji - 1 - delta_i y_i               (handoff j -> i)
         e_i >= 0,

with theta_ji the share of the handoff's model pair that passes the error:
kappa_diff + (kappa_same - kappa_diff) z_ji when passing depends only on
whether the two models coincide ('z' mode), or sum over pairs of
kappa_mm' w_ji^mm' in general ('w' mode). A handoff whose every pair passes
has theta = 1. Because every loss weight is nonnegative and the systemic cap
only bounds sum_i e_i from above, the least solution of these rows is
optimal and equals the deck's exact Boolean recourse at every binary plan.
Its value is found in one pass in topological order, the attaining row of
each output names its parent, and the subtree weights of the resulting
forest are an optimal dual solution, as in the manuscript's Proposition 1.
"""
from __future__ import annotations

import time

import numpy as np

import kp_model as km


class CutEngine:
    """Scenario losses and optimality cuts of the extended recourse for a
    fixed workflow (every task used).

    mode 'z': first-stage pair variables z (nA,), z = 1 when both tasks of a
    handoff run on one model. mode 'w': pair variables w (nA, M, M)."""

    def __init__(self, inst: km.Instance, sc: km.Scenarios, weights=None, mode: str = "z"):
        self.inst, self.sc, self.mode = inst, sc, mode
        self.ell = inst.ell if weights is None else np.asarray(weights, float)
        self.xi_f = sc.xi.astype(np.float64)
        self.delta_f = sc.delta.astype(np.float64)
        q = sc.q
        U = sc.U
        self.always = U < q.min()                      # (S, nA): every pair passes
        self.live = U < q.max()                         # some pair passes
        if mode == "z":
            sd = km.same_diff(q)
            if sd is None:
                raise ValueError("mode 'z' needs pass probabilities that depend only on same or different models")
            qs, qd = sd
            self.k_same = (U < qs).astype(np.float64)
            self.k_diff = (U < qd).astype(np.float64)
        else:
            self.kap = U[:, :, None, None] < q[None, None, :, :]     # (S, nA, M, M)

    def theta(self, zv=None, wv=None) -> np.ndarray:
        """(S, nA) transmission share of each handoff for pair values zv or wv."""
        if self.mode == "z":
            return self.k_diff + (self.k_same - self.k_diff) * np.asarray(zv)[None, :]
        return np.einsum("skab,kab->sk", self.kap.astype(np.float64), np.asarray(wv))

    def evaluate(self, xv, yv, zv=None, wv=None):
        """Least solution e (S, n), loss (S,), parent (S, n) and dual weights.

        parent: -2 output not in the forest (e = 0), -1 own-error root,
        k >= 0 the handoff whose row attains e_i."""
        inst = self.inst
        S, n = self.sc.S, inst.n
        own = np.einsum("sim,im->si", self.xi_f, np.asarray(xv, float))
        dy = self.delta_f * np.asarray(yv, float)[None, :]
        th = self.theta(zv, wv)
        th = np.where(self.always, 1.0, th)
        e = np.zeros((S, n))
        parent = np.full((S, n), -2, dtype=np.int64)
        for i in range(n):
            best = own[:, i] - dy[:, i]
            arg = np.full(S, -1, dtype=np.int64)
            for k in inst.in_arcs[i]:
                j = inst.arcs[k, 0]
                cand = np.where(self.live[:, k], e[:, j] + th[:, k] - 1.0 - dy[:, i], -np.inf)
                upd = cand > best + 1e-12
                best = np.where(upd, cand, best)
                arg = np.where(upd, k, arg)
            pos = best > 1e-9
            e[:, i] = np.where(pos, best, 0.0)
            parent[:, i] = np.where(pos, arg, -2)
        Wt = np.where(parent != -2, self.ell[None, :], 0.0)
        for i in range(n - 1, -1, -1):
            for k in inst.in_arcs[i]:
                j = inst.arcs[k, 0]
                msk = parent[:, i] == k
                Wt[msk, j] += Wt[msk, i]
        return e, e @ self.ell, parent, Wt

    def cut(self, s: int, parent, Wt):
        """Cut of scenario s: L_s >= const + sum x_terms + sum y_terms + sum pair_terms.

        x_terms (i, m, coef); y_terms (i, coef); pair_terms (k, coef) in mode z
        or (k, m, m', coef) in mode w."""
        inst, sc = self.inst, self.sc
        xt, yt, pt, const = [], [], [], 0.0
        for i in np.nonzero(parent[s] != -2)[0]:
            w = float(Wt[s, i])
            if sc.delta[s, i]:
                yt.append((int(i), -w))
            k = int(parent[s, i])
            if k == -1:
                for m in np.nonzero(sc.xi[s, i])[0]:
                    xt.append((int(i), int(m), w))
            elif self.always[s, k]:
                pass                                   # theta = 1: no pair term
            elif self.mode == "z":
                const += w * (self.k_diff[s, k] - 1.0)
                coef = w * (self.k_same[s, k] - self.k_diff[s, k])
                if coef != 0.0:
                    pt.append((k, coef))
            else:
                const -= w
                for a_, b_ in zip(*np.nonzero(self.kap[s, k])):
                    pt.append((k, int(a_), int(b_), w))
        return xt, yt, pt, const

    def cut_value(self, s, parent, Wt, xv, yv, zv=None, wv=None) -> float:
        xt, yt, pt, const = self.cut(s, parent, Wt)
        val = const + sum(c * xv[i][m] for i, m, c in xt) + sum(c * yv[i] for i, c in yt)
        if self.mode == "z":
            val += sum(c * zv[k] for k, c in pt)
        else:
            val += sum(c * wv[k][a][b] for k, a, b, c in pt)
        return float(val)


def pair_values(inst: km.Instance, a, mode: str):
    """z (nA,) or w (nA, M, M) implied by a binary assignment a."""
    a = np.asarray(a, dtype=int)
    j, i = inst.arcs[:, 0], inst.arcs[:, 1]
    if mode == "z":
        return ((a[j] == a[i]) & (a[j] >= 0)).astype(float)
    w = np.zeros((inst.nA, inst.M, inst.M))
    ok = (a[j] >= 0) & (a[i] >= 0)
    w[np.nonzero(ok)[0], a[j][ok], a[i][ok]] = 1.0
    return w


def local_search(inst: km.Instance, sc: km.Scenarios, a0, y0, lam: float, alpha: float,
                 seconds: float = 30.0, cap_beta: float | None = None, seed: int = 0,
                 max_passes: int = 50):
    """First-improvement local search on a training sample, used to improve a
    warm start before the solver: change the model of one task, add a review,
    or move a review, keeping budget, hours and quotas. The set of used tasks
    stays fixed. With cap_beta the share of systemic failures above the cap is
    penalized, so a start that meets the cap is preferred.

    Returns (a, y, value, stats); value is the training objective."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a0, dtype=int).copy()
    y = np.asarray(y0).astype(bool).copy()
    used = a >= 0

    def value(a_, y_):
        e = km.propagate(inst, a_, y_, sc)
        v = km.objective(inst, a_, y_, sc, lam, alpha, e=e)
        if cap_beta is not None:
            over = float(np.mean(e.sum(axis=1) >= inst.e_sys)) - cap_beta
            v += 1e3 * max(0.0, over)
        return v

    t0 = time.time()
    best = value(a, y)
    start_val = best
    moves = evals = passes = 0
    idx = np.nonzero(used)[0]
    while passes < max_passes and time.time() - t0 < seconds:
        passes += 1
        improved = False
        cand = [("model", int(i), int(m)) for i in idx for m in range(inst.M) if m != a[i]]
        cand += [("add", int(i), -1) for i in idx if not y[i]]
        cand += [("move", int(i), int(j)) for i in idx if y[i] for j in idx if not y[j]]
        rng.shuffle(cand)
        for kind, i, m in cand:
            if time.time() - t0 >= seconds:
                break
            a2, y2 = a, y
            if kind == "model":
                a2 = a.copy()
                a2[i] = m
            elif kind == "add":
                y2 = y.copy()
                y2[i] = True
            else:
                y2 = y.copy()
                y2[i], y2[m] = False, True
            if not km.feasible(inst, a2, y2):
                continue
            evals += 1
            v = value(a2, y2)
            if v < best - 1e-9:
                a, y, best = a2, y2, v
                moves += 1
                improved = True
        if not improved:
            break
    return a, y, best, dict(ls_start=start_val, ls_value=best, ls_moves=moves, ls_evals=evals,
                            ls_passes=passes, ls_seconds=time.time() - t0)
