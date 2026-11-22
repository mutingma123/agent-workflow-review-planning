"""Extended model of the formulation deck: instances, delegation templates,
scenarios, error propagation and scores. NumPy and SciPy only; no solver.

The layered workflow and Gaussian copula use a common stream of random draws
so the baseline instances and scenarios match across plan comparisons.
Every extension (model-pair transmission, fallible review, loss weights,
provider quotas, Student t copula, posterior error rates, delegation
templates) draws from a second random stream and leaves those draws intact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

# Model pool of the manuscript: two frontier models from different providers
# (A, B), a mid-size model (C) and a small model (D).
MODEL_NAMES = ["A", "B", "C", "D"]
MODEL_P = np.array([0.010, 0.013, 0.025, 0.050])
MODEL_COST = np.array([10.0, 9.0, 3.0, 1.0])
# Provider of each model, used only by the provider quotas. Assumption of
# this study: A and C come from provider 0, B and D from provider 1.
MODEL_PROVIDER = np.array([0, 1, 0, 1])


@dataclass
class Instance:
    n: int
    M: int
    arcs: np.ndarray              # (nA, 2): tail j, head i; task i reads task j
    layer: np.ndarray             # depth of each task (stage of a layered workflow)
    sinks: np.ndarray             # deliverables, always used
    ell: np.ndarray               # loss weight of each output
    c: np.ndarray                 # (n, M) cost of task i on model m
    p: np.ndarray                 # (n, M) marginal silent-error probability
    h: np.ndarray                 # review hours of each output
    dbar: np.ndarray              # probability that a review catches the error
    B: float                      # spend budget
    H: float                      # review-hours budget
    Q: np.ndarray                 # spend quota per provider (inf when absent)
    e_sys: int                    # wrong outputs that make a run a systemic failure
    seed: int
    kind: str = "layered"         # layered, tree or random
    and_groups: dict = field(default_factory=dict)   # task -> sub-tasks, all used
    or_groups: dict = field(default_factory=dict)    # task -> alternatives, one used
    in_arcs: list = field(default_factory=list)

    @property
    def nA(self) -> int:
        return len(self.arcs)

    @property
    def fixed(self) -> bool:
        """True when every task is used (no delegation alternatives)."""
        return not self.and_groups and not self.or_groups

    def default_use(self) -> np.ndarray:
        """Tasks used when every OR group takes its first alternative."""
        if self.fixed:
            return np.ones(self.n, dtype=bool)
        use = np.zeros(self.n, dtype=bool)
        stack = [int(r) for r in self.sinks]
        while stack:
            i = stack.pop()
            use[i] = True
            if i in self.and_groups:
                stack += list(self.and_groups[i])
            if i in self.or_groups:
                stack.append(int(self.or_groups[i][0]))
        return use


def _in_arcs(n, arcs):
    out = [[] for _ in range(n)]
    for k, (j, i) in enumerate(arcs):
        out[int(i)].append(k)
    return out


def make_instance(L: int, W: int, fan: float, beta: float, tau: float, seed: int,
                  p_scale: float = 1.0, ell_mode: str = "all", dbar: float = 1.0,
                  quota_share: float | None = None, e_sys_share: float = 0.25) -> Instance:
    """Layered workflow of the manuscript: L stages of W tasks; each task reads
    1 to 3 outputs of the previous stage (mean `fan`); the last stage holds the
    deliverables. ell_mode 'all' weighs every output 1; 'deliverables' weighs
    only the last stage. quota_share caps each provider's spend at that share
    of B (None: no quota)."""
    rng = np.random.default_rng(seed)
    n = L * W
    M = len(MODEL_NAMES)
    layer = np.repeat(np.arange(L), W)
    arcs = set()
    for l in range(1, L):
        prev = np.arange((l - 1) * W, l * W)
        cur = np.arange(l * W, (l + 1) * W)
        for i in cur:
            k = 1 + rng.binomial(2, min(1.0, max(0.0, (fan - 1) / 2)))
            k = min(k, W)
            for j in rng.choice(prev, size=k, replace=False):
                arcs.add((int(j), int(i)))
        tails = {j for (j, i) in arcs if layer[i] == l}
        for j in prev:
            if int(j) not in tails:
                arcs.add((int(j), int(rng.choice(cur))))
    arcs = np.array(sorted(arcs, key=lambda a: (a[1], a[0])), dtype=int)
    sinks = np.arange((L - 1) * W, n)
    size = rng.uniform(0.5, 1.5, n)
    diff = np.clip(np.exp(rng.normal(0.0, 0.35, n)), 0.5, 2.0)
    c = size[:, None] * MODEL_COST[None, :]
    p = np.minimum(0.5, p_scale * diff[:, None] * MODEL_P[None, :])
    h = rng.uniform(1.0, 3.0, n)
    cmin, cmax = c.min(axis=1).sum(), c.max(axis=1).sum()
    B = cmin + beta * (cmax - cmin)
    H = tau * h.sum()
    ell = np.ones(n)
    if ell_mode == "deliverables":
        ell = np.zeros(n)
        ell[sinks] = 1.0
    Q = np.full(int(MODEL_PROVIDER.max()) + 1, np.inf)
    if quota_share is not None:
        Q[:] = quota_share * B
    return Instance(n=n, M=M, arcs=arcs, layer=layer, sinks=sinks, ell=ell, c=c, p=p, h=h,
                    dbar=np.full(n, float(dbar)), B=float(B), H=float(H), Q=Q,
                    e_sys=int(math.ceil(e_sys_share * n - 1e-9)), seed=seed,
                    in_arcs=_in_arcs(n, arcs))


def make_template(kind: str, seed: int, beta: float, tau: float, p_scale: float = 1.0,
                  dbar: float = 1.0, e_sys_share: float = 0.25) -> Instance:
    """Delegation template after Baqueta and Tacla (2026): a tree of task
    occurrences rooted at one deliverable, which delegates to several
    sub-goals. Below the root, a goal has an AND group (every sub-task used)
    or, with probability 0.5, an OR group with two alternatives: a direct
    task, harder and cheaper, or a merge task with an AND group of sub-tasks.
    Outputs flow from sub-tasks to the task that delegated them.

    kind 'tree': root with 4 sub-goals, groups of 3, depth 3. kind 'random':
    root with 3 to 5 sub-goals, groups of 2 to 4, depth up to 4, and a leaf
    with probability 0.2 times the depth from depth 2."""
    rng = np.random.default_rng([seed, 31])
    nodes = []                       # [depth, kind, children]; kind leaf, and, or, direct
    if kind == "tree":
        depth_max, p_or, root_b = 3, 0.5, 4
    elif kind == "random":
        depth_max, p_or, root_b = 4, 0.5, int(rng.integers(3, 6))
    else:
        raise ValueError(kind)

    def branching():
        return 3 if kind == "tree" else int(rng.integers(2, 5))

    def build(depth):
        idx = len(nodes)
        nodes.append([depth, "leaf", []])
        if depth >= depth_max or (kind == "random" and depth >= 2 and rng.random() < 0.2 * depth):
            return idx
        if depth == 0:
            nodes[idx][1], nodes[idx][2] = "and", [build(1) for _ in range(root_b)]
        elif rng.random() < p_or:
            direct = len(nodes)
            nodes.append([depth + 1, "direct", []])
            merge = len(nodes)
            nodes.append([depth + 1, "and", []])
            nodes[merge][2] = [build(depth + 1) for _ in range(branching())]
            nodes[idx][1], nodes[idx][2] = "or", [direct, merge]
        else:
            nodes[idx][1], nodes[idx][2] = "and", [build(depth + 1) for _ in range(branching())]
        return idx

    build(0)
    # post-order numbering: sub-tasks before the task that reads them
    order, seen = [], set()

    def visit(k):
        for ch in nodes[k][2]:
            visit(ch)
        order.append(k)
        seen.add(k)

    visit(0)
    new = {old: i for i, old in enumerate(order)}
    n = len(order)
    M = len(MODEL_NAMES)
    arcs, and_groups, or_groups = [], {}, {}
    depth = np.zeros(n, dtype=int)
    kinds = [""] * n
    for old, i in new.items():
        d, kd, ch = nodes[old]
        depth[i], kinds[i] = d, kd
        kids = [new[c] for c in ch]
        for j in kids:
            arcs.append((j, i))
        if kd == "and":
            and_groups[i] = kids
        elif kd == "or":
            or_groups[i] = kids
    arcs = np.array(sorted(arcs, key=lambda a: (a[1], a[0])), dtype=int).reshape(-1, 2)
    size = rng.uniform(0.5, 1.5, n)
    diff = np.clip(np.exp(rng.normal(0.0, 0.35, n)), 0.5, 2.0)
    # A direct task does the work of the merge it replaces: harder, and cheaper
    # than the merge with its sub-tasks.
    for i, kd in enumerate(kinds):
        if kd == "direct":
            diff[i] = min(4.0, 2.5 * diff[i])
            size[i] = 1.5 * size[i]
    c = size[:, None] * MODEL_COST[None, :]
    p = np.minimum(0.5, p_scale * diff[:, None] * MODEL_P[None, :])
    h = rng.uniform(1.0, 3.0, n)
    sinks = np.array([n - 1])
    inst = Instance(n=n, M=M, arcs=arcs, layer=depth.max() - depth, sinks=sinks, ell=np.ones(n),
                    c=c, p=p, h=h, dbar=np.full(n, float(dbar)), B=0.0, H=0.0,
                    Q=np.full(int(MODEL_PROVIDER.max()) + 1, np.inf), e_sys=1, seed=seed,
                    kind=kind, and_groups=and_groups, or_groups=or_groups,
                    in_arcs=_in_arcs(n, arcs))
    # Budgets relative to the deepest decomposition (every OR group delegates),
    # so that the deepest one is affordable at share beta and shallower ones cost less.
    deep = np.zeros(n, dtype=bool)
    stack = [n - 1]
    while stack:
        i = stack.pop()
        deep[i] = True
        if i in and_groups:
            stack += and_groups[i]
        if i in or_groups:
            stack.append(or_groups[i][-1])
    cmin, cmax = c[deep].min(axis=1).sum(), c[deep].max(axis=1).sum()
    inst.B = float(cmin + beta * (cmax - cmin))
    inst.H = float(tau * h[deep].sum())
    inst.e_sys = int(math.ceil(e_sys_share * deep.sum() - 1e-9))
    return inst


def q_matrix(q_same: float, q_diff: float, M: int = len(MODEL_NAMES)) -> np.ndarray:
    """Pass probability of a handoff from model m to model m': q_same on the
    diagonal, q_diff elsewhere."""
    q = np.full((M, M), float(q_diff))
    np.fill_diagonal(q, float(q_same))
    return q


def same_diff(q: np.ndarray):
    """(q_same, q_diff) when q has that structure, else None."""
    d = np.diag(q)
    off = q[~np.eye(len(q), dtype=bool)]
    if np.allclose(d, d[0]) and np.allclose(off, off[0]):
        return float(d[0]), float(off[0])
    return None


@dataclass
class Scenarios:
    xi: np.ndarray        # (S, n, M) bool: model m errs silently on task i
    U: np.ndarray         # (S, nA) uniform draw of each handoff
    delta: np.ndarray     # (S, n) bool: a review of output i catches its error
    q: np.ndarray         # (M, M) pass probabilities of this sample
    meta: dict = field(default_factory=dict)

    @property
    def S(self) -> int:
        return self.xi.shape[0]

    def kappa(self, k: int, m: int, mm: int) -> np.ndarray:
        """(S,) bool: handoff k passes an upstream error from model m to model mm."""
        return self.U[:, k] < self.q[m, mm]


def sample_scenarios(inst: Instance, S: int, rho: float, gamma: float, q: np.ndarray, seed: int,
                     copula: str = "gaussian", nu: float = 4.0, review_tie: float | None = None,
                     posterior_k: float | None = None) -> Scenarios:
    """One-factor copula per model. rho is the latent correlation of one
    model's errors across tasks; gamma is the share of that model factor common
    to all models; marginal rates equal inst.p. A handoff passes an upstream
    error when its uniform draw is below q[m, m'] for the models of its two
    tasks, so all model pairs share common random numbers.

    copula 't' divides each scenario's latent variables by sqrt(W/nu), W
    chi-square with nu degrees, which adds tail dependence across all tasks
    and models. review_tie None draws review success independently with
    probability inst.dbar; a value r in [0, 1] draws a miss when
    sqrt(r) G + sqrt(1-r) eta falls below the quantile of 1 - dbar, with G
    the common factor of the errors, so reviews miss more often in scenarios
    where every model errs more. posterior_k draws each scenario's rates from
    Beta(k p, k (1 - p)), as from k logged outcomes."""
    rng = np.random.default_rng(seed)                 # the manuscript's stream
    rng2 = np.random.default_rng([seed, 7])           # extensions only
    n, M = inst.n, inst.M
    G = rng.standard_normal((S, 1))
    Fm = np.sqrt(gamma) * G + np.sqrt(1.0 - gamma) * rng.standard_normal((S, M))
    scale = None
    if copula == "t":
        scale = 1.0 / np.sqrt(rng2.chisquare(nu, size=S) / nu)
    elif copula != "gaussian":
        raise ValueError(copula)
    if posterior_k:
        k = float(posterior_k)
        a_ = np.clip(k * inst.p, 1e-6, None)
        b_ = np.clip(k * (1.0 - inst.p), 1e-6, None)
        p_s = rng2.beta(a_[None], b_[None], size=(S, n, M))
    else:
        p_s = None
    ppf = (lambda x: student_t.ppf(x, nu)) if copula == "t" else norm.ppf
    thr = ppf(inst.p)
    xi = np.empty((S, n, M), dtype=bool)
    chunk = max(1, 2_000_000 // (n * M))
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        eps = rng.standard_normal((s1 - s0, n, M))
        Z = np.sqrt(rho) * Fm[s0:s1, None, :] + np.sqrt(1.0 - rho) * eps
        if scale is not None:
            Z = Z * scale[s0:s1, None, None]
        th = ppf(p_s[s0:s1]) if p_s is not None else thr[None, :, :]
        xi[s0:s1] = Z < th
    U = rng.random((S, inst.nA))
    if review_tie is None and np.all(inst.dbar >= 1.0):
        delta = np.ones((S, n), dtype=bool)
    elif review_tie is None:
        delta = rng2.random((S, n)) < inst.dbar[None, :]
    else:
        r = float(review_tie)
        lat = np.sqrt(r) * G + np.sqrt(1.0 - r) * rng2.standard_normal((S, n))
        delta = lat >= norm.ppf(np.clip(1.0 - inst.dbar, 1e-12, 1 - 1e-12))[None, :]
    meta = dict(S=S, rho=rho, gamma=gamma, seed=seed, copula=copula, nu=nu,
                review_tie=review_tie, posterior_k=posterior_k)
    return Scenarios(xi=xi, U=U.astype(np.float64), delta=delta, q=np.asarray(q, float), meta=meta)


def propagate(inst: Instance, a, y, sc: Scenarios) -> np.ndarray:
    """(S, n) bool: outputs released wrong under assignment a (n,), with -1
    for an unused task, and reviews y (n,)."""
    a = np.asarray(a, dtype=int)
    y = np.asarray(y).astype(bool)
    S, n = sc.S, inst.n
    e = np.zeros((S, n), dtype=bool)
    for i in range(n):                                # index order is topological
        if a[i] < 0:
            continue
        val = sc.xi[:, i, a[i]].copy()
        for k in inst.in_arcs[i]:
            j = inst.arcs[k, 0]
            if a[j] < 0:
                continue
            val |= e[:, j] & (sc.U[:, k] < sc.q[a[j], a[i]])
        if y[i]:
            val &= ~sc.delta[:, i]
        e[:, i] = val
    return e


def cvar(loss: np.ndarray, alpha: float) -> float:
    """Rockafellar and Uryasev CVaR of an equally weighted sample."""
    S = len(loss)
    srt = np.sort(loss)[::-1]
    k = (1.0 - alpha) * S
    full = int(np.floor(k + 1e-9))
    tot = srt[:full].sum()
    if full < S and k - full > 1e-9:
        tot += (k - full) * srt[full]
    return float(tot / k)


def objective(inst, a, y, sc, lam, alpha, e=None) -> float:
    e = propagate(inst, a, y, sc) if e is None else e
    loss = e @ inst.ell
    return (1 - lam) * float(loss.mean()) + (lam * cvar(loss, alpha) if lam > 0 else 0.0)


def metrics(inst: Instance, a, y, sc: Scenarios, lam: float, alpha: float) -> dict:
    a = np.asarray(a, dtype=int)
    y = np.asarray(y).astype(bool)
    e = propagate(inst, a, y, sc)
    loss = e @ inst.ell
    count = e.sum(axis=1)
    used = a >= 0
    mean, cv = float(loss.mean()), cvar(loss, alpha)
    share = np.bincount(a[used], minlength=inst.M) / max(1, used.sum())
    both = used[inst.arcs[:, 0]] & used[inst.arcs[:, 1]] if inst.nA else np.zeros(0, bool)
    same = (a[inst.arcs[both, 0]] == a[inst.arcs[both, 1]]) if both.any() else np.zeros(0, bool)
    hrs = float(inst.h[y].sum())
    last = inst.layer == inst.layer.max()
    task_cost = np.where(used, inst.c[np.arange(inst.n), np.maximum(a, 0)], 0.0)
    prov = MODEL_PROVIDER[np.maximum(a, 0)]
    spend_k = [float(task_cost[used & (prov == k)].sum()) for k in range(len(inst.Q))]
    out = dict(
        exp_loss=mean, cvar=cv, risk_obj=(1 - lam) * mean + lam * cv,
        p_systemic=float(np.mean(count >= inst.e_sys)),
        p_clean=float(np.mean(count == 0)),
        wrong_per_run=float(count.mean()),
        p_any_deliverable=float(np.mean(e[:, inst.sinks].any(axis=1))),
        spend=float(inst.c[np.nonzero(used)[0], a[used]].sum()),
        review_hours=hrs, n_reviews=int(y.sum()), n_used=int(used.sum()),
        n_used_handoffs=int(both.sum()),
        eff_models=float(1.0 / np.sum(share ** 2)) if used.any() else 0.0,
        same_model_arc_share=float(same.mean()) if len(same) else 0.0,
        review_share_last_stage=float(y[last].sum() / y.sum()) if y.any() else 0.0,
        mean_review_depth=float(inst.layer[y].mean() / max(1, inst.layer.max())) if y.any() else -1.0,
        spend_provider=spend_k,
    )
    for m, nm in enumerate(MODEL_NAMES):
        out[f"share_{nm}"] = float(share[m])
    return out


def feasible(inst: Instance, a, y, tol: float = 1e-6) -> bool:
    """Budget, review hours, quotas and delegation groups of a plan."""
    a = np.asarray(a, dtype=int)
    y = np.asarray(y).astype(bool)
    used = a >= 0
    if np.any(y & ~used):
        return False
    if inst.c[np.nonzero(used)[0], a[used]].sum() > inst.B + tol:
        return False
    if inst.h[y].sum() > inst.H + tol:
        return False
    for k in range(len(inst.Q)):
        if np.isfinite(inst.Q[k]):
            sel = used & (MODEL_PROVIDER[np.maximum(a, 0)] == k)
            if inst.c[np.nonzero(sel)[0], a[sel]].sum() > inst.Q[k] + tol:
                return False
    if inst.fixed:
        return bool(used.all())
    if not used[inst.sinks].all():
        return False
    for i, kids in inst.and_groups.items():
        if any(used[j] != used[i] for j in kids):
            return False
    for i, kids in inst.or_groups.items():
        if sum(used[j] for j in kids) != int(used[i]):
            return False
    # a used task must be reachable from a deliverable through used groups
    reach = np.zeros(inst.n, dtype=bool)
    stack = [int(r) for r in inst.sinks]
    while stack:
        i = stack.pop()
        reach[i] = True
        for j in inst.and_groups.get(i, []) + inst.or_groups.get(i, []):
            if used[j]:
                stack.append(j)
    return bool(np.array_equal(reach, used))
