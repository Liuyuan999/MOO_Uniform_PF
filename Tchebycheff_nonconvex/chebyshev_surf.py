"""
Chebyshev (Tchebycheff) scalarization + SURF on NON-CONVEX Pareto fronts.
=========================================================================

This module transfers the SURF weight-steering idea (Jiang, Huang, Chen,
"SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto
Front", `2objective_SURF/surf_arxiv_knot.tex`) from *linear* scalarization
(LS) to *weighted Chebyshev* (Tchebycheff) scalarization, so that it can be
applied to problems whose Pareto front (PF) is genuinely **non-convex**.

Why not LS here?
----------------
LS solves   min_x  w f1(x) + (1-w) f2(x).
Its minimizers only ever land on the *convex hull* of the front, so for a
non-convex (concave) PF, sweeping w over [0,1] recovers only the two extreme
points -- the whole interior of the front is unreachable by LS, no matter how
the weights are sampled. SURF's weight steering cannot fix that; it is a
fundamental limitation of LS.

Weighted Chebyshev
------------------
    g_T(x; w) = max( w (f1(x) - z1*),  (1-w) (f2(x) - z2*) ),
where z* is the ideal (utopia) point. Every Pareto-optimal point (convex OR
non-convex) is a Chebyshev minimizer for some weight, so Chebyshev *can*
traverse a non-convex front. The remaining issue is the same one SURF was
built to solve: uniformly sampled weights give NON-uniform coverage of the
front. We apply SURF's arc-length-CDF weight steering on top of Chebyshev.

Endpoints
---------
At w in {0, 1} the Chebyshev objective ignores one objective and returns a
*weakly* Pareto point. Following the user's suggestion, we simply clip the
weight sweep to [eps, 1-eps] with a small eps (default 1e-3). The induced
endpoints are then within O(eps) of the true PF endpoints.

The SURF loop (Algorithm 1 in the paper) is scalarizer-agnostic: it only needs
the weight->PF map w |-> f_PF(w). We therefore reuse exactly the same
arc-length CDF refinement for LS and for Chebyshev.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# --------------------------------------------------------------------------- #
#  Non-convex bi-objective test fronts (minimization convention)
# --------------------------------------------------------------------------- #
#
# Each front is described by a single "front parameter" t in [0, 1] that
# traces the Pareto front F(t) = (f1(t), f2(t)). This is exact for the
# problems below: the hard part of the decision variable is separable and its
# optimum is known, so the scalarized subproblem reduces to a 1-D search over
# the front parameter t. Solving over t is therefore an *exact* inner solver.


class Front:
    """A parametrized bi-objective Pareto front, minimization convention."""

    def __init__(self, name, f1, f2, tlo=0.0, thi=1.0, ref=None):
        self.name = name
        self._f1 = f1
        self._f2 = f2
        self.tlo = tlo
        self.thi = thi
        # dense sampling of the true PF (for ideal point, IGD, plotting)
        t = np.linspace(tlo, thi, 4001)
        self.pf_dense = np.stack([f1(t), f2(t)], axis=1)          # (M_dense, 2)
        self.ideal = self.pf_dense.min(axis=0)                    # utopia z*
        self.nadir = self.pf_dense.max(axis=0)
        # reference point for hypervolume (dominated corner, minimization)
        self.ref = np.array(ref) if ref is not None else self.nadir + 0.1

    def F(self, t):
        """Objective vector(s) on the front at parameter t (array-friendly)."""
        t = np.asarray(t, dtype=float)
        return np.stack([self._f1(t), self._f2(t)], axis=-1)


def make_front(name):
    """Registry of recognized non-convex fronts (all non-convex for min)."""
    if name == "zdt2":
        # Canonical non-convex benchmark: f2 = 1 - f1^2 on the front.
        return Front("ZDT2",
                     f1=lambda t: t,
                     f2=lambda t: 1.0 - t ** 2,
                     tlo=0.0, thi=1.0, ref=[1.1, 1.1])
    if name == "circle":
        # Concave quarter circle; arc-length CDF Phi(w) has a closed form.
        half_pi = np.pi / 2.0
        return Front("Circle",
                     f1=lambda t: np.cos(t * half_pi),
                     f2=lambda t: np.sin(t * half_pi),
                     tlo=0.0, thi=1.0, ref=[1.1, 1.1])
    if name == "fonseca":
        # Fonseca-Fleming (1-D), exponential concave front.
        s = 1.0
        return Front("Fonseca",
                     f1=lambda t: 1.0 - np.exp(-((t * 2 - 1) - 1.0 / s) ** 2),
                     f2=lambda t: 1.0 - np.exp(-((t * 2 - 1) + 1.0 / s) ** 2),
                     tlo=0.0, thi=1.0, ref=[1.05, 1.05])
    raise ValueError(f"unknown front '{name}'")


# --------------------------------------------------------------------------- #
#  Scalarizers and the (exact) inner solver
# --------------------------------------------------------------------------- #

def _ls_value(Fvals, w):
    """Linear scalarization value  w f1 + (1-w) f2  (minimization)."""
    return w * Fvals[..., 0] + (1.0 - w) * Fvals[..., 1]


def _cheby_value(Fvals, w, z):
    """Weighted Chebyshev value  max(w (f1-z1), (1-w) (f2-z2))."""
    a = w * (Fvals[..., 0] - z[0])
    b = (1.0 - w) * (Fvals[..., 1] - z[1])
    return np.maximum(a, b)


def solve_scalar(front: Front, w, method="cheby", z=None, n_grid=6001, refine=True):
    """
    Exact inner solver: minimize the chosen scalarization over the front
    parameter t. Returns (t_star, f_star) with f_star = (f1, f2).

    method : "cheby" (weighted Chebyshev) or "ls" (linear scalarization).
    """
    if z is None:
        z = front.ideal
    t = np.linspace(front.tlo, front.thi, n_grid)
    Fvals = front.F(t)
    if method == "cheby":
        g = _cheby_value(Fvals, w, z)
    elif method == "ls":
        g = _ls_value(Fvals, w)
    else:
        raise ValueError(method)

    i = int(np.argmin(g))
    t_star = t[i]

    if refine:
        # golden-section refine within the neighbouring grid bracket
        lo = t[max(i - 1, 0)]
        hi = t[min(i + 1, n_grid - 1)]
        t_star = _golden_min(
            lambda tt: (_cheby_value(front.F(tt), w, z) if method == "cheby"
                        else _ls_value(front.F(tt), w)),
            lo, hi)
    return t_star, front.F(t_star)


def _golden_min(fun, a, b, tol=1e-10, maxit=200):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = float(fun(c)), float(fun(d))
    for _ in range(maxit):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(fun(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(fun(d))
    return 0.5 * (a + b)


def sweep_points(front, weights, method="cheby", z=None):
    """PF points obtained by solving the scalarization at each weight."""
    pts = np.array([solve_scalar(front, w, method=method, z=z)[1] for w in weights])
    return pts


# --------------------------------------------------------------------------- #
#  SURF: arc-length CDF refinement (Algorithm 1), scalarizer-agnostic
# --------------------------------------------------------------------------- #

def _invert_cdf(w_grid, phi_vals, q):
    """Given a monotone CDF phi on w_grid, return w = phi^{-1}(q)."""
    # phi_vals is increasing from 0 to 1; interpolate q -> w.
    return np.interp(q, phi_vals, w_grid)


def surf(front, method="cheby", N=15, T=30, alpha=0.3, eps=1e-3, z=None,
         w_grid_size=2001, record=True):
    """
    SURF (Algorithm 1) with a plug-in scalarized inner solver.

    Parameters
    ----------
    front  : Front
    method : "cheby" or "ls"           (the scalarizer to steer)
    N      : number of segments -> (N+1) PF points
    T      : number of outer refinement iterations
    alpha  : damping factor for the CDF update
    eps    : endpoint clip, weights live in [eps, 1-eps]

    Returns
    -------
    dict with per-iteration weights, PF points, CDF estimates, and metrics.
    """
    if z is None:
        z = front.ideal
    wlo, whi = eps, 1.0 - eps
    w_grid = np.linspace(wlo, whi, w_grid_size)

    # Phi_0 = identity on [wlo, whi] -> [0,1]  (t=0 is the uniform-w baseline)
    phi_vals = (w_grid - wlo) / (whi - wlo)

    q_targets = np.linspace(0.0, 1.0, N + 1)

    hist = {"weights": [], "points": [], "phi": [], "phi_tilde": [],
            "cv": [], "gap_ratio": []}

    for t in range(T + 1):
        # 1) sample weights via the current inverse CDF
        w_n = _invert_cdf(w_grid, phi_vals, q_targets)
        w_n[0], w_n[-1] = wlo, whi                       # pin endpoints

        # 2) solve the scalarized subproblem at each weight (inner solver)
        pts = np.array([solve_scalar(front, w, method=method, z=z)[1]
                        for w in w_n])                    # (N+1, 2)

        # 3) cumulative chord length  -> empirical arc-length CDF
        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # (N,)
        s = np.concatenate([[0.0], np.cumsum(chords)])
        total = s[-1] if s[-1] > 0 else 1.0
        phi_tilde_nodes = s / total

        if record:
            hist["weights"].append(w_n.copy())
            hist["points"].append(pts.copy())
            hist["phi"].append((w_grid.copy(), phi_vals.copy()))
            hist["cv"].append(_cv(chords))
            hist["gap_ratio"].append(_gap_ratio(chords))

        if t == T:
            break

        # 4) monotone (PCHIP) interpolation of the empirical CDF over [wlo,whi]
        #    nodes w_n may repeat when the front is degenerate -> guard.
        wn_u, idx = np.unique(w_n, return_index=True)
        phi_tilde_u = phi_tilde_nodes[idx]
        if wn_u[0] > wlo:
            wn_u = np.concatenate([[wlo], wn_u]); phi_tilde_u = np.concatenate([[0.0], phi_tilde_u])
        if wn_u[-1] < whi:
            wn_u = np.concatenate([wn_u, [whi]]); phi_tilde_u = np.concatenate([phi_tilde_u, [1.0]])
        pchip = PchipInterpolator(wn_u, phi_tilde_u)
        phi_tilde_grid = np.clip(pchip(w_grid), 0.0, 1.0)
        phi_tilde_grid = np.maximum.accumulate(phi_tilde_grid)   # enforce monotone

        if record:
            hist["phi_tilde"].append((w_grid.copy(), phi_tilde_grid.copy()))

        # 5) damped CDF update  Phi_{t+1} = a * Phi_tilde + (1-a) * Phi_t
        phi_vals = alpha * phi_tilde_grid + (1.0 - alpha) * phi_vals
        phi_vals = (phi_vals - phi_vals[0]) / (phi_vals[-1] - phi_vals[0])
        phi_vals = np.maximum.accumulate(phi_vals)

    hist["w_grid"] = w_grid
    hist["final_weights"] = hist["weights"][-1]
    hist["final_points"] = hist["points"][-1]
    return hist


# --------------------------------------------------------------------------- #
#  Closed-form Rule 1 for the circle (validation of SURF's estimated Phi)
# --------------------------------------------------------------------------- #

def circle_true_cdf(w):
    """
    Closed-form normalized arc-length CDF for the concave quarter-circle under
    Chebyshev scalarization. Balancing w cos(theta) = (1-w) sin(theta) gives
    theta(w) = arctan(w / (1-w)); arc length is proportional to theta.
    """
    w = np.asarray(w, dtype=float)
    return np.arctan2(w, 1.0 - w)               # theta in [0, pi/2]


def circle_rule1_weights(N, eps=1e-3):
    """Rule 1 (closed-form Phi): weights giving exactly uniform arc spacing."""
    wlo, whi = eps, 1.0 - eps
    th_lo, th_hi = circle_true_cdf(wlo), circle_true_cdf(whi)
    theta = np.linspace(th_lo, th_hi, N + 1)
    # invert theta = arctan(w/(1-w))  ->  w = tan(theta)/(1+tan(theta))
    tt = np.tan(theta)
    return tt / (1.0 + tt)


# --------------------------------------------------------------------------- #
#  Coverage / quality metrics
# --------------------------------------------------------------------------- #

def _cv(chords):
    chords = np.asarray(chords, dtype=float)
    m = chords.mean()
    return float(chords.std() / m) if m > 0 else float("nan")


def _gap_ratio(chords):
    chords = np.asarray(chords, dtype=float)
    mn = chords.min()
    return float(chords.max() / mn) if mn > 0 else float("inf")


def cv(points):
    return _cv(np.linalg.norm(np.diff(points, axis=0), axis=1))


def gap_ratio(points):
    return _gap_ratio(np.linalg.norm(np.diff(points, axis=0), axis=1))


def igd(points, pf_ref):
    """Inverted Generational Distance: mean over ref PF of nearest point dist."""
    d = np.linalg.norm(pf_ref[:, None, :] - points[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def hypervolume_2d(points, ref):
    """2-D hypervolume (minimization) of a point set w.r.t. reference `ref`."""
    pts = points[(points[:, 0] <= ref[0]) & (points[:, 1] <= ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]              # sort by f1 ascending
    hv, prev_f1 = 0.0, ref[0]
    best_f2 = ref[1]
    # sweep from largest f1 to smallest, accumulate rectangles
    for f1, f2 in pts[::-1]:
        if f2 < best_f2:
            hv += (prev_f1 - f1) * (ref[1] - f2)
            prev_f1 = f1
            best_f2 = f2
    return float(hv)


def all_metrics(points, front: Front):
    return {
        "HV": hypervolume_2d(points, front.ref),
        "IGD": igd(points, front.pf_dense),
        "CV": cv(points),
        "GapRatio": gap_ratio(points),
        "n_unique": int(len(np.unique(np.round(points, 6), axis=0))),
    }
