"""CDF refinement helpers for arc-length-uniform Pareto sampling (model-agnostic)."""

from __future__ import annotations

import numpy as np


def make_uniform_cdf_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(w_grid, F_uniform)`` with ``F(w)=w`` on ``[0,1]``."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    w = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    return w, w.copy()


def invert_cdf(
    F: np.ndarray,
    w_grid: np.ndarray,
    quantiles: np.ndarray,
    *,
    clamp: bool = True,
) -> np.ndarray:
    """Inverse-CDF sample: given monotone ``F`` on ``w_grid``, return ``F^{-1}(q)`` for each ``q``."""
    if F.shape != w_grid.shape:
        raise ValueError("F and w_grid must have same shape")
    q = np.asarray(quantiles, dtype=np.float64).ravel()
    if clamp:
        q = np.clip(q, 0.0, 1.0)
    return np.interp(q, F, w_grid)


def enforce_monotone_cdf(
    F: np.ndarray,
    *,
    eps: float,
    force_endpoints: bool,
) -> np.ndarray:
    """Project ``F`` to be non-decreasing; optionally pin ``F[0]=0``, ``F[-1]=1``."""
    x = np.asarray(F, dtype=np.float64).copy()
    if force_endpoints:
        x[0] = 0.0
        x[-1] = 1.0
    for i in range(1, len(x)):
        if x[i] < x[i - 1] + eps:
            x[i] = x[i - 1] + eps
    if force_endpoints:
        x[0] = 0.0
        x[-1] = 1.0
    m = float(np.max(x))
    if m > 1.0 + 1e-12:
        x = x / m
    return np.clip(x, 0.0, 1.0)


def compute_segment_lengths(z: np.ndarray) -> np.ndarray:
    """Chord lengths ``ell_n = ||z_{n+1}-z_n||`` for rows ``z`` shape ``(N+1, 2)``."""
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("z must have shape (N+1, 2)")
    if z.shape[0] < 2:
        raise ValueError("need at least 2 points")
    d = np.diff(z, axis=0)
    return np.linalg.norm(d, axis=1)


def build_surrogate_cdf_from_points(
    weights: np.ndarray,
    z: np.ndarray,
    w_grid: np.ndarray,
    *,
    use_pchip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Chord-length surrogate: cumulative ``s_tilde`` on ``weights``, normalized to CDF on ``w_grid``."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    zz = np.asarray(z, dtype=np.float64)
    if w.shape[0] != zz.shape[0]:
        raise ValueError("weights and z must align")
    order = np.argsort(w)
    w_s = w[order]
    z_s = zz[order]
    ell = compute_segment_lengths(z_s)
    s_at = np.zeros(len(w_s), dtype=np.float64)
    s_at[1:] = np.cumsum(ell)
    s1 = float(s_at[-1])
    if s1 <= 0.0:
        F_at = np.linspace(0.0, 1.0, len(w_s))
    else:
        F_at = s_at / s1

    wg = np.asarray(w_grid, dtype=np.float64).ravel()
    if use_pchip:
        try:
            from scipy.interpolate import PchipInterpolator

            interp = PchipInterpolator(w_s, F_at, extrapolate=False)
            F_grid = np.asarray(interp(wg), dtype=np.float64)
            nan_mask = np.isnan(F_grid)
            if np.any(nan_mask):
                F_grid[nan_mask] = np.interp(wg[nan_mask], w_s, F_at)
        except ImportError:
            F_grid = np.interp(wg, w_s, F_at)
    else:
        F_grid = np.interp(wg, w_s, F_at)

    F_grid = np.clip(F_grid, 0.0, 1.0)
    return F_grid, s_at


def blend_cdfs(F_prev: np.ndarray, F_tilde: np.ndarray, alpha: float) -> np.ndarray:
    """``F_next = (1-alpha)*F_prev + alpha*F_tilde``."""
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    return (1.0 - a) * np.asarray(F_prev) + a * np.asarray(F_tilde)


def compute_cv(ell: np.ndarray) -> float:
    """Coefficient of variation of segment lengths."""
    e = np.asarray(ell, dtype=np.float64).ravel()
    if len(e) == 0:
        return float("nan")
    m = float(np.mean(e))
    if m <= 0.0:
        return float("nan")
    return float(np.std(e, ddof=0) / m)


def compute_gap_ratio(ell: np.ndarray) -> float:
    """``max(ell) / min(ell)`` (lower is better)."""
    e = np.asarray(ell, dtype=np.float64).ravel()
    if len(e) == 0:
        return float("nan")
    mn = float(np.min(e))
    if mn <= 0.0:
        return float("inf")
    return float(np.max(e) / mn)


