"""Deterministic two-objective scalarization diagnostics for SURF.

This is an oracle-front geometry experiment, not an end-to-end optimizer
benchmark. Each scalarized subproblem is minimized over a high-resolution,
analytic Pareto-front representation. The selected points therefore isolate
the parameterization induced by each scalarizer from optimization error.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_SEGMENTS, T_STEPS, ALPHA, EPS = 15, 30, 0.3, 1e-3
N_FRONT = 200_001


@dataclass
class Front:
    name: str
    points: np.ndarray
    component: np.ndarray

    @property
    def ideal(self) -> np.ndarray:
        return self.points.min(axis=0)

    @property
    def ref(self) -> np.ndarray:
        span = self.points.max(axis=0) - self.points.min(axis=0)
        return self.points.max(axis=0) + 0.1 * np.maximum(span, 1.0)


def nondominated_curve(x: np.ndarray, y: np.ndarray, name: str) -> Front:
    """Retain a decreasing two-objective front from a dense analytic curve."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.zeros(len(x), dtype=bool)
    best_y = np.inf
    for i in range(len(x)):
        if y[i] < best_y - 1e-12:
            keep[i] = True
            best_y = y[i]
    points = np.column_stack((x[keep], y[keep]))
    # Gaps created by discarding dominated portions identify connected PF pieces.
    dx = np.diff(points[:, 0])
    typical_dx = np.median(dx[dx > 0])
    breaks = np.r_[False, dx > 10.0 * typical_dx]
    return Front(name, points, np.cumsum(breaks).astype(int))


def make_fronts() -> dict[str, Front]:
    x = np.linspace(0.0, 1.0, N_FRONT)
    zdt3 = nondominated_curve(x, 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x), "ZDT3")
    # Standard DTLZ7 has g=1 on its Pareto set, hence f2=4-f1(1+sin(3πf1)).
    dtlz7 = nondominated_curve(x, 4.0 - x * (1.0 + np.sin(3.0 * np.pi * x)), "DTLZ7 (M=2)")
    theta = np.linspace(0.0, np.pi / 2.0, N_FRONT)
    # DTLZ2 with g=0: (cos(theta), sin(theta)), reordered by f1.
    dtlz2_points = np.column_stack((np.cos(theta), np.sin(theta)))[::-1]
    dtlz2 = Front("DTLZ2 (M=2)", dtlz2_points, np.zeros(N_FRONT, dtype=int))
    # WFG M=2 zero-distance PF shapes (Huband et al. / pymoo shape functions).
    # WFG4: concave connected; WFG2: convex + disconnected (A=5).
    wfg4 = nondominated_curve(
        2.0 * np.sin(0.5 * np.pi * x),
        4.0 * np.cos(0.5 * np.pi * x),
        "WFG4 (M=2)",
    )
    wfg2 = nondominated_curve(
        2.0 * (1.0 - np.cos(0.5 * np.pi * x)),
        4.0 * (1.0 - x * np.cos(5.0 * np.pi * x) ** 2),
        "WFG2 (M=2)",
    )
    return {"ZDT3": zdt3, "DTLZ2": dtlz2, "DTLZ7": dtlz7, "WFG4": wfg4, "WFG2": wfg2}


def scalar_select(front: Front, weight: float, scalarizer: str) -> np.ndarray:
    f = front.points
    if scalarizer == "LS":
        value = weight * f[:, 0] + (1.0 - weight) * f[:, 1]
    elif scalarizer == "Chebyshev":
        z = front.ideal
        value = np.maximum(weight * (f[:, 0] - z[0]), (1.0 - weight) * (f[:, 1] - z[1]))
    else:
        raise ValueError(f"Unknown scalarizer: {scalarizer}")
    return f[int(np.argmin(value))]


def uniform_scalarization(front: Front, scalarizer: str) -> np.ndarray:
    return np.array(
        [scalar_select(front, weight, scalarizer) for weight in np.linspace(EPS, 1.0 - EPS, N_SEGMENTS + 1)]
    )


def surf(front: Front, scalarizer: str) -> np.ndarray:
    """Run the original one-dimensional SURF CDF update with a front oracle."""
    grid = np.linspace(EPS, 1.0 - EPS, 2001)
    phi = (grid - EPS) / (1.0 - 2.0 * EPS)
    quantiles = np.linspace(0.0, 1.0, N_SEGMENTS + 1)
    for step in range(T_STEPS + 1):
        weights = np.interp(quantiles, phi, grid)
        weights[[0, -1]] = EPS, 1.0 - EPS
        points = np.array([scalar_select(front, weight, scalarizer) for weight in weights])
        if step == T_STEPS:
            return points
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        empirical = np.r_[0.0, np.cumsum(chords)]
        empirical /= empirical[-1] if empirical[-1] > 0 else 1.0
        unique_w, indices = np.unique(weights, return_index=True)
        if len(unique_w) < 2:
            return points
        estimate = PchipInterpolator(unique_w, empirical[indices])(grid)
        estimate = np.maximum.accumulate(np.clip(estimate, 0.0, 1.0))
        phi = ALPHA * estimate + (1.0 - ALPHA) * phi
        phi = np.maximum.accumulate((phi - phi[0]) / (phi[-1] - phi[0]))
    raise RuntimeError("SURF loop did not terminate")


def equal_arclength_oracle(front: Front) -> np.ndarray:
    """Component-aware equal intrinsic-arc-length points from the front oracle."""
    pieces, lengths = [], []
    for component in np.unique(front.component):
        piece = front.points[front.component == component]
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
        pieces.append((piece, cumulative))
        lengths.append(cumulative[-1])
    total = sum(lengths)
    targets = np.linspace(0.0, total, N_SEGMENTS + 1)
    output = []
    offsets = np.r_[0.0, np.cumsum(lengths)]
    for target in targets:
        idx = min(np.searchsorted(offsets, target, side="right") - 1, len(pieces) - 1)
        piece, cumulative = pieces[idx]
        local = min(target - offsets[idx], cumulative[-1])
        coordinate = np.interp(local, cumulative, piece[:, 0])
        output.append(piece[int(np.argmin(np.abs(piece[:, 0] - coordinate)))])
    return np.array(output)


def nbi_oracle(front: Front) -> tuple[np.ndarray, int]:
    """NBI normal-line baseline evaluated against the front oracle.

    For each evenly spaced CHIM point, intersect its line in the CHIM normal
    direction with every connected front segment, retaining the intersection
    closest to the ideal direction. A missing intersection is reported rather
    than replaced by an invalid point; this can happen on disconnected fronts.
    """
    a = front.points[0].copy()
    b = front.points[-1].copy()
    selected, missing = [], 0
    for beta in np.linspace(0.0, 1.0, N_SEGMENTS + 1):
        p = (1.0 - beta) * a + beta * b
        target = p[0] - p[1]  # line p + t(1, 1)
        candidates = []
        for component in np.unique(front.component):
            curve = front.points[front.component == component]
            residual = curve[:, 0] - curve[:, 1] - target
            crossings = np.flatnonzero(residual[:-1] * residual[1:] <= 0.0)
            for index in crossings:
                lo, hi = residual[index], residual[index + 1]
                frac = 0.0 if abs(lo - hi) < 1e-15 else lo / (lo - hi)
                candidates.append(curve[index] + frac * (curve[index + 1] - curve[index]))
        if candidates:
            # The point furthest toward the ideal along the normal line.
            selected.append(min(candidates, key=lambda point: float(point.sum())))
        else:
            missing += 1
    return np.array(selected), missing


def ordered(points: np.ndarray) -> np.ndarray:
    return points[np.argsort(points[:, 0])]


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    points = ordered(points)
    hv, previous, best_y = 0.0, ref[0], ref[1]
    for x, y in points[::-1]:
        if y < best_y:
            hv += (previous - x) * (ref[1] - y)
            previous, best_y = x, y
    return float(hv)


def metric_values(points: np.ndarray, front: Front) -> dict[str, float | int | None]:
    points = ordered(points)
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean = chords.mean() if len(chords) else 0.0
    global_cv = float(chords.std() / mean) if mean > 0 else None
    global_gap = float(chords.max() / chords.min()) if len(chords) and chords.min() > 0 else None
    nearest = np.linalg.norm(front.points[::20, None, :] - points[None, :, :], axis=2)
    labels = np.array([front.component[np.argmin(np.linalg.norm(front.points - point, axis=1))] for point in points])
    component_chords = chords[labels[:-1] == labels[1:]]
    component_mean = component_chords.mean() if len(component_chords) else 0.0
    return {
        "HV": hypervolume_2d(points, front.ref),
        "IGD": float(nearest.min(axis=1).mean()),
        "CV": global_cv,
        "GapRatio": global_gap,
        "ComponentCV": float(component_chords.std() / component_mean) if component_mean > 0 else None,
        "ComponentGapRatio": float(component_chords.max() / component_chords.min())
        if len(component_chords) and component_chords.min() > 0 else None,
        "n_points": int(len(points)),
        "n_unique": int(len(np.unique(np.round(points, 8), axis=0))),
    }


def plot_front(front: Front, methods: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for component in np.unique(front.component):
        curve = front.points[front.component == component]
        ax.plot(curve[:, 0], curve[:, 1], color="0.65", lw=1.5, zorder=1)
    styles = {
        "Uniform LS": ("#d62728", "x"),
        "LS + SURF": ("#ff7f0e", "^"),
        "Uniform Chebyshev": ("#1f77b4", "s"),
        "Chebyshev + SURF": ("#2ca02c", "o"),
        "Equal arc-length oracle": ("#9467bd", "D"),
        "NBI normal-line oracle": ("#8c564b", "P"),
    }
    for label, points in methods.items():
        color, marker = styles[label]
        points = ordered(points)
        ax.plot(points[:, 0], points[:, 1], color=color, lw=0.8, alpha=0.45, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=38, label=label, zorder=3)
    ax.set(title=f"{front.name}: deterministic front-oracle diagnostic", xlabel="$f_1$", ylabel="$f_2$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    slug = front.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    fig.savefig(FIGURES / f"pf_{slug}.png", dpi=180)
    fig.savefig(FIGURES / f"pf_{slug}.pdf")
    plt.close(fig)


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    report = {
        "protocol": {
            "N_segments": N_SEGMENTS, "N_points": N_SEGMENTS + 1, "T": T_STEPS, "alpha": ALPHA,
            "inner_solver": "deterministic high-resolution analytic Pareto-front oracle",
            "front_samples": N_FRONT,
            "primary_metrics": ["CV", "GapRatio"],
            "secondary_metrics": ["HV", "IGD"],
        },
        "caveats": [
            "This is a front-oracle geometry diagnostic, not an end-to-end stochastic-optimizer benchmark.",
            "ZDT3, DTLZ7, and WFG2 are disconnected. Global CV and GapRatio include cross-component jumps; component-aware variants exclude those jumps.",
            "WFG4 (concave, connected) and WFG2 (convex, disconnected) use Huband/pymoo M=2 zero-distance PF shape functions.",
            "NBI is evaluated as a CHIM normal-line/front-oracle baseline. Missing normal-line intersections are reported rather than imputed.",
        ],
        "benchmarks": {},
    }
    csv_rows = []
    for key, front in make_fronts().items():
        nbi_points, nbi_missing = nbi_oracle(front)
        methods = {
            "Uniform LS": uniform_scalarization(front, "LS"),
            "LS + SURF": surf(front, "LS"),
            "Uniform Chebyshev": uniform_scalarization(front, "Chebyshev"),
            "Chebyshev + SURF": surf(front, "Chebyshev"),
            "Equal arc-length oracle": equal_arclength_oracle(front),
            "NBI normal-line oracle": nbi_points,
        }
        benchmark = {
            "n_components": int(front.component.max() + 1),
            "disconnected": bool(front.component.max() > 0),
            "NBI_missing_targets": nbi_missing,
            "methods": {},
        }
        for method, points in methods.items():
            metrics = metric_values(points, front)
            benchmark["methods"][method] = metrics
            csv_rows.append({"benchmark": key, "method": method, "NBI_missing_targets": nbi_missing, **metrics})
        report["benchmarks"][key] = benchmark
        plot_front(front, methods)
    (RESULTS / "summary.json").write_text(json.dumps(safe(report), indent=2, allow_nan=False) + "\n")
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(safe(row) for row in csv_rows)
    print(json.dumps(safe(report), indent=2, allow_nan=False))
    print(f"\nWrote {RESULTS / 'summary.json'} and {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
