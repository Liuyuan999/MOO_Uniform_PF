"""
Generate figures + metrics for Chebyshev-SURF on non-convex Pareto fronts.
Run:  python run_experiments.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chebyshev_surf as cs

FIG = os.path.join(os.path.dirname(__file__), "figure")
os.makedirs(FIG, exist_ok=True)

PyBlue = "#1f77b4"
PyRed = "#d62728"
PyGreen = "#2ca02c"
PyGray = "#7f7f7f"

N, T, ALPHA, EPS = 15, 30, 0.3, 1e-3


def run_front(name):
    front = cs.make_front(name)
    wlo, whi = EPS, 1.0 - EPS
    w_unif = np.linspace(wlo, whi, N + 1)

    res = {
        "front": front,
        "ls_unif": cs.sweep_points(front, w_unif, method="ls"),
        "ch_unif": cs.sweep_points(front, w_unif, method="cheby"),
        "surf": cs.surf(front, method="cheby", N=N, T=T, alpha=ALPHA, eps=EPS),
    }
    res["ch_surf"] = res["surf"]["final_points"]
    return res


def metrics_table(results):
    rows = []
    header = f"{'Front':8} {'Method':16} {'HV':>8} {'IGD':>8} {'CV':>10} {'GapRatio':>9} {'#uniq':>6}"
    rows.append(header)
    rows.append("-" * len(header))
    for name, res in results.items():
        front = res["front"]
        for label, pts in [("LS + uniform", res["ls_unif"]),
                           ("Cheby + uniform", res["ch_unif"]),
                           ("Cheby + SURF", res["ch_surf"])]:
            m = cs.all_metrics(pts, front)
            rows.append(f"{front.name:8} {label:16} {m['HV']:8.4f} {m['IGD']:8.4f} "
                        f"{m['CV']:10.5f} {m['GapRatio']:9.3f} {m['n_unique']:6d}")
        rows.append("-" * len(header))
    table = "\n".join(rows)
    print(table)
    with open(os.path.join(FIG, "metrics.txt"), "w") as f:
        f.write(table + "\n")


# --------------------------------------------------------------------------- #
def fig_pf_panels(results):
    """One row per front: LS-uniform | Cheby-uniform | Cheby+SURF."""
    fronts = list(results)
    fig, axes = plt.subplots(len(fronts), 3, figsize=(11, 3.5 * len(fronts)))
    if len(fronts) == 1:
        axes = axes[None, :]
    col_titles = ["LS + uniform-$w$", "Chebyshev + uniform-$w$", "Chebyshev + SURF (ours)"]
    for r, name in enumerate(fronts):
        res = results[name]
        front = res["front"]
        pf = front.pf_dense
        data = [(res["ls_unif"], PyRed, "*"),
                (res["ch_unif"], PyBlue, "o"),
                (res["ch_surf"], PyGreen, "o")]
        for c, (pts, color, mk) in enumerate(data):
            ax = axes[r, c]
            ax.plot(pf[:, 0], pf[:, 1], color=PyGray, lw=1.4, alpha=0.6,
                    label="true PF", zorder=1)
            # connect selected points to show spacing
            order = np.argsort(pts[:, 0])
            ax.plot(pts[order, 0], pts[order, 1], color=color, lw=0.8,
                    alpha=0.35, zorder=2)
            ax.scatter(pts[:, 0], pts[:, 1], s=70 if mk == "*" else 42,
                       facecolor=color, edgecolor="white", linewidth=0.7,
                       marker=mk, zorder=3)
            m = cs.all_metrics(pts, front)
            ax.set_title(f"{col_titles[c]}\nCV={m['CV']:.3f}, Gap={m['GapRatio']:.2f}",
                         fontsize=10)
            ax.set_xlabel("$f_1$")
            if c == 0:
                ax.set_ylabel(f"{front.name}\n$f_2$", fontsize=11)
            ax.grid(alpha=0.25)
    fig.suptitle("Non-convex Pareto front: LS collapses to the endpoints; "
                 "Chebyshev+SURF covers it uniformly", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"pf_panels.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cdf_circle(results):
    """Circle only: SURF's estimated CDF Phi_t vs closed-form; weight rug."""
    if "circle" not in results:
        return
    h = results["circle"]["surf"]
    w_grid = h["w_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    true_cdf = cs.circle_true_cdf(w_grid)
    true_cdf = (true_cdf - true_cdf[0]) / (true_cdf[-1] - true_cdf[0])
    for t, alpha in [(0, 0.25), (1, 0.4), (2, 0.55), (5, 0.75), (T, 1.0)]:
        wg, pv = h["phi"][t]
        ax.plot(wg, pv, color=PyBlue, alpha=alpha, lw=1.6,
                label=f"$\\Phi_{{{t}}}$ (SURF)" if t in (0, T) else None)
    ax.plot(w_grid, true_cdf, "--", color=PyRed, lw=2.0, label="closed-form $\\Phi$")
    ax.set_xlabel("weight $w$"); ax.set_ylabel("arc-length CDF $\\Phi(w)$")
    ax.set_title("SURF CDF estimate $\\Phi_t$ converges to closed form")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    wlo, whi = EPS, 1 - EPS
    w_unif = np.linspace(wlo, whi, N + 1)
    w_surf = h["final_weights"]
    w_rule1 = cs.circle_rule1_weights(N, EPS)
    ax.plot(w_unif, np.full_like(w_unif, 2), "o", color=PyGray, label="uniform-$w$")
    ax.plot(w_surf, np.full_like(w_surf, 1), "o", color=PyGreen, label="SURF weights")
    ax.plot(w_rule1, np.full_like(w_rule1, 0), "x", color=PyRed, ms=8, label="Rule-1 closed form")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Rule-1", "SURF", "uniform"])
    ax.set_ylim(-0.6, 2.6); ax.set_xlabel("weight $w$")
    ax.set_title("Steered weights (SURF $\\approx$ Rule-1)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"cdf_circle.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_convergence(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"zdt2": PyBlue, "circle": PyGreen, "fonseca": PyRed}
    for name, res in results.items():
        h = res["surf"]
        its = np.arange(len(h["cv"]))
        axes[0].semilogy(its, h["cv"], "-o", ms=3, color=colors.get(name, PyGray),
                         label=res["front"].name)
        axes[1].plot(its, h["gap_ratio"], "-o", ms=3, color=colors.get(name, PyGray),
                     label=res["front"].name)
    axes[0].set_xlabel("SURF outer iteration $t$"); axes[0].set_ylabel("CV (log)")
    axes[0].set_title("Chord-spacing CV decays geometrically"); axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(1.0, color=PyGray, ls="--", lw=1)
    axes[1].set_xlabel("SURF outer iteration $t$"); axes[1].set_ylabel("Gap Ratio")
    axes[1].set_title("Gap Ratio $\\to 1$ (uniform)"); axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"convergence.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = {name: run_front(name) for name in ["zdt2", "circle"]}
    print("\n===== Coverage / quality metrics (N=%d, T=%d, alpha=%.1f, eps=%g) =====\n"
          % (N, T, ALPHA, EPS))
    metrics_table(results)
    fig_pf_panels(results)
    fig_cdf_circle(results)
    fig_convergence(results)
    print("\nFigures written to:", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)
