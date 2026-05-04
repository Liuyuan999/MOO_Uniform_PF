"""One-off script to recompute pf_history.json and metric_history.json for a
ls_uniform or cdf_refinement run using the training-log method (tail-averaged
batch rewards + KL penalty) instead of the original eval-based approach.

Original files are backed up as *.json.bak before overwriting.

Usage
-----
    python -m qwen.recompute_pf_from_training_logs \\
        --run_root outputs/ls_uniform/ls-uniform-demo \\
        --tail_fraction 0.3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from qwen.utils.cdf_utils import compute_cv, compute_gap_ratio, compute_segment_lengths


def _estimate_from_log(
    run_dir: Path,
    tail_fraction: float = 0.3,
) -> tuple[float, float, float, float, float, float]:
    """Return (E[r1], E[r2], f1, f2, mean_kl, kl_coef) from tail-averaged training log.

    f1 = -(E[r1] - kl_coef * mean_kl)
    f2 = -(E[r2] - kl_coef * mean_kl)
    """
    # training_metrics.jsonl may sit directly in run_dir or in run_dir/logs/
    jsonl_path = run_dir / "training_metrics.jsonl"
    if not jsonl_path.is_file():
        jsonl_path = run_dir / "logs" / "training_metrics.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"training_metrics.jsonl not found in {run_dir} or {run_dir / 'logs'}. "
            "Cannot re-estimate PF point without training log."
        )

    records: list[dict[str, Any]] = []
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            records.append(obj)

    if not records:
        raise RuntimeError(f"No per-step reward records found in {jsonl_path}.")

    n_tail = max(1, int(len(records) * tail_fraction))
    tail = records[-n_tail:]

    r1 = float(np.mean([r["mean_reward_1"] for r in tail]))
    r2 = float(np.mean([r["mean_reward_2"] for r in tail]))
    kl = float(np.mean([r["mean_kl"] for r in tail if "mean_kl" in r]))
    beta_vals = [
        r["ppo_stats"]["objective_kl_coef"]
        for r in tail
        if isinstance(r.get("ppo_stats"), dict) and "objective_kl_coef" in r["ppo_stats"]
    ]
    beta = float(np.mean(beta_vals)) if beta_vals else 0.0
    f1 = -(r1 - beta * kl)
    f2 = -(r2 - beta * kl)
    return r1, r2, f1, f2, kl, beta


def recompute(run_root: Path, tail_fraction: float = 0.3) -> None:
    pf_path  = run_root / "pf_history.json"
    met_path = run_root / "metric_history.json"

    if not pf_path.is_file():
        raise FileNotFoundError(f"pf_history.json not found at {pf_path}")
    if not met_path.is_file():
        raise FileNotFoundError(f"metric_history.json not found at {met_path}")

    # Backup originals
    shutil.copy(pf_path,  pf_path.with_suffix(".json.bak"))
    shutil.copy(met_path, met_path.with_suffix(".json.bak"))
    print(f"Backed up originals to *.json.bak")

    pf_history: list[dict[str, Any]] = json.loads(pf_path.read_text(encoding="utf-8"))
    new_pf: list[dict[str, Any]] = []
    new_met: list[dict[str, Any]] = []
    baseline_cv: float | None = None
    baseline_gap: float | None = None

    for outer_entry in pf_history:
        k = int(outer_entry["outer_iter"])
        new_points: list[dict[str, Any]] = []

        for pt in outer_entry["points"]:
            run_dir = Path(pt["run_dir"])
            print(f"  outer_iter={k}  slot={pt['slot']}  weight={pt['weight']}  run_dir={run_dir}")
            r1, r2, f1, f2, kl, beta = _estimate_from_log(run_dir, tail_fraction)
            new_pt = {
                **pt,
                "E_r1": r1,
                "E_r2": r2,
                "f1": f1,
                "f2": f2,
                "mean_kl": kl,
                "kl_coef": beta,
                "pf_source": "training_log",
            }
            print(
                f"    r1={r1:.4f}  r2={r2:.4f}  kl={kl:.4f}  beta={beta:.4f}"
                f"  f1={f1:.4f}  f2={f2:.4f}"
            )
            new_points.append(new_pt)

        new_pf.append({"outer_iter": k, "points": new_points})

        # Recompute CV / GapRatio from the new f1, f2
        ordered = sorted(new_points, key=lambda p: float(p["weight"]))
        z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=np.float64)
        ell = compute_segment_lengths(z) if z.shape[0] >= 2 else np.array([], dtype=np.float64)
        cv   = compute_cv(ell)
        gap  = compute_gap_ratio(ell)
        if k == 0:
            baseline_cv, baseline_gap = cv, gap
        new_met.append({
            "outer_iter": k,
            "cv": cv,
            "gap_ratio": gap,
            "baseline_cv_iter0": baseline_cv,
            "baseline_gap_ratio_iter0": baseline_gap,
        })
        print(f"  outer_iter={k}  CV={cv:.4f}  GapRatio={gap:.4f}")

    pf_path.write_text(json.dumps(new_pf, indent=2), encoding="utf-8")
    met_path.write_text(json.dumps(new_met, indent=2), encoding="utf-8")
    print(f"\nDone.  Updated:\n  {pf_path}\n  {met_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_root", required=True, help="Path to the run directory (contains pf_history.json).")
    parser.add_argument("--tail_fraction", type=float, default=0.3, help="Fraction of training steps (from end) to average (default: 0.3).")
    args = parser.parse_args()
    recompute(Path(args.run_root), tail_fraction=args.tail_fraction)


if __name__ == "__main__":
    main()
