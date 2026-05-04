from pathlib import Path
import json
import numpy as np

from qwen.utils.cdf_utils import compute_segment_lengths, compute_cv, compute_gap_ratio

RUN_ROOT = Path("outputs/cdf_refinement/cdf-refinement-surrogate-from-ls-start4")
NUM_SEGMENTS = 5
TAIL_FRACTION = 0.3

HV_REF = np.array([2.1, 1.4])   # fixed dominated reference point (minimization)

ABS_ROOT = Path(".")


def read_training_log(log_path: Path, tail_fraction: float = 0.3):
    step_records = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            step_records.append(obj)

    if not step_records:
        raise RuntimeError(f"No step records in {log_path}")

    n_tail = max(1, int(len(step_records) * tail_fraction))
    tail = step_records[-n_tail:]

    r1 = float(np.mean([r["mean_reward_1"] for r in tail]))
    r2 = float(np.mean([r["mean_reward_2"] for r in tail]))
    kl = float(np.mean([r.get("mean_kl", 0.0) for r in tail]))

    beta_vals = [
        r["ppo_stats"]["objective_kl_coef"]
        for r in tail
        if isinstance(r.get("ppo_stats"), dict)
        and "objective_kl_coef" in r["ppo_stats"]
    ]
    beta = float(np.mean(beta_vals)) if beta_vals else 0.0

    f1 = -(r1 - beta * kl)
    f2 = -(r2 - beta * kl)

    return r1, r2, f1, f2, kl, beta


def hypervolume_2d_min(points, ref):
    """
    2D hypervolume for minimization.

    points: array shape [N,2], smaller is better.
    ref: dominated reference point (upper-right).
    """

    pts = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float)

    # keep points dominating reference
    pts = pts[(pts[:,0] <= ref[0]) & (pts[:,1] <= ref[1])]

    if len(pts) == 0:
        return 0.0

    # nondominated filter
    nd = []
    for p in pts:
        dominated = False
        for q in pts:
            if np.all(q <= p) and np.any(q < p):
                dominated = True
                break
        if not dominated:
            nd.append(p)

    pts = np.array(nd)

    # sort by f1 ascending
    pts = pts[np.argsort(pts[:,0])]

    hv = 0.0
    prev_f2 = ref[1]

    for f1, f2 in pts:
        width = ref[0] - f1
        height = prev_f2 - f2

        if width > 0 and height > 0:
            hv += width * height

        prev_f2 = min(prev_f2, f2)

    return float(hv)

def main():
    pf_history = []
    metric_history = []
    checkpoint_mapping = []

    for outer_dir in sorted(RUN_ROOT.glob("outer_iter_*"), key=lambda p: int(p.name.split("_")[-1])):
        outer = int(outer_dir.name.split("_")[-1])
        point_rows = []
        ckpt_map = {}

        for slot in range(NUM_SEGMENTS + 1):
            adapter_dir = outer_dir / f"adapter_{slot}"
            log_path = adapter_dir / "logs" / "training_metrics.jsonl"
            ckpt_final = adapter_dir / "checkpoint_final"

            if not log_path.is_file():
                raise FileNotFoundError(f"Missing log: {log_path}")

            r1, r2, f1, f2, mean_kl, kl_coef = read_training_log(
                log_path, TAIL_FRACTION
            )

            weight = slot / NUM_SEGMENTS
            row = {
                "slot": slot,
                "quantile": weight,
                "weight": weight,
                "run_dir": str(adapter_dir),
                "checkpoint_final": str(ckpt_final),
                "E_r1": r1,
                "E_r2": r2,
                "f1": f1,
                "f2": f2,
                "mean_kl": mean_kl,
                "kl_coef": kl_coef,
                "pf_source": "training_log",
            }
            point_rows.append(row)
            ckpt_map[str(slot)] = str(ckpt_final)

        point_rows = sorted(point_rows, key=lambda r: int(r["slot"]))

        with (outer_dir / "point_meta.json").open("w") as f:
            json.dump({"outer_iter": outer, "points": point_rows}, f, indent=2)

        ordered = sorted(point_rows, key=lambda r: float(r["weight"]))
        z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=float)

        ell = compute_segment_lengths(z)
        cv = compute_cv(ell)
        gap_ratio = compute_gap_ratio(ell)
        hv = hypervolume_2d_min(z, HV_REF)

        pf_history.append({"outer_iter": outer, "points": point_rows})
        checkpoint_mapping.append({"outer_iter": outer, "checkpoints": ckpt_map})
        metric_history.append(
            {
                "outer_iter": outer,
                "cv": cv,
                "gap_ratio": gap_ratio,
                "hypervolume": hv,
                "hypervolume_ref": HV_REF.tolist(),
            }
        )

        print(
            f"[ok] outer_iter_{outer}: "
            f"cv={cv:.4f}, gap={gap_ratio:.4f}, hv={hv:.4f}"
        )

    with (RUN_ROOT / "pf_history.json").open("w") as f:
        json.dump(pf_history, f, indent=2)

    with (RUN_ROOT / "metric_history.json").open("w") as f:
        json.dump(metric_history, f, indent=2)

    with (RUN_ROOT / "checkpoint_mapping.json").open("w") as f:
        json.dump(checkpoint_mapping, f, indent=2)

    print("Rebuilt:")
    print(RUN_ROOT / "pf_history.json")
    print(RUN_ROOT / "metric_history.json")
    print(RUN_ROOT / "checkpoint_mapping.json")


if __name__ == "__main__":
    main()
