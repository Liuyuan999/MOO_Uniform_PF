"""Outer-loop LS with fixed equispaced weights and CDF-like outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from qwen.tasks import summary
from qwen.train_ppo import run_ppo
from qwen.utils import args_utils, inference_utils
from qwen.utils.cdf_utils import (
    build_surrogate_cdf_from_points,
    compute_cv,
    compute_gap_ratio,
    compute_segment_lengths,
    make_uniform_cdf_grid,
)
from qwen.utils.qwen_utils import Pipelines, Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LsExecutionConfig:
    mode: Literal["sequential", "parallel"] = "sequential"
    max_concurrent_jobs: int = 1


@dataclass
class LsWarmStartConfig:
    strategy: str = "same_index_then_nearest"
    explicit_init_adapter: str | None = None
    apply_explicit_every_outer_iter: bool = True
    init_adapters_by_slot: list[str | None] | dict[int, str] | None = None


@dataclass
class LsInnerLoopConfig:
    budget_mode: Literal["epochs", "steps"] = "epochs"
    budget_value: int = 1
    num_processes: int = 1


@dataclass
class _LsInnerJob:
    """Parameters for a single inner PPO training job."""

    slot_n: int
    adapter_name: str
    w_n: float
    init_path: Path | None
    num_epochs: int
    max_updates: int | None
    seed: int


@dataclass
class LsEvaluationConfig:
    #: "training_log" (default) reads tail-averaged rewards from training_metrics.jsonl —
    #: fast, no extra GPU inference.  "eval" runs a separate inference pass on a held-out set.
    pf_source: Literal["training_log", "eval"] = "training_log"
    #: Fraction of training steps (from the end) to average when pf_source="training_log".
    tail_fraction: float = 0.3
    # ---- fields below are only used when pf_source="eval" ----
    objective_coordinates: str = "raw_rewards"
    deterministic_decoding: bool = True
    max_eval_samples: int = 512
    split: str = "train"


@dataclass
class LsResumeConfig:
    enabled: bool = False
    run_root: str | None = None
    require_config_match: bool = True


@dataclass
class LsUniformYaml:
    run_name: str
    output_root: str
    seed: int
    num_outer_iters: int
    num_segments: int
    cdf_grid_size: int
    execution: LsExecutionConfig = field(default_factory=LsExecutionConfig)
    warm_start: LsWarmStartConfig = field(default_factory=LsWarmStartConfig)
    inner_loop: LsInnerLoopConfig = field(default_factory=LsInnerLoopConfig)
    evaluation: LsEvaluationConfig = field(default_factory=LsEvaluationConfig)
    resume: LsResumeConfig = field(default_factory=LsResumeConfig)
    ppo_config_paths: list[str] = field(default_factory=list)
    ppo_overrides: dict[str, Any] = field(default_factory=dict)


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _as_path_list(paths: list[str]) -> list[Path]:
    return [_resolve_repo_path(p) for p in paths]


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _merged_ppo_dict(ls: LsUniformYaml) -> dict[str, Any]:
    merged = args_utils.merge_yaml(_as_path_list(ls.ppo_config_paths))
    return args_utils._deep_merge(merged, ls.ppo_overrides)


def _build_ppo_merged_stack(ls: LsUniformYaml, dump_path: Path) -> None:
    merged = _merged_ppo_dict(ls)
    with dump_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def _load_json_list(path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing required resume history file: {path}")
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"Expected list JSON in {path}, got {type(obj).__name__}")
    return obj


def _validate_outer_list(name: str, rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"{name}[{i}] must be an object, got {type(row).__name__}")
        if "outer_iter" not in row:
            raise KeyError(f"{name}[{i}] missing outer_iter")
        if int(row["outer_iter"]) != i:
            raise ValueError(f"{name} outer_iter mismatch at index {i}: found {row['outer_iter']}")


def _resolve_run_root(ls: LsUniformYaml) -> Path:
    if ls.resume.enabled and ls.resume.run_root:
        return _resolve_repo_path(ls.resume.run_root)
    return _resolve_repo_path(Path(ls.output_root) / ls.run_name)


def _all_resume_json_paths(run_root: Path) -> tuple[Path, ...]:
    return (
        run_root / "weight_history.json",
        run_root / "pf_history.json",
        run_root / "metric_history.json",
        run_root / "cdf_history.json",
        run_root / "checkpoint_mapping.json",
    )


def _outer_dir_has_all_checkpoint_finals(outer_dir: Path, n_points: int) -> bool:
    if not outer_dir.is_dir():
        return False
    for s in range(n_points):
        ck = outer_dir / f"adapter_{s}" / "checkpoint_final" / "adapter"
        if not ck.is_dir():
            return False
    return True


def _point_rows_complete_for_pf(points: Any, n_points: int) -> bool:
    if not isinstance(points, list) or len(points) != n_points:
        return False
    slots_seen: set[int] = set()
    for row in points:
        if not isinstance(row, dict):
            return False
        if "slot" not in row:
            return False
        slots_seen.add(int(row["slot"]))
        for k in ("f1", "f2", "E_r1", "E_r2", "checkpoint_final"):
            if k not in row:
                return False
    return slots_seen == set(range(n_points))


def _load_or_validate_json_histories(
    run_root: Path,
    w_grid: np.ndarray,
    n_points: int,
) -> tuple[
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    float | None,
] | None:
    """Return state tuple if all JSON files exist and are valid; else None."""
    paths = _all_resume_json_paths(run_root)
    if not all(p.is_file() for p in paths):
        return None
    weight_history = _load_json_list(paths[0], required=False)
    pf_history = _load_json_list(paths[1], required=False)
    metric_history = _load_json_list(paths[2], required=False)
    cdf_history = _load_json_list(paths[3], required=False)
    checkpoint_mapping = _load_json_list(paths[4], required=False)

    for name, rows in [
        ("weight_history", weight_history),
        ("pf_history", pf_history),
        ("metric_history", metric_history),
        ("cdf_history", cdf_history),
        ("checkpoint_mapping", checkpoint_mapping),
    ]:
        try:
            _validate_outer_list(name, rows)
        except (ValueError, KeyError, TypeError):
            return None

    lengths = {
        len(weight_history),
        len(pf_history),
        len(metric_history),
        len(cdf_history),
        len(checkpoint_mapping),
    }
    if len(lengths) != 1:
        return None
    if len(cdf_history) == 0:
        metric0 = metric_history[0] if metric_history else {}
        baseline_cv = float(metric0["baseline_cv_iter0"]) if "baseline_cv_iter0" in metric0 else None
        baseline_gap = float(metric0["baseline_gap_ratio_iter0"]) if "baseline_gap_ratio_iter0" in metric0 else None
        return 0, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap

    for i, row in enumerate(weight_history):
        slots = row.get("weights_slots")
        if not isinstance(slots, list) or len(slots) != n_points:
            return None
    for i, row in enumerate(pf_history):
        pts = row.get("points")
        if not isinstance(pts, list) or len(pts) != n_points:
            return None

    last_f = np.asarray(cdf_history[-1].get("F_on_grid"), dtype=np.float64)
    if last_f.size and last_f.shape != w_grid.shape:
        return None

    metric0 = metric_history[0] if metric_history else {}
    baseline_cv = float(metric0["baseline_cv_iter0"]) if "baseline_cv_iter0" in metric0 else None
    baseline_gap = float(metric0["baseline_gap_ratio_iter0"]) if "baseline_gap_ratio_iter0" in metric0 else None

    return (
        len(cdf_history),
        weight_history,
        pf_history,
        metric_history,
        cdf_history,
        checkpoint_mapping,
        baseline_cv,
        baseline_gap,
    )


def _append_synthesized_outer_histories(
    *,
    outer_k: int,
    point_rows: list[dict[str, Any]],
    w_slots: np.ndarray,
    w_grid: np.ndarray,
    F_uniform: np.ndarray,
    weight_history: list[dict[str, Any]],
    pf_history: list[dict[str, Any]],
    metric_history: list[dict[str, Any]],
    cdf_history: list[dict[str, Any]],
    checkpoint_mapping: list[dict[str, Any]],
    baseline_cv: float | None,
    baseline_gap: float | None,
) -> tuple[float | None, float | None]:
    weight_history.append(
        {
            "outer_iter": outer_k,
            "weights_raw": w_slots.tolist(),
            "weights_slots": w_slots.tolist(),
        }
    )
    ckpt_map: dict[str, str] = {}
    for row in point_rows:
        ckpt_map[str(int(row["slot"]))] = str(row["checkpoint_final"])
    checkpoint_mapping.append({"outer_iter": outer_k, "checkpoints": ckpt_map})

    ordered = sorted(point_rows, key=lambda r: float(r["weight"]))
    z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=np.float64)
    ell_arr = compute_segment_lengths(z) if z.shape[0] >= 2 else np.array([], dtype=np.float64)
    cv_r = compute_cv(ell_arr)
    gap_r = compute_gap_ratio(ell_arr)
    if outer_k == 0 and baseline_cv is None:
        baseline_cv = cv_r
        baseline_gap = gap_r

    F_tilde, _s_at = build_surrogate_cdf_from_points(
        np.array([float(p["weight"]) for p in ordered], dtype=np.float64),
        z,
        w_grid,
        use_pchip=True,
    )
    cdf_history.append(
        {
            "outer_iter": outer_k,
            "F_on_grid": F_uniform.tolist(),
            "F_tilde_on_grid": F_tilde.tolist(),
        }
    )
    metric_history.append(
        {
            "outer_iter": outer_k,
            "cv": cv_r,
            "gap_ratio": gap_r,
            "baseline_cv_iter0": baseline_cv,
            "baseline_gap_ratio_iter0": baseline_gap,
        }
    )
    pf_history.append({"outer_iter": outer_k, "points": point_rows})
    return baseline_cv, baseline_gap


def _bootstrap_ls_state_from_disk(
    run_root: Path,
    *,
    w_slots: np.ndarray,
    w_grid: np.ndarray,
    F_uniform: np.ndarray,
    n_points: int,
) -> tuple[
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    float | None,
]:
    """Rebuild resume state when JSON histories are missing or invalid (partial runs)."""
    weight_history: list[dict[str, Any]] = []
    pf_history: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    cdf_history: list[dict[str, Any]] = []
    checkpoint_mapping: list[dict[str, Any]] = []
    baseline_cv: float | None = None
    baseline_gap: float | None = None

    k = 0
    while True:
        outer_dir = run_root / f"outer_iter_{k}"
        if not outer_dir.is_dir():
            return k, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap

        if not _outer_dir_has_all_checkpoint_finals(outer_dir, n_points):
            return k, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap

        meta_path = outer_dir / "point_meta.json"
        if not meta_path.is_file():
            return k, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pts = meta.get("points", [])
        if not _point_rows_complete_for_pf(pts, n_points):
            return k, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap

        baseline_cv, baseline_gap = _append_synthesized_outer_histories(
            outer_k=k,
            point_rows=list(pts),
            w_slots=w_slots,
            w_grid=w_grid,
            F_uniform=F_uniform,
            weight_history=weight_history,
            pf_history=pf_history,
            metric_history=metric_history,
            cdf_history=cdf_history,
            checkpoint_mapping=checkpoint_mapping,
            baseline_cv=baseline_cv,
            baseline_gap=baseline_gap,
        )
        k += 1


def _load_or_init_ls_state(
    ls: LsUniformYaml,
    *,
    run_root: Path,
    w_grid: np.ndarray,
    F_uniform: np.ndarray,
    w_slots: np.ndarray,
    n_points: int,
) -> tuple[
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    float | None,
]:
    if not ls.resume.enabled:
        return 0, [], [], [], [], [], None, None

    json_state = _load_or_validate_json_histories(run_root, w_grid, n_points)
    if json_state is not None:
        return json_state

    return _bootstrap_ls_state_from_disk(
        run_root,
        w_slots=w_slots,
        w_grid=w_grid,
        F_uniform=F_uniform,
        n_points=n_points,
    )


def _row_for_slot(point_rows: list[dict[str, Any]], slot_n: int) -> dict[str, Any] | None:
    for row in point_rows:
        if int(row.get("slot", -1)) == int(slot_n):
            return row
    return None


def _row_has_eval(row: dict[str, Any]) -> bool:
    needed = {"E_r1", "E_r2", "f1", "f2", "checkpoint_final", "run_dir", "weight", "quantile", "slot"}
    return all(k in row for k in needed)


def _uniform_weights(num_segments: int) -> np.ndarray:
    n = int(num_segments)
    return np.array([k / n for k in range(n + 1)], dtype=np.float64)


def _nearest_prev_checkpoint(prev_outer_dir: Path, weight: float) -> Path | None:
    meta_path = prev_outer_dir / "point_meta.json"
    if not meta_path.is_file():
        return None
    rows = json.loads(meta_path.read_text(encoding="utf-8")).get("points", [])
    pairs: list[tuple[float, Path]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ck = Path(str(r.get("checkpoint_final", "")))
        if ck.exists():
            pairs.append((float(r["weight"]), ck))
    if not pairs:
        return None
    _, best = min(pairs, key=lambda wp: abs(wp[0] - float(weight)))
    return best.resolve()


def _per_slot_init_path(ls: LsUniformYaml, slot_n: int) -> Path | None:
    raw = ls.warm_start.init_adapters_by_slot
    if raw is None:
        return None
    if isinstance(raw, dict):
        v = raw.get(slot_n)
        if v is None:
            for k, vv in raw.items():
                if int(k) == slot_n:
                    v = vv
                    break
    else:
        if slot_n < 0 or slot_n >= len(raw):
            return None
        v = raw[slot_n]
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    p = _resolve_repo_path(str(v))
    if not p.exists():
        raise FileNotFoundError(f"warm_start.init_adapters_by_slot[{slot_n}] not found: {p}")
    return p


def _explicit_init_path(ls: LsUniformYaml) -> Path | None:
    s = ls.warm_start.explicit_init_adapter
    if not s or not str(s).strip():
        return None
    p = _resolve_repo_path(s)
    if not p.exists():
        raise FileNotFoundError(f"warm_start.explicit_init_adapter not found: {p}")
    return p


def _resolve_init_for_slot(
    ls: LsUniformYaml,
    *,
    outer: int,
    slot_n: int,
    adapter_name: str,
    weight: float,
    prev_outer: Path | None,
) -> Path | None:
    exp = _explicit_init_path(ls)
    if exp is not None and ls.warm_start.apply_explicit_every_outer_iter:
        return exp
    if outer == 0:
        per = _per_slot_init_path(ls, slot_n)
        return per if per is not None else exp

    if prev_outer is not None:
        same = prev_outer / adapter_name / "checkpoint_final"
        if (same / "adapter").is_dir():
            return same.resolve()
        near = _nearest_prev_checkpoint(prev_outer, weight)
        if near is not None:
            return near
    per = _per_slot_init_path(ls, slot_n)
    if per is not None:
        return per
    return exp


def _run_inner_ppo(
    *,
    merged_yaml: Path,
    weight: float,
    output_dir: Path,
    run_name: str,
    init_adapter: Path | None,
    num_epochs: int,
    max_updates: int | None,
    seed: int,
    num_processes: int,
) -> None:
    if num_processes <= 1:
        run_ppo(
            config_paths=[merged_yaml],
            weight=weight,
            output_dir=str(output_dir),
            run_name=run_name,
            init_adapter=str(init_adapter) if init_adapter else None,
            num_epochs=num_epochs,
            max_updates=max_updates,
            seed=seed,
        )
        return

    accel = shutil.which("accelerate")
    if not accel:
        raise RuntimeError("inner_loop.num_processes > 1 requires `accelerate` on PATH")
    cmd = [
        accel,
        "launch",
        f"--num_processes={int(num_processes)}",
        "--num_machines=1",
        "-m",
        "qwen.train_ppo",
        "--config",
        str(merged_yaml.resolve()),
        "--weight",
        str(float(weight)),
        "--output_dir",
        str(output_dir.resolve()),
        "--run_name",
        run_name,
        "--num_epochs",
        str(int(num_epochs)),
        "--seed",
        str(int(seed)),
    ]
    if max_updates is not None:
        cmd.extend(["--max_updates", str(int(max_updates))])
    if init_adapter is not None:
        cmd.extend(["--init_adapter", str(init_adapter.resolve())])
    env = os.environ.copy()
    root = str(REPO_ROOT)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{pp}" if pp else root
    subprocess.run(cmd, check=True, cwd=root, env=env)


def _build_single_gpu_cmd(
    *,
    merged_yaml: Path,
    weight: float,
    output_dir: Path,
    run_name: str,
    init_adapter: Path | None,
    num_epochs: int,
    max_updates: int | None,
    seed: int,
) -> list[str]:
    """Build a ``python -m qwen.train_ppo`` command for a single-GPU subprocess."""
    python = shutil.which("python3") or shutil.which("python") or "python3"
    cmd: list[str] = [
        python, "-m", "qwen.train_ppo",
        "--config", str(merged_yaml.resolve()),
        "--weight", str(weight),
        "--output_dir", str(output_dir.resolve()),
        "--run_name", run_name,
        "--num_epochs", str(int(num_epochs)),
        "--seed", str(int(seed)),
    ]
    if max_updates is not None:
        cmd.extend(["--max_updates", str(int(max_updates))])
    if init_adapter is not None:
        cmd.extend(["--init_adapter", str(init_adapter.resolve())])
    return cmd


def _run_inner_jobs_parallel(
    jobs: list[_LsInnerJob],
    *,
    ls: LsUniformYaml,
    merged_yaml: Path,
    outer_dir: Path,
    outer: int,
) -> None:
    """Run multiple inner PPO jobs concurrently, one GPU per job.

    Uses ``CUDA_VISIBLE_DEVICES`` to pin each subprocess to a single GPU.
    Launches at most ``execution.max_concurrent_jobs`` subprocesses at once.
    Recycles freed GPU slots as jobs complete.  Raises immediately if any job
    exits with a non-zero return code (also terminates all remaining jobs).
    """
    max_jobs = ls.execution.max_concurrent_jobs
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError(
            "execution.mode=parallel requires CUDA GPUs, but torch.cuda.device_count()==0."
        )
    if max_jobs > n_gpu:
        raise ValueError(
            f"execution.max_concurrent_jobs={max_jobs} exceeds available GPU count={n_gpu}. "
            "Reduce max_concurrent_jobs or allocate more GPUs."
        )

    root = str(REPO_ROOT)
    base_env = os.environ.copy()
    pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = f"{root}{os.pathsep}{pp}" if pp else root

    def _launch(job: _LsInnerJob, gpu_id: int) -> subprocess.Popen:
        cmd = _build_single_gpu_cmd(
            merged_yaml=merged_yaml,
            weight=job.w_n,
            output_dir=outer_dir,
            run_name=job.adapter_name,
            init_adapter=job.init_path,
            num_epochs=job.num_epochs,
            max_updates=job.max_updates,
            seed=job.seed,
        )
        env = {**base_env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        print(
            f"[ls-parallel] outer={outer} slot={job.slot_n} w={job.w_n:.4f} -> GPU {gpu_id}"
        )
        return subprocess.Popen(cmd, cwd=root, env=env)

    free_gpus: list[int] = list(range(max_jobs))
    pending = list(jobs)
    active: list[tuple[subprocess.Popen, int, _LsInnerJob]] = []

    # Fill initial GPU slots
    while pending and free_gpus:
        job = pending.pop(0)
        gpu_id = free_gpus.pop(0)
        active.append((_launch(job, gpu_id), gpu_id, job))

    # Poll until all jobs finish
    while active:
        time.sleep(2.0)
        still_active: list[tuple[subprocess.Popen, int, _LsInnerJob]] = []
        for proc, gpu_id, job in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, gpu_id, job))
            elif ret != 0:
                # Kill all remaining processes before re-raising
                for other_proc, _, _ in still_active:
                    try:
                        other_proc.terminate()
                    except Exception:
                        pass
                for other_proc, _, _ in active:
                    if other_proc is not proc:
                        try:
                            other_proc.terminate()
                        except Exception:
                            pass
                raise subprocess.CalledProcessError(ret, proc.args)
            else:
                print(
                    f"[ls-parallel] outer={outer} slot={job.slot_n} w={job.w_n:.4f} "
                    f"finished (GPU {gpu_id})"
                )
                free_gpus.append(gpu_id)
                if pending:
                    next_job = pending.pop(0)
                    next_gpu = free_gpus.pop(0)
                    still_active.append((_launch(next_job, next_gpu), next_gpu, next_job))
        active = still_active


def _estimate_pf_from_training_log(
    log_dir: Path,
    tail_fraction: float = 0.3,
) -> tuple[float, float, float, float, float, float]:
    """Return (E[r1], E[r2], f1, f2, mean_kl, kl_coef) from tail-averaged training batch rewards.

    ``f1 = -(E[r1] - kl_coef * mean_kl)``  and  ``f2 = -(E[r2] - kl_coef * mean_kl)``.

    Reads ``training_metrics.jsonl`` written by the inner PPO run and averages
    ``mean_reward_1``, ``mean_reward_2``, ``mean_kl``, and
    ``ppo_stats.objective_kl_coef`` over the last ``tail_fraction`` of
    per-step records.  Checkpoint records are skipped.  No GPU required.
    """
    jsonl_path = log_dir / "training_metrics.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"training_metrics.jsonl not found at {jsonl_path}. "
            "Set evaluation.pf_source=eval to use the inference-based estimator instead."
        )
    step_records: list[dict[str, Any]] = []
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        obj = json.loads(raw_line)
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            step_records.append(obj)
    if not step_records:
        raise RuntimeError(
            f"No per-step reward records found in {jsonl_path}. "
            "Cannot estimate PF point from training log."
        )
    n_tail = max(1, int(len(step_records) * float(tail_fraction)))
    tail = step_records[-n_tail:]
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


def evaluate_pf_point(
    *,
    merged_yaml: Path,
    checkpoint_final: Path,
    ls: LsUniformYaml,
) -> tuple[float, float, float, float]:
    cfg = args_utils.load_run_config([merged_yaml])
    if ls.evaluation.objective_coordinates != "raw_rewards":
        raise ValueError("Only objective_coordinates=raw_rewards is supported.")
    args_utils.set_seed(cfg.seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )
    n = int(ls.evaluation.max_eval_samples)
    split = ls.evaluation.split
    if split == cfg.train_split:
        ds = summary.build_dataset(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_train_samples=cfg.max_train_samples,
        )
        m = min(n, len(ds))
        rng = np.random.default_rng(int(cfg.seed))
        idxs = rng.choice(len(ds), size=m, replace=False)
        query_tensors = [ds[int(i)]["input_ids"] for i in idxs]
    else:
        query_tensors = summary.Samples.get_samples(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            bs=n,
            split=split,
        )

    base = inference_utils.Loader.load_base_model(cfg).to(device)
    adapter_dir = checkpoint_final / "adapter" if (checkpoint_final / "adapter").is_dir() else checkpoint_final
    model = inference_utils.Loader.load_peft_model(base, str(adapter_dir))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    reward_pipes = Pipelines.load_pipes(list(cfg.reward_models), device=device, cache_dir=cfg.hf_cache)
    if cfg.task_name == "reddit_summarization":
        for p in reward_pipes:
            p.tokenizer.pad_token_id = p.model.config.eos_token_id
    predictor = summary.PredictorSummary(
        reward_pipes=reward_pipes,
        tokenizer=tokenizer,
        output_max_length=cfg.eval_max_new_tokens,
        device=device,
    )
    out = inference_utils.evaluate_scalars_structured(
        predictor,
        model,
        query_tensors,
        cfg,
        include_kl=False,
        deterministic=ls.evaluation.deterministic_decoding,
    )
    r1 = float(out["reward_models"]["reward_model_1"])
    r2 = float(out["reward_models"]["reward_model_2"])
    return r1, r2, -r1, -r2


def load_ls_yaml(path: Path) -> LsUniformYaml:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ex = raw.get("execution") or {}
    ws = raw.get("warm_start") or {}
    inn = raw.get("inner_loop") or {}
    ev = raw.get("evaluation") or {}
    rs = raw.get("resume") or {}
    slot_raw = ws.get("init_adapters_by_slot")
    init_by_slot: list[str | None] | dict[int, str] | None = None
    if slot_raw is not None:
        if isinstance(slot_raw, dict):
            init_by_slot = {int(k): str(v) for k, v in slot_raw.items() if v is not None and str(v).strip()}
        elif isinstance(slot_raw, list):
            init_by_slot = [None if x is None else str(x) for x in slot_raw]
        else:
            raise TypeError("warm_start.init_adapters_by_slot must be a list or dict")
    return LsUniformYaml(
        run_name=str(raw["run_name"]),
        output_root=str(raw["output_root"]),
        seed=int(raw.get("seed", 0)),
        num_outer_iters=int(raw["num_outer_iters"]),
        num_segments=int(raw["num_segments"]),
        cdf_grid_size=int(raw.get("cdf_grid_size", 2001)),
        execution=LsExecutionConfig(
            mode=str(ex.get("mode", "sequential")),
            max_concurrent_jobs=int(ex.get("max_concurrent_jobs", 1)),
        ),
        warm_start=LsWarmStartConfig(
            strategy=str(ws.get("strategy", "same_index_then_nearest")),
            explicit_init_adapter=ws.get("explicit_init_adapter"),
            apply_explicit_every_outer_iter=bool(ws.get("apply_explicit_every_outer_iter", True)),
            init_adapters_by_slot=init_by_slot,
        ),
        inner_loop=LsInnerLoopConfig(
            budget_mode=str(inn.get("budget_mode", "epochs")),
            budget_value=int(inn.get("budget_value", 1)),
            num_processes=max(1, int(inn.get("num_processes", 1))),
        ),
        evaluation=LsEvaluationConfig(
            pf_source=str(ev.get("pf_source", "training_log")),
            tail_fraction=float(ev.get("tail_fraction", 0.3)),
            objective_coordinates=str(ev.get("objective_coordinates", "raw_rewards")),
            deterministic_decoding=bool(ev.get("deterministic_decoding", True)),
            max_eval_samples=int(ev.get("max_eval_samples", 512)),
            split=str(ev.get("split", "train")),
        ),
        resume=LsResumeConfig(
            enabled=bool(rs.get("enabled", False)),
            run_root=rs.get("run_root"),
            require_config_match=bool(rs.get("require_config_match", True)),
        ),
        ppo_config_paths=list(raw.get("ppo_config_paths") or []),
        ppo_overrides=dict(raw.get("ppo_overrides") or {}),
    )


def run_equispace_weighting(ls_yaml: Path) -> Path:
    ls = load_ls_yaml(ls_yaml)
    if ls.execution.mode not in ("sequential", "parallel"):
        raise ValueError(f"execution.mode must be 'sequential' or 'parallel', got {ls.execution.mode!r}")
    if ls.num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    if not ls.ppo_config_paths:
        raise ValueError("ls_uniform YAML must set ppo_config_paths (non-empty list).")
    n_points = ls.num_segments + 1
    slot_list = ls.warm_start.init_adapters_by_slot
    if isinstance(slot_list, list) and len(slot_list) > n_points:
        raise ValueError(f"warm_start.init_adapters_by_slot length exceeds num_segments+1={n_points}")

    run_root = _resolve_run_root(ls)
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot_cfg = run_root / "ls_uniform_config.snapshot.yaml"
    if not snapshot_cfg.exists() or not ls.resume.enabled:
        shutil.copy2(ls_yaml, snapshot_cfg)

    merged_yaml = run_root / "ppo_merged_stack.yaml"
    expected_merged = _merged_ppo_dict(ls)
    if ls.resume.enabled and merged_yaml.is_file():
        existing_merged = yaml.safe_load(merged_yaml.read_text(encoding="utf-8")) or {}
        if bool(ls.resume.require_config_match) and existing_merged != expected_merged:
            raise ValueError(
                "Resume config mismatch: existing ppo_merged_stack.yaml differs from current merged PPO config. "
                "Set resume.require_config_match=false to overwrite with current config."
            )
        if not bool(ls.resume.require_config_match) and existing_merged != expected_merged:
            _build_ppo_merged_stack(ls, merged_yaml)
    else:
        _build_ppo_merged_stack(ls, merged_yaml)

    w_grid, F_uniform = make_uniform_cdf_grid(ls.cdf_grid_size)
    w_slots = _uniform_weights(ls.num_segments)

    (
        start_outer,
        weight_history,
        pf_history,
        metric_history,
        cdf_history,
        checkpoint_mapping,
        baseline_cv,
        baseline_gap,
    ) = _load_or_init_ls_state(
        ls,
        run_root=run_root,
        w_grid=w_grid,
        F_uniform=F_uniform,
        w_slots=w_slots,
        n_points=n_points,
    )

    if int(ls.num_outer_iters) < start_outer:
        raise ValueError(
            f"num_outer_iters={ls.num_outer_iters} is smaller than completed resume outer count={start_outer}."
        )

    if (
        ls.resume.enabled
        and weight_history
        and not all(p.is_file() for p in _all_resume_json_paths(run_root))
    ):
        _write_json_atomic(run_root / "weight_history.json", weight_history)
        _write_json_atomic(run_root / "pf_history.json", pf_history)
        _write_json_atomic(run_root / "metric_history.json", metric_history)
        _write_json_atomic(run_root / "cdf_history.json", cdf_history)
        _write_json_atomic(run_root / "checkpoint_mapping.json", checkpoint_mapping)

    for outer in range(start_outer, int(ls.num_outer_iters)):
        outer_dir = run_root / f"outer_iter_{outer}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        prev_outer = run_root / f"outer_iter_{outer - 1}" if outer > 0 else None
        weight_history.append(
            {
                "outer_iter": outer,
                "weights_raw": w_slots.tolist(),
                "weights_slots": w_slots.tolist(),
            }
        )
        point_rows: list[dict[str, Any]] = []
        ckpt_map: dict[str, str] = {}
        existing_rows: list[dict[str, Any]] = []
        meta_path = outer_dir / "point_meta.json"
        if meta_path.is_file():
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            rows_raw = meta_obj.get("points", [])
            if not isinstance(rows_raw, list):
                raise TypeError(f"{meta_path} points must be a list")
            existing_rows = [r for r in rows_raw if isinstance(r, dict)]

        # ---- Phase 1: resolve budget and collect slot metadata ----
        if ls.inner_loop.budget_mode == "epochs":
            _num_epochs = int(ls.inner_loop.budget_value)
            _max_updates: int | None = None
        elif ls.inner_loop.budget_mode == "steps":
            _max_updates = int(ls.inner_loop.budget_value)
            _num_epochs = 50_000
        else:
            raise ValueError(f"Unknown inner_loop.budget_mode: {ls.inner_loop.budget_mode}")

        pending_jobs: list[_LsInnerJob] = []
        slot_status: list[str] = []  # "done" | "needs_eval" | "needs_train"

        for slot_n in range(n_points):
            w_n = float(w_slots[slot_n])
            adapter_name = f"adapter_{slot_n}"
            (outer_dir / adapter_name).mkdir(parents=True, exist_ok=True)
            ck_final = outer_dir / adapter_name / "checkpoint_final"
            prior_row = _row_for_slot(existing_rows, slot_n)

            if (ck_final / "adapter").is_dir():
                if prior_row is not None and _row_has_eval(prior_row):
                    slot_status.append("done")
                else:
                    slot_status.append("needs_eval")
            else:
                init_path = _resolve_init_for_slot(
                    ls,
                    outer=outer,
                    slot_n=slot_n,
                    adapter_name=adapter_name,
                    weight=w_n,
                    prev_outer=prev_outer,
                )
                pending_jobs.append(
                    _LsInnerJob(
                        slot_n=slot_n,
                        adapter_name=adapter_name,
                        w_n=w_n,
                        init_path=init_path,
                        num_epochs=_num_epochs,
                        max_updates=_max_updates,
                        seed=int(ls.seed) + outer * 1000 + slot_n,
                    )
                )
                slot_status.append("needs_train")

        # ---- Phase 2: run training jobs (parallel or sequential) ----
        if ls.execution.mode == "parallel" and pending_jobs:
            _run_inner_jobs_parallel(
                pending_jobs,
                ls=ls,
                merged_yaml=merged_yaml,
                outer_dir=outer_dir,
                outer=outer,
            )
        else:
            for job in pending_jobs:
                print(f"[ls-seq] outer={outer} slot={job.slot_n} w={job.w_n:.4f}")
                _run_inner_ppo(
                    merged_yaml=merged_yaml,
                    weight=job.w_n,
                    output_dir=outer_dir,
                    run_name=job.adapter_name,
                    init_adapter=job.init_path,
                    num_epochs=job.num_epochs,
                    max_updates=job.max_updates,
                    seed=job.seed,
                    num_processes=int(ls.inner_loop.num_processes),
                )

        # ---- Phase 3: evaluate and collect point rows ----
        for slot_n in range(n_points):
            w_n = float(w_slots[slot_n])
            adapter_name = f"adapter_{slot_n}"
            ck_final = outer_dir / adapter_name / "checkpoint_final"
            prior_row = _row_for_slot(existing_rows, slot_n)

            if slot_status[slot_n] == "done":
                row = dict(prior_row)  # type: ignore[arg-type]
                row["slot"] = slot_n
                row["quantile"] = w_n
                row["weight"] = w_n
                row["run_dir"] = str(ck_final.parent)
                row["checkpoint_final"] = str(ck_final)
                point_rows.append(row)
                ckpt_map[str(slot_n)] = str(ck_final)
            else:
                if not (ck_final / "adapter").is_dir():
                    raise FileNotFoundError(
                        f"Missing trained adapter checkpoint: {ck_final / 'adapter'}"
                    )
                log_dir = outer_dir / adapter_name / "logs"
                if ls.evaluation.pf_source == "training_log":
                    r1, r2, f1, f2, mean_kl, kl_coef = _estimate_pf_from_training_log(
                        log_dir, tail_fraction=ls.evaluation.tail_fraction
                    )
                else:
                    # Inference-based r1/r2; KL and beta sourced from the last training log entry.
                    r1_raw, r2_raw, _, _ = evaluate_pf_point(
                        merged_yaml=merged_yaml, checkpoint_final=ck_final, ls=ls
                    )
                    _, _, _, _, mean_kl, kl_coef = _estimate_pf_from_training_log(
                        log_dir, tail_fraction=ls.evaluation.tail_fraction
                    )
                    r1, r2 = r1_raw, r2_raw
                    f1 = -(r1 - kl_coef * mean_kl)
                    f2 = -(r2 - kl_coef * mean_kl)
                point_rows.append(
                    {
                        "slot": slot_n,
                        "quantile": w_n,
                        "weight": w_n,
                        "run_dir": str(ck_final.parent),
                        "checkpoint_final": str(ck_final),
                        "E_r1": r1,
                        "E_r2": r2,
                        "f1": f1,
                        "f2": f2,
                        "mean_kl": mean_kl,
                        "kl_coef": kl_coef,
                        "pf_source": ls.evaluation.pf_source,
                    }
                )
                ckpt_map[str(slot_n)] = str(ck_final)

        if len(point_rows) != n_points:
            raise RuntimeError(f"outer_iter {outer}: expected {n_points} PF points, got {len(point_rows)}")
        point_rows = sorted(point_rows, key=lambda r: int(r["slot"]))
        _write_json_atomic(outer_dir / "point_meta.json", {"outer_iter": outer, "points": point_rows})
        checkpoint_mapping.append({"outer_iter": outer, "checkpoints": ckpt_map})

        ordered = sorted(point_rows, key=lambda r: float(r["weight"]))
        z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=np.float64)
        ell_arr = compute_segment_lengths(z) if z.shape[0] >= 2 else np.array([], dtype=np.float64)
        cv_r = compute_cv(ell_arr)
        gap_r = compute_gap_ratio(ell_arr)
        if outer == 0 and baseline_cv is None:
            baseline_cv = cv_r
            baseline_gap = gap_r

        F_tilde, _s_at = build_surrogate_cdf_from_points(
            np.array([float(p["weight"]) for p in ordered], dtype=np.float64),
            z,
            w_grid,
            use_pchip=True,
        )
        cdf_history.append(
            {
                "outer_iter": outer,
                "F_on_grid": F_uniform.tolist(),
                "F_tilde_on_grid": F_tilde.tolist(),
            }
        )
        metric_history.append(
            {
                "outer_iter": outer,
                "cv": cv_r,
                "gap_ratio": gap_r,
                "baseline_cv_iter0": baseline_cv,
                "baseline_gap_ratio_iter0": baseline_gap,
            }
        )
        pf_history.append({"outer_iter": outer, "points": point_rows})

        _write_json_atomic(run_root / "weight_history.json", weight_history)
        _write_json_atomic(run_root / "pf_history.json", pf_history)
        _write_json_atomic(run_root / "metric_history.json", metric_history)
        _write_json_atomic(run_root / "cdf_history.json", cdf_history)
        _write_json_atomic(run_root / "checkpoint_mapping.json", checkpoint_mapping)

    return run_root


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-weight LS outer loop with CDF-like outputs")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs/train/ls_uniform.yaml"),
        help="Path to ls_uniform YAML",
    )
    ns = p.parse_args()
    out = run_equispace_weighting(_resolve_repo_path(ns.config))
    print("LS equispace run finished. Run root:", out)


if __name__ == "__main__":
    main()
