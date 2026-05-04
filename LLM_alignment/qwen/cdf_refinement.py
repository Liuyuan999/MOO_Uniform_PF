"""Outer-loop CDF refinement for arc-length-uniform Pareto sampling (see docs/cdf_refinement_skill_doc.md)."""

from __future__ import annotations

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
    blend_cdfs,
    build_surrogate_cdf_from_points,
    compute_cv,
    compute_gap_ratio,
    compute_segment_lengths,
    enforce_monotone_cdf,
    invert_cdf,
    make_uniform_cdf_grid,
)
from qwen.utils.qwen_utils import Pipelines, Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CdfExecutionConfig:
    mode: Literal["sequential", "parallel"] = "sequential"
    max_concurrent_jobs: int = 1
    pipeline_endpoints: bool = False


@dataclass
class CdfWarmStartConfig:
    strategy: str = "same_index_then_nearest"
    allow_endpoint_interp: bool = True
    explicit_init_adapter: str | None = None
    #: If True (default), ``explicit_init_adapter`` seeds every inner job on every outer iteration.
    #: If False, explicit adapter is used only when per-slot / previous-outer resolution yields nothing,
    #: except on outer_iter 0 where per-slot list is checked first, then explicit.
    apply_explicit_every_outer_iter: bool = True
    #: Length ``num_segments+1`` list (optional entries), or dict mapping slot index -> adapter path.
    init_adapters_by_slot: list[str | None] | dict[int, str] | None = None


@dataclass
class CdfInnerLoopConfig:
    budget_mode: Literal["epochs", "steps"] = "epochs"
    budget_value: int = 1
    #: 1 = in-process ``run_ppo``; >1 = ``accelerate launch --num_processes K -m qwen.train_ppo ...``.
    num_processes: int = 1


@dataclass
class _InnerJob:
    """Parameters for a single inner PPO training job."""

    slot_n: int
    adapter_name: str
    w_n: float
    init_path: Path | None
    num_epochs: int
    max_updates: int | None
    seed: int


@dataclass
class _ParallelInnerJob:
    """A single inner PPO job with its own outer output directory."""

    outer: int
    outer_dir: Path
    job: _InnerJob


@dataclass
class CdfEvaluationConfig:
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
class CdfResumeConfig:
    enabled: bool = False
    run_root: str | None = None
    require_config_match: bool = True


@dataclass
class CdfEndpointReuseConfig:
    """Optional endpoint checkpoint reuse from a prior LS run."""

    enabled: bool = False
    source_run_root: str | None = None
    #: If True, outer_iter_0 reuses all slots from source outer_iter_0.
    #: For outer_iter>0, only endpoints are reused.
    reuse_all_slots_on_outer0: bool = False
    fallback_when_missing: Literal["train"] = "train"


@dataclass
class CdfRefinementYaml:
    run_name: str
    output_root: str
    seed: int
    num_outer_iters: int
    num_segments: int
    alpha: float
    cdf_grid_size: int
    use_pchip: bool
    monotone_eps: float
    force_endpoints: bool
    execution: CdfExecutionConfig = field(default_factory=CdfExecutionConfig)
    warm_start: CdfWarmStartConfig = field(default_factory=CdfWarmStartConfig)
    inner_loop: CdfInnerLoopConfig = field(default_factory=CdfInnerLoopConfig)
    evaluation: CdfEvaluationConfig = field(default_factory=CdfEvaluationConfig)
    resume: CdfResumeConfig = field(default_factory=CdfResumeConfig)
    endpoint_reuse: CdfEndpointReuseConfig = field(default_factory=CdfEndpointReuseConfig)
    ppo_config_paths: list[str] = field(default_factory=list)
    ppo_overrides: dict[str, Any] = field(default_factory=dict)


def _as_path_list(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        out.append(pp if pp.is_absolute() else (REPO_ROOT / pp))
    return out


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolved_explicit_init(cdf: CdfRefinementYaml) -> Path | None:
    s = cdf.warm_start.explicit_init_adapter
    if not s or not str(s).strip():
        return None
    path = _resolve_repo_path(str(s))
    if not path.exists():
        raise FileNotFoundError(f"warm_start.explicit_init_adapter not found: {path}")
    return path


def _per_slot_init_path(cdf: CdfRefinementYaml, slot_n: int) -> Path | None:
    raw = cdf.warm_start.init_adapters_by_slot
    if raw is None:
        return None
    if isinstance(raw, dict):
        entry = raw.get(slot_n)
        if entry is None:
            for k, v in raw.items():
                if int(k) == slot_n:
                    entry = v
                    break
        if entry is None or (isinstance(entry, str) and not entry.strip()):
            return None
        p = str(entry)
    else:
        if slot_n < 0 or slot_n >= len(raw):
            return None
        entry = raw[slot_n]
        if entry is None or (isinstance(entry, str) and not str(entry).strip()):
            return None
        p = str(entry)
    path = _resolve_repo_path(p)
    if not path.exists():
        raise FileNotFoundError(f"warm_start.init_adapters_by_slot[{slot_n}] not found: {path}")
    return path


def _warm_start_from_previous_outer(
    *,
    cdf: CdfRefinementYaml,
    prev_outer: Path,
    adapter_name: str,
    weight: float,
    outer_dir: Path,
    merged_yaml: Path,
    slot_n: int,
) -> Path | None:
    same = prev_outer / adapter_name / "checkpoint_final"
    if (same / "adapter").is_dir():
        return same.resolve()
    # Previous outer slot may be a reused external checkpoint recorded in point_meta.json.
    prev_meta_ckpt = _checkpoint_from_point_meta_slot(prev_outer, slot_n)
    if prev_meta_ckpt is not None:
        return prev_meta_ckpt
    nearest = _nearest_prev_checkpoint(prev_outer, weight)
    if nearest is not None:
        return nearest
    return _interp_init_adapter_dir(
        cdf=cdf,
        merged_yaml=merged_yaml,
        prev_outer_dir=prev_outer,
        point_index=slot_n,
        weight=weight,
        num_segments=cdf.num_segments,
        scratch_root=outer_dir,
    )


def _resolve_init_adapter_for_slot(
    cdf: CdfRefinementYaml,
    *,
    outer: int,
    slot_n: int,
    adapter_name: str,
    weight: float,
    prev_outer: Path | None,
    outer_dir: Path,
    merged_yaml: Path,
) -> Path | None:
    """Resolve LoRA init checkpoint for this inner PPO job."""
    explicit = _resolved_explicit_init(cdf)
    if cdf.warm_start.apply_explicit_every_outer_iter and explicit is not None:
        return explicit

    if outer > 0 and prev_outer is not None:
        warm = _warm_start_from_previous_outer(
            cdf=cdf,
            prev_outer=prev_outer,
            adapter_name=adapter_name,
            weight=weight,
            outer_dir=outer_dir,
            merged_yaml=merged_yaml,
            slot_n=slot_n,
        )
        if warm is not None:
            return warm
        if explicit is not None:
            return explicit
        per = _per_slot_init_path(cdf, slot_n)
        if per is not None:
            return per
        return None

    # outer_iter == 0: per-slot overrides, then shared explicit
    per = _per_slot_init_path(cdf, slot_n)
    if per is not None:
        return per
    if explicit is not None:
        return explicit
    return None


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
    """Run one scalarized PPO job (in-process or via Accelerate for multi-GPU)."""
    if num_processes <= 1:
        run_ppo(
            config_paths=[merged_yaml],
            weight=weight,
            output_dir=str(output_dir),
            run_name=run_name,
            init_adapter=str(init_adapter) if init_adapter else None,
            resume=None,
            num_epochs=num_epochs,
            max_updates=max_updates,
            seed=seed,
        )
        return

    accel = shutil.which("accelerate")
    if not accel:
        raise RuntimeError(
            "inner_loop.num_processes > 1 requires the `accelerate` CLI on PATH "
            "(e.g. pip install accelerate). Or set inner_loop.num_processes: 1."
        )
    cmd: list[str] = [
        accel,
        "launch",
        f"--num_processes={int(num_processes)}",
        "--num_machines=1",
        "-m",
        "qwen.train_ppo",
        "--config",
        str(merged_yaml.resolve()),
        "--weight",
        str(weight),
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


class _ParallelInnerJobScheduler:
    """Small subprocess scheduler for one-GPU inner PPO jobs."""

    def __init__(self, *, cdf: CdfRefinementYaml, merged_yaml: Path) -> None:
        self.cdf = cdf
        self.merged_yaml = merged_yaml
        self.max_jobs = int(cdf.execution.max_concurrent_jobs)
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            raise RuntimeError(
                "execution.mode=parallel requires CUDA GPUs, but torch.cuda.device_count()==0."
            )
        if self.max_jobs > n_gpu:
            raise ValueError(
                f"execution.max_concurrent_jobs={self.max_jobs} exceeds available GPU count={n_gpu}. "
                "Reduce max_concurrent_jobs or allocate more GPUs."
            )

        root = str(REPO_ROOT)
        self.root = root
        self.base_env = os.environ.copy()
        pp = self.base_env.get("PYTHONPATH", "")
        self.base_env["PYTHONPATH"] = f"{root}{os.pathsep}{pp}" if pp else root
        self.free_gpus: list[int] = list(range(self.max_jobs))
        self.pending: list[_ParallelInnerJob] = []
        self.active: list[tuple[subprocess.Popen, int, _ParallelInnerJob]] = []
        self.active_keys: set[tuple[int, int]] = set()
        self.completed_keys: set[tuple[int, int]] = set()

    @staticmethod
    def key(spec: _ParallelInnerJob) -> tuple[int, int]:
        return (int(spec.outer), int(spec.job.slot_n))

    @staticmethod
    def checkpoint_exists(spec: _ParallelInnerJob) -> bool:
        return (spec.outer_dir / spec.job.adapter_name / "checkpoint_final" / "adapter").is_dir()

    def has_job(self, outer: int, slot_n: int) -> bool:
        key = (int(outer), int(slot_n))
        if key in self.active_keys or key in self.completed_keys:
            return True
        return any(self.key(spec) == key for spec in self.pending)

    def submit(self, spec: _ParallelInnerJob) -> bool:
        key = self.key(spec)
        if key in self.active_keys or key in self.completed_keys:
            return False
        if any(self.key(p) == key for p in self.pending):
            return False
        if self.checkpoint_exists(spec):
            self.completed_keys.add(key)
            return False
        self.pending.append(spec)
        self._fill_free_gpus()
        return True

    def _launch(self, spec: _ParallelInnerJob, gpu_id: int) -> subprocess.Popen:
        job = spec.job
        cmd = _build_single_gpu_cmd(
            merged_yaml=self.merged_yaml,
            weight=job.w_n,
            output_dir=spec.outer_dir,
            run_name=job.adapter_name,
            init_adapter=job.init_path,
            num_epochs=job.num_epochs,
            max_updates=job.max_updates,
            seed=job.seed,
        )
        env = {**self.base_env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        print(
            f"[cdf-parallel] outer={spec.outer} slot={job.slot_n} "
            f"w={job.w_n:.4f} -> GPU {gpu_id}"
        )
        return subprocess.Popen(cmd, cwd=self.root, env=env)

    def _fill_free_gpus(self) -> None:
        while self.pending and self.free_gpus:
            spec = self.pending.pop(0)
            gpu_id = self.free_gpus.pop(0)
            proc = self._launch(spec, gpu_id)
            self.active.append((proc, gpu_id, spec))
            self.active_keys.add(self.key(spec))

    def poll_once(self) -> list[_ParallelInnerJob]:
        completed: list[_ParallelInnerJob] = []
        still_active: list[tuple[subprocess.Popen, int, _ParallelInnerJob]] = []
        for proc, gpu_id, spec in self.active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, gpu_id, spec))
                continue
            self.active_keys.discard(self.key(spec))
            if ret != 0:
                for other_proc, _, _ in still_active:
                    try:
                        other_proc.terminate()
                    except Exception:
                        pass
                for other_proc, _, other_spec in self.active:
                    if other_spec is not spec:
                        try:
                            other_proc.terminate()
                        except Exception:
                            pass
                raise subprocess.CalledProcessError(ret, proc.args)
            print(
                f"[cdf-parallel] outer={spec.outer} slot={spec.job.slot_n} "
                f"w={spec.job.w_n:.4f} finished (GPU {gpu_id})"
            )
            self.completed_keys.add(self.key(spec))
            self.free_gpus.append(gpu_id)
            completed.append(spec)
        self.active = still_active
        self._fill_free_gpus()
        return completed

    def wait_for_outer(self, outer: int, slots: set[int], *, on_complete: Any = None) -> None:
        wanted = {(int(outer), int(s)) for s in slots}
        while True:
            missing = [key for key in wanted if key not in self.completed_keys]
            if not missing:
                return
            if not self.active and not self.pending:
                raise RuntimeError(f"No active jobs left while waiting for outer={outer}, slots={sorted(slots)}")
            time.sleep(2.0)
            for spec in self.poll_once():
                if on_complete is not None:
                    on_complete(spec)

    def wait_all(self, *, on_complete: Any = None) -> None:
        while self.active or self.pending:
            time.sleep(2.0)
            for spec in self.poll_once():
                if on_complete is not None:
                    on_complete(spec)


def _run_inner_jobs_parallel(
    jobs: list[_InnerJob],
    *,
    cdf: CdfRefinementYaml,
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
    scheduler = _ParallelInnerJobScheduler(cdf=cdf, merged_yaml=merged_yaml)
    required_slots = {int(job.slot_n) for job in jobs}
    for job in jobs:
        scheduler.submit(_ParallelInnerJob(outer=outer, outer_dir=outer_dir, job=job))
    scheduler.wait_for_outer(outer, required_slots)


def load_cdf_yaml(path: Path) -> CdfRefinementYaml:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ex = raw.get("execution") or {}
    ws = raw.get("warm_start") or {}
    inn = raw.get("inner_loop") or {}
    slot_raw = ws.get("init_adapters_by_slot")
    init_by_slot: list[str | None] | dict[int, str] | None = None
    if slot_raw is not None:
        if isinstance(slot_raw, dict):
            init_by_slot = {}
            for k, v in slot_raw.items():
                if v is None or (isinstance(v, str) and not v.strip()):
                    continue
                init_by_slot[int(k)] = str(v)
        elif isinstance(slot_raw, list):
            init_by_slot = [None if x is None else str(x) for x in slot_raw]
        else:
            raise TypeError("warm_start.init_adapters_by_slot must be a list or dict")
    ev = raw.get("evaluation") or {}
    rs = raw.get("resume") or {}
    er = raw.get("endpoint_reuse") or {}
    return CdfRefinementYaml(
        run_name=str(raw["run_name"]),
        output_root=str(raw["output_root"]),
        seed=int(raw.get("seed", 0)),
        num_outer_iters=int(raw["num_outer_iters"]),
        num_segments=int(raw["num_segments"]),
        alpha=float(raw["alpha"]),
        cdf_grid_size=int(raw.get("cdf_grid_size", 2001)),
        use_pchip=bool(raw.get("use_pchip", True)),
        monotone_eps=float(raw.get("monotone_eps", 1e-8)),
        force_endpoints=bool(raw.get("force_endpoints", True)),
        execution=CdfExecutionConfig(
            mode=str(ex.get("mode", "sequential")),
            max_concurrent_jobs=int(ex.get("max_concurrent_jobs", 1)),
            pipeline_endpoints=bool(ex.get("pipeline_endpoints", False)),
        ),
        warm_start=CdfWarmStartConfig(
            strategy=str(ws.get("strategy", "same_index_then_nearest")),
            allow_endpoint_interp=bool(ws.get("allow_endpoint_interp", True)),
            explicit_init_adapter=ws.get("explicit_init_adapter"),
            apply_explicit_every_outer_iter=bool(ws.get("apply_explicit_every_outer_iter", True)),
            init_adapters_by_slot=init_by_slot,
        ),
        inner_loop=CdfInnerLoopConfig(
            budget_mode=str(inn.get("budget_mode", "epochs")),
            budget_value=int(inn.get("budget_value", 1)),
            num_processes=max(1, int(inn.get("num_processes", 1))),
        ),
        evaluation=CdfEvaluationConfig(
            pf_source=str(ev.get("pf_source", "training_log")),
            tail_fraction=float(ev.get("tail_fraction", 0.3)),
            objective_coordinates=str(ev.get("objective_coordinates", "raw_rewards")),
            deterministic_decoding=bool(ev.get("deterministic_decoding", True)),
            max_eval_samples=int(ev.get("max_eval_samples", 512)),
            split=str(ev.get("split", "train")),
        ),
        resume=CdfResumeConfig(
            enabled=bool(rs.get("enabled", False)),
            run_root=rs.get("run_root"),
            require_config_match=bool(rs.get("require_config_match", True)),
        ),
        endpoint_reuse=CdfEndpointReuseConfig(
            enabled=bool(er.get("enabled", False)),
            source_run_root=er.get("source_run_root"),
            reuse_all_slots_on_outer0=bool(er.get("reuse_all_slots_on_outer0", False)),
            fallback_when_missing=str(er.get("fallback_when_missing", "train")),
        ),
        ppo_config_paths=list(raw.get("ppo_config_paths") or []),
        ppo_overrides=dict(raw.get("ppo_overrides") or {}),
    )


def _build_merged_ppo_yaml(cdf: CdfRefinementYaml, dump_path: Path) -> None:
    merged = _merged_ppo_dict(cdf)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def _merged_ppo_dict(cdf: CdfRefinementYaml) -> dict[str, Any]:
    paths = _as_path_list(cdf.ppo_config_paths)
    merged = args_utils.merge_yaml(paths)
    return args_utils._deep_merge(merged, cdf.ppo_overrides)


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


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


def _resolve_run_root(cdf: CdfRefinementYaml) -> Path:
    if cdf.resume.enabled and cdf.resume.run_root:
        return _resolve_repo_path(cdf.resume.run_root)
    return (REPO_ROOT / cdf.output_root / cdf.run_name).resolve()


def _outer_dir_has_all_checkpoint_finals(outer_dir: Path, n_points: int) -> bool:
    if not outer_dir.is_dir():
        return False
    for s in range(n_points):
        if not (outer_dir / f"adapter_{s}" / "checkpoint_final" / "adapter").is_dir():
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


def _bootstrap_cdf_state_from_disk(
    run_root: Path,
    *,
    w_grid: np.ndarray,
    cdf: CdfRefinementYaml,
    n_points: int,
) -> tuple[
    np.ndarray,
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    float | None,
]:
    """Rebuild resume state by scanning outer_iter_k directories when JSON histories are missing.

    For each fully completed outer iteration found on disk (all adapter checkpoints present
    and point_meta.json valid), replays the CDF update to reconstruct the F array and all
    history lists.  Stops at the first incomplete or missing iteration.
    """
    F = w_grid.copy()
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
            break
        if not _outer_dir_has_all_checkpoint_finals(outer_dir, n_points):
            break
        meta_path = outer_dir / "point_meta.json"
        if not meta_path.is_file():
            break
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pts = meta.get("points", [])
        if not _point_rows_complete_for_pf(pts, n_points):
            break

        ordered = sorted(pts, key=lambda r: float(r["weight"]))
        w_slots_k = np.array([float(p["weight"]) for p in ordered], dtype=np.float64)
        z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=np.float64)

        # Reconstruct slot-ordered weights for weight_history
        pts_by_slot = sorted(pts, key=lambda r: int(r["slot"]))
        w_by_slot = [float(p["weight"]) for p in pts_by_slot]
        weight_history.append({
            "outer_iter": k,
            "weights_raw": w_by_slot,
            "weights_slots": w_by_slot,
        })

        ckpt_map: dict[str, str] = {}
        for row in pts:
            ckpt_map[str(int(row["slot"]))] = str(row["checkpoint_final"])
        checkpoint_mapping.append({"outer_iter": k, "checkpoints": ckpt_map})

        ell_arr = compute_segment_lengths(z) if z.shape[0] >= 2 else np.array([], dtype=np.float64)
        cv_r = compute_cv(ell_arr)
        gap_r = compute_gap_ratio(ell_arr)
        if k == 0 and baseline_cv is None:
            baseline_cv = cv_r
            baseline_gap = gap_r

        F_tilde, _ = build_surrogate_cdf_from_points(w_slots_k, z, w_grid, use_pchip=cdf.use_pchip)
        F_next = blend_cdfs(F, F_tilde, cdf.alpha)
        F_next = enforce_monotone_cdf(F_next, eps=cdf.monotone_eps, force_endpoints=cdf.force_endpoints)

        cdf_history.append({
            "outer_iter": k,
            "F_on_grid": F_next.tolist(),
            "F_tilde_on_grid": F_tilde.tolist(),
        })
        metric_history.append({
            "outer_iter": k,
            "cv": cv_r,
            "gap_ratio": gap_r,
            "baseline_cv_iter0": baseline_cv,
            "baseline_gap_ratio_iter0": baseline_gap,
        })
        pf_history.append({"outer_iter": k, "points": list(pts)})

        F = F_next
        k += 1

    print(f"[cdf-bootstrap] Reconstructed {k} outer iteration(s) from disk. start_outer={k}")
    return F, k, weight_history, pf_history, metric_history, cdf_history, checkpoint_mapping, baseline_cv, baseline_gap


def _load_or_init_state(
    cdf: CdfRefinementYaml,
    *,
    run_root: Path,
    w_grid: np.ndarray,
    n_points: int,
) -> tuple[
    np.ndarray,
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    float | None,
]:
    if not cdf.resume.enabled:
        return (
            w_grid.copy(),
            0,
            [],
            [],
            [],
            [],
            [],
            None,
            None,
        )

    json_paths = [
        run_root / "weight_history.json",
        run_root / "pf_history.json",
        run_root / "metric_history.json",
        run_root / "cdf_history.json",
        run_root / "checkpoint_mapping.json",
    ]
    if not all(p.is_file() for p in json_paths):
        print(
            "[cdf-resume] One or more JSON history files are missing — "
            "falling back to disk bootstrap from outer_iter_k directories."
        )
        state = _bootstrap_cdf_state_from_disk(
            run_root, w_grid=w_grid, cdf=cdf, n_points=n_points
        )
        # Persist bootstrapped histories so future resumes find the JSON files.
        if state[1] > 0:  # start_outer > 0 means we found something
            _, _, wh, pfh, mh, ch, ckm, _, _ = state
            _write_json_atomic(run_root / "weight_history.json", wh)
            _write_json_atomic(run_root / "pf_history.json", pfh)
            _write_json_atomic(run_root / "metric_history.json", mh)
            _write_json_atomic(run_root / "cdf_history.json", ch)
            _write_json_atomic(run_root / "checkpoint_mapping.json", ckm)
        return state

    weight_history = _load_json_list(run_root / "weight_history.json", required=True)
    pf_history = _load_json_list(run_root / "pf_history.json", required=True)
    metric_history = _load_json_list(run_root / "metric_history.json", required=True)
    cdf_history = _load_json_list(run_root / "cdf_history.json", required=True)
    checkpoint_mapping = _load_json_list(run_root / "checkpoint_mapping.json", required=True)

    for name, rows in [
        ("weight_history", weight_history),
        ("pf_history", pf_history),
        ("metric_history", metric_history),
        ("cdf_history", cdf_history),
        ("checkpoint_mapping", checkpoint_mapping),
    ]:
        _validate_outer_list(name, rows)

    lengths = {len(weight_history), len(pf_history), len(metric_history), len(cdf_history), len(checkpoint_mapping)}
    if len(lengths) != 1:
        raise ValueError(
            "Resume histories have different lengths: "
            f"weight={len(weight_history)}, pf={len(pf_history)}, metric={len(metric_history)}, "
            f"cdf={len(cdf_history)}, checkpoint={len(checkpoint_mapping)}"
        )
    if len(cdf_history) == 0:
        return (
            w_grid.copy(),
            0,
            weight_history,
            pf_history,
            metric_history,
            cdf_history,
            checkpoint_mapping,
            None,
            None,
        )

    for i, row in enumerate(weight_history):
        slots = row.get("weights_slots")
        if not isinstance(slots, list) or len(slots) != n_points:
            raise ValueError(
                f"weight_history[{i}] weights_slots must have length {n_points}; "
                f"got {None if not isinstance(slots, list) else len(slots)}"
            )
    for i, row in enumerate(pf_history):
        pts = row.get("points")
        if not isinstance(pts, list) or len(pts) != n_points:
            raise ValueError(
                f"pf_history[{i}].points must have length {n_points}; "
                f"got {None if not isinstance(pts, list) else len(pts)}"
            )

    f_last = np.asarray(cdf_history[-1].get("F_on_grid"), dtype=np.float64)
    if f_last.shape != w_grid.shape:
        raise ValueError(
            f"Resume CDF grid shape mismatch: expected {w_grid.shape}, got {f_last.shape}. "
            "Use matching cdf_grid_size."
        )
    f_last = enforce_monotone_cdf(f_last, eps=float(cdf.monotone_eps), force_endpoints=bool(cdf.force_endpoints))
    metric0 = metric_history[0] if metric_history else {}
    baseline_cv = float(metric0["baseline_cv_iter0"]) if "baseline_cv_iter0" in metric0 else None
    baseline_gap = float(metric0["baseline_gap_ratio_iter0"]) if "baseline_gap_ratio_iter0" in metric0 else None
    return (
        f_last,
        len(cdf_history),
        weight_history,
        pf_history,
        metric_history,
        cdf_history,
        checkpoint_mapping,
        baseline_cv,
        baseline_gap,
    )


def _row_for_slot(point_rows: list[dict[str, Any]], slot_n: int) -> dict[str, Any] | None:
    for row in point_rows:
        if int(row.get("slot", -1)) == int(slot_n):
            return row
    return None


def _row_has_eval(row: dict[str, Any]) -> bool:
    needed = {"E_r1", "E_r2", "f1", "f2", "checkpoint_final", "run_dir", "weight", "quantile", "slot"}
    return all(k in row for k in needed)


def _checkpoint_from_point_meta_slot(outer_dir: Path, slot_n: int) -> Path | None:
    """Resolve checkpoint path for a slot from outer_dir/point_meta.json."""
    meta_path = outer_dir / "point_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pts = meta.get("points")
    if not isinstance(pts, list):
        return None
    for row in pts:
        if not isinstance(row, dict):
            continue
        if int(row.get("slot", -1)) != int(slot_n):
            continue
        ck = row.get("checkpoint_final")
        if isinstance(ck, str) and ck.strip():
            p = Path(ck)
            if (p / "adapter").is_dir():
                return p.resolve()
    return None


def _resolve_endpoint_reuse_checkpoint(
    cdf: CdfRefinementYaml,
    *,
    outer: int,
    slot_n: int,
    n_points: int,
) -> Path | None:
    """Return reusable endpoint checkpoint for this outer/slot if configured."""
    er = cdf.endpoint_reuse
    if not er.enabled:
        return None
    if not (outer == 0 and er.reuse_all_slots_on_outer0):
        # Default behavior: only endpoints are reused.
        if slot_n not in (0, n_points - 1):
            return None
    # Special surrogate mode: outer_iter_0 can reuse all slots from source.
    if outer < 0:
        return None
    if not er.source_run_root:
        raise ValueError("endpoint_reuse.enabled=true requires endpoint_reuse.source_run_root")
    src_root = _resolve_repo_path(er.source_run_root)
    ck = src_root / f"outer_iter_{outer}" / f"adapter_{slot_n}" / "checkpoint_final"
    if (ck / "adapter").is_dir():
        return ck.resolve()
    if er.fallback_when_missing == "train":
        return None
    raise ValueError(f"Unknown endpoint_reuse.fallback_when_missing: {er.fallback_when_missing!r}")


def _quantile_grid(num_segments: int) -> np.ndarray:
    n = int(num_segments)
    return np.array([k / n for k in range(n + 1)], dtype=np.float64)


def _monotone_weights_on_slots(w_raw: np.ndarray, *, force_endpoints: bool, eps: float = 1e-9) -> np.ndarray:
    """Sort into non-decreasing order and pin endpoints (quantile slots ``0..N``)."""
    w = np.sort(np.clip(np.asarray(w_raw, dtype=np.float64).ravel(), 0.0, 1.0))
    if len(w) == 0:
        return w
    if force_endpoints:
        w[0] = 0.0
        w[-1] = 1.0
    for i in range(1, len(w)):
        if w[i] <= w[i - 1]:
            w[i] = min(w[i - 1] + eps, 1.0)
    return w


def _nearest_prev_checkpoint(prev_outer_dir: Path, weight: float) -> Path | None:
    meta_path = prev_outer_dir / "point_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pairs: list[tuple[float, Path]] = []
    for row in meta.get("points", []):
        ck = row.get("checkpoint_final")
        p = Path(ck) if ck else Path(row["run_dir"]) / "checkpoint_final"
        pairs.append((float(row["weight"]), p))
    if not pairs:
        return None
    _, best = min(pairs, key=lambda wp: abs(wp[0] - float(weight)))
    return best.resolve() if best.exists() else None


def _interp_init_adapter_dir(
    *,
    cdf: CdfRefinementYaml,
    merged_yaml: Path,
    prev_outer_dir: Path,
    point_index: int,
    weight: float,
    num_segments: int,
    scratch_root: Path,
) -> Path | None:
    warm = cdf.warm_start
    if not warm.allow_endpoint_interp or prev_outer_dir is None:
        return None
    (scratch_root / "_interp_cache").mkdir(parents=True, exist_ok=True)
    p0 = prev_outer_dir / "adapter_0" / "checkpoint_final"
    pN = prev_outer_dir / f"adapter_{num_segments}" / "checkpoint_final"
    a0 = p0 / "adapter" if (p0 / "adapter").is_dir() else p0
    aN = pN / "adapter" if (pN / "adapter").is_dir() else pN
    if not a0.is_dir() or not aN.is_dir():
        return None
    meta_all = prev_outer_dir / "point_meta.json"
    if not meta_all.is_file():
        return None
    pts = json.loads(meta_all.read_text(encoding="utf-8")).get("points", [])
    if len(pts) < 2:
        return None
    pts_sorted = sorted(pts, key=lambda r: float(r["weight"]))
    w0 = float(pts_sorted[0]["weight"])
    w1 = float(pts_sorted[-1]["weight"])
    denom = max(w1 - w0, 1e-12)
    lam = float(np.clip((float(weight) - w0) / denom, 0.0, 1.0))
    cfg = args_utils.load_run_config([merged_yaml])
    base = inference_utils.Loader.load_base_model(cfg)
    wa = inference_utils.WeightAverager.build_wa(cfg, [str(a0), str(aN)], [1.0 - lam, lam])
    out = scratch_root / "_interp_cache" / f"n{point_index}_w{weight:.6f}".replace(".", "p")
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    wa.save_pretrained(str(out))
    del wa
    del base
    return out.resolve()


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
    cdf: CdfRefinementYaml,
) -> tuple[float, float, float, float]:
    """Return (E[r1], E[r2], f1, f2) with f1=-E[r1], f2=-E[r2] for raw minimization coordinates."""
    cfg = args_utils.load_run_config([merged_yaml])
    if cdf.evaluation.objective_coordinates != "raw_rewards":
        raise ValueError("Only objective_coordinates=raw_rewards is supported in this milestone.")
    args_utils.set_seed(cfg.seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )
    n = cdf.evaluation.max_eval_samples
    split = cdf.evaluation.split
    if split == cfg.train_split:
        train_ds = summary.build_dataset(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_train_samples=cfg.max_train_samples,
        )
        m = min(int(n), len(train_ds))
        rng = np.random.default_rng(int(cfg.seed))
        idxs = rng.choice(len(train_ds), size=m, replace=False)
        query_tensors = [train_ds[int(i)]["input_ids"] for i in idxs]
    else:
        query_tensors = summary.Samples.get_samples(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            bs=n,
            split=split,
        )

    base_model = inference_utils.Loader.load_base_model(cfg).to(device)
    adapter_dir = checkpoint_final / "adapter" if (checkpoint_final / "adapter").is_dir() else checkpoint_final
    model = inference_utils.Loader.load_peft_model(base_model, str(adapter_dir))
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
        deterministic=cdf.evaluation.deterministic_decoding,
    )
    rm = out["reward_models"]
    r1 = float(rm["reward_model_1"])
    r2 = float(rm["reward_model_2"])
    return r1, r2, -r1, -r2


def run_cdf_refinement(cdf_yaml: Path) -> Path:
    cdf = load_cdf_yaml(cdf_yaml)
    if cdf.execution.mode not in ("sequential", "parallel"):
        raise ValueError(f"execution.mode must be 'sequential' or 'parallel', got {cdf.execution.mode!r}")
    if cdf.endpoint_reuse.fallback_when_missing not in ("train",):
        raise ValueError(
            "endpoint_reuse.fallback_when_missing must be 'train', got "
            f"{cdf.endpoint_reuse.fallback_when_missing!r}"
        )
    if cdf.num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    n_points = cdf.num_segments + 1
    slot_list = cdf.warm_start.init_adapters_by_slot
    if isinstance(slot_list, list) and len(slot_list) > n_points:
        raise ValueError(
            f"warm_start.init_adapters_by_slot has length {len(slot_list)} but num_segments+1 is {n_points}"
        )
    if not cdf.ppo_config_paths:
        raise ValueError("cdf_refinement YAML must set ppo_config_paths (non-empty list).")
    if not (0.0 < cdf.alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    run_root = _resolve_run_root(cdf)
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot_cfg = run_root / "cdf_refinement_config.snapshot.yaml"
    if not snapshot_cfg.exists() or not cdf.resume.enabled:
        shutil.copy2(cdf_yaml, snapshot_cfg)

    merged_yaml = run_root / "ppo_merged_stack.yaml"
    expected_merged = _merged_ppo_dict(cdf)
    if cdf.resume.enabled and merged_yaml.is_file():
        existing_merged = yaml.safe_load(merged_yaml.read_text(encoding="utf-8")) or {}
        if bool(cdf.resume.require_config_match) and existing_merged != expected_merged:
            raise ValueError(
                "Resume config mismatch: existing ppo_merged_stack.yaml differs from current merged PPO config. "
                "Set resume.require_config_match=false to overwrite with current config."
            )
        if not bool(cdf.resume.require_config_match) and existing_merged != expected_merged:
            _build_merged_ppo_yaml(cdf, merged_yaml)
    else:
        _build_merged_ppo_yaml(cdf, merged_yaml)

    w_grid, _f_uniform = make_uniform_cdf_grid(cdf.cdf_grid_size)
    q = _quantile_grid(cdf.num_segments)

    (
        F,
        start_outer,
        weight_history,
        pf_history,
        metric_history,
        cdf_history,
        checkpoint_mapping,
        baseline_cv,
        baseline_gap,
    ) = _load_or_init_state(cdf, run_root=run_root, w_grid=w_grid, n_points=n_points)
    if int(cdf.num_outer_iters) < start_outer:
        raise ValueError(
            f"num_outer_iters={cdf.num_outer_iters} is smaller than completed resume outer count={start_outer}."
        )
    pipeline_scheduler: _ParallelInnerJobScheduler | None = None
    if cdf.execution.mode == "parallel" and cdf.execution.pipeline_endpoints:
        pipeline_scheduler = _ParallelInnerJobScheduler(cdf=cdf, merged_yaml=merged_yaml)

    for outer in range(start_outer, int(cdf.num_outer_iters)):
        outer_dir = run_root / f"outer_iter_{outer}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        prev_outer = run_root / f"outer_iter_{outer - 1}" if outer > 0 else None

        w_raw = invert_cdf(F, w_grid, q, clamp=True)
        w_slots = _monotone_weights_on_slots(w_raw, force_endpoints=cdf.force_endpoints)
        weight_history.append({"outer_iter": outer, "weights_raw": w_raw.tolist(), "weights_slots": w_slots.tolist()})

        point_rows: list[dict[str, Any]] = []
        ckpt_map: dict[str, str] = {}
        existing_meta: dict[str, Any] | None = None
        existing_rows: list[dict[str, Any]] = []
        meta_path = outer_dir / "point_meta.json"
        if meta_path.is_file():
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows_raw = existing_meta.get("points", [])
            if not isinstance(rows_raw, list):
                raise TypeError(f"{meta_path} points must be a list")
            existing_rows = [r for r in rows_raw if isinstance(r, dict)]

        # ---- Phase 1: resolve budget and collect slot metadata ----
        if cdf.inner_loop.budget_mode == "epochs":
            _num_epochs = int(cdf.inner_loop.budget_value)
            _max_updates: int | None = None
        elif cdf.inner_loop.budget_mode == "steps":
            _max_updates = int(cdf.inner_loop.budget_value)
            _num_epochs = 50_000
        else:
            raise ValueError(f"Unknown inner_loop.budget_mode: {cdf.inner_loop.budget_mode}")

        pending_jobs: list[_InnerJob] = []
        slot_status: list[str] = []  # "done" | "needs_eval" | "needs_train"
        endpoint_ckpt_by_slot: dict[int, Path] = {}
        wait_train_slots: set[int] = set()

        def _maybe_submit_next_endpoint(slot_n: int, init_ckpt: Path) -> None:
            if pipeline_scheduler is None:
                return
            if slot_n not in (0, n_points - 1):
                return
            next_outer = outer + 1
            if next_outer >= int(cdf.num_outer_iters):
                return
            if _resolve_endpoint_reuse_checkpoint(cdf, outer=next_outer, slot_n=slot_n, n_points=n_points) is not None:
                return
            next_outer_dir = run_root / f"outer_iter_{next_outer}"
            next_outer_dir.mkdir(parents=True, exist_ok=True)
            adapter_name = f"adapter_{slot_n}"
            if (next_outer_dir / adapter_name / "checkpoint_final" / "adapter").is_dir():
                return
            if pipeline_scheduler.has_job(next_outer, slot_n):
                return
            next_job = _InnerJob(
                slot_n=slot_n,
                adapter_name=adapter_name,
                w_n=0.0 if slot_n == 0 else 1.0,
                init_path=init_ckpt,
                num_epochs=_num_epochs,
                max_updates=_max_updates,
                seed=int(cdf.seed) + next_outer * 1000 + slot_n,
            )
            submitted = pipeline_scheduler.submit(
                _ParallelInnerJob(outer=next_outer, outer_dir=next_outer_dir, job=next_job)
            )
            if submitted:
                print(
                    f"[cdf-pipeline] submitted endpoint lookahead outer={next_outer} "
                    f"slot={slot_n} from {init_ckpt}"
                )

        def _on_parallel_job_complete(spec: _ParallelInnerJob) -> None:
            ck = spec.outer_dir / spec.job.adapter_name / "checkpoint_final"
            if (ck / "adapter").is_dir():
                # Endpoint weights do not depend on the CDF update, so they can
                # safely train one iteration ahead.
                saved_outer = outer
                try:
                    nonlocal_outer = spec.outer
                    # Temporarily submit based on the completed job's outer.
                    next_outer = nonlocal_outer + 1
                    if (
                        spec.job.slot_n in (0, n_points - 1)
                        and next_outer < int(cdf.num_outer_iters)
                        and _resolve_endpoint_reuse_checkpoint(
                            cdf, outer=next_outer, slot_n=spec.job.slot_n, n_points=n_points
                        ) is None
                    ):
                        next_outer_dir = run_root / f"outer_iter_{next_outer}"
                        next_outer_dir.mkdir(parents=True, exist_ok=True)
                        adapter_name = f"adapter_{spec.job.slot_n}"
                        if not (next_outer_dir / adapter_name / "checkpoint_final" / "adapter").is_dir():
                            if pipeline_scheduler is not None and not pipeline_scheduler.has_job(next_outer, spec.job.slot_n):
                                next_job = _InnerJob(
                                    slot_n=spec.job.slot_n,
                                    adapter_name=adapter_name,
                                    w_n=0.0 if spec.job.slot_n == 0 else 1.0,
                                    init_path=ck,
                                    num_epochs=spec.job.num_epochs,
                                    max_updates=spec.job.max_updates,
                                    seed=int(cdf.seed) + next_outer * 1000 + spec.job.slot_n,
                                )
                                if pipeline_scheduler.submit(
                                    _ParallelInnerJob(outer=next_outer, outer_dir=next_outer_dir, job=next_job)
                                ):
                                    print(
                                        f"[cdf-pipeline] submitted endpoint lookahead outer={next_outer} "
                                        f"slot={spec.job.slot_n} from {ck}"
                                    )
                finally:
                    _ = saved_outer

        for slot_n in range(n_points):
            w_n = float(w_slots[slot_n])
            adapter_name = f"adapter_{slot_n}"
            (outer_dir / adapter_name).mkdir(parents=True, exist_ok=True)
            ck_final = outer_dir / adapter_name / "checkpoint_final"
            prior_row = _row_for_slot(existing_rows, slot_n)
            endpoint_ckpt = _resolve_endpoint_reuse_checkpoint(
                cdf, outer=outer, slot_n=slot_n, n_points=n_points
            )
            if endpoint_ckpt is not None:
                endpoint_ckpt_by_slot[slot_n] = endpoint_ckpt
                if prior_row is not None and _row_has_eval(prior_row):
                    slot_status.append("done")
                else:
                    slot_status.append("needs_eval")
                _maybe_submit_next_endpoint(slot_n, endpoint_ckpt)
                continue

            if pipeline_scheduler is not None and pipeline_scheduler.has_job(outer, slot_n):
                if (ck_final / "adapter").is_dir():
                    if prior_row is not None and _row_has_eval(prior_row):
                        slot_status.append("done")
                    else:
                        slot_status.append("needs_eval")
                else:
                    wait_train_slots.add(slot_n)
                    slot_status.append("needs_train")
                continue

            if (ck_final / "adapter").is_dir():
                if prior_row is not None and _row_has_eval(prior_row):
                    slot_status.append("done")
                else:
                    slot_status.append("needs_eval")
            else:
                init_path = _resolve_init_adapter_for_slot(
                    cdf,
                    outer=outer,
                    slot_n=slot_n,
                    adapter_name=adapter_name,
                    weight=w_n,
                    prev_outer=prev_outer,
                    outer_dir=outer_dir,
                    merged_yaml=merged_yaml,
                )
                pending_jobs.append(
                    _InnerJob(
                        slot_n=slot_n,
                        adapter_name=adapter_name,
                        w_n=w_n,
                        init_path=init_path,
                        num_epochs=_num_epochs,
                        max_updates=_max_updates,
                        seed=int(cdf.seed) + outer * 1000 + slot_n,
                    )
                )
                wait_train_slots.add(slot_n)
                slot_status.append("needs_train")

        # ---- Phase 2: run training jobs (parallel or sequential) ----
        if cdf.execution.mode == "parallel" and (pending_jobs or wait_train_slots):
            if pipeline_scheduler is not None:
                for job in pending_jobs:
                    pipeline_scheduler.submit(_ParallelInnerJob(outer=outer, outer_dir=outer_dir, job=job))
                pipeline_scheduler.wait_for_outer(
                    outer,
                    wait_train_slots,
                    on_complete=_on_parallel_job_complete,
                )
            else:
                _run_inner_jobs_parallel(
                    pending_jobs,
                    cdf=cdf,
                    merged_yaml=merged_yaml,
                    outer_dir=outer_dir,
                    outer=outer,
                )
        else:
            for job in pending_jobs:
                print(f"[cdf-seq] outer={outer} slot={job.slot_n} w={job.w_n:.4f}")
                _run_inner_ppo(
                    merged_yaml=merged_yaml,
                    weight=job.w_n,
                    output_dir=outer_dir,
                    run_name=job.adapter_name,
                    init_adapter=job.init_path,
                    num_epochs=job.num_epochs,
                    max_updates=job.max_updates,
                    seed=job.seed,
                    num_processes=int(cdf.inner_loop.num_processes),
                )

        # ---- Phase 3: evaluate and collect point rows ----
        for slot_n in range(n_points):
            w_n = float(w_slots[slot_n])
            adapter_name = f"adapter_{slot_n}"
            ck_final = outer_dir / adapter_name / "checkpoint_final"
            prior_row = _row_for_slot(existing_rows, slot_n)
            eval_ck = endpoint_ckpt_by_slot.get(slot_n, ck_final)

            if slot_status[slot_n] == "done":
                row = dict(prior_row)  # type: ignore[arg-type]
                row["slot"] = slot_n
                row["quantile"] = float(q[slot_n])
                row["weight"] = w_n
                row["run_dir"] = str(eval_ck.parent)
                row["checkpoint_final"] = str(eval_ck)
                point_rows.append(row)
                ckpt_map[str(slot_n)] = str(eval_ck)
            else:
                if slot_n not in endpoint_ckpt_by_slot and not (ck_final / "adapter").is_dir():
                    raise FileNotFoundError(
                        f"Inner PPO finished but missing adapter checkpoint: {ck_final / 'adapter'}"
                    )
                log_dir = eval_ck.parent / "logs"
                if cdf.evaluation.pf_source == "training_log":
                    r1, r2, f1, f2, mean_kl, kl_coef = _estimate_pf_from_training_log(
                        log_dir, tail_fraction=cdf.evaluation.tail_fraction
                    )
                else:
                    # Inference-based r1/r2; KL and beta sourced from the last training log entry.
                    r1_raw, r2_raw, _, _ = evaluate_pf_point(
                        merged_yaml=merged_yaml, checkpoint_final=eval_ck, cdf=cdf
                    )
                    _, _, _, _, mean_kl, kl_coef = _estimate_pf_from_training_log(
                        log_dir, tail_fraction=cdf.evaluation.tail_fraction
                    )
                    r1, r2 = r1_raw, r2_raw
                    f1 = -(r1 - kl_coef * mean_kl)
                    f2 = -(r2 - kl_coef * mean_kl)
                point_rows.append(
                    {
                        "slot": slot_n,
                        "quantile": float(q[slot_n]),
                        "weight": w_n,
                        "run_dir": str(eval_ck.parent),
                        "checkpoint_final": str(eval_ck),
                        "E_r1": r1,
                        "E_r2": r2,
                        "f1": f1,
                        "f2": f2,
                        "mean_kl": mean_kl,
                        "kl_coef": kl_coef,
                        "pf_source": cdf.evaluation.pf_source,
                    }
                )
                ckpt_map[str(slot_n)] = str(eval_ck)

        if len(point_rows) != n_points:
            raise RuntimeError(f"outer_iter {outer}: expected {n_points} PF points, got {len(point_rows)}")
        point_rows = sorted(point_rows, key=lambda r: int(r["slot"]))
        _write_json_atomic(outer_dir / "point_meta.json", {"outer_iter": outer, "points": point_rows})
        checkpoint_mapping.append({"outer_iter": outer, "checkpoints": ckpt_map})

        ordered = sorted(point_rows, key=lambda r: float(r["weight"]))
        z = np.array([[p["f1"], p["f2"]] for p in ordered], dtype=np.float64)
        w_ord = np.array([float(p["weight"]) for p in ordered], dtype=np.float64)
        ell_arr = compute_segment_lengths(z) if z.shape[0] >= 2 else np.array([], dtype=np.float64)
        cv_r = compute_cv(ell_arr)
        gap_r = compute_gap_ratio(ell_arr)

        if outer == 0 and baseline_cv is None:
            baseline_cv = cv_r
            baseline_gap = gap_r

        F_tilde, _s_at = build_surrogate_cdf_from_points(
            w_ord,
            z,
            w_grid,
            use_pchip=cdf.use_pchip,
        )
        F_next = blend_cdfs(F, F_tilde, cdf.alpha)
        F_next = enforce_monotone_cdf(F_next, eps=cdf.monotone_eps, force_endpoints=cdf.force_endpoints)
        cdf_history.append({"outer_iter": outer, "F_on_grid": F_next.tolist(), "F_tilde_on_grid": F_tilde.tolist()})

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

        F = F_next

        _write_json_atomic(run_root / "weight_history.json", weight_history)
        _write_json_atomic(run_root / "pf_history.json", pf_history)
        _write_json_atomic(run_root / "metric_history.json", metric_history)
        _write_json_atomic(run_root / "cdf_history.json", cdf_history)
        _write_json_atomic(run_root / "checkpoint_mapping.json", checkpoint_mapping)

    return run_root


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="CDF refinement outer loop over PPO")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs/train/cdf_refinement.yaml"),
        help="Path to cdf_refinement YAML",
    )
    ns = p.parse_args()
    out = run_cdf_refinement(Path(ns.config).resolve())
    print("CDF refinement finished. Run root:", out)


if __name__ == "__main__":
    main()
