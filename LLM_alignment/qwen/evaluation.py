from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from datasets import disable_progress_bars
except Exception:  # pragma: no cover - optional dependency guard
    disable_progress_bars = None

from qwen.tasks import summary
from qwen.utils import args_utils, inference_utils
from qwen.utils.cdf_utils import compute_cv, compute_gap_ratio, compute_segment_lengths
from qwen.utils.qwen_utils import Pipelines, Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
HV_REF = np.array([2.1, 1.4], dtype=float)
OUTPUT_MODE_CHOICES = ("in_place", "separate_root", "both")


@dataclass
class EvalJob:
    outer_dir: Path
    run_root: Path
    slot: int
    weight: float
    checkpoint_final: Path
    log_path: Path
    seed: int
    job_key: str


def _adapter_subdir(checkpoint_final: Path) -> Path:
    return checkpoint_final / "adapter" if (checkpoint_final / "adapter").is_dir() else checkpoint_final


def _backup_file_if_exists(path: Path) -> None:
    if path.is_file():
        bak_path = path.with_suffix(path.suffix + ".bak")
        bak_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bak_path)


def _count_done_batches(log_path: Path) -> int:
    """Count completed evaluation step records in an existing log file."""
    if not log_path.is_file():
        return 0
    count = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            count += 1
    return count


def _read_step_rows_optional(log_path: Path) -> list[dict]:
    """Read step records silently; returns empty list if file absent or has no step rows."""
    if not log_path.is_file():
        return []
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            rows.append(obj)
    return rows


def _parse_checkpoint_args(raw_values: list[str] | None) -> list[Path]:
    if not raw_values:
        return []
    parts: list[str] = []
    for raw in raw_values:
        for piece in raw.split(","):
            piece = piece.strip()
            if piece:
                parts.append(piece)
    return [Path(p) for p in parts]


def _parse_weights(raw_weights: str | None, n: int) -> list[float]:
    if raw_weights is None:
        if n <= 1:
            return [0.0]
        return [i / (n - 1) for i in range(n)]
    vals = [x.strip() for x in raw_weights.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"len(weights)={len(vals)} must equal number of checkpoints={n}.")
    try:
        return [float(v) for v in vals]
    except ValueError as exc:
        raise ValueError(f"Malformed --weights string: {raw_weights}") from exc


def _discover_outer_checkpoints(outer_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for p in outer_dir.glob("adapter_*/checkpoint_final"):
        parent = p.parent.name
        if not parent.startswith("adapter_"):
            continue
        try:
            slot = int(parent.split("_")[-1])
        except ValueError:
            continue
        items.append((slot, p))
    if not items:
        raise FileNotFoundError(f"No checkpoints discovered under {outer_dir}/adapter_*/checkpoint_final")
    items = sorted(items, key=lambda x: x[0])
    return items


def _parse_outer_dirs(raw_values: list[str] | None) -> list[Path]:
    if not raw_values:
        return []
    out: list[Path] = []
    for raw in raw_values:
        for piece in raw.split(","):
            piece = piece.strip()
            if piece:
                out.append(Path(piece))
    return out


def _resolve_outer_dirs(single_outer_dir: str | None, multi_outer_dirs: list[str] | None) -> list[Path]:
    dirs: list[Path] = []
    if single_outer_dir:
        dirs.append(Path(single_outer_dir))
    dirs.extend(_parse_outer_dirs(multi_outer_dirs))
    if not dirs:
        return []
    uniq: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def _resolve_logs_root_for_job(
    *,
    job_outer_dir: Path,
    job_checkpoint_final: Path,
    output_mode: str,
    output_root: Path | None,
) -> Path:
    if output_mode not in OUTPUT_MODE_CHOICES:
        raise ValueError(f"Unsupported output mode: {output_mode}")
    if output_mode in ("in_place", "both"):
        return job_checkpoint_final.parent / "logs"
    if output_root is None:
        raise ValueError(f"output_mode={output_mode} requires --output_root.")
    rel_outer = job_outer_dir
    if job_outer_dir.is_absolute():
        rel_outer = Path(*job_outer_dir.parts[1:])
    mirrored_outer = output_root / rel_outer
    return mirrored_outer / job_checkpoint_final.parent.name / "logs"


def _parse_weights_per_outer(raw_weights: str | None, outer_dirs: list[Path]) -> dict[str, list[float]]:
    if raw_weights is None:
        out: dict[str, list[float]] = {}
        for od in outer_dirs:
            discovered = _discover_outer_checkpoints(od)
            out[str(od.resolve())] = _parse_weights(None, len(discovered))
        return out
    vals = [x.strip() for x in raw_weights.split(",") if x.strip()]
    if not outer_dirs:
        return {}
    parsed = [float(v) for v in vals]
    out: dict[str, list[float]] = {}
    for od in outer_dirs:
        discovered = _discover_outer_checkpoints(od)
        n = len(discovered)
        if len(parsed) != n:
            raise ValueError(
                f"When using outer_dirs, --weights must have {n} values for each outer dir. "
                f"Got {len(parsed)} for {od}."
            )
        out[str(od.resolve())] = list(parsed)
    return out


def _resolve_jobs(
    *,
    outer_dirs: list[Path],
    checkpoints: list[Path],
    weights: list[float] | None,
    weights_by_outer: dict[str, list[float]] | None,
    seed: int,
    output_mode: str,
    output_root: Path | None,
) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    if outer_dirs:
        idx_global = 0
        for outer_dir in outer_dirs:
            discovered = _discover_outer_checkpoints(outer_dir)
            ws = (weights_by_outer or {}).get(str(outer_dir.resolve()))
            if ws is None:
                ws = _parse_weights(None, len(discovered))
            if len(ws) != len(discovered):
                raise ValueError(f"Weight count mismatch for {outer_dir}.")
            for (slot, ckpt), w in zip(discovered, ws):
                if not ckpt.is_dir():
                    raise FileNotFoundError(ckpt)
                if not _adapter_subdir(ckpt).is_dir():
                    raise FileNotFoundError(f"Missing adapter directory under {ckpt}")
                logs_root = _resolve_logs_root_for_job(
                    job_outer_dir=outer_dir,
                    job_checkpoint_final=ckpt,
                    output_mode=output_mode,
                    output_root=output_root,
                )
                log_path = logs_root / "training_metrics.jsonl"
                jobs.append(
                    EvalJob(
                        outer_dir=outer_dir,
                        run_root=outer_dir.parent,
                        slot=slot,
                        weight=float(w),
                        checkpoint_final=ckpt,
                        log_path=log_path,
                        seed=int(seed) + idx_global,
                        job_key=f"{outer_dir.resolve()}::{slot}",
                    )
                )
                idx_global += 1
        return jobs

    if weights is None:
        raise ValueError("weights must be provided for checkpoint list mode.")
    if len(checkpoints) != len(weights):
        raise ValueError("Number of weights does not match provided checkpoints.")
    for idx, (ckpt, w) in enumerate(zip(checkpoints, weights)):
        if not ckpt.is_dir():
            raise FileNotFoundError(ckpt)
        if not _adapter_subdir(ckpt).is_dir():
            raise FileNotFoundError(f"Missing adapter directory under {ckpt}")
        slot = idx
        synthetic_outer = ckpt.parent.parent
        logs_root = _resolve_logs_root_for_job(
            job_outer_dir=synthetic_outer,
            job_checkpoint_final=ckpt,
            output_mode=output_mode,
            output_root=output_root,
        )
        log_path = logs_root / "training_metrics.jsonl"
        jobs.append(
            EvalJob(
                outer_dir=synthetic_outer,
                run_root=synthetic_outer.parent,
                slot=slot,
                weight=float(w),
                checkpoint_final=ckpt,
                log_path=log_path,
                seed=int(seed) + idx,
                job_key=f"{synthetic_outer.resolve()}::{slot}",
            )
        )
    return jobs


def _read_step_rows(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "mean_reward_1" in obj and "mean_reward_2" in obj:
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"No step rows found in {log_path}")
    return rows


def _estimate_from_log(log_path: Path, fallback_kl_coef: float) -> tuple[float, float, float, float, float, float]:
    rows = _read_step_rows(log_path)
    r1 = float(np.mean([float(r["mean_reward_1"]) for r in rows]))
    r2 = float(np.mean([float(r["mean_reward_2"]) for r in rows]))
    kl_vals: list[float] = []
    for r in rows:
        if "mean_kl" not in r:
            raise RuntimeError(f"Missing mean_kl in {log_path}. KL must be present for every evaluated checkpoint.")
        kl_vals.append(float(r["mean_kl"]))
    mean_kl = float(np.mean(kl_vals))
    beta_vals = [
        float(r["ppo_stats"]["objective_kl_coef"])
        for r in rows
        if isinstance(r.get("ppo_stats"), dict) and "objective_kl_coef" in r["ppo_stats"]
    ]
    kl_coef = float(np.mean(beta_vals)) if beta_vals else float(fallback_kl_coef)
    f1 = -(r1 - kl_coef * mean_kl)
    f2 = -(r2 - kl_coef * mean_kl)
    return r1, r2, f1, f2, mean_kl, kl_coef


def _hypervolume_2d_min(points: np.ndarray, ref: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float)
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1])]
    if len(pts) == 0:
        return 0.0
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
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in pts:
        width = ref[0] - f1
        height = prev_f2 - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f2 = min(prev_f2, f2)
    return float(hv)


def _parse_outer_iter(outer_dir: Path) -> int:
    name = outer_dir.name
    if name.startswith("outer_iter_"):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 0
    return 0


def _compute_outer_rows(
    *,
    jobs: list[EvalJob],
    outer_iter: int,
    fallback_kl_coef: float,
) -> tuple[dict, dict, dict]:
    points: list[dict] = []
    ckpt_map: dict[str, str] = {}
    for j in sorted(jobs, key=lambda x: x.slot):
        r1, r2, f1, f2, mean_kl, kl_coef = _estimate_from_log(j.log_path, fallback_kl_coef)
        row = {
            "slot": int(j.slot),
            "quantile": float(j.weight),
            "weight": float(j.weight),
            "run_dir": str(j.checkpoint_final.parent.resolve()),
            "checkpoint_final": str(j.checkpoint_final.resolve()),
            "E_r1": r1,
            "E_r2": r2,
            "f1": f1,
            "f2": f2,
            "mean_kl": mean_kl,
            "kl_coef": kl_coef,
            "pf_source": "training_log",
        }
        points.append(row)
        ckpt_map[str(j.slot)] = str(j.checkpoint_final.resolve())

    points = sorted(points, key=lambda r: int(r["slot"]))
    z = np.array([[p["f1"], p["f2"]] for p in points], dtype=float)
    ell = compute_segment_lengths(z)
    metric_row = {
        "outer_iter": outer_iter,
        "cv": compute_cv(ell),
        "gap_ratio": compute_gap_ratio(ell),
        "hypervolume": _hypervolume_2d_min(z, HV_REF),
        "hypervolume_ref": HV_REF.tolist(),
    }
    pf_row = {"outer_iter": outer_iter, "points": points}
    ckpt_row = {"outer_iter": outer_iter, "checkpoints": ckpt_map}
    return pf_row, metric_row, ckpt_row


def _write_histories(
    *,
    jobs: list[EvalJob],
    output_mode: str,
    output_root: Path | None,
    fallback_kl_coef: float,
) -> None:
    by_outer: dict[str, list[EvalJob]] = {}
    for j in jobs:
        by_outer.setdefault(str(j.outer_dir.resolve()), []).append(j)

    in_place_run_rows: dict[str, dict[str, list[dict]]] = {}
    separate_run_rows: dict[str, dict[str, list[dict]]] = {}

    for outer_key, outer_jobs in by_outer.items():
        outer_path = Path(outer_key)
        outer_iter = _parse_outer_iter(outer_path)
        pf_row, metric_row, ckpt_row = _compute_outer_rows(
            jobs=outer_jobs,
            outer_iter=outer_iter,
            fallback_kl_coef=fallback_kl_coef,
        )

        if output_mode in ("in_place", "both"):
            _backup_file_if_exists(outer_path / "point_meta.json")
            (outer_path / "point_meta.json").write_text(
                json.dumps({"outer_iter": outer_iter, "points": pf_row["points"]}, indent=2),
                encoding="utf-8",
            )
            run_key = str(outer_path.parent.resolve())
            bucket = in_place_run_rows.setdefault(run_key, {"pf": [], "metric": [], "ckpt": []})
            bucket["pf"].append(pf_row)
            bucket["metric"].append(metric_row)
            bucket["ckpt"].append(ckpt_row)

        if output_mode in ("separate_root", "both"):
            if output_root is None:
                raise ValueError(f"output_mode={output_mode} requires --output_root.")
            rel_outer = outer_path if not outer_path.is_absolute() else Path(*outer_path.parts[1:])
            mirrored_outer = output_root / rel_outer
            mirrored_outer.mkdir(parents=True, exist_ok=True)
            if output_mode == "both":
                for j in outer_jobs:
                    rel_logs = j.log_path.relative_to(outer_path)
                    dst_log = mirrored_outer / rel_logs
                    dst_log.parent.mkdir(parents=True, exist_ok=True)
                    if j.log_path.is_file():
                        shutil.copy2(j.log_path, dst_log)
            _backup_file_if_exists(mirrored_outer / "point_meta.json")
            (mirrored_outer / "point_meta.json").write_text(
                json.dumps({"outer_iter": outer_iter, "points": pf_row["points"]}, indent=2),
                encoding="utf-8",
            )
            run_key = str(mirrored_outer.parent.resolve())
            bucket = separate_run_rows.setdefault(run_key, {"pf": [], "metric": [], "ckpt": []})
            bucket["pf"].append(pf_row)
            bucket["metric"].append(metric_row)
            bucket["ckpt"].append(ckpt_row)

    for run_key, rows in in_place_run_rows.items():
        run_root = Path(run_key)
        run_root.mkdir(parents=True, exist_ok=True)
        pf_rows = sorted(rows["pf"], key=lambda r: int(r["outer_iter"]))
        metric_rows = sorted(rows["metric"], key=lambda r: int(r["outer_iter"]))
        ckpt_rows = sorted(rows["ckpt"], key=lambda r: int(r["outer_iter"]))
        _backup_file_if_exists(run_root / "pf_history.json")
        _backup_file_if_exists(run_root / "metric_history.json")
        _backup_file_if_exists(run_root / "checkpoint_mapping.json")
        (run_root / "pf_history.json").write_text(json.dumps(pf_rows, indent=2), encoding="utf-8")
        (run_root / "metric_history.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
        (run_root / "checkpoint_mapping.json").write_text(json.dumps(ckpt_rows, indent=2), encoding="utf-8")

    for run_key, rows in separate_run_rows.items():
        run_root = Path(run_key)
        run_root.mkdir(parents=True, exist_ok=True)
        pf_rows = sorted(rows["pf"], key=lambda r: int(r["outer_iter"]))
        metric_rows = sorted(rows["metric"], key=lambda r: int(r["outer_iter"]))
        ckpt_rows = sorted(rows["ckpt"], key=lambda r: int(r["outer_iter"]))
        _backup_file_if_exists(run_root / "pf_history.json")
        _backup_file_if_exists(run_root / "metric_history.json")
        _backup_file_if_exists(run_root / "checkpoint_mapping.json")
        (run_root / "pf_history.json").write_text(json.dumps(pf_rows, indent=2), encoding="utf-8")
        (run_root / "metric_history.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
        (run_root / "checkpoint_mapping.json").write_text(json.dumps(ckpt_rows, indent=2), encoding="utf-8")


def _progress_reader(proc: subprocess.Popen, progress_q: "queue.Queue[str]", log_buf: list[str]) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("__EVAL_PROGRESS__"):
            progress_q.put(line)
        else:
            log_buf.append(line)


def _run_worker_jobs(
    *,
    jobs: list[EvalJob],
    merged_yaml: Path,
    num_batches: int,
    batch_size: int,
    deterministic: bool,
    num_gpus: int,
    max_concurrent_jobs: int,
    overwrite: bool,
    skip_existing: bool,
) -> list[EvalJob]:
    if max_concurrent_jobs > num_gpus:
        raise ValueError(
            f"max_concurrent_jobs ({max_concurrent_jobs}) must be <= num_gpus ({num_gpus})."
        )
    if max_concurrent_jobs <= 0:
        raise ValueError("max_concurrent_jobs must be positive.")

    runnable: list[EvalJob] = []
    job_remaining: list[int] = []
    for job in jobs:
        if skip_existing and job.log_path.is_file():
            print(f"[skip] {job.job_key} (skip_existing=True)")
            continue
        if overwrite and job.log_path.is_file():
            _backup_file_if_exists(job.log_path)
            job.log_path.unlink()
        done = _count_done_batches(job.log_path)
        remaining = int(num_batches) - done
        if remaining <= 0:
            print(f"[skip] {job.job_key} already complete ({done}/{num_batches} batches)")
            continue
        if done > 0:
            print(f"[resume] {job.job_key}: {done} done, {remaining} remaining")
        runnable.append(job)
        job_remaining.append(remaining)

    total_steps = sum(job_remaining)
    if total_steps == 0:
        return jobs

    progress_re = re.compile(r"^__EVAL_PROGRESS__\s+job_key=(.+)\s+batch=(\d+)/(\d+)\s*$")
    progress_q: "queue.Queue[str]" = queue.Queue()
    slot_progress: dict[str, int] = {j.job_key: 0 for j in runnable}
    pbar = tqdm(total=total_steps, desc="adapter eval", leave=True)

    pending = list(runnable)
    free_gpus = list(range(num_gpus))
    active: list[tuple[EvalJob, int, subprocess.Popen, threading.Thread, list[str]]] = []
    current_done = 0
    try:
        while pending or active:
            while pending and free_gpus and len(active) < max_concurrent_jobs:
                job = pending.pop(0)
                gpu_id = free_gpus.pop(0)
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                cmd = [
                    "python",
                    "-u",
                    "-m",
                    "qwen.evaluation",
                    "--worker",
                    "--merged_yaml",
                    str(merged_yaml),
                    "--checkpoint_final",
                    str(job.checkpoint_final),
                    "--log_path",
                    str(job.log_path),
                    "--weight",
                    str(job.weight),
                    "--num_batches",
                    str(num_batches),
                    "--batch_size",
                    str(batch_size),
                    "--seed",
                    str(job.seed),
                    "--slot",
                    str(job.slot),
                    "--job_key",
                    str(job.job_key),
                ]
                if deterministic:
                    cmd.append("--deterministic")
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                log_buf: list[str] = []
                reader = threading.Thread(
                    target=_progress_reader,
                    args=(proc, progress_q, log_buf),
                    daemon=True,
                )
                reader.start()
                active.append((job, gpu_id, proc, reader, log_buf))

            while True:
                try:
                    msg = progress_q.get_nowait()
                except queue.Empty:
                    break
                m = progress_re.match(msg)
                if not m:
                    continue
                job_key = str(m.group(1)).strip()
                batch = int(m.group(2))
                if job_key in slot_progress and batch > slot_progress[job_key]:
                    slot_progress[job_key] = batch
            done_steps = int(sum(slot_progress.values()))
            if done_steps > current_done:
                pbar.update(done_steps - current_done)
                current_done = done_steps

            next_active: list[tuple[EvalJob, int, subprocess.Popen, threading.Thread, list[str]]] = []
            failure: tuple[EvalJob, subprocess.Popen, list[str]] | None = None
            for job, gpu_id, proc, reader, log_buf in active:
                ret = proc.poll()
                if ret is None:
                    next_active.append((job, gpu_id, proc, reader, log_buf))
                    continue
                reader.join(timeout=1.0)
                free_gpus.append(gpu_id)
                if ret != 0 and failure is None:
                    failure = (job, proc, log_buf)
            if failure is not None:
                failed_job, failed_proc, failed_log = failure
                for _, _, proc, _, _ in next_active:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                tail = "\n".join(failed_log[-80:]) if failed_log else "(no worker stdout captured)"
                raise subprocess.CalledProcessError(
                    failed_proc.returncode if failed_proc.returncode is not None else 1,
                    failed_proc.args,
                    output=(
                        f"Evaluation worker failed for slot={failed_job.slot}, checkpoint={failed_job.checkpoint_final}\n"
                        f"---- worker output tail ----\n{tail}"
                    ),
                )
            active = next_active
            free_gpus = sorted(set(free_gpus))
            if pending or active:
                time.sleep(0.1)
    finally:
        pbar.close()

    return jobs


def _evaluate_worker(
    *,
    merged_yaml: Path,
    checkpoint_final: Path,
    log_path: Path,
    weight: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    slot: int,
    job_key: str,
    deterministic: bool,
) -> None:
    if not merged_yaml.is_file():
        raise FileNotFoundError(f"Missing config: {merged_yaml}")

    cfg = args_utils.load_run_config([merged_yaml])
    args_utils.set_seed(seed)
    if disable_progress_bars is not None:
        disable_progress_bars()

    device = 0 if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Detect previously completed batches so we can resume.
    existing_rows: list[dict] = _read_step_rows_optional(log_path)
    done_batches = len(existing_rows)
    remaining = num_batches - done_batches
    if remaining <= 0:
        print(f"[skip worker] slot={slot} already has {done_batches}>={num_batches} batches.", flush=True)
        return

    eval_subset_size = int(remaining * batch_size)
    train_ds = summary.build_dataset(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.train_split,
        max_train_samples=eval_subset_size,
    )
    if len(train_ds) < eval_subset_size:
        raise RuntimeError(
            f"Eval subset too small: requested {eval_subset_size}, got {len(train_ds)}."
        )
    # Use a shifted seed when resuming to avoid sampling the same examples again.
    rng_seed = seed if done_batches == 0 else seed + done_batches * 7919
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(eval_subset_size)

    policy_base = inference_utils.Loader.load_base_model(cfg).to(device)
    adapter_dir = _adapter_subdir(checkpoint_final)
    policy_model = inference_utils.Loader.load_peft_model(policy_base, str(adapter_dir))
    ref_model = inference_utils.Loader.load_base_model(cfg).to(device)
    if torch.cuda.is_available():
        policy_model = policy_model.to("cuda")
        ref_model = ref_model.to("cuda")
    policy_model.eval()
    ref_model.eval()

    reward_pipes = Pipelines.load_pipes(
        list(cfg.reward_models),
        device=device,
        cache_dir=cfg.hf_cache,
    )
    if cfg.task_name == "reddit_summarization":
        for p in reward_pipes:
            p.tokenizer.pad_token_id = p.model.config.eos_token_id

    predictor = summary.PredictorSummary(
        reward_pipes=reward_pipes,
        tokenizer=tokenizer,
        output_max_length=cfg.eval_max_new_tokens,
        device=device,
    )
    beta = float(getattr(cfg, "init_kl_coef", 0.05))
    if hasattr(cfg, "rl") and hasattr(cfg.rl, "init_kl_coef"):
        beta = float(cfg.rl.init_kl_coef)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    with log_path.open("a", encoding="utf-8") as f:
        for i in range(remaining):
            b = done_batches + i  # global batch index
            batch_idxs = order[i * batch_size : (i + 1) * batch_size]
            query_tensors = [train_ds[int(idx)]["input_ids"] for idx in batch_idxs]
            out = inference_utils.evaluate_scalars_structured(
                predictor,
                policy_model,
                query_tensors,
                cfg,
                include_kl=True,
                ref_model=ref_model,
                deterministic=bool(deterministic),
            )
            rm = out["reward_models"]
            r1 = float(rm["reward_model_1"])
            r2 = float(rm["reward_model_2"])
            mean_kl = None
            for k in ("kl_mean", "mean_kl", "kl", "kl_surrogate"):
                if k in out:
                    mean_kl = float(out[k])
                    break
            if mean_kl is None:
                raise RuntimeError(
                    f"Missing KL in evaluator output for checkpoint {checkpoint_final}. "
                    "Expected one of: kl_mean, mean_kl, kl, kl_surrogate."
                )
            rec = {
                "global_update_step": b + 1,
                "epoch_index": 0,
                "epoch_one_based": 1,
                "batch_in_epoch": b,
                "n_batches_in_epoch": num_batches,
                "within_epoch_progress": float((b + 1) / num_batches),
                "weight": float(weight),
                "mean_reward_1": r1,
                "mean_reward_2": r2,
                "mean_scalarized_reward": float((1.0 - weight) * r1 + weight * r2),
                "mean_kl": mean_kl,
                "ppo_stats": {
                    "objective_kl": mean_kl,
                    "objective_kl_coef": beta,
                },
            }
            new_rows.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            # Report local progress (i+1 out of remaining) so parent tqdm is accurate.
            print(f"__EVAL_PROGRESS__ job_key={job_key} batch={i + 1}/{remaining}", flush=True)

        # Aggregate final_eval over ALL batches (existing + new).
        all_rows = existing_rows + new_rows
        E_r1 = float(np.mean([float(r["mean_reward_1"]) for r in all_rows]))
        E_r2 = float(np.mean([float(r["mean_reward_2"]) for r in all_rows]))
        mean_kl_all = float(np.mean([float(r["mean_kl"]) for r in all_rows]))
        f1 = -(E_r1 - beta * mean_kl_all)
        f2 = -(E_r2 - beta * mean_kl_all)
        final_rec = {
            "record_type": "final_eval",
            "weight": float(weight),
            "E_r1": E_r1,
            "E_r2": E_r2,
            "mean_kl": mean_kl_all,
            "objective_kl_coef": beta,
            "f1": f1,
            "f2": f2,
            "checkpoint_path": str(checkpoint_final.resolve()),
            "global_update_step": len(all_rows),
        }
        f.write(json.dumps(final_rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    del policy_model
    del ref_model
    del policy_base
    del reward_pipes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _default_output_root(outer_dir: Path | None, checkpoints: list[Path]) -> Path:
    if outer_dir is not None:
        return outer_dir.parent
    if not checkpoints:
        raise ValueError("Cannot infer output root without outer_dir or checkpoints.")
    return checkpoints[0].parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outer_dir", type=str, default=None)
    p.add_argument("--outer_dirs", action="append", default=None)
    p.add_argument("--checkpoints", action="append", default=None)
    p.add_argument("--merged_yaml", type=str, required=True)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_concurrent_jobs", type=int, default=1)
    p.add_argument("--num_batches", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--output_mode", choices=OUTPUT_MODE_CHOICES, default="in_place")
    p.add_argument("--worker", action="store_true")

    # Worker-only args
    p.add_argument("--checkpoint_final", type=str, default=None)
    p.add_argument("--log_path", type=str, default=None)
    p.add_argument("--weight", type=float, default=None)
    p.add_argument("--slot", type=int, default=None)
    p.add_argument("--job_key", type=str, default=None)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    merged_yaml = Path(ns.merged_yaml)
    if not merged_yaml.is_file():
        raise FileNotFoundError(f"Missing config: {merged_yaml}")

    if ns.worker:
        if (
            ns.checkpoint_final is None
            or ns.log_path is None
            or ns.weight is None
            or ns.slot is None
            or ns.job_key is None
        ):
            raise ValueError("--worker requires --checkpoint_final, --log_path, --weight, --slot, --job_key.")
        _evaluate_worker(
            merged_yaml=merged_yaml,
            checkpoint_final=Path(ns.checkpoint_final),
            log_path=Path(ns.log_path),
            weight=float(ns.weight),
            num_batches=int(ns.num_batches),
            batch_size=int(ns.batch_size),
            seed=int(ns.seed),
            slot=int(ns.slot),
            job_key=str(ns.job_key),
            deterministic=bool(ns.deterministic),
        )
        return

    outer_dirs = _resolve_outer_dirs(ns.outer_dir, ns.outer_dirs)
    checkpoints = _parse_checkpoint_args(ns.checkpoints)
    if (len(outer_dirs) == 0) == (len(checkpoints) == 0):
        raise ValueError("Provide exactly one input mode: outer_dir(s) OR checkpoints.")
    if ns.overwrite and ns.skip_existing:
        raise ValueError("--overwrite and --skip_existing cannot both be set.")
    if ns.num_gpus <= 0:
        raise ValueError("--num_gpus must be positive.")
    if ns.output_mode in ("separate_root", "both") and not ns.output_root:
        raise ValueError(f"--output_mode {ns.output_mode} requires --output_root.")

    weights = None
    weights_by_outer = None
    if outer_dirs:
        weights_by_outer = _parse_weights_per_outer(ns.weights, outer_dirs)
    else:
        weights = _parse_weights(ns.weights, len(checkpoints))

    output_root = Path(ns.output_root) if ns.output_root else None
    jobs = _resolve_jobs(
        outer_dirs=outer_dirs,
        checkpoints=checkpoints,
        weights=weights,
        weights_by_outer=weights_by_outer,
        seed=int(ns.seed),
        output_mode=ns.output_mode,
        output_root=output_root,
    )

    cfg = args_utils.load_run_config([merged_yaml])
    fallback_kl_coef = float(getattr(cfg, "init_kl_coef", 0.05))
    if hasattr(cfg, "rl") and hasattr(cfg.rl, "init_kl_coef"):
        fallback_kl_coef = float(cfg.rl.init_kl_coef)

    _run_worker_jobs(
        jobs=jobs,
        merged_yaml=merged_yaml,
        num_batches=int(ns.num_batches),
        batch_size=int(ns.batch_size),
        deterministic=bool(ns.deterministic),
        num_gpus=int(ns.num_gpus),
        max_concurrent_jobs=int(ns.max_concurrent_jobs),
        overwrite=bool(ns.overwrite),
        skip_existing=bool(ns.skip_existing),
    )

    _write_histories(
        jobs=jobs,
        output_mode=ns.output_mode,
        output_root=output_root,
        fallback_kl_coef=fallback_kl_coef,
    )
    final_root = output_root if output_root is not None else "in-place run roots"
    print("Done. Wrote evaluation outputs to:", final_root)


if __name__ == "__main__":
    main()
