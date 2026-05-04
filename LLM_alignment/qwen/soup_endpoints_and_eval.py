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


def adapter_subdir(ckpt: Path) -> Path:
    return ckpt / "adapter" if (ckpt / "adapter").is_dir() else ckpt


def copy_endpoint(src_ckpt: Path, dst_ckpt: Path, src_log: Path, dst_log: Path) -> None:
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if dst_ckpt.exists():
        shutil.rmtree(dst_ckpt)
    shutil.copytree(src_ckpt, dst_ckpt)

    dst_log.parent.mkdir(parents=True, exist_ok=True)
    if src_log.is_file():
        shutil.copy2(src_log, dst_log)


def soup_one(
    *,
    merged_yaml: Path,
    endpoint0: Path,
    endpoint1: Path,
    out_ckpt: Path,
    lam: float,
) -> None:
    cfg = args_utils.load_run_config([merged_yaml])

    a0 = adapter_subdir(endpoint0)
    a1 = adapter_subdir(endpoint1)

    if not a0.is_dir():
        raise FileNotFoundError(a0)
    if not a1.is_dir():
        raise FileNotFoundError(a1)

    if out_ckpt.exists():
        shutil.rmtree(out_ckpt)
    (out_ckpt / "adapter").mkdir(parents=True, exist_ok=True)

    base = inference_utils.Loader.load_base_model(cfg)
    wa = inference_utils.WeightAverager.build_wa(
        cfg,
        [str(a0), str(a1)],
        [1.0 - lam, lam],
    )
    wa.save_pretrained(str(out_ckpt / "adapter"))

    del wa
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_to_log(
    *,
    merged_yaml: Path,
    checkpoint_final: Path,
    log_path: Path,
    weight: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    progress_outer: int | None = None,
    progress_slot: int | None = None,
) -> None:
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

    eval_subset_size = int(num_batches * batch_size)
    train_ds = summary.build_dataset(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.train_split,
        max_train_samples=eval_subset_size,
    )
    if len(train_ds) < eval_subset_size:
        raise RuntimeError(
            f"Eval subset too small: requested {eval_subset_size}, got {len(train_ds)} from build_dataset."
        )
    # Lightweight randomization over the already small subset.
    rng = np.random.default_rng(seed)
    order = rng.permutation(eval_subset_size)

    policy_base_model = inference_utils.Loader.load_base_model(cfg).to(device)
    adapter_dir = adapter_subdir(checkpoint_final)
    model = inference_utils.Loader.load_peft_model(policy_base_model, str(adapter_dir))
    ref_model = inference_utils.Loader.load_base_model(cfg).to(device)
    if torch.cuda.is_available():
        model = model.to("cuda")
        ref_model = ref_model.to("cuda")
    model.eval()
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

    with log_path.open("w", encoding="utf-8") as f:
        for b in range(num_batches):
            batch_idxs = order[b * batch_size : (b + 1) * batch_size]
            query_tensors = [train_ds[int(i)]["input_ids"] for i in batch_idxs]

            out = inference_utils.evaluate_scalars_structured(
                predictor,
                model,
                query_tensors,
                cfg,
                include_kl=True,
                ref_model=ref_model,
                deterministic=True,
            )

            rm = out["reward_models"]
            r1 = float(rm["reward_model_1"])
            r2 = float(rm["reward_model_2"])

            mean_kl = None
            for k in ("kl_mean", "kl", "mean_kl", "kl_surrogate"):
                if k in out:
                    mean_kl = float(out[k])
                    break
            if mean_kl is None:
                raise RuntimeError(
                    f"KL was not produced during soup evaluation for {checkpoint_final}. "
                    "Expected one of keys: kl_mean, kl, mean_kl, kl_surrogate."
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
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if progress_outer is not None and progress_slot is not None:
                print(
                    f"__SOUP_PROGRESS__ outer={int(progress_outer)} "
                    f"slot={int(progress_slot)} batch={b + 1}/{num_batches}",
                    flush=True,
                )

        ckpt_rec = {
            "record_type": "checkpoint",
            "checkpoint_tag": "checkpoint_final",
            "checkpoint_path": str(checkpoint_final.resolve()),
            "global_update_step": num_batches,
            "epoch_index": 0,
            "batch_in_epoch": num_batches - 1,
            "n_batches_in_epoch": num_batches,
            "within_epoch_progress": 1.0,
            "weight": float(weight),
        }
        f.write(json.dumps(ckpt_rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    del model
    del ref_model
    del policy_base_model
    del reward_pipes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_step_records(log_path: Path) -> list[dict]:
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
        raise RuntimeError(f"No reward step rows found in {log_path}")
    return rows


def _estimate_point_from_log(
    log_path: Path,
    *,
    fallback_kl_coef: float,
) -> tuple[float, float, float, float, float, float]:
    rows = _extract_step_records(log_path)
    r1 = float(np.mean([float(r["mean_reward_1"]) for r in rows]))
    r2 = float(np.mean([float(r["mean_reward_2"]) for r in rows]))
    kl_vals: list[float] = []
    for r in rows:
        if "mean_kl" not in r:
            raise RuntimeError(
                f"Missing mean_kl in a training/eval row at {log_path}. "
                "KL must be explicitly computed for all slots."
            )
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


def _write_outer_metadata(
    *,
    outer: int,
    outer_dir: Path,
    num_segments: int,
    hv_ref: np.ndarray,
    fallback_kl_coef: float,
) -> tuple[dict, dict, dict]:
    points: list[dict] = []
    checkpoints: dict[str, str] = {}
    for slot in range(num_segments + 1):
        adapter_dir = outer_dir / f"adapter_{slot}"
        checkpoint_final = adapter_dir / "checkpoint_final"
        log_path = adapter_dir / "logs" / "training_metrics.jsonl"
        r1, r2, f1, f2, mean_kl, kl_coef = _estimate_point_from_log(
            log_path,
            fallback_kl_coef=fallback_kl_coef,
        )
        weight = slot / num_segments
        points.append(
            {
                "slot": slot,
                "weight": weight,
                "quantile": weight,
                "run_dir": str(adapter_dir.resolve()),
                "checkpoint_final": str(checkpoint_final.resolve()),
                "E_r1": r1,
                "E_r2": r2,
                "f1": f1,
                "f2": f2,
                "mean_kl": mean_kl,
                "kl_coef": kl_coef,
                "pf_source": "training_log",
            }
        )
        checkpoints[str(slot)] = str(checkpoint_final.resolve())

    points = sorted(points, key=lambda r: int(r["slot"]))
    (outer_dir / "point_meta.json").write_text(
        json.dumps({"outer_iter": outer, "points": points}, indent=2),
        encoding="utf-8",
    )

    z = np.array([[p["f1"], p["f2"]] for p in points], dtype=float)
    ell = compute_segment_lengths(z)
    metric_row = {
        "outer_iter": outer,
        "cv": compute_cv(ell),
        "gap_ratio": compute_gap_ratio(ell),
        "hypervolume": _hypervolume_2d_min(z, hv_ref),
        "hypervolume_ref": hv_ref.tolist(),
    }
    return {"outer_iter": outer, "points": points}, metric_row, {"outer_iter": outer, "checkpoints": checkpoints}


def _write_run_histories(
    *,
    output_root: Path,
    num_segments: int,
    fallback_kl_coef: float,
) -> None:
    hv_ref = np.array([2.1, 1.4], dtype=float)
    outer_dirs = sorted(
        [p for p in output_root.glob("outer_iter_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    pf_history: list[dict] = []
    metric_history: list[dict] = []
    checkpoint_mapping: list[dict] = []
    for outer_dir in outer_dirs:
        outer = int(outer_dir.name.split("_")[-1])
        pf_row, metric_row, ckpt_row = _write_outer_metadata(
            outer=outer,
            outer_dir=outer_dir,
            num_segments=num_segments,
            hv_ref=hv_ref,
            fallback_kl_coef=fallback_kl_coef,
        )
        pf_history.append(pf_row)
        metric_history.append(metric_row)
        checkpoint_mapping.append(ckpt_row)
    (output_root / "pf_history.json").write_text(json.dumps(pf_history, indent=2), encoding="utf-8")
    (output_root / "metric_history.json").write_text(json.dumps(metric_history, indent=2), encoding="utf-8")
    (output_root / "checkpoint_mapping.json").write_text(json.dumps(checkpoint_mapping, indent=2), encoding="utf-8")


def worker_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged_yaml", required=True)
    p.add_argument("--endpoint0", required=True)
    p.add_argument("--endpoint1", required=True)
    p.add_argument("--out_ckpt", required=True)
    p.add_argument("--log_path", required=True)
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--num_batches", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress_outer", type=int, default=None)
    p.add_argument("--progress_slot", type=int, default=None)
    ns = p.parse_args()

    soup_one(
        merged_yaml=Path(ns.merged_yaml),
        endpoint0=Path(ns.endpoint0),
        endpoint1=Path(ns.endpoint1),
        out_ckpt=Path(ns.out_ckpt),
        lam=float(ns.weight),
    )

    evaluate_to_log(
        merged_yaml=Path(ns.merged_yaml),
        checkpoint_final=Path(ns.out_ckpt),
        log_path=Path(ns.log_path),
        weight=float(ns.weight),
        num_batches=int(ns.num_batches),
        batch_size=int(ns.batch_size),
        seed=int(ns.seed),
        progress_outer=ns.progress_outer,
        progress_slot=ns.progress_slot,
    )


def _progress_reader(
    proc: subprocess.Popen,
    progress_q: "queue.Queue[int]",
    log_buf: list[str],
) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("__SOUP_PROGRESS__"):
            progress_q.put(line)
        else:
            log_buf.append(line)


def _run_outer_workers(
    *,
    outer: int,
    merged_yaml: Path,
    endpoint0: Path,
    endpoint1: Path,
    dst_outer: Path,
    num_segments: int,
    num_batches: int,
    batch_size: int,
    base_seed: int,
    num_gpus: int,
    overwrite: bool,
) -> None:
    middle_slots = list(range(1, num_segments))
    total_eval = num_batches
    progress_q: "queue.Queue[str]" = queue.Queue()
    progress_re = re.compile(r"^__SOUP_PROGRESS__\s+outer=(\d+)\s+slot=(\d+)\s+batch=(\d+)/(\d+)\s*$")
    slot_progress: dict[int, int] = {slot: 0 for slot in middle_slots}
    current_round = 0

    procs: list[tuple[int, subprocess.Popen, threading.Thread, list[str]]] = []
    pbar = tqdm(total=total_eval, desc=f"outer_iter_{outer} eval", leave=True)

    try:
        for slot in middle_slots:
            w = slot / num_segments
            out_ckpt = dst_outer / f"adapter_{slot}" / "checkpoint_final"
            log_path = dst_outer / f"adapter_{slot}" / "logs" / "training_metrics.jsonl"

            if (
                not overwrite
                and (out_ckpt / "adapter").is_dir()
                and log_path.is_file()
            ):
                print(f"[skip] outer={outer} slot={slot} already exists")
                slot_progress[slot] = num_batches
                continue

            if overwrite:
                if out_ckpt.exists():
                    shutil.rmtree(out_ckpt)
                if log_path.exists():
                    log_path.unlink()

            gpu_id = slot - 1  # slot 1->0, 2->1, 3->2, 4->3
            if gpu_id >= num_gpus:
                raise ValueError(
                    f"slot {slot} needs GPU {gpu_id}, but num_gpus={num_gpus}. "
                    "Set --num_gpus >= num_segments-1 (4 for num_segments=5)."
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                "python",
                "-u",
                "-m",
                "qwen.soup_endpoints_and_eval",
                "--worker",
                "--merged_yaml",
                str(merged_yaml),
                "--endpoint0",
                str(endpoint0),
                "--endpoint1",
                str(endpoint1),
                "--out_ckpt",
                str(out_ckpt),
                "--log_path",
                str(log_path),
                "--weight",
                str(w),
                "--num_batches",
                str(num_batches),
                "--batch_size",
                str(batch_size),
                "--seed",
                str(base_seed + outer * 1000 + slot),
                "--progress_outer",
                str(outer),
                "--progress_slot",
                str(slot),
            ]

            print(f"[launch] outer={outer} slot={slot} w={w:.2f} gpu={gpu_id}")
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
            procs.append((slot, proc, reader, log_buf))

        while procs:
            while True:
                try:
                    msg = progress_q.get_nowait()
                except queue.Empty:
                    break
                m = progress_re.match(msg)
                if not m:
                    continue
                m_outer = int(m.group(1))
                m_slot = int(m.group(2))
                m_batch = int(m.group(3))
                if m_outer != outer or m_slot not in slot_progress:
                    continue
                if m_batch > slot_progress[m_slot]:
                    slot_progress[m_slot] = m_batch

            completed_round = min(slot_progress.values()) if slot_progress else num_batches
            if completed_round > current_round:
                pbar.update(completed_round - current_round)
                current_round = completed_round

            alive: list[tuple[int, subprocess.Popen, threading.Thread, list[str]]] = []
            failure: tuple[int, subprocess.Popen, list[str]] | None = None
            for slot, proc, reader, log_buf in procs:
                ret = proc.poll()
                if ret is None:
                    alive.append((slot, proc, reader, log_buf))
                    continue
                reader.join(timeout=1.0)
                if ret != 0 and failure is None:
                    failure = (slot, proc, log_buf)

            if failure is not None:
                failed_slot, failed_proc, failed_log = failure
                for slot, proc, _, _ in alive:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                for slot, proc, _, _ in alive:
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                tail = "\n".join(failed_log[-80:]) if failed_log else "(no worker stdout captured)"
                raise subprocess.CalledProcessError(
                    failed_proc.returncode if failed_proc.returncode is not None else 1,
                    failed_proc.args,
                    output=(
                        f"Worker failed at outer={outer}, slot={failed_slot}\n"
                        f"---- worker output tail ----\n{tail}"
                    ),
                )

            procs = alive
            if procs:
                time.sleep(0.1)

        while True:
            try:
                msg = progress_q.get_nowait()
            except queue.Empty:
                break
            m = progress_re.match(msg)
            if not m:
                continue
            m_outer = int(m.group(1))
            m_slot = int(m.group(2))
            m_batch = int(m.group(3))
            if m_outer == outer and m_slot in slot_progress and m_batch > slot_progress[m_slot]:
                slot_progress[m_slot] = m_batch
        completed_round = min(slot_progress.values()) if slot_progress else num_batches
        if completed_round > current_round:
            pbar.update(completed_round - current_round)
    finally:
        pbar.close()


def run_all() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_root", default="outputs/ls_uniform/ls-uniform-demo")
    p.add_argument("--output_root", default="outputs/souping")
    p.add_argument(
        "--merged_yaml",
        default="outputs/cdf_refinement/cdf-refinement-surrogate-from-ls-start4/ppo_merged_stack.yaml",
    )
    p.add_argument("--num_segments", type=int, default=5)
    p.add_argument("--outer_start", type=int, default=10)
    p.add_argument("--outer_end", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    ns = p.parse_args()

    source_root = Path(ns.source_root)
    output_root = Path(ns.output_root)
    merged_yaml = Path(ns.merged_yaml)
    cfg = args_utils.load_run_config([merged_yaml])
    fallback_kl_coef = float(getattr(cfg, "init_kl_coef", 0.05))
    if hasattr(cfg, "rl") and hasattr(cfg.rl, "init_kl_coef"):
        fallback_kl_coef = float(cfg.rl.init_kl_coef)

    outer_iters = list(range(ns.outer_start, ns.outer_end - 1, -1))
    for outer in tqdm(outer_iters, desc="outer loop", leave=True):
        print(f"\n[soup] outer_iter_{outer}")

        src_outer = source_root / f"outer_iter_{outer}"
        dst_outer = output_root / f"outer_iter_{outer}"

        endpoint0 = src_outer / "adapter_0" / "checkpoint_final"
        endpoint1 = src_outer / f"adapter_{ns.num_segments}" / "checkpoint_final"

        if not endpoint0.is_dir():
            raise FileNotFoundError(endpoint0)
        if not endpoint1.is_dir():
            raise FileNotFoundError(endpoint1)

        # Copy endpoints for complete 0..N point_meta / pf_history compatibility.
        copy_endpoint(
            endpoint0,
            dst_outer / "adapter_0" / "checkpoint_final",
            src_outer / "adapter_0" / "logs" / "training_metrics.jsonl",
            dst_outer / "adapter_0" / "logs" / "training_metrics.jsonl",
        )
        copy_endpoint(
            endpoint1,
            dst_outer / f"adapter_{ns.num_segments}" / "checkpoint_final",
            src_outer / f"adapter_{ns.num_segments}" / "logs" / "training_metrics.jsonl",
            dst_outer / f"adapter_{ns.num_segments}" / "logs" / "training_metrics.jsonl",
        )

        _run_outer_workers(
            outer=outer,
            merged_yaml=merged_yaml,
            endpoint0=endpoint0,
            endpoint1=endpoint1,
            dst_outer=dst_outer,
            num_segments=ns.num_segments,
            num_batches=ns.num_batches,
            batch_size=ns.batch_size,
            base_seed=ns.seed,
            num_gpus=ns.num_gpus,
            overwrite=bool(ns.overwrite),
        )

    _write_run_histories(
        output_root=output_root,
        num_segments=int(ns.num_segments),
        fallback_kl_coef=fallback_kl_coef,
    )
    print("\nDone. Souping outputs written to:", output_root)


def main() -> None:
    import sys

    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker_main()
    else:
        run_all()


if __name__ == "__main__":
    main()
