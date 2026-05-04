"""REINFORCE training for Qwen summarization with reward scalarization.

Standalone entrypoint; does not share any runtime state with train_ppo.py.

KL regularization design
-------------------------
KL is included in the reward signal, not as a separate loss term.  Per-sample:

    R_total = (1 - w) * r1  +  w * r2  -  beta * KL_seq

where KL_seq = sum_t [ log pi_theta(y_t | x, y_<t)  -  log pi_ref(y_t | x, y_<t) ]

The KL values used in the advantage are stop-gradient (detached), so the policy
gradient only differentiates through log pi_theta in the REINFORCE term:

    loss = -E[ stop_grad(R_total - baseline)  *  sum_t log pi_theta(y_t | context) ]

Baseline: batch-mean total reward (simplest unbiased variance reduction).

Usage
-----
Single-GPU:

    python -m qwen.train_reinforce \\
        --config configs/model/qwen_0p5b_dora.yaml \\
                 configs/task/reddit_summarization.yaml \\
                 configs/rl/reinforce.yaml \\
        --weight 0.5

Multi-GPU (Accelerate):

    accelerate launch --num_processes 4 -m qwen.train_reinforce \\
        --config ... --weight 0.5

Resume from a saved checkpoint:

    python -m qwen.train_reinforce ... --resume /path/to/run/checkpoint_epoch_0001

Optional --init_adapter loads adapter weights (but not optimizer) before training:

    python -m qwen.train_reinforce ... --init_adapter /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qwen.tasks import summary
from qwen.utils import args_utils, reinforce_utils
from qwen.utils.ppo_utils import assert_only_expected_trainables, collator
from qwen.utils.qwen_utils import Pipelines, Tokenizer


# ---------------------------------------------------------------------------
# Process helpers (mirrors train_ppo.py pattern)
# ---------------------------------------------------------------------------

def _is_main_process_env() -> bool:
    return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0


def _main_print(*args: object, **kwargs: object) -> None:
    if _is_main_process_env():
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# argv pre-processing: extract --init_adapter before argparse runs
# (same approach as train_ppo.py)
# ---------------------------------------------------------------------------

def _pop_init_adapter_from_argv() -> str | None:
    """Remove ``--init_adapter PATH`` from ``sys.argv`` before argparse runs."""
    path: str | None = None
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--init_adapter" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            i += 2
        elif a.startswith("--init_adapter="):
            path = a.split("=", 1)[1]
            i += 1
        else:
            new_argv.append(a)
            i += 1
    sys.argv = new_argv
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REINFORCE Qwen summarization with reward scalarization (1-w)*r1 + w*r2"
    )
    parser.add_argument(
        "--weight",
        type=float,
        required=True,
        help="Scalarization weight w in [0,1]: (1-w)*r1 + w*r2.",
    )
    parser.add_argument("--config", nargs="*", type=str, default=None, help="YAML files (merged L→R)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config output_dir")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a REINFORCE checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--save_per_epoch_fraction",
        type=int,
        default=None,
        help="Override: save a mid-epoch checkpoint every ~1/N epoch.",
    )
    parser.add_argument(
        "--save_every_n_updates",
        type=int,
        default=None,
        help="Override: save a mid-epoch checkpoint every N update steps (takes precedence).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reward model selection (mirrors train_ppo.py)
# ---------------------------------------------------------------------------

def _reward_lists_for_weight(
    cfg: args_utils.RunConfig, weight: float
) -> tuple[list[str], list[str]]:
    """Select reward heads for the given weight: (1-w)*r1 + w*r2."""
    if len(cfg.reward_models) < 2 or len(cfg.reward_formats) < 2:
        raise ValueError("Config must list two task.reward_models and task.reward_formats.")
    w = float(weight)
    if w <= 0.0:
        return [cfg.reward_models[0]], [cfg.reward_formats[0]]
    if w >= 1.0:
        return [cfg.reward_models[1]], [cfg.reward_formats[1]]
    return list(cfg.reward_models[:2]), list(cfg.reward_formats[:2])


# ---------------------------------------------------------------------------
# Resume metadata parsing
# ---------------------------------------------------------------------------

def _parse_resume_meta(
    resume_path: Path, expected_weight: float, num_epochs: int
) -> tuple[dict, int, int, int]:
    """Read meta.json and return (meta, start_epoch, skip_batches, global_update_start)."""
    meta_file = resume_path / "meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Resume path must contain meta.json: {resume_path}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rw = float(meta.get("weight", expected_weight))
    if abs(rw - float(expected_weight)) > 1e-5:
        raise ValueError(
            f"Checkpoint weight {rw} != --weight {expected_weight}. "
            "Resume with the same scalarization weight."
        )
    nb = meta.get("next_batch_index_in_epoch")
    if nb is not None and int(nb) > 0:
        start_epoch = int(meta.get("epoch_index", 0))
        skip_batches = int(nb)
    else:
        start_epoch = int(meta.get("completed_epochs", 0))
        skip_batches = 0
    global_start = int(meta.get("global_update_step", 0))
    if skip_batches > 0:
        if start_epoch >= num_epochs:
            raise ValueError(
                f"Checkpoint epoch_index={start_epoch} >= num_epochs={num_epochs}."
            )
    elif start_epoch >= num_epochs:
        raise ValueError(
            f"completed_epochs={start_epoch} >= num_epochs={num_epochs}. "
            "Increase num_epochs in YAML to train further."
        )
    return meta, start_epoch, skip_batches, global_start


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_loop(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    dataloader: DataLoader,
    reward_pipes: list,
    reward_formats: list[str],
    reward_weight: float,
    optimizer: torch.optim.Optimizer,
    cfg: args_utils.RunConfig,
    run_name: str,
    save_root: Path,
    start_epoch: int,
    start_batch_skip: int,
    global_update_start: int,
    device: torch.device,
    is_main: bool,
) -> None:
    """Core REINFORCE training loop.

    One 'update step' (global_update) corresponds to one optimizer step,
    which happens every ``cfg.gradient_accumulation_steps`` batches.
    Metrics are logged every batch for fine-grained visibility.
    """
    w = reward_weight
    beta = cfg.init_kl_coef          # KL penalty coefficient
    grad_accum = cfg.gradient_accumulation_steps
    max_new_tokens = cfg.output_max_length

    run_dir = save_root / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        dl_len: int | None = len(dataloader)
    except TypeError:
        dl_len = None

    # --- metrics log file ---
    metrics_dir = run_dir / "logs"
    metrics_path = metrics_dir / "training_metrics.jsonl"
    log_f = None
    if is_main:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        append_log = (
            (start_epoch > 0 or start_batch_skip > 0 or global_update_start > 0)
            and metrics_path.is_file()
        )
        log_f = open(metrics_path, "a" if append_log else "w", encoding="utf-8", buffering=1)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }

    global_update = global_update_start
    accum_count = 0  # batches accumulated since last optimizer step

    try:
        optimizer.zero_grad()

        for epoch in range(start_epoch, cfg.num_epochs):
            if is_main:
                print(f"Begin epoch {epoch + 1}/{cfg.num_epochs} (index {epoch})")
            pbar = tqdm(
                dataloader,
                desc=f"REINFORCE epoch {epoch + 1}/{cfg.num_epochs}",
                total=dl_len,
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
                leave=True,
                disable=not is_main,
            )

            for batch_idx, batch in enumerate(pbar):
                if epoch == start_epoch and batch_idx < start_batch_skip:
                    continue

                t0 = time.time()
                queries: list[torch.Tensor] = batch["input_ids"]

                # ----------------------------------------------------------
                # 1. Generate responses  (policy, no grad, eval mode)
                # ----------------------------------------------------------
                model.eval()
                responses: list[torch.Tensor] = []
                with torch.no_grad():
                    for q in queries:
                        out_ids = model.generate(
                            input_ids=q.to(device).unsqueeze(0),
                            **gen_kwargs,
                        )
                        # Keep only newly generated tokens.
                        responses.append(out_ids[0, len(q) :].cpu())

                batch_with_resp = dict(batch)
                batch_with_resp["response"] = [tokenizer.decode(r) for r in responses]

                # ----------------------------------------------------------
                # 2. Reward model scoring  (no grad, external classifiers)
                # ----------------------------------------------------------
                r1_vec = reinforce_utils.reward_vector_for_pipe(
                    0, batch_with_resp, reward_pipes, reward_formats,
                    summary.transform_text_summary, cfg.mini_batch_size,
                )
                r2_vec = (
                    reinforce_utils.reward_vector_for_pipe(
                        1, batch_with_resp, reward_pipes, reward_formats,
                        summary.transform_text_summary, cfg.mini_batch_size,
                    )
                    if len(reward_pipes) >= 2
                    else None
                )
                r_scalarized_np = (
                    (1.0 - w) * r1_vec + w * r2_vec if r2_vec is not None else r1_vec
                )

                # ----------------------------------------------------------
                # 3. Log-probability computation
                #    - Reference: no grad (frozen model)
                #    - Policy:    with grad (differentiable for REINFORCE)
                # ----------------------------------------------------------
                queries_dev = [q.to(device) for q in queries]
                responses_dev = [r.to(device) for r in responses]

                ref_log_probs = reinforce_utils.compute_sequence_log_probs(
                    ref_model, queries_dev, responses_dev,
                    device=device, no_grad=True,
                )  # [B], no grad

                model.train()
                policy_log_probs = reinforce_utils.compute_sequence_log_probs(
                    model, queries_dev, responses_dev,
                    device=device, no_grad=False,
                )  # [B], has grad

                # ----------------------------------------------------------
                # 4. KL penalty (detached — used as reward signal, not as a
                #    direct regularization gradient term)
                #
                #    KL_seq_i = sum_t [log pi_theta(y_t|.) - log pi_ref(y_t|.)]
                #
                #    Detaching means the advantage does not back-propagate
                #    through the KL values; gradients only flow through the
                #    REINFORCE log-prob term.
                # ----------------------------------------------------------
                kl_seq = (policy_log_probs - ref_log_probs).detach()  # [B]

                # ----------------------------------------------------------
                # 5. Total reward and advantage
                # ----------------------------------------------------------
                r_scalarized = torch.tensor(
                    r_scalarized_np, dtype=torch.float32, device=device
                )
                r_total = r_scalarized - beta * kl_seq  # [B], all values detached

                # Batch-mean baseline for variance reduction.
                baseline = r_total.mean()
                advantage = r_total - baseline  # [B]

                # ----------------------------------------------------------
                # 6. REINFORCE loss
                #    loss = -E[ stop_grad(advantage) * sum_t log pi(y_t | ctx) ]
                # ----------------------------------------------------------
                loss = -(advantage.detach() * policy_log_probs).mean()

                # ----------------------------------------------------------
                # 7. Backward (scale by grad_accum for correct effective LR)
                # ----------------------------------------------------------
                (loss / grad_accum).backward()
                accum_count += 1

                # ----------------------------------------------------------
                # 8. Optimizer step
                # ----------------------------------------------------------
                if accum_count % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_update += 1

                t_elapsed = time.time() - t0

                # ----------------------------------------------------------
                # 9. Metrics logging (every batch)
                # ----------------------------------------------------------
                if is_main and log_f is not None:
                    resp_lens = [len(r) for r in responses]
                    rec: dict = {
                        "record_type": "step",
                        "global_update_step": global_update,
                        "epoch_index": epoch,
                        "epoch_one_based": epoch + 1,
                        "batch_in_epoch": batch_idx,
                        "n_batches_in_epoch": dl_len,
                        "within_epoch_progress": (
                            (batch_idx + 1) / dl_len if dl_len else None
                        ),
                        "weight": w,
                        "mean_reward_1": float(np.mean(r1_vec)),
                        "mean_reward_2": (
                            float(np.mean(r2_vec)) if r2_vec is not None else None
                        ),
                        "mean_scalarized_reward": float(np.mean(r_scalarized_np)),
                        "mean_kl": float(kl_seq.mean().item()),
                        "mean_total_reward_after_kl": float(r_total.mean().item()),
                        "reinforce_loss": float(loss.item()),
                        "baseline": float(baseline.item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "mean_response_len": float(np.mean(resp_lens)),
                        "min_response_len": int(min(resp_lens)),
                        "max_response_len": int(max(resp_lens)),
                        "elapsed_seconds": round(t_elapsed, 3),
                    }
                    log_f.write(json.dumps(rec, default=str) + "\n")
                    log_f.flush()

                # ----------------------------------------------------------
                # 10. Mid-epoch checkpoint
                # ----------------------------------------------------------
                is_last = dl_len is not None and batch_idx == dl_len - 1
                if is_main and reinforce_utils.should_save_mid_epoch(
                    cfg,
                    global_update=global_update,
                    batch_idx=batch_idx,
                    dl_len=dl_len,
                    is_last_batch=is_last,
                ):
                    mid_dir = run_dir / f"checkpoint_step_{global_update:07d}"
                    reinforce_utils.save_checkpoint(
                        model, optimizer, mid_dir,
                        meta={
                            "completed_epochs": epoch,
                            "num_epochs": cfg.num_epochs,
                            "epoch_index": epoch,
                            "next_batch_index_in_epoch": batch_idx + 1,
                            "n_batches_in_epoch": dl_len,
                            "within_epoch_progress": (
                                (batch_idx + 1) / dl_len if dl_len else None
                            ),
                            "global_update_step": global_update,
                            "weight": w,
                            "run_name": run_name,
                            "checkpoint_mid_epoch": True,
                        },
                        is_main=is_main,
                    )
                    if log_f is not None:
                        log_f.write(
                            json.dumps(
                                {
                                    "record_type": "checkpoint",
                                    "checkpoint_tag": f"checkpoint_step_{global_update:07d}",
                                    "checkpoint_path": str(mid_dir),
                                    "global_update_step": global_update,
                                    "epoch_index": epoch,
                                    "batch_in_epoch": batch_idx,
                                    "n_batches_in_epoch": dl_len,
                                    "within_epoch_progress": (
                                        (batch_idx + 1) / dl_len if dl_len else None
                                    ),
                                    "weight": w,
                                },
                                default=str,
                            )
                            + "\n"
                        )
                        log_f.flush()

            # --- end of epoch: flush any partially accumulated gradients ---
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_update += 1

            # --- per-epoch checkpoint ---
            ckpt_dir = run_dir / f"checkpoint_epoch_{epoch:04d}"
            reinforce_utils.save_checkpoint(
                model, optimizer, ckpt_dir,
                meta={
                    "completed_epochs": epoch + 1,
                    "num_epochs": cfg.num_epochs,
                    "epoch_index": epoch,
                    "next_batch_index_in_epoch": 0,
                    "n_batches_in_epoch": dl_len,
                    "within_epoch_progress": 1.0,
                    "global_update_step": global_update,
                    "weight": w,
                    "run_name": run_name,
                    "checkpoint_mid_epoch": False,
                },
                is_main=is_main,
            )
            if is_main and log_f is not None:
                log_f.write(
                    json.dumps(
                        {
                            "record_type": "checkpoint",
                            "checkpoint_tag": f"checkpoint_epoch_{epoch:04d}",
                            "checkpoint_path": str(ckpt_dir),
                            "global_update_step": global_update,
                            "epoch_index": epoch,
                            "batch_in_epoch": 0 if dl_len is None else dl_len - 1,
                            "n_batches_in_epoch": dl_len,
                            "within_epoch_progress": 1.0,
                            "weight": w,
                        },
                        default=str,
                    )
                    + "\n"
                )
                log_f.flush()

        # --- final checkpoint ---
        final_dir = run_dir / "checkpoint_final"
        reinforce_utils.save_checkpoint(
            model, optimizer, final_dir,
            meta={
                "completed_epochs": cfg.num_epochs,
                "num_epochs": cfg.num_epochs,
                "epoch_index": cfg.num_epochs - 1,
                "next_batch_index_in_epoch": 0,
                "n_batches_in_epoch": dl_len,
                "within_epoch_progress": 1.0,
                "global_update_step": global_update,
                "weight": w,
                "run_name": run_name,
                "final": True,
                "checkpoint_mid_epoch": False,
            },
            is_main=is_main,
        )
        if is_main and log_f is not None:
            log_f.write(
                json.dumps(
                    {
                        "record_type": "checkpoint",
                        "checkpoint_tag": "checkpoint_final",
                        "checkpoint_path": str(final_dir),
                        "global_update_step": global_update,
                        "epoch_index": cfg.num_epochs - 1,
                        "batch_in_epoch": 0 if dl_len is None else dl_len - 1,
                        "n_batches_in_epoch": dl_len,
                        "within_epoch_progress": 1.0,
                        "weight": w,
                    },
                    default=str,
                )
                + "\n"
            )
            log_f.flush()

    finally:
        if log_f is not None:
            log_f.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    init_adapter_arg = _pop_init_adapter_from_argv()
    cli = _parse_args()

    if init_adapter_arg and cli.resume:
        raise ValueError("Use either --init_adapter or --resume, not both.")

    w = float(cli.weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"--weight must be in [0, 1], got {w}")

    # --- config ---
    paths = [Path(p) for p in cli.config] if cli.config else None
    cfg = args_utils.load_run_config(paths)
    if cli.output_dir:
        cfg = dataclasses.replace(cfg, output_dir=cli.output_dir)
    if cli.save_per_epoch_fraction is not None:
        cfg = dataclasses.replace(
            cfg, save_per_epoch_fraction=max(1, int(cli.save_per_epoch_fraction))
        )
    if cli.save_every_n_updates is not None:
        cfg = dataclasses.replace(
            cfg, save_every_n_updates=max(1, int(cli.save_every_n_updates))
        )

    args_utils.set_seed(cfg.seed)
    reward_models, reward_formats = _reward_lists_for_weight(cfg, w)

    _main_print(
        args_utils.Naming.str_dict(
            {
                **dataclasses.asdict(cfg),
                "weight": w,
                "reward_models": reward_models,
                "reward_formats": reward_formats,
                "init_adapter": init_adapter_arg,
            }
        )
    )

    # --- device (no device_map='auto') ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = _is_main_process_env()

    # --- tokenizer ---
    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )

    # --- dataset + dataloader ---
    dataset = summary.build_dataset(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.train_split,
        max_train_samples=cfg.max_train_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # --- reward pipes ---
    reward_pipes = Pipelines.load_pipes(reward_models, device, cfg.hf_cache)
    if cfg.task_name == "reddit_summarization":
        for pipe in reward_pipes:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    # --- reference model (frozen) ---
    ref_model = reinforce_utils.load_ref_model(cfg, device)

    # --- policy model + run bookkeeping ---
    start_epoch = 0
    start_batch_skip = 0
    global_update_start = 0
    save_root: Path
    run_name: str

    if cli.resume:
        resume_path = Path(cli.resume).resolve()
        meta, start_epoch, start_batch_skip, global_update_start = _parse_resume_meta(
            resume_path, w, cfg.num_epochs
        )
        run_dir = resume_path.parent
        save_root = run_dir.parent
        run_name = run_dir.name
        if meta.get("run_name") and meta["run_name"] != run_name:
            raise ValueError(
                f"Checkpoint meta run_name {meta['run_name']!r} != directory {run_name!r}."
            )
        adapter_dir = resume_path / "adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")
        model = reinforce_utils.load_policy_from_adapter(cfg, adapter_dir, device)
        _main_print(
            f"Resuming run {run_name} at epoch={start_epoch}, "
            f"skip_batches={start_batch_skip}, global_update={global_update_start} "
            f"(target num_epochs={cfg.num_epochs})."
        )

    elif init_adapter_arg:
        init_path = Path(init_adapter_arg).resolve()
        adapter_dir = init_path / "adapter" if (init_path / "adapter").is_dir() else init_path
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"--init_adapter: adapter path is not a directory: {adapter_dir}"
            )
        model = reinforce_utils.load_policy_from_adapter(cfg, adapter_dir, device)
        _main_print("Initialized policy from --init_adapter", init_path)
        run_name = reinforce_utils.run_name_reinforce(cfg, w)
        save_root = Path(cfg.output_dir).resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    else:
        model = reinforce_utils.load_policy_model(cfg, device)
        run_name = reinforce_utils.run_name_reinforce(cfg, w)
        save_root = Path(cfg.output_dir).resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    # Verify that only LoRA / DoRA parameters are trainable.
    assert_only_expected_trainables(model)
    if is_main:
        from qwen.utils.ppo_utils import Loader
        Loader.print_trainable_parameters(model)

    # --- optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
    )

    # Load optimizer state when resuming (done after optimizer construction so
    # parameter groups are already set up correctly).
    if cli.resume:
        opt_file = resume_path / "optimizer.pt"  # type: ignore[possibly-undefined]
        if opt_file.is_file():
            try:
                sd = torch.load(opt_file, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(opt_file, map_location="cpu")
            optimizer.load_state_dict(sd)
            _main_print("Loaded optimizer state from", opt_file)
        else:
            _main_print("No optimizer.pt in checkpoint; optimizer re-initialized.")

    # --- training ---
    _train_loop(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        reward_pipes=reward_pipes,
        reward_formats=reward_formats,
        reward_weight=w,
        optimizer=optimizer,
        cfg=cfg,
        run_name=run_name,
        save_root=save_root,
        start_epoch=start_epoch,
        start_batch_skip=start_batch_skip,
        global_update_start=global_update_start,
        device=device,
        is_main=is_main,
    )


if __name__ == "__main__":
    main()
