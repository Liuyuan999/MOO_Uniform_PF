"""PPO training for Qwen summarization with reward scalarization (adapted from RS train_ppo.py)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import torch

from qwen.tasks import summary
from qwen.utils import args_utils, ppo_utils, trl_compat
from qwen.utils.qwen_utils import Tokenizer


def _is_main_process_env() -> bool:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank == 0 and local_rank == 0


def _main_print(*args: object, **kwargs: object) -> None:
    if _is_main_process_env():
        print(*args, **kwargs)


def _apply_init_adapter_weights(model: torch.nn.Module, init_path: Path) -> None:
    """Load PEFT adapter (and optional value head); fresh run — no meta or optimizer."""
    p = init_path.resolve()
    adapter_dir = p / "adapter" if (p / "adapter").is_dir() else p
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"--init_adapter: adapter path is not a directory: {adapter_dir}")
    ppo_utils.Loader.load_trained_adapter_into_policy(model, adapter_dir)
    ppo_utils.Loader.load_value_head_if_present(model, p)
    if adapter_dir == p:
        ppo_utils.Loader.load_value_head_if_present(model, p.parent)
    _main_print("Initialized policy from --init_adapter", p)


def _reward_lists_for_weight(cfg: args_utils.RunConfig, weight: float) -> tuple[list[str], list[str]]:
    """Always load both reward heads; scalarization still uses (1-w)*r1 + w*r2."""
    del weight  # kept for API compatibility with callers
    if len(cfg.reward_models) < 2 or len(cfg.reward_formats) < 2:
        raise ValueError("Config must list two task.reward_models and task.reward_formats.")
    return [cfg.reward_models[0], cfg.reward_models[1]], [cfg.reward_formats[0], cfg.reward_formats[1]]


def _load_resume_checkpoint(
    *,
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_weight: float,
    num_epochs: int,
) -> tuple[int, int, int]:
    """Load adapter, optional value head & optimizer.

    Returns ``(start_epoch, batches_to_skip_in_first_epoch, global_update_start)``.
    """
    meta_file = resume_path / "meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Resume path must contain meta.json: {resume_path}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rw = float(meta.get("weight", expected_weight))
    if abs(rw - float(expected_weight)) > 1e-5:
        raise ValueError(
            f"Checkpoint weight {rw} does not match --weight {expected_weight}. "
            "Use the same scalarization weight as the run you are resuming."
        )
    adapter_dir = resume_path / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")
    ppo_utils.Loader.load_trained_adapter_into_policy(model, adapter_dir)
    ppo_utils.Loader.load_value_head_if_present(model, resume_path)
    opt_file = resume_path / "optimizer.pt"
    if opt_file.is_file():
        try:
            sd = torch.load(opt_file, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(opt_file, map_location="cpu")
        optimizer.load_state_dict(sd)
        _main_print("Loaded optimizer state from", opt_file)
    else:
        _main_print("No optimizer.pt in checkpoint; optimizer re-initialized.")

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
                f"Checkpoint epoch_index={start_epoch} >= num_epochs={num_epochs}; nothing to resume."
            )
    elif start_epoch >= num_epochs:
        raise ValueError(
            f"Checkpoint reports completed_epochs={start_epoch} >= num_epochs={num_epochs} in your config. "
            "Increase num_epochs in YAML to train further, or pick an earlier checkpoint."
        )

    return start_epoch, skip_batches, global_start


def run_ppo(
    *,
    config_paths: list[Path | str] | None,
    weight: float,
    output_dir: Path | str | None = None,
    run_name: str | None = None,
    init_adapter: Path | str | None = None,
    resume: Path | str | None = None,
    num_epochs: int | None = None,
    max_updates: int | None = None,
    save_per_epoch_fraction: int | None = None,
    save_every_n_updates: int | None = None,
    seed: int | None = None,
) -> Path:
    """Run one PPO job; returns the run directory containing ``checkpoint_final``.

    Intended for programmatic use (e.g. CDF outer loop). CLI ``main()`` delegates here.
    """
    if not trl_compat.LEGACY_PPO_TRAINER:
        raise RuntimeError(trl_compat.TRL_PIN_MESSAGE)
    if init_adapter is not None and resume is not None:
        raise ValueError("Use either init_adapter or resume, not both.")
    w = float(weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"weight must be in [0, 1], got {w}")

    paths = [Path(p) for p in config_paths] if config_paths else None
    cfg = args_utils.load_run_config(paths)
    if output_dir is not None:
        cfg = dataclasses.replace(cfg, output_dir=str(Path(output_dir).resolve()))
    if save_per_epoch_fraction is not None:
        cfg = dataclasses.replace(cfg, save_per_epoch_fraction=max(1, int(save_per_epoch_fraction)))
    if save_every_n_updates is not None:
        cfg = dataclasses.replace(cfg, save_every_n_updates=max(1, int(save_every_n_updates)))
    if num_epochs is not None:
        cfg = dataclasses.replace(cfg, num_epochs=int(num_epochs))
    if seed is not None:
        cfg = dataclasses.replace(cfg, seed=int(seed))

    args_utils.set_seed(cfg.seed)
    reward_models, reward_formats = _reward_lists_for_weight(cfg, w)

    _main_print(
        args_utils.Naming.str_dict(
            {
                **dataclasses.asdict(cfg),
                "weight": w,
                "reward_models": reward_models,
                "reward_formats": reward_formats,
                "init_adapter": str(init_adapter) if init_adapter else None,
                "resume": str(resume) if resume else None,
                "max_updates": max_updates,
            }
        )
    )

    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )

    _main_print("tokenizer:", tokenizer)

    model = ppo_utils.Loader.load_policy_with_value_head(cfg)
    ref_model = ppo_utils.Loader.load_ref_value_head(cfg)
    if init_adapter is not None:
        _apply_init_adapter_weights(model, Path(init_adapter))
    if _is_main_process_env():
        ppo_utils.Loader.print_trainable_parameters(model)
    ppo_utils.assert_only_expected_trainables(model)

    dataset = summary.build_dataset(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.train_split,
        max_train_samples=cfg.max_train_samples,
    )

    ppo_config = trl_compat.PPOConfig(
        learning_rate=cfg.learning_rate,
        init_kl_coef=cfg.init_kl_coef,
        adap_kl_ctrl=cfg.adaptive_kl,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ppo_config.learning_rate,
    )

    start_epoch = 0
    start_batch_skip = 0
    global_update_start = 0
    save_root: Path
    resolved_run_name: str
    if resume:
        resume_path = Path(resume).resolve()
        start_epoch, start_batch_skip, global_update_start = _load_resume_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            expected_weight=w,
            num_epochs=cfg.num_epochs,
        )
        run_dir = resume_path.parent
        save_root = run_dir.parent
        resolved_run_name = run_dir.name
        meta = json.loads((resume_path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("run_name") and meta["run_name"] != resolved_run_name:
            raise ValueError(
                f"Checkpoint meta run_name {meta['run_name']!r} does not match directory {resolved_run_name!r}."
            )
        _main_print(
            f"Resuming run {resolved_run_name} at epoch={start_epoch}, skip_batches={start_batch_skip}, "
            f"global_update={global_update_start} (target num_epochs={cfg.num_epochs})."
        )
    else:
        resolved_run_name = run_name if run_name else args_utils.Naming.run_name_qwen(cfg, w)
        save_root = Path(cfg.output_dir).resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    try:
        ppo_trainer = trl_compat.PPOTrainer(
            ppo_config,
            model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=ppo_utils.collator,
            optimizer=optimizer,
        )
    except TypeError:
        ppo_trainer = trl_compat.PPOTrainer(
            ppo_config,
            model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=ppo_utils.collator,
        )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    runner = ppo_utils.Runner(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        reward_models=reward_models,
        reward_formats=reward_formats,
        transform_text_summary=summary.transform_text_summary,
        reward_weight=w,
        optimizer=optimizer,
    )

    if cfg.task_name == "reddit_summarization":
        for pipe in runner.reward_pipes:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    runner.train_ppo(
        model,
        num_epochs=cfg.num_epochs,
        save_root=save_root,
        run_name=resolved_run_name,
        start_epoch=start_epoch,
        start_batch_skip=start_batch_skip,
        global_update_start=global_update_start,
        max_global_updates=max_updates,
    )
    return save_root / resolved_run_name


def main() -> None:
    cli = args_utils.parse_train_ppo_args()
    if cli.init_adapter and cli.resume:
        raise ValueError("Use either --init_adapter or --resume, not both.")
    run_ppo(
        config_paths=[Path(p) for p in cli.config] if cli.config else None,
        weight=float(cli.weight),
        output_dir=cli.output_dir,
        run_name=cli.run_name,
        init_adapter=cli.init_adapter,
        resume=cli.resume,
        num_epochs=cli.num_epochs,
        seed=cli.seed,
        save_per_epoch_fraction=cli.save_per_epoch_fraction,
        save_every_n_updates=cli.save_every_n_updates,
        max_updates=cli.max_updates,
    )


if __name__ == "__main__":
    main()
