"""Inference and soup-style evaluation (adapted from RS inference_rewardedsoups.py; summary only)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from qwen.tasks import summary
from qwen.utils import args_utils, inference_utils, ppo_utils
from qwen.utils.qwen_utils import Pipelines, Tokenizer

device = 0 if torch.cuda.is_available() else "cpu"


def _extract_extra_inference_flags() -> tuple[str | None, bool]:
    """Strip ``--save_soups_dir`` / ``--include_kl`` from ``sys.argv`` before ``parse_inference_args``."""
    save: str | None = None
    include_kl = False
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--save_soups_dir" and i + 1 < len(sys.argv):
            save = sys.argv[i + 1]
            i += 2
        elif a.startswith("--save_soups_dir="):
            save = a.split("=", 1)[1]
            i += 1
        elif a == "--include_kl":
            include_kl = True
            i += 1
        elif a.startswith("--include_kl="):
            include_kl = a.split("=", 1)[1].strip().lower() in ("1", "true", "yes")
            i += 1
        else:
            new_argv.append(a)
            i += 1
    sys.argv = new_argv
    return save, include_kl


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def main() -> None:
    save_soups_dir, include_kl = _extract_extra_inference_flags()
    cli = args_utils.parse_inference_args()
    paths = [Path(p) for p in cli.config] if cli.config else None
    cfg = args_utils.load_run_config(paths)

    if cli.output_dir:
        cfg = dataclasses.replace(cfg, output_dir=cli.output_dir)

    args_utils.set_seed(cfg.seed)
    print(args_utils.Naming.str_dict({**vars(cli), **dataclasses.asdict(cfg)}))

    tokenizer = Tokenizer.load_tokenizer(
        cfg.tokenizer_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
    )

    dataset_name = cli.dataset_name or cfg.dataset_name
    num_samples = cli.num_samples if cli.num_samples is not None else cfg.eval_num_samples

    if dataset_name == "samples":
        query_tensors = summary.Samples.get_fake_samples(bs=num_samples, tokenizer=tokenizer)
    else:
        query_tensors = summary.Samples.get_samples(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            bs=num_samples,
            split=cfg.eval_split,
        )

    print("First decoded query:", tokenizer.decode(query_tensors[0]))

    base_model = inference_utils.Loader.load_base_model(cfg)
    base_model = base_model.to(device)
    kl_ref = None
    if include_kl:
        kl_ref = ppo_utils.Loader.load_ref_value_head(cfg)
        if cfg.device_map is None:
            kl_ref = kl_ref.to(device)

    # Reward models: use eval-specific ones if provided, otherwise fall back to training models.
    train_reward_models = list(cfg.reward_models)
    train_reward_formats = list(cfg.reward_formats)
    if cli.eval_reward_models:
        eval_reward_models = list(cli.eval_reward_models)
        eval_reward_formats = list(cli.eval_reward_formats) if cli.eval_reward_formats else train_reward_formats
        if len(eval_reward_formats) != len(eval_reward_models):
            raise ValueError(
                f"--eval_reward_formats length ({len(eval_reward_formats)}) must match "
                f"--eval_reward_models length ({len(eval_reward_models)})."
            )
        # Override cfg so that evaluate_scalars_structured uses the eval formats.
        cfg = dataclasses.replace(cfg, reward_models=tuple(eval_reward_models), reward_formats=tuple(eval_reward_formats))
        print(f"Using eval reward models: {eval_reward_models}")
    else:
        eval_reward_models = train_reward_models
        eval_reward_formats = train_reward_formats

    reward_pipes = Pipelines.load_pipes(eval_reward_models, device=device, cache_dir=cfg.hf_cache)
    if cfg.task_name == "reddit_summarization":
        for p in reward_pipes:
            p.tokenizer.pad_token_id = p.model.config.eos_token_id

    predictor = summary.PredictorSummary(
        reward_pipes=reward_pipes,
        tokenizer=tokenizer,
        output_max_length=cfg.eval_max_new_tokens,
        device=device,
    )
    resultscomputer = inference_utils.ResultsComputer(
        cfg=cfg,
        predictor=predictor,
        base_model=base_model,
        query_tensors=query_tensors,
        verbose=bool(cli.verbose),
        include_kl=include_kl,
        kl_ref_model=kl_ref,
    )

    results = inference_utils.get_results_rewards(
        resultscomputer,
        peft_names=cli.peft_names,
        num_lambdas=cli.num_lambdas,
        save_soups_dir=save_soups_dir,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "peft_names": cli.peft_names,
        "num_lambdas": cli.num_lambdas,
        "num_samples": num_samples,
        "dataset_name": dataset_name,
        "train_reward_model_names": train_reward_models,
        "eval_reward_model_names": eval_reward_models,
        "eval_reward_formats": list(cfg.reward_formats),
        "save_soups_dir": save_soups_dir,
        "include_kl": include_kl,
        "kl_beta": float(cfg.init_kl_coef) if include_kl else None,
    }
    payload: dict[str, Any] = {"meta": meta}
    payload.update(to_jsonable(results))

    out_path = out_dir / "rewardedsoups_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {out_path}")

    # Plot with KL regularization (if available in results)
    plot_path_kl = out_dir / "rewardedsoups_frontier_kl.png"
    inference_utils.plot_soup_frontier(results, plot_path_kl, include_kl=True)

    # Plot without KL (raw rewards only)
    plot_path_raw = out_dir / "rewardedsoups_frontier_raw.png"
    inference_utils.plot_soup_frontier(results, plot_path_raw, include_kl=False)


if __name__ == "__main__":
    main()
