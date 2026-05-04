"""CLI for Rewarded Soup interpolation (two endpoint adapters)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qwen.tasks import summary
from qwen.utils import args_utils, inference_utils, ppo_utils
from qwen.utils.qwen_utils import Pipelines, Tokenizer

device = 0 if torch.cuda.is_available() else "cpu"


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def main() -> None:
    cli = args_utils.parse_soup_args()
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

    num_samples = cli.num_samples if cli.num_samples is not None else cfg.eval_num_samples
    split = cli.split or cfg.train_split
    if cli.fake_samples:
        query_tensors = summary.Samples.get_fake_samples(bs=num_samples, tokenizer=tokenizer)
    elif split == cfg.train_split:
        # Match PPO candidate pool: training split with the same max_train_samples truncation.
        train_ds = summary.build_dataset(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_train_samples=cfg.max_train_samples,
        )
        n = min(int(num_samples), len(train_ds))
        rng = np.random.default_rng(int(cfg.seed))
        idxs = rng.choice(len(train_ds), size=n, replace=False)
        query_tensors = [train_ds[int(i)]["input_ids"] for i in idxs]
    else:
        query_tensors = summary.Samples.get_samples(
            dataset_name=cfg.dataset_name,
            tokenizer=tokenizer,
            bs=num_samples,
            split=split,
        )

    base_model = inference_utils.Loader.load_base_model(cfg)
    base_model = base_model.to(device)
    kl_ref = None
    if cli.include_kl:
        kl_ref = ppo_utils.Loader.load_ref_value_head(cfg)
        if cfg.device_map is None:
            kl_ref = kl_ref.to(device)
    reward_pipes = Pipelines.load_pipes(list(cfg.reward_models), device=device, cache_dir=cfg.hf_cache)
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
        verbose=False,
        include_kl=bool(cli.include_kl),
        kl_ref_model=kl_ref,
    )
    results = inference_utils.get_results_rewards(
        resultscomputer,
        peft_names=[cli.peft_a, cli.peft_b],
        num_lambdas=cli.num_lambdas,
        save_soups_dir=cli.save_soups_dir,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "meta": {
            "peft_names": [cli.peft_a, cli.peft_b],
            "num_lambdas": int(cli.num_lambdas),
            "num_samples": int(num_samples),
            "dataset_name": cfg.dataset_name,
            "split": split,
            "reward_model_names": list(cfg.reward_models),
            "reward_formats": list(cfg.reward_formats),
            "save_soups_dir": cli.save_soups_dir,
            "include_kl": bool(cli.include_kl),
            "kl_beta": float(cfg.init_kl_coef) if cli.include_kl else None,
        }
    }
    payload.update(_to_jsonable(results))

    out_path = out_dir / "rewardedsoups_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved results to:", out_path)

    plot_path_kl = out_dir / "rewardedsoups_frontier_kl.png"
    inference_utils.plot_soup_frontier(results, plot_path_kl, include_kl=True)
    plot_path_raw = out_dir / "rewardedsoups_frontier_raw.png"
    inference_utils.plot_soup_frontier(results, plot_path_raw, include_kl=False)


if __name__ == "__main__":
    main()
