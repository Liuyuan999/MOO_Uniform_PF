"""CLI defaults, YAML merge (config-driven runs), and naming helpers."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from trl import set_seed as trl_set_seed
except (ImportError, AttributeError):
    from accelerate.utils import set_seed as trl_set_seed

# Repo root: qwen/utils/args_utils.py -> parents[2] = LLM_FT
REPO_ROOT = Path(__file__).resolve().parents[2]

LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
FOLDER_EXPE = os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "outputs"))


def default_config_paths() -> list[Path]:
    return [
        REPO_ROOT / "configs/base.yaml",
        REPO_ROOT / "configs/task/reddit_summarization.yaml",
        REPO_ROOT / "configs/rl/ppo.yaml",
        REPO_ROOT / "configs/eval/default.yaml",
        REPO_ROOT / "configs/model/qwen_0p5b_dora.yaml",
    ]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {p}")
    return data


def merge_yaml(paths: list[Path | str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _deep_merge(merged, load_yaml(path))
    return merged


def _positive_int_or_none(raw: Any) -> int | None:
    if raw is None:
        return None
    i = int(raw)
    return i if i > 0 else None


@dataclass
class RunConfig:
    """Flattened view of merged YAML for Qwen PPO / inference."""

    seed: int
    output_dir: str
    model_name: str
    tokenizer_name: str
    trust_remote_code: bool
    dtype: str
    device_map: str | dict[str, Any]
    hf_cache: str
    peft_method: str
    peft_r: int
    peft_alpha: int
    peft_dropout: float
    peft_target_modules: tuple[str, ...]
    task_name: str
    dataset_name: str
    train_split: str
    eval_split: str
    max_train_samples: int | None
    reward_models: tuple[str, ...]
    reward_formats: tuple[str, ...]
    init_kl_coef: float
    learning_rate: float
    batch_size: int
    mini_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    output_min_length: int
    output_max_length: int
    log_with: str | None
    eval_max_new_tokens: int
    eval_num_samples: int
    save_per_epoch_fraction: int | None
    save_every_n_updates: int | None
    adaptive_kl: bool


def run_config_from_merged(data: dict[str, Any]) -> RunConfig:
    m, p, h = data["model"], data["peft"], data["hf"]
    tm = p.get("target_modules") or []
    if not isinstance(tm, list) or not tm:
        raise ValueError("peft.target_modules must be a non-empty list in model YAML")
    t = data.get("task") or {}
    r = data.get("rl") or {}
    e = data.get("eval") or {}
    b = data
    rm = t.get("reward_models") or []
    rf = t.get("reward_formats") or []
    if len(rm) != len(rf):
        raise ValueError(
            f"task.reward_models length ({len(rm)}) must match task.reward_formats length ({len(rf)})."
        )
    log_with = r.get("log_with")
    if isinstance(log_with, str) and log_with.lower() in ("none", "null", ""):
        log_with = None
    elif log_with is not None:
        log_with = str(log_with)
    return RunConfig(
        seed=int(b.get("seed", 42)),
        output_dir=str(b.get("output_dir", "outputs/run")),
        model_name=str(m["name"]),
        tokenizer_name=str(m["tokenizer_name"]),
        trust_remote_code=bool(m.get("trust_remote_code", True)),
        dtype=str(m.get("dtype", "bfloat16")),
        device_map=m.get("device_map", None),
        hf_cache=str(h["local_cache"]),
        peft_method=str(p["method"]).lower(),
        peft_r=int(p["r"]),
        peft_alpha=int(p["alpha"]),
        peft_dropout=float(p["dropout"]),
        peft_target_modules=tuple(str(x) for x in tm),
        task_name=str(t.get("name", "reddit_summarization")),
        dataset_name=str(t.get("dataset_name", "openai")),
        train_split=str(t.get("train_split", "train")),
        eval_split=str(t.get("eval_split", "validation")),
        max_train_samples=t.get("max_train_samples"),
        reward_models=tuple(str(x) for x in rm),
        reward_formats=tuple(str(x) for x in rf),
        init_kl_coef=float(r.get("init_kl_coef", 0.2)),
        learning_rate=float(r.get("learning_rate", 1.41e-5)),
        batch_size=int(r.get("batch_size", 32)),
        mini_batch_size=int(r.get("mini_batch_size", 4)),
        gradient_accumulation_steps=int(r.get("gradient_accumulation_steps", 1)),
        num_epochs=int(r.get("num_epochs", 1)),
        output_min_length=int(r.get("output_min_length", 16)),
        output_max_length=int(r.get("output_max_length", 32)),
        log_with=log_with,
        eval_max_new_tokens=int(e.get("max_new_tokens", 128)),
        eval_num_samples=int(e.get("num_samples", 64)),
        save_per_epoch_fraction=_positive_int_or_none(r.get("save_per_epoch_fraction")),
        save_every_n_updates=_positive_int_or_none(r.get("save_every_n_updates")),
        adaptive_kl=bool(r.get("adaptive_kl", False)),
    )


def load_run_config(paths: list[Path | str] | None = None) -> RunConfig:
    paths = paths or default_config_paths()
    merged = merge_yaml(paths)
    need = {"model", "peft", "hf"}
    if not need.issubset(merged.keys()):
        raise ValueError(f"Merged config must include keys {need}, got {sorted(merged.keys())}")
    return run_config_from_merged(merged)


def set_seed(seed: int) -> None:
    trl_set_seed(seed)


class Naming:
    @staticmethod
    def str_dict(dict_args: dict[str, Any]) -> str:
        str_out = "\n" + "#" * 40
        col_width = max(len(str(word)) for word in dict_args) + 2
        for arg in sorted(dict_args.keys()):
            if str(arg).startswith("__"):
                continue
            str_print = str(dict_args[arg])
            str_out += "\n" + "".join([str(arg).ljust(col_width), str_print])
        str_out += "\n" + "#" * 40 + "\n"
        return str_out

    @staticmethod
    def get_name_model(name: str) -> str:
        list_reward_suffix = re.split(r"-|_", name.split("/")[-1])
        list_reward_suffix = [t for t in list_reward_suffix if t]
        return "".join(t[0] for t in list_reward_suffix)

    @staticmethod
    def weight_tag(weight: float) -> str:
        """Filesystem-friendly tag for scalarization weight, e.g. 0.5 -> w0p5, 1.0 -> w1p0."""
        s = f"{float(weight):.4f}".rstrip("0").rstrip(".")
        if s == "":
            s = "0"
        return "w" + s.replace(".", "p").replace("-", "m")

    @staticmethod
    def run_name_qwen(cfg: RunConfig, weight: float) -> str:
        suffix = Naming.weight_tag(weight)
        return f"{cfg.model_name.split('/')[-1]}-ppo-summary-{suffix}-{datetime.now().strftime('%m%d-%H%M%S')}"[:92]


def parse_train_ppo_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Qwen summarization with reward scalarization (1-w)*r1 + w*r2"
    )
    parser.add_argument(
        "--weight",
        type=float,
        required=True,
        help="Scalarization in [0,1]: (1-w)*reward_1 + w*reward_2. w=0 => only reward 1; w=1 => only reward 2.",
    )
    parser.add_argument("--config", nargs="*", type=str, default=None, help="YAML files (merged L→R)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config output_dir")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory (e.g. .../run_dir/checkpoint_epoch_0001) to resume training.",
    )
    parser.add_argument(
        "--save_per_epoch_fraction",
        type=int,
        default=None,
        help="Optional override: save a mid-epoch checkpoint every ~1/N epoch.",
    )
    parser.add_argument(
        "--save_every_n_updates",
        type=int,
        default=None,
        help="Optional override: save a mid-epoch checkpoint every N PPO updates (takes precedence).",
    )
    parser.add_argument(
        "--max_updates",
        type=int,
        default=None,
        help="Optional cap on total PPO update steps (global_update); stops early after this many steps.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subdirectory name under output_dir for this run (default: auto from model + weight + time).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override config num_epochs for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for this run.",
    )
    parser.add_argument(
        "--init_adapter",
        type=str,
        default=None,
        help="Load PEFT adapter (and optional value head) from this path before training; mutually exclusive with --resume.",
    )
    return parser.parse_args()


def parse_inference_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference / Rewarded Soup evaluation")
    parser.add_argument("--config", nargs="*", type=str, default=None)
    parser.add_argument("--peft_names", nargs="+", type=str, required=True)
    parser.add_argument("--num_lambdas", type=int, default=11)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help="Override task dataset; use 'samples' for fake")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument(
        "--eval_reward_models",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional eval-time reward models (space-separated HF paths). "
            "Defaults to the training reward models from the task config."
        ),
    )
    parser.add_argument(
        "--eval_reward_formats",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Score format strings for --eval_reward_models (one per model, same syntax as reward_formats in task YAML). "
            "Defaults to the training reward_formats when --eval_reward_models is not set; "
            "must be provided when --eval_reward_models specifies models with different output formats."
        ),
    )
    return parser.parse_args()


def parse_soup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soup interpolation between two endpoints")
    parser.add_argument("--config", nargs="*", type=str, default=None)
    parser.add_argument("--peft_a", type=str, required=True)
    parser.add_argument("--peft_b", type=str, required=True)
    parser.add_argument("--num_lambdas", type=int, default=11)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to sample prompts from (defaults to task.train_split).",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSON + plots.")
    parser.add_argument(
        "--save_soups_dir",
        type=str,
        default=None,
        help="Optional directory to persist each interpolated soup adapter.",
    )
    parser.add_argument(
        "--include_kl",
        dest="include_kl",
        action="store_true",
        default=True,
        help="Include KL computations/regularized objectives (default: enabled).",
    )
    parser.add_argument(
        "--no_include_kl",
        dest="include_kl",
        action="store_false",
        help="Disable KL computations/regularized objectives.",
    )
    parser.add_argument("--fake_samples", action="store_true")
    return parser.parse_args()
