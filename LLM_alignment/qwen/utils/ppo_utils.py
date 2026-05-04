"""PPO Loader, Runner, and training helpers (adapted from RS ppo_utils; Qwen + YAML PEFT)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from qwen.utils.args_utils import RunConfig
from qwen.utils.trl_compat import (
    AutoModelForCausalLMWithValueHead,
    LengthSampler,
    PPOConfig,
    PPOTrainer,
)
from qwen.utils.qwen_utils import Instructions, Pipelines

COGCOMP_FAITHFUL_PREFIX = "CogComp/bart-faithful-summary-detector"


def _is_cogcomp_faithful_pipe(reward_pipe: Any) -> bool:
    return getattr(getattr(reward_pipe, "model", None), "name_or_path", "").startswith(COGCOMP_FAITHFUL_PREFIX)


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    k = dtype_str.lower().strip()
    if k not in m:
        raise ValueError(dtype_str)
    return m[k]


def build_lora_config(cfg: RunConfig) -> LoraConfig:
    if cfg.peft_method not in ("lora", "dora"):
        raise ValueError(f"peft.method must be lora or dora, got {cfg.peft_method}")
    return LoraConfig(
        r=cfg.peft_r,
        lora_alpha=cfg.peft_alpha,
        lora_dropout=cfg.peft_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg.peft_target_modules),
        use_dora=cfg.peft_method == "dora",
    )


class Loader:
    @staticmethod
    def load_base_causal(cfg: RunConfig) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.hf_cache,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=resolve_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
        )

    @staticmethod
    def load_policy_with_value_head(cfg: RunConfig) -> Any:
        lora = build_lora_config(cfg)
        return AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model_name,
            peft_config=lora,
            cache_dir=cfg.hf_cache,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=resolve_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
        )

    @staticmethod
    def load_ref_value_head(cfg: RunConfig) -> Any:
        """Frozen π_ref: pretrained Qwen instruct + value head, no PEFT."""
        ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.hf_cache,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=resolve_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
        )
        ref.eval()
        ref.requires_grad_(False)
        return ref

    @staticmethod
    def load_trained_adapter_into_policy(model: Any, adapter_dir: str | Path) -> None:
        """Load LoRA adapter weights from a PEFT folder saved via ``save_pretrained``."""
        path = str(Path(adapter_dir).resolve())
        inner = getattr(model, "pretrained_model", None)
        if inner is None:
            raise TypeError("Expected model with pretrained_model (PeftModel).")
        if not isinstance(inner, PeftModel):
            raise TypeError(f"pretrained_model must be PeftModel, got {type(inner)}")
        inner.load_adapter(path, adapter_name="default", is_trainable=True)
        inner.set_adapter("default")

    @staticmethod
    def save_value_head_if_present(model: Any, folder: Path) -> None:
        vh = getattr(model, "v_head", None)
        if vh is not None:
            torch.save(vh.state_dict(), folder / "value_head.pt")

    @staticmethod
    def load_value_head_if_present(model: Any, folder: Path) -> None:
        p = folder / "value_head.pt"
        if not p.is_file():
            return
        vh = getattr(model, "v_head", None)
        if vh is None:
            return
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(p, map_location="cpu")
        vh.load_state_dict(sd)

    @staticmethod
    def print_trainable_parameters(model: Any) -> None:
        trainable = all_p = 0
        for _, p in model.named_parameters():
            all_p += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"trainable params: {trainable} || all params: {all_p} || trainable%: {100 * trainable / max(all_p, 1):.4f}")

def assert_only_expected_trainables(model: Any) -> None:
    bad: list[str] = []
    allowed_keywords = ("lora_", "dora", "v_head")

    for name, p in model.named_parameters():
        if p.requires_grad and not any(k in name for k in allowed_keywords):
            bad.append(name)

    if bad:
        preview = bad[:20]
        raise RuntimeError(f"Unexpected trainable parameters: {preview}")
        
def collator(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [d[key] for d in data] for key in data[0]}


def _normalize_id2label(raw: Any) -> dict[int, str] | None:
    if not raw:
        return None
    out: dict[int, str] = {}
    for k, v in dict(raw).items():
        ik = int(k) if isinstance(k, str) and str(k).isdigit() else int(k)
        out[ik] = str(v)
    return out


def _scores_vector_from_output(output: Any, *, id2label: dict[int, str] | None) -> list[float]:
    """Map pipeline output to [score for class 0, score for class 1, ...].

    Newer ``transformers`` text-classification returns a single ``dict`` when
    ``top_k`` defaults (legacy top-1). Passing ``top_k=None`` yields a list of
    per-class dicts; that list may be sorted by score, so we match by ``label``
    when ``id2label`` is available (Rewarded Soups indices refer to class id).
    """
    if isinstance(output, list) and output and isinstance(output[0], dict) and "score" in output[0]:
        if id2label is not None:
            by_label = {str(d["label"]): float(d["score"]) for d in output if "label" in d}
            n = len(id2label)
            return [by_label[str(id2label[j])] for j in range(n)]
        return [float(output[j]["score"]) for j in range(len(output))]
    if isinstance(output, dict) and "score" in output and "label" in output:
        if id2label is not None and len(id2label) > 1:
            raise ValueError(
                "Reward pipeline returned only one label dict; use top_k=None (all labels) for multi-class reward_formats."
            )
        return [float(output["score"])]
    raise TypeError(f"Unexpected reward pipeline output: {type(output)!r}")


def _stats_scalar(stats: Any, *keys: str) -> float | None:
    if not isinstance(stats, dict):
        return None
    for k in keys:
        if k not in stats:
            continue
        v = stats[k]
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().item())
    return None


def _ppo_stats_summary(stats: Any) -> dict[str, float]:
    """Flatten a few common TRL PPO stat keys for logging."""
    if not isinstance(stats, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in stats.items():
        if not isinstance(k, str) or "/" not in k:
            continue
        if isinstance(v, (int, float)):
            out[k.replace("/", "_")] = float(v)
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k.replace("/", "_")] = float(v.detach().cpu().item())
    return out


def get_score_from_output(
    output: Any,
    reward_format: str,
    *,
    id2label: dict[int, str] | None = None,
) -> float:
    if reward_format == "":
        return 0.0
    fmt = reward_format.split("x")[0] if "x" in reward_format else reward_format
    vec = _scores_vector_from_output(output, id2label=id2label)
    if "-" in fmt:
        a, b = fmt.split("-", 1)
        return float(vec[int(a)]) - float(vec[int(b)])
    return float(vec[int(fmt)])


class Runner:
    """PPO rollout + reward; scalarization (1-w)*r1 + w*r2 when two reward heads are loaded."""

    def __init__(
        self,
        ppo_trainer: PPOTrainer,
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        cfg: RunConfig,
        reward_models: list[str],
        reward_formats: list[str],
        transform_text_summary: Any,
        reward_weight: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.ppo_trainer = ppo_trainer
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        self.reward_models = reward_models
        self.reward_formats = reward_formats
        self.reward_weight = float(reward_weight)
        self.optimizer = optimizer
        self.transform_text_summary = transform_text_summary
        # Map local loaded reward heads (possibly a subset) to global cfg.reward_models slots.
        self._local_to_global_reward_idx: list[int] = []
        used_global: set[int] = set()
        cfg_models = list(cfg.reward_models)
        cfg_formats = list(cfg.reward_formats)
        for local_model, local_fmt in zip(self.reward_models, self.reward_formats):
            g_idx = None
            for i, (gm, gf) in enumerate(zip(cfg_models, cfg_formats)):
                if i in used_global:
                    continue
                if gm == local_model and gf == local_fmt:
                    g_idx = i
                    break
            if g_idx is not None:
                used_global.add(g_idx)
                self._local_to_global_reward_idx.append(g_idx)
        self.generation_kwargs: dict[str, Any] = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        self.output_length_sampler = LengthSampler(cfg.output_min_length, cfg.output_max_length)
        self.reward_pipes = Pipelines.load_pipes(reward_models, device, cfg.hf_cache)
        # top_k=None: return all classes (see transformers TextClassificationPipeline.postprocess).
        # ``return_all_scores`` is not handled by current pipelines and was effectively ignored.
        self.sent_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": cfg.mini_batch_size,
        }

    def _reward_vector_for_pipe(self, i: int, batch: dict[str, list[Any]]) -> np.ndarray:
        reward_pipe = self.reward_pipes[i]
        fmt = self.reward_formats[i].split("x")[0]
        texts = [
            self.transform_text_summary(
                reward_pipe=reward_pipe,
                post=Instructions.get_input(query),
                response=Instructions.get_response(query) + response,
            )
            for query, response in zip(batch["query"], batch["response"])
        ]
        pipe_kw = dict(self.sent_kwargs)
        if _is_cogcomp_faithful_pipe(reward_pipe):
            pipe_kw["batch_size"] = 1
        outs = reward_pipe(texts, **pipe_kw)
        id2l = _normalize_id2label(getattr(reward_pipe.model.config, "id2label", None))
        if not id2l:
            id2l = None
        rewards = [get_score_from_output(o, fmt, id2label=id2l) for o in outs]
        if "x" in self.reward_formats[i]:
            c = float(self.reward_formats[i].split("x")[1])
            rewards = [c * r for r in rewards]
        return np.array(rewards, dtype=np.float64)

    def apply_reward(self, batch: dict[str, list[Any]]) -> np.ndarray:
        w = self.reward_weight
        if len(self.reward_pipes) == 1:
            return self._reward_vector_for_pipe(0, batch)
        if len(self.reward_pipes) == 2:
            r1 = self._reward_vector_for_pipe(0, batch)
            r2 = self._reward_vector_for_pipe(1, batch)
            return (1.0 - w) * r1 + w * r2
        raise ValueError(f"Expected 1 or 2 reward pipes, got {len(self.reward_pipes)}")

    def _global_reward_means_from_step(self, step_out: dict[str, Any]) -> tuple[float | None, float | None]:
        """Return (mean_reward_1, mean_reward_2) in global cfg.reward_models indexing."""
        local_means: list[float | None] = [step_out.get("mean_r1"), step_out.get("mean_r2")]
        g_means: list[float | None] = [None, None]
        if self._local_to_global_reward_idx:
            for local_i, global_i in enumerate(self._local_to_global_reward_idx[:2]):
                if 0 <= global_i <= 1 and local_i < len(local_means):
                    g_means[global_i] = local_means[local_i]
            return g_means[0], g_means[1]
        # Fallback to legacy behavior if mapping is unavailable.
        return step_out.get("mean_r1"), step_out.get("mean_r2")

    def _save_training_checkpoint(
        self,
        model: Any,
        checkpoint_dir: Path,
        *,
        meta: dict[str, Any],
    ) -> None:
        acc = self.ppo_trainer.accelerator
        is_main = getattr(acc, "is_main_process", getattr(acc, "is_local_main_process", True))
        if not is_main:
            return
        unwrapped = acc.unwrap_model(model) if hasattr(acc, "unwrap_model") else model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = checkpoint_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        inner = getattr(unwrapped, "pretrained_model", unwrapped)
        inner.save_pretrained(str(adapter_dir))
        Loader.save_value_head_if_present(unwrapped, checkpoint_dir)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        with (checkpoint_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        acc.print("Saved checkpoint to", checkpoint_dir)

    def _mid_epoch_interval_batches(self, dl_len: int | None) -> int | None:
        """Batches between mid-epoch saves when using ``save_per_epoch_fraction``."""
        f = self.cfg.save_per_epoch_fraction
        if not f or not dl_len or dl_len <= 0:
            return None
        return max(1, math.ceil(dl_len / f))

    def _should_save_mid_epoch(
        self,
        *,
        global_update: int,
        batch_idx: int,
        dl_len: int | None,
        is_last_batch: bool,
    ) -> bool:
        if is_last_batch:
            return False
        if self.cfg.save_every_n_updates:
            return global_update > 0 and global_update % int(self.cfg.save_every_n_updates) == 0
        interval = self._mid_epoch_interval_batches(dl_len)
        if interval is None:
            return False
        return (batch_idx + 1) % interval == 0

    def train_ppo(
        self,
        model: Any,
        num_epochs: int,
        save_root: Path,
        run_name: str,
        *,
        start_epoch: int = 0,
        start_batch_skip: int = 0,
        global_update_start: int = 0,
        max_global_updates: int | None = None,
    ) -> None:
        run_dir = save_root / run_name

        acc = self.ppo_trainer.accelerator
        is_main = getattr(acc, "is_main_process", getattr(acc, "is_local_main_process", True))
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(acc, "wait_for_everyone"):
            acc.wait_for_everyone()

        try:
            dl_len = len(self.ppo_trainer.dataloader)
        except TypeError:
            dl_len = None

        legacy_metrics_path = run_dir / "training_metrics.jsonl"
        metrics_dir = run_dir / "logs"
        metrics_path = metrics_dir / "training_metrics.jsonl"
        log_f = None
        if is_main:
            # Backward-compatible: if an older run wrote metrics at run_dir root, keep appending there.
            if not metrics_path.is_file() and legacy_metrics_path.is_file():
                metrics_path = legacy_metrics_path
            else:
                metrics_dir.mkdir(parents=True, exist_ok=True)
            append_log = (start_epoch > 0 or start_batch_skip > 0 or global_update_start > 0) and metrics_path.is_file()
            log_f = open(metrics_path, "a" if append_log else "w", encoding="utf-8", buffering=1)

        global_update = int(global_update_start)
        stop_early = False

        try:
            for epoch in range(start_epoch, num_epochs):
                acc.print(f"Begin epoch {epoch + 1}/{num_epochs} (index {epoch})")
                pbar = tqdm(
                    self.ppo_trainer.dataloader,
                    desc=f"PPO epoch {epoch + 1}/{num_epochs}",
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

                    step_out = self.step_ppo(model, batch)
                    global_update += 1

                    stats = step_out.get("stats")
                    mean_kl = _stats_scalar(
                        stats,
                        "ppo/mean_kl",
                        "objective/kl",
                        "kl",
                    )
                    ppo_flat = _ppo_stats_summary(stats)

                    if is_main and log_f is not None:
                        mean_reward_1, mean_reward_2 = self._global_reward_means_from_step(step_out)
                        rec: dict[str, Any] = {
                            "global_update_step": global_update,
                            "epoch_index": epoch,
                            "epoch_one_based": epoch + 1,
                            "batch_in_epoch": batch_idx,
                            "n_batches_in_epoch": dl_len,
                            "within_epoch_progress": (batch_idx + 1) / dl_len if dl_len else None,
                            "weight": self.reward_weight,
                            "mean_reward_1": mean_reward_1,
                            "mean_reward_2": mean_reward_2,
                            "mean_scalarized_reward": step_out["mean_scalarized"],
                            "mean_kl": mean_kl,
                            "ppo_stats": ppo_flat,
                        }
                        log_f.write(json.dumps(rec, default=str) + "\n")
                        log_f.flush()

                    if max_global_updates is not None and global_update >= int(max_global_updates):
                        stop_early = True
                        break

                    is_last = dl_len is not None and batch_idx == dl_len - 1
                    if is_main and self._should_save_mid_epoch(
                        global_update=global_update,
                        batch_idx=batch_idx,
                        dl_len=dl_len,
                        is_last_batch=is_last,
                    ):
                        mid_dir = run_dir / f"checkpoint_step_{global_update:07d}"
                        self._save_training_checkpoint(
                            model,
                            mid_dir,
                            meta={
                                "completed_epochs": epoch,
                                "num_epochs": num_epochs,
                                "epoch_index": epoch,
                                "next_batch_index_in_epoch": batch_idx + 1,
                                "n_batches_in_epoch": dl_len,
                                "within_epoch_progress": (batch_idx + 1) / dl_len if dl_len else None,
                                "global_update_step": global_update,
                                "weight": self.reward_weight,
                                "run_name": run_name,
                                "checkpoint_mid_epoch": True,
                            },
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
                                        "within_epoch_progress": (batch_idx + 1) / dl_len if dl_len else None,
                                        "weight": self.reward_weight,
                                    },
                                    default=str,
                                )
                                + "\n"
                            )
                            log_f.flush()

                if stop_early:
                    break

                ckpt_dir = run_dir / f"checkpoint_epoch_{epoch:04d}"
                self._save_training_checkpoint(
                    model,
                    ckpt_dir,
                    meta={
                        "completed_epochs": epoch + 1,
                        "num_epochs": num_epochs,
                        "epoch_index": epoch,
                        "next_batch_index_in_epoch": 0,
                        "n_batches_in_epoch": dl_len,
                        "within_epoch_progress": 1.0,
                        "global_update_step": global_update,
                        "weight": self.reward_weight,
                        "run_name": run_name,
                        "checkpoint_mid_epoch": False,
                    },
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
                                "weight": self.reward_weight,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                    log_f.flush()
                if hasattr(acc, "wait_for_everyone"):
                    acc.wait_for_everyone()

            final_dir = run_dir / "checkpoint_final"
            self._save_training_checkpoint(
                model,
                final_dir,
                meta={
                    "completed_epochs": num_epochs,
                    "num_epochs": num_epochs,
                    "epoch_index": num_epochs - 1,
                    "next_batch_index_in_epoch": 0,
                    "n_batches_in_epoch": dl_len,
                    "within_epoch_progress": 1.0,
                    "global_update_step": global_update,
                    "weight": self.reward_weight,
                    "run_name": run_name,
                    "final": True,
                    "checkpoint_mid_epoch": False,
                },
            )
            if is_main and log_f is not None:
                log_f.write(
                    json.dumps(
                        {
                            "record_type": "checkpoint",
                            "checkpoint_tag": "checkpoint_final",
                            "checkpoint_path": str(final_dir),
                            "global_update_step": global_update,
                            "epoch_index": num_epochs - 1,
                            "batch_in_epoch": 0 if dl_len is None else dl_len - 1,
                            "n_batches_in_epoch": dl_len,
                            "within_epoch_progress": 1.0,
                            "weight": self.reward_weight,
                        },
                        default=str,
                    )
                    + "\n"
                )
                log_f.flush()
            if hasattr(acc, "wait_for_everyone"):
                acc.wait_for_everyone()
        finally:
            if log_f is not None:
                log_f.close()

    def step_ppo(self, model: Any, batch: dict[str, list[Any]]) -> dict[str, Any]:
        query_tensors = batch["input_ids"]
        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        response_tensors: list[torch.Tensor] = []
        for query in query_tensors:
            gen_len = self.output_length_sampler()
            self.generation_kwargs["max_new_tokens"] = gen_len
            response = self.ppo_trainer.generate(query, **self.generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

        r1_vec = self._reward_vector_for_pipe(0, batch)
        r2_vec = self._reward_vector_for_pipe(1, batch) if len(self.reward_pipes) >= 2 else None
        r_vec = self.apply_reward(batch)
        rewards = [torch.tensor(float(r_vec[i])) for i in range(len(r_vec))]

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        try:
            self.ppo_trainer.log_stats(stats, batch, rewards)
        except Exception as exc:
            self.ppo_trainer.accelerator.print("log_stats:", exc)

        mean_r1 = float(np.mean(r1_vec))
        mean_r2 = float(np.mean(r2_vec)) if r2_vec is not None else None
        mean_scalar = float(np.mean(r_vec))
        return {
            "mean_r1": mean_r1,
            "mean_r2": mean_r2,
            "mean_scalarized": mean_scalar,
            "stats": stats,
        }
