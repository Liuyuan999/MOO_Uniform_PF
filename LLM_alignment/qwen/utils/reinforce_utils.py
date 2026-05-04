"""REINFORCE training utilities for Qwen summarization with reward scalarization.

KL regularization design
------------------------
KL divergence is included in the *reward signal*, not as a separate loss term.
Per-sample total reward is:

    R_total_i = (1 - w) * r1_i + w * r2_i  -  beta * KL_seq_i

where the sampled sequence-level KL surrogate is:

    KL_seq_i = sum_t [ log pi_theta(y_t | x, y_<t)  -  log pi_ref(y_t | x, y_<t) ]

This matches the conceptual RL objective in the PPO path while keeping the REINFORCE
gradient purely through the log-probability score multiplied by stop-gradient advantages.

Baseline
--------
Variance reduction uses the batch-mean total reward as a simple but unbiased baseline:

    advantage_i = R_total_i  -  mean_j(R_total_j)

The REINFORCE loss is then:

    loss = -E_i [ stop_grad(advantage_i) * policy_log_probs_i ]

Model loading
-------------
REINFORCE uses plain ``AutoModelForCausalLM + PEFT`` (no TRL value-head wrapper) because
no value head is needed.  Adapter checkpoints are saved in PEFT-native format and are
therefore compatible with PPO checkpoints (which also save the PEFT inner model).
"""

from __future__ import annotations

import contextlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from qwen.utils.args_utils import Naming, RunConfig
from qwen.utils.ppo_utils import (
    _is_cogcomp_faithful_pipe,
    _normalize_id2label,
    build_lora_config,
    get_score_from_output,
    resolve_torch_dtype,
)
from qwen.utils.qwen_utils import Instructions, Pipelines


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_policy_model(cfg: RunConfig, device: torch.device) -> PeftModel:
    """Fresh PEFT policy model (no value head).

    Avoids ``device_map='auto'``; loads on CPU then moves to ``device``
    so the caller controls device placement explicitly.
    """
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=resolve_torch_dtype(cfg.dtype),
    )
    model = get_peft_model(base, build_lora_config(cfg))
    # Required so that PEFT computes input-embedding gradients during backprop.
    model.enable_input_require_grads()
    return model.to(device)


def load_policy_from_adapter(
    cfg: RunConfig, adapter_dir: Path, device: torch.device
) -> PeftModel:
    """Load a trainable PEFT policy from a saved adapter directory.

    The adapter directory must contain ``adapter_config.json`` and weight files.
    Compatible with adapters saved by both REINFORCE and PPO checkpoints (both
    write the PEFT inner model via ``PeftModel.save_pretrained``).
    """
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=resolve_torch_dtype(cfg.dtype),
    )
    model = PeftModel.from_pretrained(
        base, str(adapter_dir.resolve()), is_trainable=True
    )
    model.enable_input_require_grads()
    return model.to(device)


def load_ref_model(cfg: RunConfig, device: torch.device) -> AutoModelForCausalLM:
    """Frozen reference model (no PEFT).  Used only for KL log-prob computation."""
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=resolve_torch_dtype(cfg.dtype),
    )
    ref.eval()
    ref.requires_grad_(False)
    return ref.to(device)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def run_name_reinforce(cfg: RunConfig, weight: float) -> str:
    """Unique run name for REINFORCE; uses 'rf' tag to distinguish from PPO runs."""
    suffix = Naming.weight_tag(weight)
    ts = datetime.now().strftime("%m%d-%H%M%S")
    return f"{cfg.model_name.split('/')[-1]}-rf-summary-{suffix}-{ts}"[:92]


# ---------------------------------------------------------------------------
# Sequence log-probability computation (core KL math)
# ---------------------------------------------------------------------------

def compute_sequence_log_probs(
    model: torch.nn.Module,
    queries: list[torch.Tensor],
    responses: list[torch.Tensor],
    *,
    device: torch.device,
    no_grad: bool = False,
) -> torch.Tensor:
    """Return the summed log-prob of each response under ``model``.

    Processes each sample individually to cleanly handle variable-length sequences.

    The sampled sequence-level KL surrogate for sample i is:

        KL_seq_i = policy_log_probs_i  -  ref_log_probs_i

    where both terms are obtained by calling this function on the respective model.

    Parameters
    ----------
    model:
        Policy (with gradients) or reference (frozen).
    queries:
        List of 1-D token tensors for the prompt portion.
    responses:
        List of 1-D token tensors for the generated response portion.
    device:
        Target device; tensors are moved here before the forward pass.
    no_grad:
        If True, wraps the forward pass in ``torch.no_grad()``.

    Returns
    -------
    Tensor of shape [batch_size] with summed log-probs (float32).
    """
    ctx: Any = torch.no_grad() if no_grad else contextlib.nullcontext()
    results: list[torch.Tensor] = []
    with ctx:
        for q, r in zip(queries, responses):
            q_dev = q.to(device)
            r_dev = r.to(device)
            if len(r_dev) == 0:
                results.append(torch.tensor(0.0, device=device))
                continue
            # Concatenate prompt + response into a single sequence for one forward pass.
            full = torch.cat([q_dev, r_dev]).unsqueeze(0)  # [1, q_len + r_len]
            out = model(input_ids=full)
            # logits[0, t] predicts token t+1.
            # Response tokens sit at positions [q_len .. q_len+r_len-1] in ``full``.
            # In the shifted logits that means positions [q_len-1 .. q_len+r_len-2].
            logits = out.logits[0, len(q_dev) - 1 : len(q_dev) - 1 + len(r_dev), :]
            lp = F.log_softmax(logits, dim=-1)                      # [r_len, vocab]
            token_lp = lp[torch.arange(len(r_dev), device=device), r_dev]  # [r_len]
            results.append(token_lp.sum())
    return torch.stack(results)  # [batch_size]


# ---------------------------------------------------------------------------
# Reward computation (mirrors ppo_utils.Runner._reward_vector_for_pipe)
# ---------------------------------------------------------------------------

def reward_vector_for_pipe(
    pipe_idx: int,
    batch: dict[str, list[Any]],
    reward_pipes: list[Any],
    reward_formats: list[str],
    transform_text_summary: Any,
    mini_batch_size: int,
) -> np.ndarray:
    """Compute reward scores from pipeline ``pipe_idx`` for the batch.

    Mirrors ``ppo_utils.Runner._reward_vector_for_pipe`` without depending on
    PPOTrainer / Runner.
    """
    reward_pipe = reward_pipes[pipe_idx]
    fmt = reward_formats[pipe_idx].split("x")[0]
    texts = [
        transform_text_summary(
            reward_pipe=reward_pipe,
            post=Instructions.get_input(query),
            response=Instructions.get_response(query) + response,
        )
        for query, response in zip(batch["query"], batch["response"])
    ]
    pipe_kw: dict[str, Any] = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": mini_batch_size,
    }
    if _is_cogcomp_faithful_pipe(reward_pipe):
        pipe_kw["batch_size"] = 1
    outs = reward_pipe(texts, **pipe_kw)
    id2l = _normalize_id2label(getattr(reward_pipe.model.config, "id2label", None)) or None
    rewards = [get_score_from_output(o, fmt, id2label=id2l) for o in outs]
    if "x" in reward_formats[pipe_idx]:
        c = float(reward_formats[pipe_idx].split("x")[1])
        rewards = [c * v for v in rewards]
    return np.array(rewards, dtype=np.float64)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    meta: dict[str, Any],
    *,
    is_main: bool = True,
) -> None:
    """Save PEFT adapter weights, optimizer state, and ``meta.json``."""
    if not is_main:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    with (checkpoint_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved REINFORCE checkpoint to", checkpoint_dir)


# ---------------------------------------------------------------------------
# Mid-epoch checkpoint cadence (mirrors ppo_utils.Runner logic)
# ---------------------------------------------------------------------------

def mid_epoch_interval_batches(cfg: RunConfig, dl_len: int | None) -> int | None:
    """Batches between mid-epoch saves when ``save_per_epoch_fraction`` is set."""
    f = cfg.save_per_epoch_fraction
    if not f or not dl_len or dl_len <= 0:
        return None
    return max(1, math.ceil(dl_len / f))


def should_save_mid_epoch(
    cfg: RunConfig,
    *,
    global_update: int,
    batch_idx: int,
    dl_len: int | None,
    is_last_batch: bool,
) -> bool:
    """True if a mid-epoch checkpoint should be saved after this batch."""
    if is_last_batch:
        return False
    if cfg.save_every_n_updates:
        return global_update > 0 and global_update % int(cfg.save_every_n_updates) == 0
    interval = mid_epoch_interval_batches(cfg, dl_len)
    if interval is None:
        return False
    return (batch_idx + 1) % interval == 0
