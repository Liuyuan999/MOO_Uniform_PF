"""TRL imports for Rewarded Soups–style PPO and TRL 0.29+ layout.

TRL 0.29+ no longer exports ``PPOConfig`` / ``PPOTrainer`` / ``AutoModelForCausalLMWithValueHead``
at the top level; they live under ``trl.experimental.ppo``. That trainer is a different API
(separate reward/value models, HuggingFace ``Trainer``-style loop) and is **not** a drop-in for
this codebase, which targets the legacy TRL PPO (``dataset=``, ``tokenizer=``, ``.step()``).

See ``LEGACY_PPO_TRAINER`` and ``TRL_PIN_MESSAGE`` for the supported install line.
"""

from __future__ import annotations

import random

import trl

TRL_VERSION = getattr(trl, "__version__", "unknown")

LEGACY_PPO_TRAINER = False
try:
    from trl import PPOTrainer as PPOTrainer

    LEGACY_PPO_TRAINER = True
except (ImportError, AttributeError):
    try:
        from trl.trainer.ppo_trainer import PPOTrainer as PPOTrainer

        LEGACY_PPO_TRAINER = True
    except (ImportError, AttributeError):
        from trl.experimental.ppo import PPOTrainer as PPOTrainer

if LEGACY_PPO_TRAINER:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig
else:
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead, PPOConfig

TRL_PIN_MESSAGE = (
    "This project uses the legacy TRL PPO API (Rewarded Soups–style: value-head policy, "
    "custom reward pipelines, ``PPOTrainer.step``). TRL 0.29+ only ships an experimental "
    "PPO trainer with a different contract. Pin TRL to the last line that still provides "
    'the legacy top-level imports, e.g. ``pip install "trl>=0.28,<0.29"`` '
    "(test against your transformers/accelerate versions)."
)

try:
    from trl.core import LengthSampler as LengthSampler
except ImportError:

    class LengthSampler:
        """Random output length in ``[low, high]`` (replacement for removed ``trl.core.LengthSampler``)."""

        def __init__(self, low: int, high: int) -> None:
            self.low = int(low)
            self.high = int(high)

        def __call__(self) -> int:
            return random.randint(self.low, self.high)


__all__ = [
    "AutoModelForCausalLMWithValueHead",
    "LEGACY_PPO_TRAINER",
    "LengthSampler",
    "PPOConfig",
    "PPOTrainer",
    "TRL_PIN_MESSAGE",
    "TRL_VERSION",
]
