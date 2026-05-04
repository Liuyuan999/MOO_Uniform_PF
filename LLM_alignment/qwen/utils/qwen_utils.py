"""Tokenizer, prompts, and reward pipelines for Qwen (replaces llama_utils for this project)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, Pipeline, pipeline as hf_pipeline


class Tokenizer:
    @staticmethod
    def load_tokenizer(
        tokenizer_name: str,
        *,
        cache_dir: str | None = None,
        trust_remote_code: bool = True,
    ) -> AutoTokenizer:
        tok = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok


class Pipelines:
    @staticmethod
    def load_pipes(reward_models: list[str], device: int | str | torch.device, cache_dir: str | None) -> list[Pipeline]:
        return [Pipelines.load_pipe(m, device, cache_dir) for m in reward_models]

    @staticmethod
    def load_pipe(reward_model: str, device: int | str | torch.device, cache_dir: str | None) -> Pipeline:
        print(f"Load reward model: {reward_model}")
        kw: dict[str, Any] = {"model": reward_model, "tokenizer": reward_model, "device": device}
        if cache_dir:
            kw["model_kwargs"] = {"cache_dir": cache_dir}
        return hf_pipeline("text-classification", **kw)


class Instructions:
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def get_prompt_summary(cls, post: str) -> str:
        return cls.get_prompt_input(cls.instruction_summary, post)

    @classmethod
    def get_prompt_input(cls, instruction: str, input_text: str) -> str:
        return f"### Instruction: {instruction} ### Input: {input_text} ### Response: "

    @staticmethod
    def get_input(query: str) -> str:
        after_input = ". ".join(query.split(Instructions.input_split)[1:]).replace("\n", " ").strip()
        return after_input.split(Instructions.response_split)[0]

    @staticmethod
    def get_response(full_text: str) -> str:
        return ". ".join(full_text.split(Instructions.response_split)[1:]).replace("\n", " ").strip()
