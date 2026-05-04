"""Summarization dataset and predictors (adapted from RS tasks/summary.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from qwen.utils.trl_compat import LengthSampler

from qwen.utils import inference_utils
from qwen.utils.qwen_utils import Instructions

MIN_SIZE = 100
MAX_SIZE_NEWS = 1500


def build_dataset(dataset_name: str, tokenizer: Any, split: str = "train", max_train_samples: int | None = None):
    if dataset_name == "news":
        ds = _build_news_dataset(tokenizer=tokenizer, split=split)
    else:
        ds = _build_openai_dataset(tokenizer=tokenizer, split=split)
    if max_train_samples is not None:
        n = min(max_train_samples, len(ds))
        ds = ds.select(range(n))
    print(f"Loaded dataset {dataset_name}:", ds)
    return ds


def _build_news_dataset(tokenizer: Any, split: str = "train"):
    split_use = {"train": "test", "validation": "train"}.get(split, split)
    ds = load_dataset("argilla/news-summary", name="comparisons", split=split_use)
    ds_filtered = ds.filter(
        lambda x: x["text"] is not None and MIN_SIZE < len(x["text"]) < MAX_SIZE_NEWS and x["id"] is not None,
        batched=False,
    )

    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x["id"]})
        _, unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        return duplicated_dataset.select(unique_indices.tolist())

    ds_deduplicated = remove_duplicate(ds_filtered)
    input_size_sampler = LengthSampler(2, 8)

    def tokenize(sample):
        info_post = "-".join(sample["text"].replace("\n", " ").split("(Reuters) -")[1:]).strip()
        prompt_summary = Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = 0
        response = sample["prediction"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds_deduplicated.map(tokenize, batched=False, load_from_cache_file=False)
    ds_mapped.set_format(type="torch")
    return ds_mapped


def _build_openai_dataset(tokenizer: Any, split: str = "train", max_size: int = 1200):
    ds = load_dataset("openai/summarize_from_feedback", name="comparisons", split=split)
    ds = ds.filter(
        lambda x: x["info"]["post"] is not None
        and MIN_SIZE < len(x["info"]["post"]) < max_size
        and x["info"]["id"] is not None,
        batched=False,
    )

    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x["info"]["id"]})
        _, unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        return duplicated_dataset.select(unique_indices.tolist())

    ds = remove_duplicate(ds)
    input_size_sampler = LengthSampler(2, 8)

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = sample["choice"]
        response = sample["summaries"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds


def transform_text_summary(reward_pipe: Any, post: str, response: str) -> str | dict[str, str]:
    response = response.split(".")[0] + "."
    name = reward_pipe.model.name_or_path
    if name.startswith("CogComp/bart-faithful-summary-detector"):
        # BART seq. classification pools on <eos> positions; manual eos concatenation yields a variable
        # eos-token count per example when batched → ValueError. Use tokenizer pair encoding instead.
        return {"text": response.strip(), "text_pair": post.strip()}
    if name.startswith("Tristan/gpt2_reward_summarization"):
        return response + " " + reward_pipe.tokenizer.bos_token + " " + post
    raise ValueError(f"Unsupported reward model for summary transform: {name}")


class PredictorSummary(inference_utils.Predictor):
    def get_rewards(self, texts: list[str]) -> Any:
        queries_responses = [
            (Instructions.get_input(text), Instructions.get_response(text)) for text in texts
        ]
        rewards = [
            [
                reward_pipe(
                    transform_text_summary(reward_pipe=reward_pipe, post=query, response=response),
                    **self.sent_kwargs,
                )
                for reward_pipe in self.reward_pipes
            ]
            for query, response in queries_responses
        ]
        return [self.transform_reward(r) for r in rewards]


class Samples:
    @staticmethod
    def get_fake_samples(bs: int, tokenizer: Any) -> list[Any]:
        list_posts = [
            "Zinedine Yazid Zidane popularly known as Zizou, is a French professional football manager.",
            "Thierry Daniel Henry is a French professional football coach and former player.",
            "Pablo Escobar was a Colombian drug lord.",
        ]
        list_responses = [
            "Zinedine Zidane is a footballer",
            "Thierry Henry is a footballer",
            "The mafia is",
        ]
        list_texts = [
            Instructions.get_prompt_summary(post=post) + response for post, response in zip(list_posts, list_responses)
        ]
        return [np.array(tokenizer.encode(text), dtype=np.int32)[:-1] for text in list_texts][:bs]

    @staticmethod
    def get_samples(
        dataset_name: str,
        tokenizer: Any,
        bs: int = 16,
        split: str = "validation",
        max_train_samples: int | None = None,
    ) -> list[Any]:
        """Load ``split`` then take the first ``bs`` rows (after optional ``max_train_samples`` cap).

        ``max_train_samples`` is passed to ``build_dataset`` (same as PPO training pool cap).  Default
        ``None`` uses the full filtered split before taking ``bs`` prompts — avoid ``max_train_samples=bs``,
        which incorrectly capped the underlying dataset to only ``bs`` rows total.
        """
        ds = build_dataset(
            dataset_name=dataset_name, tokenizer=tokenizer, split=split, max_train_samples=max_train_samples
        )
        n = min(bs, len(ds))
        return [ds[i]["input_ids"] for i in range(n)]
