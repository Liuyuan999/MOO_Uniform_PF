"""Inference, soup weight-averaging, and evaluation (adapted from RS inference_utils; Qwen)."""

from __future__ import annotations

import copy
import glob
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import AutoModelForCausalLM

from qwen.utils import args_utils
from qwen.utils.ppo_utils import _is_cogcomp_faithful_pipe, _normalize_id2label, get_score_from_output
from qwen.utils.trl_compat import LengthSampler


def format_lambda_key(coeff: float) -> str:
    """JSON key for a soup coefficient (e.g. 0.0 -> '0.0', 0.5 -> '0.5')."""
    c = float(coeff)
    s = f"{c:.10f}".rstrip("0").rstrip(".")
    if not s:
        return "0.0"
    if s == "-0":
        return "0.0"
    return s


def lambda_tag_for_path(coeff: float) -> str:
    """Filesystem-safe tag from lambda (e.g. 0.5 -> '0p5')."""
    s = f"{float(coeff):.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p").replace("-", "m")


def _logits_from_causal_forward(out: Any) -> torch.Tensor:
    """Support ``CausalLMOutput.logits`` and TRL value-head ``forward`` returning ``(logits, ...)``."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"Cannot read logits from model output type {type(out)!r}")


def mean_logratio_kl_on_response(
    policy: Any,
    ref: Any,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> float:
    """Mean (log π_policy(a|s) − log π_ref(a|s)) over generated response tokens (local PPO-style KL surrogate)."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    dev_p = next(policy.parameters()).device
    input_ids_p = input_ids.to(dev_p)
    dev_r = next(ref.parameters()).device
    input_ids_r = input_ids.to(dev_r)
    L = int(input_ids_p.size(1))
    if prompt_len >= L:
        return 0.0
    with torch.no_grad():
        pol_out = policy(input_ids=input_ids_p)
        ref_out = ref(input_ids=input_ids_r)
        pol_lp = F.log_softmax(_logits_from_causal_forward(pol_out), dim=-1)
        ref_lp = F.log_softmax(_logits_from_causal_forward(ref_out), dim=-1).to(pol_lp.device)
        terms: list[float] = []
        for pos in range(prompt_len, L):
            tid = int(input_ids_p[0, pos].item())
            terms.append(float((pol_lp[0, pos - 1, tid] - ref_lp[0, pos - 1, tid]).item()))
    return float(sum(terms) / len(terms)) if terms else 0.0


class Predictor:
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    def __init__(self, reward_pipes: list[Any], tokenizer: Any, output_max_length: int, device: Any):
        self.reward_pipes = reward_pipes
        self.tokenizer = tokenizer
        self.output_max_length = output_max_length
        self.device = device

    def get_rewards(self, texts: list[str]) -> Any:
        raise NotImplementedError

    @staticmethod
    def transform_reward(reward: list[Any]) -> list[dict[str, float]]:
        d_reward: list[dict[str, float]] = []
        for rew in reward:
            d: dict[str, float] = {}
            items = rew[0] if isinstance(rew, list) and rew and isinstance(rew[0], list) else rew
            if isinstance(items, list):
                for r in items:
                    if isinstance(r, dict) and "label" in r:
                        d[str(r["label"])] = float(r["score"])
            d_reward.append(d)
        return d_reward

    def average_rewards(self, rewards: list[list[dict[str, float]]]) -> list[dict[str, float]]:
        avg_reward: list[dict[str, float]] | None = None
        for reward in rewards:
            if avg_reward is None:
                avg_reward = copy.deepcopy(reward)
            else:
                for a_dict, r_dict in zip(avg_reward, reward):
                    for label in a_dict:
                        a_dict[label] = a_dict[label] + r_dict[label]
        assert avg_reward is not None
        n = len(rewards)
        for a_dict in avg_reward:
            for label in list(a_dict.keys()):
                if label == "n":
                    continue
                a_dict[label] = a_dict[label] / n
        return avg_reward

    def get_prediction_rewards(self, model: Any, query_tensors: list[Any]) -> tuple[list[str], Any, list[Any]]:
        texts: list[str] = []
        for i in range(len(query_tensors)):
            query_tensor = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(self.device)
            output = model.generate(
                input_ids=query_tensor,
                max_new_tokens=self.output_max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            ).squeeze()
            texts.append(self.tokenizer.decode(output, skip_special_tokens=True))

        rewards = self.get_rewards(texts)
        merged: list[dict[str, float]] = []
        for r in rewards:
            m: dict[str, float] = {}
            for d in r:
                m.update(d)
            merged.append(m)
        merged.append({"length": float(len(query_tensors))})
        return texts, rewards, merged

    def predict(self, model: Any, query_tensors: list[Any], verbose: bool = False) -> list[Any]:
        texts, rewards, avg_reward = self.get_prediction_rewards(model, query_tensors)
        for text, reward in zip(texts, rewards):
            print("=== text:", text.replace("\n", "[NEWLINE] "), reward)
            if not verbose:
                break
        return avg_reward


def evaluate_scalars_structured(
    predictor: Predictor,
    model: Any,
    query_tensors: list[Any],
    cfg: args_utils.RunConfig,
    *,
    include_kl: bool = False,
    ref_model: Any | None = None,
    beta: float | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Mean scalar rewards (aligned with training ``reward_formats``) and mean response length.

    If ``include_kl`` and ``ref_model`` are set, also reports mean sequence KL (policy vs base)
    and regularized objectives using ``beta`` (same role as ``init_kl_coef`` in PPO).

    If ``deterministic`` is True, uses greedy decoding with ``cfg.eval_max_new_tokens`` new tokens
    (for evaluation / CDF outer loop). Otherwise matches PPO rollout sampling and length sampler.
    """
    from qwen.tasks.summary import Instructions, transform_text_summary

    formats = list(cfg.reward_formats)
    n_rm = len(formats)
    if len(predictor.reward_pipes) != n_rm:
        raise ValueError(
            f"Number of reward pipes ({len(predictor.reward_pipes)}) must match reward_formats ({n_rm})."
        )
    if include_kl and ref_model is None:
        raise ValueError("include_kl requires ref_model (base causal LM).")
    b = float(beta if beta is not None else cfg.init_kl_coef)

    eos_id = getattr(predictor.tokenizer, "eos_token_id", None)
    length_sampler = LengthSampler(cfg.output_min_length, cfg.output_max_length)

    texts: list[str] = []
    kl_samples: list[float] = []
    for i in range(len(query_tensors)):
        query_tensor = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(predictor.device)
        prompt_len = int(query_tensor.size(1))
        if deterministic:
            gen_len = int(cfg.eval_max_new_tokens)
            gen_kw: dict[str, Any] = {
                "max_new_tokens": gen_len,
                "pad_token_id": predictor.tokenizer.pad_token_id,
                "eos_token_id": eos_id,
                "do_sample": False,
            }
        else:
            gen_len = int(length_sampler())
            gen_kw = {
                "max_new_tokens": gen_len,
                "min_length": -1,
                "pad_token_id": predictor.tokenizer.pad_token_id,
                "eos_token_id": eos_id,
                "do_sample": True,
                "top_k": 0,
                "top_p": 1.0,
            }
        with torch.inference_mode():
            gen_out = model.generate(input_ids=query_tensor, **gen_kw)
        out_ids = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        if out_ids.dim() == 2:
            full_ids = out_ids.squeeze(0)
        else:
            full_ids = out_ids.flatten()
        texts.append(predictor.tokenizer.decode(full_ids, skip_special_tokens=True))
        if include_kl and ref_model is not None:
            kl_samples.append(
                mean_logratio_kl_on_response(model, ref_model, full_ids, prompt_len)
            )

    per_model: list[list[float]] = [[] for _ in range(n_rm)]
    lengths: list[float] = []
    for text in texts:
        post = Instructions.get_input(text)
        resp = Instructions.get_response(text)
        lengths.append(float(len(predictor.tokenizer.encode(resp))))
        for j, pipe in enumerate(predictor.reward_pipes):
            pipe_kw: dict[str, Any] = {
                "top_k": None,
                "function_to_apply": "none",
                "batch_size": cfg.mini_batch_size,
            }
            if _is_cogcomp_faithful_pipe(pipe):
                pipe_kw["batch_size"] = 1
            tin = transform_text_summary(reward_pipe=pipe, post=post, response=resp)
            raw = pipe(tin, **pipe_kw)
            id2l = _normalize_id2label(getattr(pipe.model.config, "id2label", None))
            if not id2l:
                id2l = None
            fmt = formats[j]
            base_fmt = fmt.split("x")[0]
            s = get_score_from_output(raw, base_fmt, id2label=id2l)
            if "x" in fmt:
                s *= float(fmt.split("x")[1])
            per_model[j].append(s)

    n = len(texts)
    reward_models = {
        f"reward_model_{i + 1}": float(sum(per_model[i]) / n) if n else 0.0 for i in range(n_rm)
    }
    mean_len = float(sum(lengths) / n) if n else 0.0
    out: dict[str, Any] = {"reward_models": reward_models, "length": mean_len}

    if include_kl and kl_samples:
        kl_mean = float(sum(kl_samples) / len(kl_samples))
        out["kl_mean"] = kl_mean
        out["beta"] = b
        reg: dict[str, float] = {}
        plot_min: dict[str, float] = {}
        for i in range(n_rm):
            key = f"reward_model_{i + 1}"
            r = reward_models[key]
            reg[key] = r - b * kl_mean
            plot_min[f"neg_reg_{i + 1}"] = -r + b * kl_mean
        out["regularized_reward_models"] = reg
        out["plot_minimization"] = plot_min

    return out


def plot_soup_frontier(
    lambda_results: dict[str, Any],
    out_path: Path,
    *,
    title: str = "Soup evaluation: minimization axes",
    include_kl: bool = False,
) -> None:
    """Scatter for Pareto-style visualization: raw (−R1, −R2) or KL-adjusted ``plot_minimization``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping soup frontier plot.")
        return

    pts: list[tuple[float, float, str]] = []
    items = [(k, v) for k, v in lambda_results.items() if k != "meta"]
    for lam_key, payload in sorted(items, key=lambda kv: float(kv[0])):
        if not isinstance(payload, dict):
            continue
        if include_kl:
            pm = payload.get("plot_minimization")
            if isinstance(pm, dict) and "neg_reg_1" in pm and "neg_reg_2" in pm:
                pts.append((float(pm["neg_reg_1"]), float(pm["neg_reg_2"]), lam_key))
        else:
            rm = payload.get("reward_models")
            if isinstance(rm, dict) and "reward_model_1" in rm and "reward_model_2" in rm:
                pts.append((-float(rm["reward_model_1"]), -float(rm["reward_model_2"]), lam_key))

    if len(pts) < 1:
        print("Not enough reward pairs for frontier plot; skipping.")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(xs, ys, c=range(len(pts)), cmap="viridis", s=72, edgecolors="k", linewidths=0.4)
    for x, y, lab in pts:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if include_kl:
        ax.set_xlabel("−R1 + β·KL (minimization; β = init_kl_coef)")
        ax.set_ylabel("−R2 + β·KL")
        ax.set_title(title + "\n(KL-regularized objectives)", fontsize=10)
    else:
        ax.set_xlabel("−R1 (minimize; higher reward → more negative here)")
        ax.set_ylabel("−R2 (minimize; higher reward → more negative here)")
        ax.set_title(title + "\n(raw rewards, no KL)", fontsize=10)
    fig.colorbar(sc, ax=ax, label="sorted lambda order")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved soup frontier plot to", out_path)


class ResultsComputer:
    def __init__(
        self,
        cfg: args_utils.RunConfig,
        predictor: Predictor,
        base_model: Any,
        query_tensors: list[Any],
        verbose: bool,
        include_kl: bool = False,
        kl_ref_model: Any | None = None,
    ):
        self.cfg = cfg
        self.predictor = predictor
        self.base_model = base_model
        self.query_tensors = query_tensors
        self.verbose = verbose
        self.include_kl = include_kl
        self.kl_ref_model = kl_ref_model

    def eval_weighted_model(self, model: Any) -> dict[str, Any]:
        ref = self.kl_ref_model if self.kl_ref_model is not None else self.base_model
        return evaluate_scalars_structured(
            self.predictor,
            model,
            self.query_tensors,
            self.cfg,
            include_kl=self.include_kl,
            ref_model=ref if self.include_kl else None,
            beta=float(self.cfg.init_kl_coef),
        )

    def single(self, peft_name: str) -> Any:
        print("Single adapter")
        wa = Loader.load_peft_model(self.base_model, peft_name)
        if torch.cuda.is_available():
            wa = wa.to("cuda")
        wa.eval()
        structured = self.eval_weighted_model(wa)
        if self.verbose:
            self.predictor.predict(wa, self.query_tensors[:1], verbose=True)
        print("== single", structured, "\n")
        del wa
        torch.cuda.empty_cache()
        return structured

    def interpolation(self, peft_names: list[str], num_lambdas: int, save_soups_dir: Path | None = None) -> dict[str, Any]:
        print("Interpolation")
        peft_names = [get_last_epoch(p) for p in peft_names]
        coeffs = [x / max(num_lambdas - 1, 1) for x in range(num_lambdas)]
        out: dict[str, Any] = {}
        for coeff in coeffs:
            lam_key = format_lambda_key(coeff)
            save_path: Path | None = None
            if save_soups_dir is not None:
                save_path = Path(save_soups_dir) / f"soup_lambda_{lambda_tag_for_path(coeff)}"
            out[lam_key] = self.create_and_call_wa(
                peft_names,
                [1.0 - coeff, coeff],
                name=lam_key,
                save_dir=save_path,
            )
        return out

    def create_and_call_wa(
        self,
        peft_names: list[str],
        coefficients: list[float],
        name: str | None = None,
        save_dir: Path | None = None,
    ) -> Any:
        wa = WeightAverager.build_wa(self.cfg, peft_names, coefficients)
        wa.eval()
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            wa.save_pretrained(str(save_dir))
            print("Saved souped PEFT adapter to", save_dir)
        structured = self.eval_weighted_model(wa)
        print("==", name or "wa", structured, "\n")
        del wa
        torch.cuda.empty_cache()
        return structured


LOAD_ONLY_LORA = True


class WeightAverager:
    @staticmethod
    def average_weights(
        cfg: args_utils.RunConfig,
        peft_names: list[str],
        coefficients: list[float],
    ) -> OrderedDict[str, torch.Tensor]:
        weights_averaged: OrderedDict[str, torch.Tensor] | None = None
        for peft_name, coefficient in zip(peft_names, coefficients):
            if coefficient == 0.0:
                continue
            base = Loader.load_base_model(cfg)
            current = Loader.load_peft_model(base, peft_name)
            current_weights = get_peft_model_state_dict(current, state_dict=None)
            if weights_averaged is None:
                weights_averaged = OrderedDict()
                for key in current_weights:
                    weights_averaged[key] = coefficient * current_weights[key]
            else:
                for key in current_weights:
                    weights_averaged[key] = weights_averaged[key] + coefficient * current_weights[key]
            del current
            del base
            torch.cuda.empty_cache()
        if weights_averaged is None:
            raise ValueError("No weights averaged")
        return weights_averaged

    @staticmethod
    def build_wa(cfg: args_utils.RunConfig, peft_names: list[str], coefficients: list[float]) -> PeftModel:
        weights_averaged = WeightAverager.average_weights(cfg, peft_names, coefficients)
        base = Loader.load_base_model(cfg)
        wa = Loader.load_peft_model(base, peft_names[0])
        wa.load_state_dict(weights_averaged, strict=not LOAD_ONLY_LORA)
        # Move to GPU if available; base models load on CPU when device_map is unset.
        import torch as _torch
        if _torch.cuda.is_available():
            wa = wa.to("cuda")
        return wa


class Loader:
    @staticmethod
    def load_base_model(cfg: args_utils.RunConfig) -> AutoModelForCausalLM:
        from qwen.utils.ppo_utils import resolve_torch_dtype

        return AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.hf_cache,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=resolve_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
        )

    @staticmethod
    def load_peft_model(base_model: AutoModelForCausalLM, peft_name: str) -> PeftModel:
        return PeftModel.from_pretrained(
            base_model,
            peft_name,
            local_files_only=args_utils.LOCAL_FILES_ONLY,
        )


def get_last_epoch(peft_name: str) -> str:
    if os.path.split(peft_name)[-1].startswith("epoch"):
        return peft_name
    list_folder = os.listdir(peft_name)
    dict_epochs = {int(path.split("epoch")[1]): path for path in list_folder if "epoch" in path}
    if not dict_epochs:
        return peft_name
    last_epoch = os.path.join(peft_name, dict_epochs[max(dict_epochs.keys())])
    print("detected", last_epoch)
    return last_epoch


def get_results_rewards(
    resultscomputer: ResultsComputer,
    peft_names: list[str],
    num_lambdas: int,
    *,
    save_soups_dir: str | Path | None = None,
) -> dict[str, Any]:
    save_path = Path(save_soups_dir).resolve() if save_soups_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    if len(peft_names) == 2:
        peft_names = [get_last_epoch(p) for p in peft_names]
        return resultscomputer.interpolation(peft_names, num_lambdas, save_soups_dir=save_path)
    if len(peft_names) == 1:
        return {format_lambda_key(0.0): resultscomputer.single(peft_names[0])}
    raise ValueError("Use one or two peft paths for this milestone")
