# `qwen/` — Package Overview

This package implements multi-objective PPO fine-tuning of Qwen language models for the Reddit summarization task, together with the SURF/CDF-refinement outer loop and all supporting evaluation and analysis utilities.

---

## Top-level scripts

| File | Purpose |
|---|---|
| `train_ppo.py` | **Inner-loop PPO trainer.** Loads a Qwen model with a PEFT/LoRA adapter, runs PPO for a fixed budget (epochs or steps) under a single scalarization weight, saves `checkpoint_final/`. Called by the outer loops as a subprocess. |
| `train_reinforce.py` | Alternative inner-loop trainer using REINFORCE instead of PPO. Same interface as `train_ppo.py`; kept for ablations. |
| `cdf_refinement.py` | **SURF/CDF outer loop.** Iteratively updates the arc-length CDF, inverts it to obtain scalarization weights, launches inner PPO jobs (sequential or parallel GPU queue), estimates PF points from training logs, and writes `pf_history.json`, `metric_history.json`, and `cdf_history.json` under `outputs/cdf_refinement/<run>/`. |
| `equispace_weighting.py` | **LS (equispaced) outer loop.** Same structure and output format as `cdf_refinement.py` but with fixed uniform weights — no CDF update. Used to produce the LS baseline. |
| `evaluation.py` | **Standalone parallel adapter evaluator.** Accepts one or many `outer_iter_*` directories (or explicit checkpoint paths), evaluates each `adapter_*/checkpoint_final` using the same reward+KL procedure as training, writes `logs/training_metrics.jsonl` per adapter, `point_meta.json` per outer dir, and run-level `pf_history.json` / `metric_history.json`. Supports resume (appends missing batches), backup-on-overwrite, and in-place or mirrored output modes. |
| `soup_endpoints_and_eval.py` | **Endpoint souping + lightweight evaluation.** For each outer iteration of an LS run, copies `adapter_0` and `adapter_N` directly, interpolates middle adapters by parameter averaging (model souping), evaluates each souped adapter on a small fixed subset of training data, and writes compatible `training_metrics.jsonl` logs. Drives a shared per-outer-iteration progress bar. |
| `soup.py` | Minimal CLI wrapper for two-endpoint Rewarded Soups interpolation (original RS-style; single outer evaluation call). |
| `inference_rewardedsoups.py` | CLI for running inference/evaluation in the original Rewarded Soups style. Produces JSON result files. |
| `calculate_pf_history_n_metrics.py` | **Post-hoc metric script.** Reads `training_metrics.jsonl` logs for all outer iterations of a run, estimates KL-penalized PF objectives by tail-averaging, and rewrites `pf_history.json`, `metric_history.json`, and `checkpoint_mapping.json`. Configurable via constants at the top of the file. |
| `recompute_pf_from_training_logs.py` | One-off script to recompute PF histories for existing runs using the training-log-based, KL-penalized objective convention. Handles the case where logs live under `logs/training_metrics.jsonl` rather than directly in the adapter directory. |

---

## `tasks/`

| File | Purpose |
|---|---|
| `tasks/summary.py` | Dataset builder (`build_dataset`) and task-specific predictor (`PredictorSummary`) for the Reddit summarization task (`openai/summarize_from_feedback`). Handles tokenization, prompt formatting, and reward scoring for both reward models. |

---

## `utils/`

| File | Purpose |
|---|---|
| `utils/args_utils.py` | Config loading and merging (YAML stack → `RunConfig` dataclass), CLI argument parsers for PPO / inference / soup, seed setting, and run-naming helpers. Central place for all configuration handling. |
| `utils/ppo_utils.py` | Low-level PPO machinery: LoRA model builder (`build_lora_config`), `Loader` (base model + PEFT loading), `Runner` (PPO training loop with TRL), reward scoring helpers, and `training_metrics.jsonl` logging. |
| `utils/inference_utils.py` | Inference and evaluation utilities: `evaluate_scalars_structured` (rewards + KL from policy vs reference), `mean_logratio_kl_on_response` (token-level KL surrogate), `WeightAverager` (parameter interpolation / souping), and `Loader` (model loading wrappers). |
| `utils/cdf_utils.py` | **Model-agnostic CDF math.** Pure NumPy functions for the SURF algorithm: uniform grid construction, CDF inversion, monotonicity enforcement, PCHIP surrogate CDF from PF polyline, CDF blending, segment-length computation, CV, and gap ratio. No model or torch dependencies. |
| `utils/qwen_utils.py` | Qwen-specific utilities: tokenizer loader (`Tokenizer`), prompt/response parsing (`Instructions`), reward pipeline loader (`Pipelines`). Replaces the `llama_utils` module from the original codebase. |
| `utils/reinforce_utils.py` | REINFORCE-specific training helpers (analogous to `ppo_utils.py` for the REINFORCE trainer). |
| `utils/trl_compat.py` | Thin compatibility shim for TRL version differences (e.g. `LengthSampler`). |

---

## Output layout

```
outputs/
  cdf_refinement/<run_name>/        # SURF/CDF outer loop outputs
    outer_iter_k/
      adapter_n/
        checkpoint_final/adapter/   # PEFT adapter weights
        logs/training_metrics.jsonl # per-batch reward+KL logs
      point_meta.json               # PF point estimates for this outer iteration
    pf_history.json
    metric_history.json
    cdf_history.json
    checkpoint_mapping.json
    ppo_merged_stack.yaml           # merged config snapshot

  ls_uniform/<run_name>/            # LS equispaced outer loop (same structure)

  souping/<run_name>/               # Souped adapters (same structure, no PPO)

  analysis/                         # Comparison plots and metric tables
```

---

## Typical workflow

1. **Warm-start**: run `train_ppo.py` at `w=0.5` for several epochs to obtain a shared initialization adapter.
2. **LS baseline**: run `equispace_weighting.py` with fixed weights `{0, 0.2, 0.4, 0.6, 0.8, 1.0}`.
3. **SURF/CDF refinement**: run `cdf_refinement.py`, optionally reusing LS endpoints via `endpoint_reuse`.
4. **Souping baseline**: run `soup_endpoints_and_eval.py` to produce interpolated adapters from LS endpoints.
5. **Evaluation**: run `evaluation.py` on any set of outer dirs to get consistent reward/KL estimates.
6. **Metrics**: run `calculate_pf_history_n_metrics.py` or the inline analysis commands to compute CV, Gap Ratio, and Hypervolume.
