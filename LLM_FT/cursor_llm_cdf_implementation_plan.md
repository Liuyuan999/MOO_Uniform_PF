# Cursor Implementation Plan: LLM RLFT + CDF Refinement

## Goal
Implement a small, reliable benchmark for **arc-length-uniform Pareto front sampling** in **LLM fine-tuning via RL**, using a **policy-space scalarized reward objective with KL-to-reference regularization** and comparing:

1. **Uniform scalarization weights**
2. **Random scalarization weights**
3. **Rewarded Soup** baseline
4. **CDF refinement** (our method)

The first target should be a **small text-only instruct model** with **LoRA**. Keep the design modular so DoRA or head-only FT can be added later.

---

## Recommended first scope

### Model
Start with one small instruct model only.

Recommended order:
1. `Qwen2.5-0.5B-Instruct` or similar ~0.5B model
2. Then scale to ~1B or ~3B only after the pipeline is stable

### PEFT mode
Start with **LoRA** only.

Do **not** start with multiple PEFT methods at once. Make PEFT pluggable, but implement and validate LoRA first.

### RL trainer
Use one trainer only in the first version. Keep it simple.

Recommended progression:
1. **REINFORCE with sampled completions + KL penalty**
2. Then optionally move to PPO/GRPO if needed

Reason: CDF refinement already adds enough moving pieces; the first objective is a correct and inspectable benchmark, not the most sophisticated RL stack.

### Number of objectives
Start with **2 objectives only**.

The entire CDF refinement algorithm, arc-length metrics, and visualizations are clearest in the bi-objective case.

---

## Core mathematical object
For prompt distribution `x ~ D`, response `y ~ π_θ(.|x)`, two rewards `r1(x,y), r2(x,y)`, reference policy `π_ref`, scalarization weight `w ∈ [0,1]`, and KL coefficient `β > 0`, define:

\[
J_w(\theta)
:=
\mathbb E_{x\sim D,\ y\sim \pi_\theta(\cdot|x)}
\left[
(1-w) r_1(x,y) + w r_2(x,y)
- \beta\, \mathrm{KL}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))
\right].
\]

For implementation, use the standard sampled sequence-level KL surrogate:

\[
\widehat{\mathrm{KL}}(x,y)
:=
\sum_{t=1}^T
\left(
\log \pi_\theta(y_t\mid x,y_{<t})
-
\log \pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
\right),
\]

and define the sampled scalarized return:

\[
R_w(x,y)
:=
(1-w) r_1(x,y) + w r_2(x,y) - \beta \, \widehat{\mathrm{KL}}(x,y).
\]

Then train one policy per weight `w`.

---

## High-level stages

### Stage 0: pick one clean benchmark
Pick a benchmark where both rewards are cheap and deterministic to compute.

Good first options:
- **Summarization**: reward 1 = faithfulness, reward 2 = brevity / fluency / style
- **Instruction following**: reward 1 = task correctness proxy, reward 2 = length or format adherence
- **Simple QA**: reward 1 = exact / soft correctness, reward 2 = concise answer reward

Do not begin with expensive judge-model-heavy evaluation for every reward.

### Stage 1: build a minimal RLFT pipeline
Implement:
- dataset loader
- prompt formatting
- generation
- reward computation
- reference logprob computation
- REINFORCE objective with KL penalty
- LoRA finetuning
- checkpoint save/load

Acceptance check:
- can train endpoint experts for `w=0` and `w=1`
- rewards move in expected directions
- KL remains finite and monitored

### Stage 2: implement scalarized runs
Implement a function:
- `train_scalarized_policy(weight, init_ckpt=None, config=...) -> checkpoint + metrics`

This should produce one model for a given `w`.

Acceptance check:
- interpolating between endpoint weights in objective space should show a trade-off
- endpoint specialists differ in objective values

### Stage 3: implement evaluation in objective space
For each trained checkpoint, evaluate the **true bi-objective coordinates**:

\[
f(\theta) = (f_1(\theta), f_2(\theta))
\]

where `f1, f2` are the evaluation metrics without scalarization and without training-time KL penalty.

Important:
- training objective uses reward + KL regularization
- PF plotting should usually use **task objectives only** unless explicitly studying regularized objectives
- keep both options available in code

Acceptance check:
- every checkpoint gets a 2D point
- plots are reproducible from saved JSON/CSV results

### Stage 4: implement baselines
Implement:
- **Uniform weights**: `w_n = n / N`
- **Random weights**: `w_n ~ Uniform[0,1]`, then sort
- **Rewarded Soup**: interpolate LoRA parameters between endpoint experts
- optional: scalarization gap heuristic later

For Rewarded Soup in the bi-objective case:

\[
\theta_{\text{RS}}(\lambda) = (1-\lambda)\theta^{(0)} + \lambda\theta^{(1)}
\]

For LoRA, average only trainable adapter parameters, not the full frozen base model weights.

Acceptance check:
- can evaluate a dense `lambda` grid for rewarded soup without retraining
- objective-space curve is produced

### Stage 5: implement CDF refinement
Implement iterative CDF refinement over scalarization weights.

Use:
- `num_segments = N`
- `num_points = N + 1`
- quantiles `q_n = n / N`, `n=0,...,N`

At outer iteration `t`:
1. compute weights from inverse CDF
2. train or continue training each scalarized policy from previous checkpoint
3. evaluate objective-space points
4. build polyline cumulative arc-length
5. interpolate cumulative arc-length over `[0,1]`
6. normalize to a CDF surrogate
7. blend with previous CDF

Acceptance check:
- `Err_arc_tilde` decreases over outer iterations on at least one clean benchmark
- point spacing visibly improves against uniform weights

---

## Recommended repository structure

```text
project/
  configs/
    base.yaml
    model/
    task/
    train/
  data/
  src/
    data/
      datasets.py
      prompts.py
    model/
      load_model.py
      peft.py
      generation.py
      logprobs.py
    rewards/
      reward_base.py
      reward_task.py
      reward_length.py
      reward_faithfulness.py
      combine.py
    rl/
      reinforce_trainer.py
      rollout.py
      losses.py
      checkpointing.py
    scalarization/
      train_scalarized.py
      soup.py
    cdf/
      refinement.py
      interpolation.py
      metrics.py
    eval/
      evaluate_policy.py
      front_metrics.py
      plotting.py
    utils/
      seed.py
      io.py
      device.py
  scripts/
    train_endpoint.py
    train_scalarized_grid.py
    run_rewarded_soup.py
    run_cdf_refinement.py
    evaluate_all.py
  outputs/
```

---

## Concrete implementation order for Cursor

### Task 1: scaffold configs and interfaces
Have Cursor define typed interfaces first.

Needed objects:
- `TrainConfig`
- `ModelConfig`
- `TaskConfig`
- `RewardConfig`
- `CDFConfig`
- `EvalResult`
- `FrontResult`

### Task 2: implement reward interface
Required API:

```python
class RewardFn(Protocol):
    def __call__(self, prompts: list[str], responses: list[str], **kwargs) -> np.ndarray:
        ...
```

Then implement:
- `Reward1`
- `Reward2`
- `ScalarizedReward`

with

\[
r_w = (1-w) r_1 + w r_2.
\]

### Task 3: implement KL computation
Required API:

```python
def sequence_logprob(model, prompt_ids, completion_ids) -> torch.Tensor:
    ...

def sampled_kl(current_model, ref_model, prompt_ids, completion_ids) -> torch.Tensor:
    ...
```

Return per-sample sequence KL.

### Task 4: implement REINFORCE loss
Given sampled scalarized return `R_w(x,y)`, compute advantage with a simple batch baseline:

\[
A_i = R_i - \frac{1}{B}\sum_{j=1}^B R_j.
\]

Loss:

\[
\mathcal L(\theta) = - \frac1B \sum_{i=1}^B A_i \log \pi_\theta(y^{(i)}|x^{(i)}).
\]

where the reward already includes the KL penalty. Keep this version first.

### Task 5: endpoint experts
Train:
- `w=0`
- `w=1`

These are needed for:
- front sanity check
- Rewarded Soup
- warm starts

### Task 6: scalarized training API
Implement:

```python
def train_scalarized_policy(weight: float, init_checkpoint: str | None, cfg: TrainConfig) -> TrainRunResult:
    ...
```

This is the central callable used by uniform/random/CDF schedules.

### Task 7: evaluation API
Implement:

```python
def evaluate_objectives(checkpoint: str, eval_cfg: EvalConfig) -> dict:
    return {
        "f1": ...,
        "f2": ...,
        "regularized_f1": ...,
        "regularized_f2": ...,
    }
```

### Task 8: front metrics
Implement:
- surrogate cumulative arc-length
- `Err_arc_tilde`
- `CV`
- `GapRatio`
- optional `IGD`

### Task 9: Rewarded Soup baseline
Implement LoRA parameter interpolation between two checkpoints.

### Task 10: CDF refinement outer loop
Implement saveable outer-loop state:
- `F_grid`
- sampled weights
- checkpoint paths
- objective history
- metric history

---

## Warm-start policy for CDF refinement
For outer iteration `t`, point `n`, initialize from previous outer iteration checkpoint at the same index:

\[
\theta_n^{(t)} \leftarrow \theta_n^{(t-1)}.
\]

If absent, initialize from nearest previously trained weight, or from endpoint interpolation in adapter space.

This warm start is important; otherwise CDF refinement becomes too expensive.

---

## Off-by-one convention
Your algorithm text mixes `N` as budget/segments and uses indices `n=0,...,N`. In code, make this explicit:

- `num_segments = N`
- `num_points = N + 1`
- `quantiles = np.linspace(0.0, 1.0, num_points)`

This avoids repeated indexing bugs.

---

## Evaluation protocol

### Plot 1: objective-space frontier
Plot points for:
- uniform
- random
- rewarded soup
- CDF refinement

### Plot 2: all metrics vs scalarization/interpolation parameter
Useful for interpretation; inspired by Rewarded Soup style plots.

### Plot 3: CDF refinement convergence
Plot over outer iteration:
- surrogate `Err_arc_tilde`
- `CV`
- `GapRatio`

### Plot 4: sampled weight movement
Plot `w_n^{(t)}` across outer iterations.

---

## Suggested first success criterion
The first milestone is **not** “best benchmark performance.”
It is this:

1. endpoint specialists differ
2. uniform scalarization is visibly non-uniform in objective space
3. CDF refinement reduces `Err_arc_tilde` compared with uniform weights
4. Rewarded Soup gives a meaningful path baseline

If those 4 happen on a small task, the prototype is successful.

---

## Common failure modes to tell Cursor to avoid

1. **Confusing training reward with evaluation objective**
   - keep scalarized training reward separate from reported `(f1, f2)`

2. **Applying full-model interpolation when using LoRA**
   - only interpolate adapter parameters unless full FT is explicitly intended

3. **Using unsorted random weights**
   - always sort before front metrics

4. **Forgetting to normalize the surrogate CDF**
   - `F_tilde(w) = s_tilde(w) / s_tilde(1)`

5. **Breaking monotonicity in interpolation**
   - use monotone interpolation such as PCHIP when available

6. **No checkpoint reuse between outer iterations**
   - warm start from previous iteration

7. **Reporting IGD to a weak reference front**
   - if using IGD, build a dense reference front carefully

---

## Minimal first experiment
Use a small dataset subset and short generations.

Suggested debug budget:
- 2 rewards
- 11 points (`N=10` segments)
- 2 outer iterations
- tiny train subset
- fixed decoding settings

Get the whole loop working before scaling data or model size.

---

## What to ask Cursor to do first
Give Cursor this exact order:

1. Scaffold the project structure and configs.
2. Implement sequence reward + KL objective with REINFORCE.
3. Train endpoint experts at `w=0` and `w=1`.
4. Implement evaluation to get `(f1, f2)` points.
5. Implement uniform/random scalarization baselines.
6. Implement LoRA Rewarded Soup interpolation.
7. Implement CDF refinement outer loop.
8. Add front metrics and plots.
9. Only then optimize speed or add PPO/GRPO.

