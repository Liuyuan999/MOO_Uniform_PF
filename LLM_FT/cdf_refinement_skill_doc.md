# Skill Doc for Cursor: LLM RLFT with Scalarized Reward + KL and CDF Refinement

## Purpose
You are implementing a **bi-objective LLM fine-tuning via RL** benchmark whose goal is to compare scalarization-based Pareto-front sampling methods and, in particular, implement **Iterative CDF Refinement for Arc-Length Uniform PF Sampling**.

The implementation must be mathematically faithful, modular, and easy to debug.

---

# 1. Core optimization objective

We have:
- prompt/input `x ~ D`
- generated response `y ~ π_θ(.|x)`
- two task rewards `r1(x,y)` and `r2(x,y)`
- a frozen reference policy `π_ref`
- scalarization weight `w ∈ [0,1]`
- KL coefficient `β > 0`

For each scalarization weight `w`, define the policy-space regularized objective

$$
J_w(\theta)
:=
\mathbb E_{x\sim D,\ y\sim \pi_\theta(\cdot|x)}
\left[
(1-w) r_1(x,y) + w r_2(x,y) - \beta \mathrm{KL}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))
\right].
$$

The code must optimize one policy for each `w`.

## Important separation
- **Training objective:** scalarized reward plus KL regularization
- **Evaluation objectives:** usually report the two task objectives `(f1, f2)` separately
- Do **not** accidentally collapse all analysis to the scalarized training reward only

---

# 2. Sequence-level implementation of KL regularization

For a sampled completion $y = (y_1,\dots,y_T)$, the standard sampled sequence-level KL surrogate is

$$
\widehat{\mathrm{KL}}(x,y) := \sum_{t=1}^T 
$$

This is the practical term to use in the sampled RL objective.
Then define sampled return
$$
R_w(x,y)
:=
(1-w) r_1(x,y) + w r_2(x,y) - \beta  \widehat{\mathrm{KL}}(x,y).
$$

The implementation must compute:
1. sampled completion from current policy
2. per-sample reward 1
3. per-sample reward 2
4. per-sample KL to reference
5. combined scalarized return

---

# 3. Recommended first trainer: REINFORCE

For the first implementation, use plain REINFORCE with a batch baseline.

For a batch of sampled trajectories, define

$$
A_i = R_i - \bar R,
\qquad
\bar R = \frac1B\sum_{j=1}^B R_j.
$$

Let

$$
\log \pi_\theta(y^{(i)}|x^{(i)}) = \sum_{t=1}^{T_i} \log \pi_\theta(y_t^{(i)}\mid x^{(i)}, y_{<t}^{(i)}).
$$

Then optimize the loss

$$
\mathcal L_{\mathrm{RL}}(\theta)
= -rac1B 
\sum_{i=1}^B
A_i  \log \pi_\theta(y^{(i)}|x^{(i)}).
$$

The KL penalty is already included inside `R_i`; do not subtract it again inside the loss.

## Why this is the correct first choice
- easy to inspect
- no clipped PPO terms yet
- easier to verify scalarization logic and CDF refinement outer loop
- can later be upgraded to PPO/GRPO without changing the outer algorithm

---

# 4. PEFT requirement

Start with **LoRA** only.

The code should be written so that PEFT mode is pluggable, but the first fully working implementation must target:
- small text-only instruct model
- frozen base model
- trainable LoRA adapters only

Do not complicate the first version with multiple PEFT modes.

---

# 5. Rewarded Soup baseline

In the bi-objective setting, if the two endpoint experts are

$$
\theta^{(0)} \approx \arg\max_\theta J_0(\theta),
\qquad
\theta^{(1)} \approx \arg\max_\theta J_1(\theta),
$$

then the Rewarded Soup baseline is the adapter interpolation path

$$
\theta_{\mathrm{RS}}(\lambda)= (1-\lambda)\theta^{(0)} + \lambda\theta^{(1)},
\qquad \lambda\in[0,1].
$$

## Implementation rule
When using LoRA, interpolate **only adapter parameters**, not the frozen base weights.

Given two LoRA state dicts `A` and `B`, define:

```python
mixed[k] = (1 - lam) * A[k] + lam * B[k]
```

for every trainable adapter parameter key `k`.

---

# 6. CDF refinement algorithm to implement

You must implement the following iterative algorithm faithfully.

## Indexing convention
To avoid off-by-one ambiguity, interpret the algorithm as:
- `num_segments = N`
- `num_points = N + 1`
- points indexed by `n = 0,1,...,N`
- quantiles `q_n = n / N`

## Outer iteration
At iteration `t`, given current CDF `F_t` on `[0,1]`:

$$
w_n^{(t)} = F_t^{-1}(q_n), \qquad q_n = \frac{n}{N}, \quad n=0,\dots,N.
$$

For each point `n`, solve or continue solving the scalarized problem:

$$
\mathbf x_n^{(t)} = \textsc{InnerSolver}\left(w_n^{(t)};\ \mathbf x_{\mathrm{init}} = \mathbf x_n^{(t-1)}\right).
$$

In the LLM RLFT setting, `x_n^{(t)}` should be interpreted as the trainable model parameters / checkpoint associated with weight `w_n^{(t)}`.

Then evaluate objective coordinates:

$$
\mathbf f(\mathbf x_n^{(t)}) = \big(f_1(\mathbf x_n^{(t)}), f_2(\mathbf x_n^{(t)})\big).
$$

---

# 7. Surrogate cumulative arc-length

From the sampled points ordered by `w_n^{(t)}`, define

$$
\tilde s_w(w_0^{(t)}) := 0,
$$

and recursively

$$
\tilde s_w(w_{n+1}^{(t)})
=
\tilde s_w(w_n^{(t)})
+
\left\| \mathbf f(\mathbf x_{n+1}^{(t)}) - \mathbf f(\mathbf x_n^{(t)}) \right\|_2,
\qquad n=0,\dots,N-1.
$$

This is the sampled polyline cumulative arc-length.

Then extend `\tilde s_w` from the sampled weights to all of `[0,1]` using a **monotone interpolation method**, preferably **PCHIP**.

Then normalize:

$$
\tilde F_t(w) = \frac{\tilde s_w(w)}{\tilde s_w(1)}.
$$

This `\tilde F_t` is the surrogate arc-length CDF.

---

# 8. CDF update rule

Update by convex blending:

$$
F_{t+1}(w) = \alpha  \tilde F_t(w) + (1-\alpha) F_t(w),
\qquad \alpha \in (0,1].
$$

## Implementation notes
- represent `F_t` on a dense grid over `[0,1]`
- ensure monotonicity after blending if numerical noise appears
- invert `F_t` by monotone interpolation
- use `np.interp` only for a first version; prefer monotone interpolation wrappers when available

---

# 9. Warm-start rule

This algorithm relies on warm starts.

For point `n` at outer iteration `t`, initialize from the previous outer iteration checkpoint at the same index when possible:

$$
\theta_n^{(t)} \leftarrow \theta_n^{(t-1)}.
$$

If no prior checkpoint exists:
- use nearest available trained checkpoint by weight, or
- use interpolated endpoint LoRA weights as initialization

Do not retrain every scalarized problem from scratch at every outer iteration unless explicitly requested.

---

# 10. Metrics to implement

## 10.1 True relative RMS arc-length error
Given exact cumulative arc-length `s_w`, total length `L = s_w(1)`, and weights `w_0 < \cdots < w_N`, define

$$
\mathrm{Err}_{\mathrm{arc}}(\{w_n\}_{n=0}^N)
:=
\sqrt{
\frac1N
\sum_{n=0}^{N-1}
\left(
\frac{s_w(w_{n+1}) - s_w(w_n)}{L/N} - 1
\right)^2
}.
$$

This is the ideal metric, but generally unavailable because true `s_w` is unknown.

## 10.2 Computable surrogate error
Replace `s_w` by `\tilde s_w` to obtain the surrogate metric:

$$
\widetilde{\mathrm{Err}}_{\mathrm{arc}}(\{w_n\}_{n=0}^N)
:=
\sqrt{
\frac1N
\sum_{n=0}^{N-1}
\left(
\frac{\tilde s_w(w_{n+1}) - \tilde s_w(w_n)}{\tilde L/N} - 1
\right)^2
},
\qquad \tilde L := \tilde s_w(1).
$$

This is the main computable metric for the algorithm.

## 10.3 CV of segment lengths
Let segment lengths be

$$
\ell_n := \left\| \mathbf f(\mathbf x_{n+1}) - \mathbf f(\mathbf x_n) \right\|_2,
\qquad n=0,\dots,N-1.
$$

Define coefficient of variation:

$$
\mathrm{CV} := \frac{\mathrm{std}(\ell_0,\dots,\ell_{N-1})}{\mathrm{mean}(\ell_0,\dots,\ell_{N-1})}.
$$

Lower is better.

## 10.4 GapRatio
Define

$$
\mathrm{GapRatio} := \frac{\max_n \ell_n}{\min_n \ell_n}.
$$

Lower is better.

## 10.5 Optional IGD
If a reliable dense reference front `P_ref` is available, define IGD in the standard way:

$$
\mathrm{IGD}(P, P_{\mathrm{ref}})
:=
\frac{1}{|P_{\mathrm{ref}}|}
\sum_{z\in P_{\mathrm{ref}}}
\min_{p\in P}
\|z-p\|_2.
$$

Use IGD only if the reference front is genuinely dense and high quality.

---

# 11. Required outputs for every run

For every method and run, save:
- sampled weights
- checkpoint paths
- objective-space points `(f1, f2)`
- segment lengths
- `Err_arc_tilde`
- `CV`
- `GapRatio`
- optional `IGD`
- config snapshot
- random seed

For CDF refinement, also save:
- full `F_t` grid per outer iteration
- `weight_history`
- `pf_history`
- `metric_history`

---

# 12. Baselines to support

You must support the following baselines.

## Uniform scalarization

$$
w_n = \frac{n}{N}, \qquad n=0,\dots,N.
$$

## Random scalarization
Sample iid `u_n ~ Uniform[0,1]`, then sort and force endpoints if desired.

## Rewarded Soup
Interpolate endpoint LoRA adapters over a dense `\lambda` grid.

## CDF refinement
Use the iterative CDF update described above.

---

# 13. Plotting requirements

The code must generate:

1. **Objective-space Pareto plots** comparing methods
2. **Metric-vs-parameter curves** for scalarization weight or soup coefficient
3. **CDF refinement convergence plots** over outer iteration
4. **Weight movement plots** showing how sampled `w_n^{(t)}` change

---

# 14. Guardrails and correctness checks

## Check 1: monotonicity
Ensure weights are sorted and cumulative arc-length is monotone.

## Check 2: endpoint consistency
`w=0` and `w=1` should correspond to endpoint specialists.

## Check 3: separation of concerns
Do not mix:
- task reward definitions
- scalarization
- KL penalty
- evaluation metrics

Keep each in a separate function/module.

## Check 4: deterministic evaluation
For evaluation of objective-space fronts, decoding settings should be fixed and documented.

## Check 5: off-by-one
Always treat `N` as number of segments and `N+1` as number of points.

---

# 15. Preferred implementation style

- write typed Python code
- use dataclasses for configs/results
- separate training and evaluation code
- avoid giant notebooks as the primary implementation
- write small testable functions
- save intermediate artifacts as JSON/CSV/PT
- prefer explicit names over abbreviations

---

# 16. Immediate first milestone

The first milestone is a small end-to-end run that demonstrates:
1. endpoint experts train successfully
2. uniform scalarization yields visibly non-uniform front spacing
3. Rewarded Soup produces a path baseline
4. CDF refinement reduces `Err_arc_tilde` relative to uniform weights

Only after this works should you optimize speed, add PPO/GRPO, or scale the model.

