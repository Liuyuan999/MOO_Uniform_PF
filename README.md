# SURF: Steering Scalarization Weights to Uniformly Traverse the Pareto Front

This repository contains the code for SURF, a scalarization-based framework for uniformly sampling a Pareto front.

This repository contains the submitted-paper experiments, which use SURF with linear scalarization, plus separate post-submission diagnostics that extend SURF to Chebyshev scalarization on non-convex and benchmark Pareto fronts.

## Repository Layout

**Submitted-paper experiments**

- `uniform_PF.ipynb`: Offline bandit toy problem comparing uniform weights with arc-length-uniform weights.
- `DST/Policy_update_DST_with_baseline.ipynb`: Deep Sea Treasure policy-optimization experiment with SURF, uniform-weight, OLS, souping, UMOD, NBI, epsilon-constraint, continuation, and equal-spacing baselines. Some cells contain follow-up baseline extensions.
- `Fishwood/Policy_update_Fishwood_with_baseline.ipynb`: Tabular FishWood experiment using the same CDF refinement and baseline structure. Some cells contain follow-up baseline extensions.
- `Mountaincar/Policy_update_MountainCar_with_baseline.ipynb`: MO-MountainCar experiment with policy-gradient inner solves and seed-wise comparison tables.
- `LLM_alignment/`: Qwen PPO/REINFORCE code for Reddit summarization alignment with two reward models, LS baselines, SURF/CDF refinement, evaluation, and souping utilities.

**Post-submission diagnostics**

- `Tchebycheff_nonconvex/`: Weighted Chebyshev + SURF diagnostic on non-convex fronts such as ZDT2 and Circle.
- `benchmark_moo/`: Deterministic front-oracle benchmark suite for ZDT3, DTLZ2, DTLZ7, WFG4, and WFG2.
## Post-submission diagnostics

- `Tchebycheff_nonconvex/run_experiments.py` reproduces the exact-inner-solver
  ZDT2/Circle Tchebysheff-SURF diagnostic and writes figures plus
  `figure/metrics.txt`.
- `benchmark_moo/run_benchmarks.py` runs the fixed \(N=15\), \(T=30\),
  \(\alpha=0.3\) two-objective ZDT3/DTLZ2/DTLZ7 front-oracle suite with LS,
  weighted Chebyshev, SURF, equal-arc-length, and NBI normal-line baselines:

  ```bash
  MPLCONFIGDIR=/tmp/matplotlib ../.venv/bin/python benchmark_moo/run_benchmarks.py
  ```

  It writes JSON/CSV metrics and PF figures under `benchmark_moo/`. This is a
  deterministic front-oracle geometry diagnostic—not an end-to-end
  stochastic-optimizer comparison—and reports component-aware metrics for
  the disconnected ZDT3 and DTLZ7 fronts.
- `DST/Policy_update_DST_with_baseline.ipynb` and
  `Fishwood/Policy_update_Fishwood_with_baseline.ipynb` contain follow-up
  NBI, epsilon-constraint, continuation, and equal-spacing implementations.
  Only submitted-paper baselines have multi-seed tables; rerun the extended
  cells before presenting aggregate results.

## Setup

Use Python 3.11 with NumPy, SciPy, Matplotlib,
`pymoo`, and `mo-gymnasium`:

```bash
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/bin/python numpy scipy matplotlib pymoo mo-gymnasium
```

The LLM workflow downloads Hugging Face models, datasets, and reward models
unless they are already cached. Configure GPU resources, local cache paths,
and checkpoint paths in `LLM_alignment/configs/` before launching large runs.


## Metrics

The repository reports spacing, coverage, and quality metrics, including:

- `CV`: coefficient of variation of consecutive Pareto-front segment lengths.
- `GapRatio`: ratio between the largest and smallest consecutive segment.
- `IGD`: inverted generational distance against a dense or oracle reference
  front.
- `HV`: 2D hypervolume, with sign conventions adjusted for maximization or
  minimization settings.




## Acknowledgements

This repository builds on common multi-objective optimization and learning
tools, including PyTorch, SciPy, pymoo, MO-Gymnasium, MORL-Baselines, Hugging
Face Transformers, PEFT, TRL, and Qwen model tooling. Please also cite the
upstream tools, datasets, and baselines when using them in experiments.
