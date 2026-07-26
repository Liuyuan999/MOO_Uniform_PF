# SURF: Steering Scalarization Weights to Uniformly Traverse the Pareto Front

SURF is a scalarization-based framework for sampling a Pareto front uniformly.
This repository contains the submitted-paper experiments, which use linear
scalarization, and separate post-submission diagnostics that extend SURF to
weighted Chebyshev scalarization on non-convex and benchmark Pareto fronts.

## Repository layout

### Submitted-paper experiments

- `uniform_PF.ipynb`: Offline bandit toy problem comparing uniform weights
  with arc-length-uniform weights.
- `DST/Policy_update_DST_with_baseline.ipynb`: Deep Sea Treasure
  policy-optimization experiment with SURF, uniform-weight, OLS, souping,
  UMOD, NBI, epsilon-constraint, continuation, and equal-spacing baselines.
- `Fishwood/Policy_update_Fishwood_with_baseline.ipynb`: Tabular FishWood
  experiment using the same CDF refinement and baseline structure.
- `Mountaincar/Policy_update_MountainCar_with_baseline.ipynb`:
  MO-MountainCar experiment with policy-gradient inner solves and seed-wise
  comparison tables.
- `LLM_alignment/`: Qwen PPO/REINFORCE workflow for Reddit summarization
  alignment with two reward models, linear-scalarization baselines, SURF/CDF
  refinement, evaluation, and souping utilities.

The DST and FishWood notebooks also contain follow-up baseline extensions.

### Post-submission diagnostics

- `Tchebycheff_nonconvex/`: Weighted Chebyshev + SURF diagnostic on the
  non-convex ZDT2 and Circle fronts.
- `benchmark_moo/`: Deterministic front-oracle benchmark suite for ZDT3,
  DTLZ2, DTLZ7, WFG4, and WFG2.

`Tchebycheff_nonconvex/run_experiments.py` reproduces the exact-inner-solver
ZDT2/Circle diagnostic and writes figures and `figure/metrics.txt`.

Run the two-objective benchmark suite from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib ../.venv/bin/python benchmark_moo/run_benchmarks.py
```

The benchmark uses 15 segments (16 points), 30 SURF updates, and
`alpha=0.3`. It compares uniform linear scalarization, linear
scalarization + SURF, uniform weighted Chebyshev, Chebyshev + SURF,
equal-intrinsic-arc-length, and NBI normal-line baselines. JSON and CSV
metrics are written to `benchmark_moo/results/`, with PNG and PDF
Pareto-front figures in `benchmark_moo/figures/`.

This suite is a deterministic front-oracle geometry diagnostic, not an
end-to-end stochastic-optimizer comparison. It reports component-aware
spacing metrics for the disconnected ZDT3, DTLZ7, and WFG2 fronts.

## Setup

Use Python 3.11 with NumPy, SciPy, Matplotlib, `pymoo`, and
`mo-gymnasium`:

```bash
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/bin/python numpy scipy matplotlib pymoo mo-gymnasium
```

The LLM alignment workflow also downloads Hugging Face models, datasets, and
reward models unless they are already cached. Configure GPU resources, local
cache directories, and checkpoint paths in `LLM_alignment/configs/` before
launching large runs.



## Reproducibility notes

The DST and FishWood notebooks include follow-up implementations of NBI,
epsilon-constraint, continuation, and equal-spacing baselines. Only the
submitted-paper baselines have multi-seed tables; rerun the extended cells
before reporting aggregate results for the follow-up baselines.

## Metrics

- `CV`: Coefficient of variation of consecutive Pareto-front segment lengths.
- `GapRatio`: Ratio of the largest to the smallest consecutive segment.
- `IGD`: Inverted generational distance to a dense or oracle reference front.
- `HV`: Two-dimensional hypervolume, with sign conventions adjusted for
  maximization or minimization.

For disconnected fronts, `ComponentCV` and `ComponentGapRatio` exclude jumps
between components.

## Acknowledgements

This repository builds on PyTorch, SciPy, pymoo, MO-Gymnasium,
MORL-Baselines, Hugging Face Transformers, PEFT, TRL, and Qwen model tooling.
Please cite the relevant upstream tools, datasets, and baselines when using
the experiments.
