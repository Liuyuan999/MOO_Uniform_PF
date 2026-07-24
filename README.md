# SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front

This repository contains both the submitted-paper experiments and clearly
separated post-submission diagnostics for **SURF**, a scalarization-based
framework for uniformly sampling a Pareto front.

## Experiment map

```text
.
├── uniform_PF.ipynb       # Submitted bandit toy
├── DST/                   # Submitted Deep Sea Treasure experiment
├── Fishwood/              # Submitted Fishwood experiment
├── Mountaincar/           # Submitted MO-MountainCar experiment
├── LLM_alignment/         # Submitted LLM-alignment experiment
├── Tchebycheff_nonconvex/ # New exact-inner-solver ZDT2/Circle diagnostic
└── benchmark_moo/         # New ZDT3/DTLZ2/DTLZ7 front-oracle benchmark suite
```

`uniform_PF.ipynb`, `DST/`, `Fishwood/`, `Mountaincar/`, and
`LLM_alignment/` correspond to the submitted paper. The last two directories
are post-submission rebuttal diagnostics and must be reported as such.

## Environment

Use Python 3.11 with NumPy, SciPy, Matplotlib,
`pymoo`, and `mo-gymnasium`:

```bash
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/bin/python numpy scipy matplotlib pymoo mo-gymnasium
```

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
