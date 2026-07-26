# SURF: Steering Scalarization Weights to Uniformly Traverse the Pareto Front

This repository contains the code for SURF, a scalarization-based framework for uniformly sampling a Pareto front.

This repository contains the submitted-paper experiments, which use SURF with linear scalarization, plus separate post-submission diagnostics that extend SURF to Chebyshev scalarization on non-convex and benchmark Pareto fronts.


## Repository Layout

| Path | Description |
| --- | --- |
| `uniform_PF.ipynb` | Submitted-paper offline bandit toy problem comparing uniform weights with arc-length-uniform weights. |
| `DST/Policy_update_DST_with_baseline.ipynb` | Submitted-paper Deep Sea Treasure policy-optimization experiment with SURF, uniform-weight, OLS, souping, UMOD, NBI, epsilon-constraint, continuation, and equal-spacing baselines. Some cells contain follow-up baseline extensions. |
| `Fishwood/Policy_update_Fishwood_with_baseline.ipynb` | Submitted-paper tabular FishWood experiment using the same CDF refinement and baseline structure. Some cells contain follow-up baseline extensions. |
| `Mountaincar/Policy_update_MountainCar_with_baseline.ipynb` | Submitted-paper MO-MountainCar experiment with policy-gradient inner solves and seed-wise comparison tables. |
| `LLM_alignment/` | Submitted-paper Qwen PPO/REINFORCE code for Reddit summarization alignment with two reward models, LS baselines, SURF/CDF refinement, evaluation, and souping utilities. |
| `Tchebycheff_nonconvex/` | Post-submission weighted Chebyshev + SURF diagnostic on non-convex fronts such as ZDT2 and Circle. |
| `benchmark_moo/` | Post-submission deterministic front-oracle benchmark suite for ZDT3, DTLZ2, DTLZ7, WFG4, and WFG2. |




## Setup

The lightweight diagnostics and notebooks use Python 3.10+ or 3.11. A typical
setup is:

```bash
git clone https://github.com/Liuyuan999/MOO_Uniform_PF.git
cd MOO_Uniform_PF

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib pymoo mo-gymnasium jupyter
```

Install PyTorch according to your hardware and CUDA version. For the policy
notebooks, install it before running the notebooks:

```bash
pip install torch
```

Some baseline cells use MORL-Baselines:

```bash
pip install morl-baselines
```

For the LLM-alignment experiment:

```bash
cd LLM_alignment
pip install -e .
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
