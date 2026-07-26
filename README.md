# SURF: Steering Scalarization Weights to Uniformly Traverse the Pareto Front

This repository contains the code for SURF, a scalarization-based framework for uniformly sampling a Pareto front.

The repository includes the experiments used in the submitted paper, together
with clearly separated post-submission diagnostics for additional non-convex
and benchmark-front analysis. 


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




## Installation

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

## Quick Start

### 1. Run the notebook experiments

```bash
jupyter lab uniform_PF.ipynb
jupyter lab DST/Policy_update_DST_with_baseline.ipynb
jupyter lab Fishwood/Policy_update_Fishwood_with_baseline.ipynb
jupyter lab Mountaincar/Policy_update_MountainCar_with_baseline.ipynb
```

The notebooks include plotting and metric cells. Generated figures are written
inside each experiment folder, for example `DST/figure/`.

### 2. Run the non-convex Chebyshev diagnostic

```bash
python Tchebycheff_nonconvex/run_experiments.py
```

Outputs are written to:

- `Tchebycheff_nonconvex/figure/metrics.txt`
- `Tchebycheff_nonconvex/figure/*.png`
- `Tchebycheff_nonconvex/figure/*.pdf`

This diagnostic illustrates why linear scalarization can collapse to endpoints
on non-convex fronts, while weighted Chebyshev scalarization plus SURF can
cover the front more uniformly.

### 3. Run the deterministic benchmark-front diagnostic

```bash
MPLCONFIGDIR=/tmp/matplotlib python benchmark_moo/run_benchmarks.py
```

Outputs are written to:

- `benchmark_moo/results/summary.json`
- `benchmark_moo/results/summary.csv`
- `benchmark_moo/figures/*.png`
- `benchmark_moo/figures/*.pdf`

This benchmark is a deterministic front-oracle geometry diagnostic. It
minimizes scalarized objectives over dense analytic Pareto-front
representations, so it isolates scalarizer-induced spacing from optimizer
error. It should not be interpreted as an end-to-end stochastic optimizer
benchmark.

### 4. Run LLM-alignment outer loops

From `LLM_alignment/`:

```bash
python -m qwen.equispace_weighting --config configs/train/ls_uniform.yaml
python -m qwen.cdf_refinement --config configs/train/cdf_refinement.yaml
```

The default configs use Qwen2.5-0.5B-Instruct, LoRA adapters, PPO, the Reddit
summarization task, and two reward models. Outputs are written under
`LLM_alignment/outputs/` by default.

For lower-level single-weight PPO training:

```bash
python -m qwen.train_ppo \
  --config configs/model/qwen_0p5b_dora.yaml \
           configs/task/reddit_summarization.yaml \
           configs/rl/ppo.yaml \
           configs/eval/default.yaml \
  --weight 0.5
```

See `LLM_alignment/qwen/README.md` for the package-level workflow and output
layout.

## Metrics

The repository reports spacing, coverage, and quality metrics, including:

- `CV`: coefficient of variation of consecutive Pareto-front segment lengths.
- `GapRatio`: ratio between the largest and smallest consecutive segment.
- `IGD`: inverted generational distance against a dense or oracle reference
  front.
- `HV`: 2D hypervolume, with sign conventions adjusted for maximization or
  minimization settings.
- Component-aware `CV` and `GapRatio` in disconnected-front diagnostics.

For disconnected fronts such as ZDT3, DTLZ7, and WFG2, global spacing metrics
include jumps between disconnected components. Use the component-aware metrics
when discussing within-component uniformity.

## Reproducibility Notes

- The submitted-paper notebooks and LLM-alignment experiments are stochastic
  and should be reported with the seed protocol used in the corresponding
  cells/configs.
- The `Tchebycheff_nonconvex/` and `benchmark_moo/` folders are
  post-submission diagnostics. Label them as such in papers, rebuttals, or
  presentations.
- The benchmark suite is an oracle-front diagnostic, not a replacement for
  full optimizer comparisons.
- LLM runs can be expensive and checkpoint-heavy. Review `output_root`,
  `hf.local_cache`, `execution`, `warm_start`, and `resume` settings before
  launching.
- Some notebooks include additional baseline cells that should be rerun before
  presenting aggregate tables.

## Citation

If this repository supports your research, please cite the SURF paper. Add the
official BibTeX entry here once the preprint or proceedings version is
available.

```bibtex
@misc{surf_uniform_pf,
  title = {SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front},
  author = {Jiang et al.},
  note = {Code repository},
  year = {2026}
}
```

## Acknowledgements

This repository builds on common multi-objective optimization and learning
tools, including PyTorch, SciPy, pymoo, MO-Gymnasium, MORL-Baselines, Hugging
Face Transformers, PEFT, TRL, and Qwen model tooling. Please also cite the
upstream tools, datasets, and baselines when using them in experiments.
