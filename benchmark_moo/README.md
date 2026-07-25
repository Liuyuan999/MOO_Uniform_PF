# Two-objective oracle benchmark suite

This directory runs a deterministic geometry diagnostic on ZDT3, DTLZ2
(\(M=2\)), DTLZ7 (\(M=2\)), WFG4 (\(M=2\), concave connected), and WFG2
(\(M=2\), convex disconnected). It compares uniform linear scalarization,
LS+SURF, uniform weighted Chebyshev, Chebyshev+SURF, an equal-intrinsic-arc
length front-oracle reference, and an NBI normal-line/front-oracle baseline.

Run from `exp_2_obj_SURF-main/`:

```bash
MPLCONFIGDIR=/tmp/matplotlib ../.venv/bin/python benchmark_moo/run_benchmarks.py
```

The fixed configuration is 15 segments (16 points), 30 SURF updates, and
`alpha=0.3`. Results are written as valid JSON and CSV in `results/`; one PNG
and PDF Pareto-front figure is produced for each benchmark in `figures/`.

## Interpretation and caveats

The scalarized subproblems are minimized over a deterministic, high-resolution
analytic Pareto-front representation (200,001 parameter samples). Thus this
is a **front-oracle diagnostic**, not an end-to-end stochastic-optimizer
comparison, and it does not establish a general scalarizer-level theorem.

CV and GapRatio are the primary spacing measures; HV and IGD are secondary
coverage/quality measures. ZDT3, DTLZ7, and WFG2 have disconnected fronts.
Their global CV and GapRatio include Euclidean jumps between components, so
`ComponentCV` and `ComponentGapRatio` are also reported after excluding those
jumps. WFG4/WFG2 use Huband/pymoo \(M=2\) zero-distance PF shape functions.
The equal-spacing reference allocates points by total intrinsic length over
components. The NBI baseline is a valid CHIM normal-line construction
evaluated against the same front oracle; if a normal line has no intersection,
the script reports the missing target rather than fabricating a solution.
