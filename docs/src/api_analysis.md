# Analysis

Spatial measures, map comparison, calibration, likelihoods, observation models and plotting.

## Measures

```@docs
state_counts
cluster_sizes
morans_i
```

## Evaluation

Comparing simulated and observed maps.

```@docs
confusion_counts
cohens_kappa
auc
figure_of_merit
```

## Calibration

Rejection ABC on ensembles, with the evaluation metrics or summaries as
distances.

```@docs
abc_rejection
ABCResult
```

## Likelihoods

The exact likelihood of a path observed completely, at every step, without
error. A rule written as a [`ProbabilisticRule`](@ref) states each cell's
transition probabilities, which drive both its simulation and its likelihood. A
[`Synchronous`](@ref) or [`Stochastic`](@ref) step then has the product over
cells as its probability; other schemes raise [`NoLikelihoodError`](@ref).

```@docs
ProbabilisticRule
transition_probability
transition_logpdf
path_logpdf
```

## Observation models

What a sensor or survey reports of a state. A [`Footprint`](@ref) maps model
cells to observation units (pixels, plots, inspected plants), by the exact
overlap of two [`MapGrid`](@ref)s or from weights computed elsewhere, such as a
point-spread function; a link gives the distribution of each unit's report.
The same [`Observation`](@ref) simulates reports, for simulation-based fitting,
and scores them, for likelihood-based fitting. Missing reports (cloud, masks)
are skipped, which assumes that whether a unit is observed does not depend on
its unobserved state.

```@docs
MapGrid
Footprint
footprint
coverage
aggregate
Observation
Detection
Misclassification
GaussianError
observation_logpdf
simulate_observation
```

## Plotting

With Plots.jl loaded, `plot(sol)` draws a solution (a space-time diagram for a
one-dimensional lattice, otherwise the last state; keywords `snapshot`, the
index of the saved state to draw, and `layer`), and `plot(space, u)` draws a
state: a heatmap on a square lattice and hexagons on a hexagonal one.

```@docs
hexagon_vertices
```
