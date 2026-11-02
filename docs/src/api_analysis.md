# Analysis

Spatial measures, map comparison, calibration and plotting.

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

## Plotting

With Plots.jl loaded, `plot(sol)` draws a solution (a space-time diagram for a
one-dimensional lattice, otherwise the last state; keywords `snapshot`, the
index of the saved state to draw, and `layer`), and `plot(space, u)` draws a
state: a heatmap on a square lattice and hexagons on a hexagonal one.

```@docs
hexagon_vertices
```
