# CellularAutomataCore.jl

[![Build Status](https://github.com/ecorecipes/CellularAutomataCore.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/CellularAutomataCore.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/CellularAutomataCore.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Cellular automata with one model specification and pluggable backends (arrays, GPU arrays, sparse and event-driven, Agents.jl, Ark.jl): the Catlab-free core of the ecorecipes cellular-automata ecosystem.

Part of the ecorecipes cellular-automata ecosystem: `CellularAutomataCore.jl` → `CategoricalCellularAutomata.jl` (applied category theory) and `CellularAutomataCore.jl` → `EcologicalCellularAutomata.jl` (model zoo).

## Features

- **Spaces**: periodic and bounded square lattices of any dimension (`Torus`, `Grid`), hexagonal grids
  (`HexGrid`) and arbitrary networks (`GraphSpace`), with per-axis `Periodic`, `Reflecting` or `Constant` edges.
- **Neighbourhoods**: `Moore(r)`, `VonNeumann(r)`, `Hexagonal(r)`, arbitrary `Offsets`, and graph neighbours.
- **Rules**: Wolfram and totalistic codes, Life-like rules including Larger-than-Life (`Life`, with presets),
  `Generations`, Margolus `BlockRule`s, continuous `Lenia`, any function (`LocalRule`), and transition rates
  (`EventRule`) for continuous-time models.
- **States**: integers, `Bool`s, `Symbol`s, isbits structs, or several layers as a `NamedTuple` of arrays.
- **Environment**: covariate layers (fixed or seasonal), masks, and dispersal-kernel fields (`KernelField` with
  Gaussian, exponential, power-law, anisotropic, empirical and Lenia kernels; direct or FFT).
- **Rasters**: with Rasters.jl, spaces, masks and covariates from GeoTIFF or NetCDF rasters, and states
  written back as rasters (`raster_space`, `raster_mask`, `raster_covariate`, `state_raster`).
- **Update schemes**: `Synchronous`, `Stochastic(p)`, `Checkerboard`, `RandomSequential`, `Margolus`, and
  exact event-driven simulation in continuous time with `Gillespie` (next-reaction or direct method).
- **Backends**: threaded halo-padded `ArrayBackend`, tiled `SparseBackend` for invasions on large landscapes,
  `GPUBackend` on any KernelAbstractions device (CUDA, ROCm, Metal, oneAPI), `AgentsBackend` (Agents.jl) and
  `ArkBackend` (Ark.jl) for hybrid models with mobile agents, each with the state as a grid or as one agent or
  entity per cell, and a `ReferenceBackend`; all give identical results, stochastic models included, thanks to
  counter-based Philox random numbers.
- **SciML interface**: `CAProblem(rule, u0, tspan, p)`, `solve(prob, alg; saveat, callback)`, integrators with
  `u`, `t`, `p`, SciMLBase callbacks (`DiscreteCallback`, `PeriodicCallback`, `PresetTimeCallback`,
  `SavingCallback`), `remake`, and `EnsembleProblem` ensembles.
- **Models**: ModelingToolkit-style `@camodel` / `CAModel` declarations of named states (with domains) and
  parameters (with defaults), built into problems with `CAProblem(model, u0map, tspan, pmap; space)`.
- **Apple GPUs**: Metal through `GPUBackend(MetalBackend())`, with single-precision random draws, schemes,
  fields and Lenia for devices without Float64.
- **Measures, patterns and plots**: state counts, cluster sizes, Moran's I; agreement between simulated and
  observed maps (Cohen's kappa, ROC AUC, figure of merit); RLE pattern I/O; Plots recipes.
- **Calibration**: rejection approximate Bayesian computation on ensembles (`abc_rejection`), with map or
  summary-statistic distances.
- **Your own backends**: a small public interface (`allocate_state`, `advance!`, ...) and `update_cell`, which
  applies a rule to one cell, for backends written outside the package.

## Quick Start

```julia
using CellularAutomataCore

sol = solve(CAProblem(Wolfram(30), single_seed(101), 50))   # rule 30 on a ring
trajectory(sol)                                             # 101 × 51, time last

u0 = rand(Int8[0, 1], 128, 128)
sol = solve(CAProblem(Life(:highlife), u0, 200); saveat=20)  # on a 128 × 128 torus

fire = LocalRule(VonNeumann(1)) do cell, nbrs, ctx           # any rule, with random numbers
    cell == :fire && return :empty
    cell == :empty && return bernoulli(ctx, ctx.p.p, 1) ? :tree : :empty
    any(==(:fire), nbrs) ? :fire : (bernoulli(ctx, ctx.p.f, 2) ? :fire : :tree)
end
prob = CAProblem(fire, fill(:empty, 100, 100), 500, (p=0.03, f=1e-4); seed=1)
sol = solve(prob, Synchronous(); backend=ArrayBackend(), saveat=10)
ens = solve(EnsembleProblem(prob), Synchronous(), EnsembleThreads(); trajectories=20)
```

## Installation

The ecosystem packages are not registered. This package has no ecosystem dependencies, so install it
by URL on its own:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/CellularAutomataCore.jl")
```

Requires Julia ≥ 1.12.

## Vignettes

| # | Vignette | Description |
|---|---|---|
| 1 | [Getting started](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/01_getting_started/01_getting_started.md) | Elementary rules, space-time diagrams, saving and stepping, the Game of Life |
| 2 | [Rule families](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/02_rule_families/02_rule_families.md) | Wolfram and totalistic codes, Life-like and Generations rules, RLE patterns |
| 3 | [Spaces and neighbourhoods](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/03_spaces_and_neighbourhoods/03_spaces_and_neighbourhoods.md) | Edges, neighbourhood shapes, Larger than Life, hexagonal grids, three dimensions, graphs |
| 4 | [Custom local rules](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/04_custom_local_rules/04_custom_local_rules.md) | Integer, Symbol, struct and layered states; random numbers; space- and time-varying parameters |
| 5 | [Stochastic rules and ensembles](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/05_stochastic_rules/05_stochastic_rules.md) | Counter-based randomness, directed percolation, lattice SIR, SciML ensembles |
| 6 | [Update schemes](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/06_update_schemes/06_update_schemes.md) | Synchronous, stochastic, checkerboard and random-sequential updating; Margolus block rules |
| 7 | [Dispersal kernels](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/07_dispersal_kernels/07_dispersal_kernels.md) | Kernel shapes, kernel fields, invasions with thin, fat-tailed and wind-driven dispersal, FFT |
| 8 | [Landscapes and interventions](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/08_covariates_and_interventions/08_covariates_and_interventions.md) | Covariates, seasons, masks, spraying, releases and culling with callbacks, spatial measures |
| 9 | [Continuous states and Lenia](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/09_continuous_and_lenia/09_continuous_and_lenia.md) | Diffusion, Gray-Scott reaction-diffusion, the Lenia glider Orbium |
| 10 | [Backends](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/10_backends/10_backends.md) | Reference, array, sparse and GPU (including Apple Metal) backends: identical results, timings, choosing one |
| 11 | [Models and problems](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/11_models/11_models.md) | ModelingToolkit-style `@camodel` declarations turned into problems with `CAProblem(model, u0map, tspan, pmap)` |
| 12 | [Event-driven dynamics](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/12_event_driven/12_event_driven.md) | Exact continuous-time simulation with `EventRule` and `Gillespie`, sparse landscapes |
| 13 | [Hybrid models with Agents.jl](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/13_agents_jl/13_agents_jl.md) | An aphid-borne virus: the crop as a cellular automaton, the vectors as Agents.jl agents; insecticide from a callback |
| 14 | [Hybrid models with Ark.jl](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/14_ark/14_ark.md) | The same model as entities; scouting and roguing systems on one entity per plant; the cost of the agent backends |
| 15 | [Landscapes from rasters](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/15_landscapes_from_rasters/15_landscapes_from_rasters.md) | Spaces, masks and covariates from GeoTIFF and NetCDF rasters with Rasters.jl; writing states back as rasters |
| 16 | [Evaluation and calibration](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/16_calibration/16_calibration.md) | Kappa, figure of merit and ROC AUC against surveys; rejection ABC with `abc_rejection`; forecasting with the posterior |
| 17 | [Writing a backend](https://github.com/ecorecipes/CellularAutomataCore.jl/blob/main/vignettes/17_writing_a_backend/17_writing_a_backend.md) | The backend interface: a minimal backend, one that visits only active cells, a timing wrapper, testing |

Each vignette is also published as a tutorial in the
[documentation](https://ecorecipes.github.io/CellularAutomataCore.jl/); the sources are the `.qmd` files in
[`vignettes/`](vignettes/).

## References

The full bibliography for the ecosystem is `docs/src/references.bib` (also in
`vignettes/references.bib`); the rendered list is the References page of the
[documentation](https://ecorecipes.github.io/CellularAutomataCore.jl/).
