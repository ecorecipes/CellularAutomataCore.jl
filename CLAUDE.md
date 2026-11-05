# CellularAutomataCore.jl

Cellular automata with one model specification and pluggable backends (arrays, GPU arrays, sparse and event-driven, Agents.jl, Ark.jl): the Catlab-free core of the ecorecipes cellular-automata ecosystem.

## Place in the ecosystem

Dependency order (arrows = depends on): CategoricalCellularAutomata → CellularAutomataCore and
EcologicalCellularAutomata → CellularAutomataCore. The zoo does not depend on the categorical package.
This package depends on: nothing else in the ecosystem. `Project.toml` has no `[sources]` section.

The shared bibliography lives in the development workspace that holds the three packages side by side;
the `references.bib` files here are synced copies.

## Invariants that must not be broken

- **No Catlab, GATlab or ACSets, ever, in `[deps]`, `[weakdeps]`, `[extras]`, the docs or the vignette
  environment**. The categorical layer is `CategoricalCellularAutomata.jl`,
  which works through this package's public API (rules, `neighbourhood_offsets`, `CellContext`, `LocalRule`,
  `CAProblem`). Any operation added here for it gets a plain name, never `compose`, `otimes`, `id`, `pair`,
  `proj1`, `proj2`, `mcopy` or `delete`.
- Backend packages (KernelAbstractions, Agents, Ark, FFTW, Rasters, Makie) are `[weakdeps]` loaded through
  `ext/`, never hard dependencies. Hard dependencies are Adapt, CommonSolve, RecipesBase and SciMLBase only.
- The public interface follows SciML: `CAProblem(rule, u0, tspan, p)`, `solve(prob, alg; saveat, callback)`,
  integrators with `u`/`t`/`p`, SciMLBase callbacks and `EnsembleProblem`. The update scheme is the algorithm
  (`AbstractScheme <: SciMLBase.AbstractSciMLAlgorithm`). Do not subtype SciML's time-series solution or
  integrator types: their recipes and array semantics conflict with spatial states.
- Draw number `k = 0` is reserved for update schemes; rules use `k >= 1`.
- Rule closures must not assign to variables of the enclosing scope: such a variable is shared by all cells and
  races under threaded and GPU backends (it happened in the conformance suite).
- A model is specified once and is backend-agnostic. Every backend passes the shared conformance suite and
  agrees **exactly** with the array backend on deterministic rules and on stochastic rules driven by the
  counter-based RNG. `test/backend_conformance.jl` holds the battery; add every new backend to `backends()`.
- Random numbers in rules come only from `uniform(ctx, k)` / `bernoulli(ctx, p, k)` (Philox4x32-10 keyed by seed,
  cell, time and draw number `k`). Never use a global or sequential RNG inside a rule or an engine.
- Neighbour order is the order of `neighbourhood_offsets` (first axis fastest; `-r, …, -1, 1, …, r` in one
  dimension). `Wolfram` and `Totalistic` depend on it.
- Continuous time (`Gillespie` with an `EventRule`) also uses counter-based draws only: the next-reaction
  method keys cell `i`'s `n`-th waiting time and transition choice on `(seed, i, n, k = 0)`; the direct method
  uses the reserved cell id 0 and its draw count. Integrator and solution times are `Float64` there and `Int`
  for every other scheme.
- Initial states and neighbourhoods are validated once, in the `CAProblem` constructor; engines do no checking.
- GPU code must be Float64-free for Metal: `uniform(Float32, ...)`, `Stochastic{Float32}`, `KernelField`/`Lenia`
  `precision=Float32`; no hex-float literals (they are Float64), no `Symbol` fields in anything adapted to a
  device. `test/metal.jl` runs conformance on an Apple GPU by hand (not in CI).
- `AgentsBackend` and `ArkBackend` sweep on an `ArrayBackend` working state. In `:grid` mode that state is the
  state of record; in `:cells` mode the cell agents/entities are, and each step gathers them into the working
  state and scatters the result back. Mobile agents act after the sweep and may draw from the sequential RNG
  seeded from the problem's seed (`abmrng(model)`, the world's `Xoshiro` resource); the sweep never does.
  Without mobile agents both modes are in the conformance battery.
- `integ.u` is a cached copy: `_sync!` applies it only while it is fresh, and `agent_model(integ)` marks it
  stale, so that changes made through the model or world are not overwritten.
- The backend interface (`check_problem`, `allocate_state`, `advance!`, `current_state`, `set_state!`,
  `update_cell`) is declared `public`, not exported: the names are generic. `update_cell` must agree with a
  synchronous step of `ReferenceBackend`; `CellLoopBackend` in `test/test_backend_interface.jl` is written only
  against the public interface and is in the conformance battery, so a change that breaks outside backends
  fails there. Plots.jl exports its own `AbstractBackend`; code that loads both imports ours explicitly.
- `abc_rejection` runs proposal `i` with the seed `seed_trajectory` gives trajectory `i`, so results do not
  depend on the ensemble algorithm. Priors are sampled with a user-supplied `rng`: Random is not a dependency.
- Time is the last axis of `trajectory`. The plot keyword for a saved state is `snapshot`, not `frame`, which
  Plots reads as `framestyle`.

## Layout

- `src/CellularAutomataCore.jl`: module docstring, exports, includes.
- `src/errors.jl`: the root `CellularAutomataCoreError` and every exception under it.
- `src/spaces.jl`: boundaries, `Lattice` (`Torus`, `Grid`, `HexGrid`), `GraphSpace`.
- `src/neighbourhoods.jl`: `Moore`, `VonNeumann`, `Hexagonal`, `Offsets`, `GraphNeighbours`, resolution to offsets.
- `src/state.jl`: single-array and layered (`NamedTuple`) state helpers, `single_seed`, `place!`.
- `src/random.jl`: `philox4x32`, `CellContext`, `uniform`, `bernoulli`.
- `src/rules.jl`: the rule interface, `LocalRule`, `Wolfram`, `Totalistic`, `Life`, `Generations`, presets.
- `src/rle.jl`: RLE pattern I/O.
- `src/kernels.jl`, `src/environment.jl`: dispersal kernels, `KernelField`, covariates, masks, the per-step `_Env`.
- `src/rasters.jl`: `raster_space`, `raster_mask`, `raster_covariate`, `state_raster` (fallbacks that ask for
  Rasters); their methods are in `ext/CellularAutomataCoreRastersExt.jl` (X and Y are the lattice's axes, Ti
  a covariate's time axis).
- `src/schemes.jl`: update schemes and `BlockRule`. `src/events.jl`: `EventRule`, `Gillespie`;
  `src/event_engine.jl`: the continuous-time engine (indexed heap, dependents, stepping, tstops).
  `src/lenia.jl`: `Lenia`, `ORBIUM`. `src/measures.jl`, `src/evaluation.jl`: summaries and map comparison.
  `src/calibration.jl`: `abc_rejection`, `ABCResult` (its plot recipe is in `src/recipes.jl`).
- `src/backends.jl`: the backend contract (`supports`, `check_problem`, `allocate_state`, `advance!`,
  `current_state`, `set_state!`, `read_cell`, `write_cell!`) and `update_cell`, `ReferenceBackend`,
  `ArrayBackend`, `GPUBackend`'s type. `src/sparse.jl`: `SparseBackend`. `src/agents.jl`: `AgentsBackend`, `ArkBackend`, `CACell`,
  `agent_model`. `ext/`: the KernelAbstractions (GPU), FFTW, Agents and Ark extensions (the last two also
  need Random, a stdlib that both packages load).
- `src/problem.jl`: `CAProblem`, `CAIntegrator`, `CASolution`, SciML methods, callbacks, ensembles.
- `src/recipes.jl`: RecipesBase plot recipes.
- `test/`: one file per source file, `test_backend_interface.jl` (`CellLoopBackend`), `backend_conformance.jl`, `test_errors.jl`, `test_docstrings.jl`,
  `test_aqua.jl`; seeded RNGs only. Rule families are tested against naive implementations written from their
  definitions, never against the package's own code paths.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite
julia --project=docs docs/make.jl --no-deploy                     # build docs locally
julia scripts/render_vignettes.jl                                 # render out-of-date vignettes to html/gfm/pdf, running each one's code once (cached; PDF needs lualatex + ../fonts/JuliaMono)
julia scripts/sync_vignettes.jl [--check]                         # copy vignettes into docs/src/tutorials
```

## Files not to edit by hand

- `docs/src/tutorials/` is generated by `scripts/sync_vignettes.jl`.
- `vignettes/*/*.md`, `*.html`, `*.pdf` and `*_files/` are quarto output; edit the `.qmd`.
- `docs/src/references.bib` and `vignettes/references.bib` are copied from the workspace
  `docs/references.bib` by `scripts/sync_references.jl`; edit the workspace copy.

## Style

JuliaFormatter `yas`; docstrings on every exported name, which `test/test_docstrings.jl` enforces; the docs
build is strict (no `warnonly`), so a docstring left out of the manual or a broken `@ref` fails it; typed
exceptions with the offending names in the message: they live in `src/errors.jl` and
subtype `CellularAutomataCoreError`, invalid arguments and keywords raise `ArgumentError`, typed errors from a lower package pass
through unchanged and documented, and another package's type is named as a code span, never with `@ref`;
seeded RNGs only in tests; every optimised code path is tested against a slower reference; no emojis in code
or docs.
