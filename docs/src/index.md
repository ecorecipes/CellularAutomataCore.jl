# CellularAutomataCore.jl

Cellular automata with one model specification and pluggable backends: the
Catlab-free core of the ecorecipes cellular-automata ecosystem
(`CellularAutomataCore.jl` → `CategoricalCellularAutomata.jl` for applied
category theory, and `CellularAutomataCore.jl` →
`EcologicalCellularAutomata.jl` for a model zoo of plant-disease and pest
models).

A cellular automaton here is a **rule** (a neighbourhood and a local rule in
the sense of [CeccheriniSilberstein2010](@cite)) applied to a **state** on a
**space**. How it runs, the **update scheme** and the **backend**, is chosen
when the problem is solved, so the same model runs serially, on all threads or,
in later releases, on GPUs and agent-based backends.

## Quick start

```julia
using CellularAutomataCore

# Rule 30 from a single live cell on a ring of 101 cells, for 50 steps
sol = solve(CAProblem(Wolfram(30), single_seed(101), 50))
trajectory(sol)              # 101 × 51 array, time last

# Conway's Life on a 64 × 64 torus from a glider
glider, _ = parse_rle("x = 3, y = 3\nbo\$2bo\$3o!")
u0 = place!(zeros(Int8, 64, 64), glider)
sol = solve(CAProblem(Life(), u0, 100); saveat=10)

# Any local rule: a stochastic SIR epidemic with Symbol states
sir = LocalRule(VonNeumann(1)) do cell, nbrs, ctx
    if cell == :S
        k = count(==(:I), nbrs)
        bernoulli(ctx, 1 - (1 - ctx.p.β)^k, 1) ? :I : :S
    elseif cell == :I
        bernoulli(ctx, ctx.p.γ, 2) ? :R : :I
    else
        cell
    end
end
u0 = fill(:S, 100, 100); u0[50, 50] = :I
sol = solve(CAProblem(sir, u0, 200, (β=0.3, γ=0.1); seed=1))
```

With Plots.jl loaded, `plot(sol)` draws a solution.

## Concepts

| concept | types |
|---|---|
| space | [`Torus`](@ref), [`Grid`](@ref), [`HexGrid`](@ref) (all [`Lattice`](@ref)), [`GraphSpace`](@ref) |
| boundary | [`Periodic`](@ref), [`Reflecting`](@ref), [`Constant`](@ref), one per lattice axis |
| neighbourhood | [`Moore`](@ref), [`VonNeumann`](@ref), [`Hexagonal`](@ref), [`Offsets`](@ref), [`GraphNeighbours`](@ref) |
| rule | [`Wolfram`](@ref), [`Totalistic`](@ref), [`Life`](@ref), [`Generations`](@ref), [`Lenia`](@ref), [`BlockRule`](@ref), [`LocalRule`](@ref) |
| state | an array the size of the space, or a `NamedTuple` of arrays (layers) |
| environment | covariates ([`covariate`](@ref)), masks, kernel fields ([`KernelField`](@ref), [`field`](@ref)) |
| scheme (algorithm) | [`Synchronous`](@ref), [`Stochastic`](@ref), [`Checkerboard`](@ref), [`RandomSequential`](@ref), [`Margolus`](@ref) |
| backend | [`ArrayBackend`](@ref), [`SparseBackend`](@ref), [`GPUBackend`](@ref), [`ReferenceBackend`](@ref) |

A local rule receives the state of its cell, the states of its neighbours (a
tuple on a lattice, in the order of [`neighbourhood_offsets`](@ref): first axis
fastest, so `-r, …, -1, 1, …, r` in one dimension) and a [`CellContext`](@ref)
carrying the problem's parameters `p`, the cell's index and the time `t` of the state
being read. It returns the state at `t + 1`.

## The SciML interface

Problems, solutions, integrators, callbacks and ensembles follow SciML
conventions: `CAProblem(rule, u0, tspan, p)` with `remake`; `solve(prob, alg;
saveat, save_everystep, callback, backend)` returning a solution with `u`, `t`,
`retcode` and `sol(t)`; `init`/`step!` integrators with `u`, `t` and `p`;
SciMLBase's `DiscreteCallback` and `CallbackSet` plus [`PeriodicCallback`](@ref),
[`PresetTimeCallback`](@ref) and [`SavingCallback`](@ref); and
`EnsembleProblem`, whose default `prob_func` gives each trajectory its own seed
([`seed_trajectory`](@ref)).

## Random numbers and reproducibility

Stochastic rules draw from [`uniform`](@ref)`(ctx, k)` and
[`bernoulli`](@ref)`(ctx, p, k)`. The `k`-th draw of a cell at time `t` is a
pure function of the problem's `seed`, the cell, `t` and `k`, computed with the
Philox4x32-10 counter-based generator of [Salmon2011](@cite). It does not depend
on the order in which cells are visited, so a model gives the same result on
every backend and thread count, and the test suite checks every backend
against [`ReferenceBackend`](@ref) for exact equality. Use a different `k` for
each independent draw a rule makes in one step.

## Backends

[`ArrayBackend`](@ref) pads a lattice state with a halo as wide as the
neighbourhood, refreshed from the boundary conditions once per step, so the
inner loop does no boundary arithmetic and does not allocate; with
`threaded=true` the lattice is split into slabs along its last axis, one task
per slab. [`ReferenceBackend`](@ref) applies the boundary conditions on every
access and is the definition the others are tested against.

## Errors

All failures are typed exceptions under [`CellularAutomataCoreError`](@ref):
[`RuleParseError`](@ref), [`RLEParseError`](@ref), [`StateShapeError`](@ref),
[`InvalidStateError`](@ref), [`IncompatibleSpaceError`](@ref) and
[`UnsupportedCombinationError`](@ref). Initial states and neighbourhoods are
checked once, when a [`CAProblem`](@ref) is built, never during a run. Invalid
arguments raise Base's `ArgumentError`.

See the Tutorials section for the vignettes, the API Reference for docstrings
and the [References](references.md) page for full bibliographic entries.
