"""
    CellularAutomataCore

Cellular automata with one model specification and pluggable backends: the
Catlab-free core of the ecorecipes cellular-automata ecosystem.

The package separates *what* a cellular automaton is from *how* it runs:

- a **space** ([`Torus`](@ref), [`Grid`](@ref), [`HexGrid`](@ref),
  [`GraphSpace`](@ref)) with boundary conditions ([`Periodic`](@ref),
  [`Reflecting`](@ref), [`Constant`](@ref));
- a **rule**: a neighbourhood ([`Moore`](@ref), [`VonNeumann`](@ref),
  [`Hexagonal`](@ref), [`Offsets`](@ref)) and a local rule, either a built-in
  family ([`Wolfram`](@ref), [`Totalistic`](@ref), [`Life`](@ref),
  [`Generations`](@ref), [`Lenia`](@ref)) or any function
  ([`LocalRule`](@ref)). A local rule in the sense of
  [CeccheriniSilberstein2010](@cite), a finite neighbourhood `S` and a map
  `μ : A^S → B`, is written once and runs everywhere;
- a **state**: an array the size of the space, or a `NamedTuple` of arrays for
  several layers;
- an **environment**: covariate layers, dispersal-kernel fields
  ([`KernelField`](@ref)) and masks;
- an **update scheme**, the algorithm ([`Synchronous`](@ref),
  [`Stochastic`](@ref), [`Checkerboard`](@ref), [`RandomSequential`](@ref),
  [`Margolus`](@ref), [`Gillespie`](@ref)), and a **backend**
  ([`ArrayBackend`](@ref), [`SparseBackend`](@ref), [`GPUBackend`](@ref),
  [`AgentsBackend`](@ref), [`ArkBackend`](@ref), [`ReferenceBackend`](@ref)),
  chosen when the problem is solved.

Stochastic rules draw counter-based random numbers ([`uniform`](@ref)), so a
model gives the same result on every backend and thread count. Problems follow
the SciML interface: `solve(CAProblem(rule, u0, tspan, p), alg; saveat,
callback)`, with SciMLBase callbacks, `remake` and ensembles;
[`abc_rejection`](@ref) calibrates a model against observations on an ensemble.

This package depends on no category-theory library.
`CategoricalCellularAutomata.jl` builds the categorical layer on top of it.

Part of the ecorecipes cellular-automata ecosystem.
"""
module CellularAutomataCore

using Adapt: Adapt
using CommonSolve: CommonSolve, init, solve, solve!, step!
using RecipesBase: RecipesBase, @recipe, @series
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, EnsembleDistributed,
                 EnsembleProblem,
                 EnsembleSerial, EnsembleThreads, NullParameters, ReturnCode, reinit!,
                 remake,
                 terminate!, u_modified!

# Spaces and boundaries
export AbstractSpace, AbstractBoundary, Periodic, Reflecting, Constant
export AbstractGeometry, Square, Hex, Lattice, Torus, Grid, HexGrid, GraphSpace
export cells, boundaries, geometry
# Neighbourhoods
export AbstractNeighbourhood, Moore, VonNeumann, Hexagonal, Offsets, GraphNeighbours
export default_neighbourhood, neighbourhood_offsets
# States and patterns
export single_seed, place!, parse_rle, read_rle, rle_string, write_rle
# Rules
export AbstractRule, local_rule, neighbourhood, validate_state, LocalRule
export Wolfram, Totalistic, Life, Generations, LIFE_PRESETS, GENERATIONS_PRESETS, rulestring
export BlockRule, Lenia, ORBIUM, EventRule
# Randomness
export CellContext, uniform, bernoulli
# Environment: covariates, kernels and fields
export covariate, field, fields, KernelField, kernel_weights
# Landscapes from rasters (methods in the Rasters extension)
export raster_space, raster_mask, raster_covariate, state_raster
export DispersalKernel, GaussianKernel, ExponentialKernel, PowerLawKernel, UniformKernel
export LeniaKernel, AnisotropicKernel, EmpiricalKernel
# Models (ModelingToolkit-style)
export CAModel, StateVariable, Parameter, @camodel, states, parameters, defaults
# Problems, integrators, solutions (SciML interface)
export CAProblem, CASolution, CAIntegrator, final_state, trajectory
export solve, init, step!, solve!, remake, reinit!, terminate!, u_modified!
export NullParameters, ReturnCode
export DiscreteCallback, CallbackSet, PeriodicCallback, PresetTimeCallback, SavingCallback,
       SavedValues
export EnsembleProblem, EnsembleThreads, EnsembleSerial, EnsembleDistributed,
       seed_trajectory
# Update schemes (algorithms) and backends
export AbstractScheme, Synchronous, Stochastic, Checkerboard, RandomSequential, Margolus,
       Gillespie
export AbstractBackend, ArrayBackend, ReferenceBackend, SparseBackend, GPUBackend
export supports, GraphNeighbourStates, stored_tiles, read_cell, write_cell!
export AgentsBackend, ArkBackend, CACell, agent_model
# The backend interface: public but not exported, as the names are generic
public check_problem, allocate_state, advance!, current_state, set_state!, update_cell
# Measures and plotting helpers
export state_counts, cluster_sizes, morans_i, hexagon_vertices
export confusion_counts, cohens_kappa, auc, figure_of_merit
# Calibration
export abc_rejection, ABCResult
# Exceptions: the root and the concrete types under it
export CellularAutomataCoreError, RuleParseError, RLEParseError, StateShapeError
export InvalidStateError, IncompatibleSpaceError, UnsupportedCombinationError

include("errors.jl")
include("spaces.jl")
include("neighbourhoods.jl")
include("state.jl")
include("random.jl")
include("rules.jl")
include("kernels.jl")
include("environment.jl")
include("rasters.jl")
include("schemes.jl")
include("events.jl")
include("lenia.jl")
include("rle.jl")
include("backends.jl")
include("sparse.jl")
include("problem.jl")
include("event_engine.jl")
include("sparse_events.jl")
include("agents.jl")
include("model.jl")
include("measures.jl")
include("evaluation.jl")
include("calibration.jl")
include("recipes.jl")

end # module
