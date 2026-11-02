# Agent backends: the state lives in an Agents.jl model or an Ark.jl world, so that mobile
# agents (vectors, farmers, beetle swarms) can share it. The types are defined here and the
# engines in ext/. In both packages the sweep itself runs on an ArrayBackend working state,
# so every mode agrees exactly with ReferenceBackend:
#
#   :grid   the working state is the state of record, held in the model's properties (Agents)
#           or a world resource (Ark).
#   :cells  each cell is an agent (Agents) or an entity with a `CACell` component (Ark), which
#           is the state of record; a step gathers the cells into the working state, sweeps it
#           and scatters the result back.
#
# Mobile agents act after each sweep. They may draw from a sequential RNG seeded from the
# problem's seed (`abmrng(model)`, the world's `Xoshiro` resource); the sweep never does.

function _check_mode(mode::Symbol)
    mode in (:grid, :cells) ||
        throw(ArgumentError("mode must be :grid or :cells, got :$mode"))
    return mode
end

"""
    AgentsBackend(mode=:grid; agent_type=nothing, init! = nothing, agent_step! = nothing,
                  model_step! = nothing, threaded=Threads.nthreads() > 1)

Keep the state in an Agents.jl model (`StandardABM`), so that mobile agents can
read and change the cells. Needs `using Agents`.

- `mode = :grid`: the state is an array in the model's properties.
- `mode = :cells`: every cell is an agent with fields `id`, `pos` (on a
  lattice) and `state`; the agent with id `i` is the cell with linear index `i`,
  so `model[i].state` is its state. Mobile agents get ids after the cells.
  The cell agents occupy every position, so Agents.jl functions that look for
  empty positions find none (pass `ifempty=false` to `randomwalk!`), and
  `nearby_agents` returns cells as well as mobile agents.

The model's space is a `GridSpace` of the lattice's size, periodic along the
axes whose boundary is [`Periodic`](@ref) (no space on a [`GraphSpace`](@ref)).
Its properties give the parameters as `model.p` and the time as `model.t`, and
`abmrng(model)` is seeded from the problem's seed.

Mobile agents are of type `agent_type` (an `@agent` type for the space).
`init!(model)` adds them once the cells are set, at the start of a run and on
`reinit!`. Each step runs the cellular automaton's sweep, then
`agent_step!(agent, model)` for every mobile agent in order of id, then
`model_step!(model)`. Inside them, [`read_cell`](@ref)`(model, pos)` and
[`write_cell!`](@ref)`(model, pos, v)` read and change a cell. From a callback,
[`agent_model`](@ref)`(integ)` returns the model.

Without mobile agents the result equals that of [`ArrayBackend`](@ref) exactly.
Runs every discrete-time scheme; the sweep uses `ArrayBackend(; threaded)`.
"""
struct AgentsBackend{A,I,G,M} <: AbstractBackend
    mode::Symbol
    agent_type::A
    init!::I
    agent_step!::G
    model_step!::M
    threaded::Bool
end
function AgentsBackend(mode::Symbol=:grid; agent_type=nothing, (init!)=nothing,
                       (agent_step!)=nothing, (model_step!)=nothing,
                       threaded::Bool=Threads.nthreads() > 1)
    return AgentsBackend(_check_mode(mode), agent_type, init!, agent_step!, model_step!,
                         threaded)
end

"""
    ArkBackend(mode=:grid; components=(), init! = nothing, update! = nothing,
               threaded=Threads.nthreads() > 1)

Keep the state in an Ark.jl `World` (an entity component system), so that
mobile agents, which are entities, can read and change the cells. Needs
`using Ark`.

- `mode = :grid`: the state is an array in a world resource.
- `mode = :cells`: every cell is an entity with a [`CACell`](@ref) component
  holding its linear index and state, which queries can select:
  `Query(world, (CACell{T},))` for cell type `T`.

`components` lists the component types of the mobile agents, which the world
must know when it is created. `init!(world, p)` creates the agents once the
cells are set, at the start of a run and on `reinit!`. Each step runs the
cellular automaton's sweep, then `update!(world, p, t)` with the parameters and
the new time. Inside them, [`read_cell`](@ref)`(world, I)` and
[`write_cell!`](@ref)`(world, I, v)` read and change a cell, and
`get_resource(world, Xoshiro)` is a random number generator seeded from the
problem's seed. From a callback, [`agent_model`](@ref)`(integ)` returns the
world.

Without mobile agents the result equals that of [`ArrayBackend`](@ref) exactly.
Runs every discrete-time scheme; the sweep uses `ArrayBackend(; threaded)`.
"""
struct ArkBackend{C<:Tuple,I,U} <: AbstractBackend
    mode::Symbol
    components::C
    init!::I
    update!::U
    threaded::Bool
end
function ArkBackend(mode::Symbol=:grid; components::Tuple=(), (init!)=nothing,
                    (update!)=nothing, threaded::Bool=Threads.nthreads() > 1)
    return ArkBackend(_check_mode(mode), components, init!, update!, threaded)
end

const _AgentBackend = Union{AgentsBackend,ArkBackend}

function Base.show(io::IO, b::AgentsBackend)
    return print(io, "AgentsBackend(:", b.mode,
                 b.agent_type === nothing ? "" : "; agent_type=$(b.agent_type)", ")")
end
Base.show(io::IO, b::ArkBackend) = print(io, "ArkBackend(:", b.mode, ")")

"""
    CACell{S}(index, state)

The component of a cell entity in [`ArkBackend`](@ref)`(:cells)`: the cell's
linear index (its vertex on a [`GraphSpace`](@ref)) and its state, of the type
a cell has in the problem (a `NamedTuple` for a layered state).
"""
struct CACell{S}
    index::Int
    state::S
end

"""
    agent_model(integ::CAIntegrator)

The Agents.jl model or the Ark.jl world that holds the state of an integrator
solved with [`AgentsBackend`](@ref) or [`ArkBackend`](@ref), for callbacks that
add, inspect or remove mobile agents. Changes made through it are seen by
`integ.u` afterwards, and changes made to `integ.u` before the call are kept.
"""
function agent_model(integ::CAIntegrator)
    _sync!(integ)
    setfield!(integ, :_fresh, false)
    setfield!(integ, :_modified, true)
    return _host(integ.backend, integ.state)
end
function _host(b::AbstractBackend, st)
    return throw(ArgumentError("agent_model needs AgentsBackend or ArkBackend, not $b"))
end

"""
    read_cell(model, I)
    read_cell(backend, state, I)

The state of the cell at `I` (a `CartesianIndex`, a tuple such as an agent's
`pos`, or a linear index; a vertex on a [`GraphSpace`](@ref)). The first form
reads from the Agents.jl model or Ark.jl world of [`AgentsBackend`](@ref) or
[`ArkBackend`](@ref), for mobile agents; the second is part of the interface
every backend implements.
"""
function read_cell end

"""
    write_cell!(model, I, v)
    write_cell!(backend, state, I, v)

Set the cell at `I` to `v`, converted to the state's cell type; the forms are
those of [`read_cell`](@ref).
"""
function write_cell! end

supports(::_AgentBackend, ::Union{_GatedScheme,RandomSequential,Margolus}) = true

function allocate_state(b::AgentsBackend, prob, scheme)
    return throw(ArgumentError("AgentsBackend needs Agents.jl: run `using Agents`"))
end
function allocate_state(b::ArkBackend, prob, scheme)
    return throw(ArgumentError("ArkBackend needs Ark.jl: run `using Ark`"))
end

# Shared by the extensions: the cell index of a position given as a tuple, a CartesianIndex
# or a linear index, and the type of one cell of the state.
_cell_index(::GraphSpace, i::Integer) = Int(i)
_cell_index(::Lattice, I::CartesianIndex) = I
_cell_index(::Lattice, I::Tuple) = CartesianIndex(I)
_cell_index(s::Lattice, i::Integer) = CartesianIndices(s.dims)[i]
_linear_index(::GraphSpace, i) = Int(i)
_linear_index(s::Lattice, I) = LinearIndices(s.dims)[_cell_index(s, I)]
_cell_type(u0) = typeof(_read(u0, 1))
