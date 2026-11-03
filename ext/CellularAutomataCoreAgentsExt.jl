# AgentsBackend: the state of a cellular automaton in an Agents.jl model, shared with mobile
# agents. The sweep runs on an ArrayBackend working state (see src/agents.jl); the integrator's
# state is the model itself.
module CellularAutomataCoreAgentsExt

using Agents: Agents, AbstractAgent, GridSpace, Graphs, StandardABM, abmproperties, abmrng,
              add_agent!,
              allids, dummystep, remove_agent!
using CellularAutomataCore
using CellularAutomataCore: Lattice, _EventState, _cell_index, _cell_type, _convert_cell,
                            _linear_index, _read, _template
using Random: Xoshiro

const CAC = CellularAutomataCore

# A cell as an agent: on a lattice it has a position; on a graph its id is its vertex.
mutable struct LatticeCell{D,S} <: AbstractAgent
    const id::Int
    pos::NTuple{D,Int}
    state::S
end
mutable struct GraphCell{S} <: AbstractAgent
    const id::Int
    pos::Int
    state::S
end
_cell_agent(::Lattice{D}, S) where {D} = LatticeCell{D,S}
_cell_agent(::GraphSpace, S) = GraphCell{S}

_abm_space(s::Lattice) = GridSpace(s.dims; periodic=map(b -> b isa Periodic, s.boundaries))
# The graph the cells read along: undirected when every neighbour list is mirrored, otherwise
# directed with an edge i -> j when cell i reads cell j.
function _abm_space(s::CAC.GraphSpace)
    ns = s.neighbours
    n = length(ns)
    symmetric = all(i -> all(j -> i in ns[j], ns[i]), 1:n)
    g = symmetric ? Graphs.SimpleGraph(n) : Graphs.SimpleDiGraph(n)
    for i in 1:n, j in ns[i]
        Graphs.add_edge!(g, i, j)
    end
    return Agents.GraphSpace(g)
end

# The model's properties. `cells` holds the cell agents by linear index in :cells mode and is
# `nothing` in :grid mode, where `work` is the state of record.
mutable struct CAProperties{P,W,Sp,C,I}
    p::P
    t::Int
    const space::Sp
    const work::W
    const array::ArrayBackend
    const cells::C
    const index::Vector{I}
    const init!::Any
end
const GridProperties = CAProperties{<:Any,<:Any,<:Any,Nothing}

_ncells(props::CAProperties) = props.cells === nothing ? 0 : length(props.cells)

# Mobile agents in order of id; cell agents (ids up to `offset`) are never scheduled.
struct MobileAgents
    offset::Int
end
(m::MobileAgents)(model) = sort!([id for id in allids(model) if id > m.offset])

function CAC.allocate_state(b::AgentsBackend, prob::CAProblem, scheme::AbstractScheme)
    array = ArrayBackend(b.threaded)
    work = CAC.allocate_state(array, prob, scheme)
    index = vec(collect(cells(prob.space)))
    C = _cell_agent(prob.space, _cell_type(prob.u0))
    cellvec = b.mode === :cells ? Vector{C}(undef, length(index)) : nothing
    A = b.agent_type === nothing ? C :
        b.mode === :cells ? Union{C,b.agent_type} : b.agent_type
    props = CAProperties(prob.p, floor(Int, prob.tspan[1]), prob.space, work, array,
                         cellvec,
                         index,
                         b.init!)
    model = StandardABM(A, _abm_space(prob.space);
                        (agent_step!)=something(b.agent_step!, dummystep),
                        (model_step!)=something(b.model_step!, dummystep),
                        scheduler=MobileAgents(_ncells(props)), properties=props,
                        rng=Xoshiro(prob.seed), warn=false, warn_deprecation=false)
    if cellvec !== nothing
        for (i, I) in enumerate(index)
            cellvec[i] = _add_cell!(model, C, I, _read(prob.u0, I))
        end
    end
    b.init! === nothing || b.init!(model)
    return model
end

_add_cell!(model, C::Type{<:LatticeCell}, I, v) = add_agent!(Tuple(I), C, model, v)
_add_cell!(model, C::Type{<:GraphCell}, i, v) = add_agent!(i, C, model, v)

# Copy the cell agents into the working state, and back.
function _gather!(props::CAProperties)
    for (a, I) in zip(props.cells, props.index)
        CAC.write_cell!(props.array, props.work, I, a.state)
    end
    return nothing
end
function _scatter!(props::CAProperties)
    for (a, I) in zip(props.cells, props.index)
        a.state = CAC.read_cell(props.array, props.work, I)
    end
    return nothing
end
_gather!(::GridProperties) = nothing
_scatter!(::GridProperties) = nothing

function CAC.advance!(b::AgentsBackend, model::StandardABM, prob, scheme, params, t)
    props = abmproperties(model)
    props.p = params
    _gather!(props)
    CAC.advance!(props.array, props.work, prob, scheme, params, t)
    _scatter!(props)
    props.t = t + 1
    (b.agent_step! === nothing && b.model_step! === nothing) || Agents.step!(model, 1)
    return nothing
end

function CAC.current_state(::AgentsBackend, model::StandardABM)
    props = abmproperties(model)
    # in continuous time the cell agents follow the working state event by event
    props.work isa _EventState || _gather!(props)
    return CAC.current_state(props.array, props.work)
end

function CAC.set_state!(::AgentsBackend, model::StandardABM, u)
    props = abmproperties(model)
    if props.cells === nothing
        CAC.set_state!(props.array, props.work, u)
    else
        for (a, I) in zip(props.cells, props.index)
            a.state = _convert_cell(_template(props.work), _read(u, I))
        end
        props.work isa _EventState && _gather!(props)
    end
    return nothing
end

CAC.read_cell(::AgentsBackend, model::StandardABM, I) = CAC.read_cell(model, I)
CAC.write_cell!(::AgentsBackend, model::StandardABM, I, v) = CAC.write_cell!(model, I, v)

CAC.read_cell(model::StandardABM, I) = _read_host(abmproperties(model), I)
function _read_host(props::GridProperties, I)
    return CAC.read_cell(props.array, props.work, _cell_index(props.space, I))
end
_read_host(props::CAProperties, I) = props.cells[_linear_index(props.space, I)].state

CAC.write_cell!(model::StandardABM, I, v) = _write_host!(abmproperties(model), I, v)
function _write_host!(props::GridProperties, I, v)
    return CAC.write_cell!(props.array, props.work, _cell_index(props.space, I), v)
end
function _write_host!(props::CAProperties, I, v)
    props.cells[_linear_index(props.space, I)].state = _convert_cell(_template(props.work),
                                                                     v)
    return nothing
end

# Continuous time (Gillespie): the event state is the working state; after an event the cell
# that changed is copied to its cell agent, and mobile agents act at every whole time.
CAC._events(::AgentsBackend, model::StandardABM) = abmproperties(model).work
function CAC._after_fire!(::AgentsBackend, model::StandardABM, id)
    return _copy_cell!(abmproperties(model), id)
end
_copy_cell!(::GridProperties, id) = nothing
function _copy_cell!(props::CAProperties, id)
    props.cells[id].state = CAC.read_cell(props.array, props.work, props.index[id])
    return nothing
end
function CAC._mobile_agents(b::AgentsBackend)
    return b.agent_step! !== nothing || b.model_step! !== nothing
end
function CAC._agents_act!(::AgentsBackend, model::StandardABM, prob, p, t)
    props = abmproperties(model)
    props.p = p
    props.t = floor(Int, t)
    Agents.step!(model, 1)
    _gather!(props)
    return nothing
end

CAC._host(::AgentsBackend, model::StandardABM) = model

# reinit!: remove the mobile agents, restart ids, time and random numbers, and rerun init!.
function CAC._restart!(model::StandardABM, prob, p, t0)
    props = abmproperties(model)
    n = _ncells(props)
    for id in collect(allids(model))
        id > n && remove_agent!(id, model)
    end
    getfield(model, :maxid)[] = n
    getfield(model, :time)[] = 0
    copy!(abmrng(model), Xoshiro(prob.seed))
    props.p = p
    props.t = floor(Int, t0)
    CAC._restart!(props.work, prob, p, t0)
    props.init! === nothing || props.init!(model)
    return nothing
end

end # module
