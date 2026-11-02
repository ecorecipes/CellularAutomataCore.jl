# ArkBackend: the state of a cellular automaton in an Ark.jl world, shared with mobile agents
# (entities). The sweep runs on an ArrayBackend working state (see src/agents.jl). The world
# holds the engine's data as a resource, so that read_cell(world, I) can find it.
module CellularAutomataCoreArkExt

using Ark: Ark, Entity, Query, World, add_resource!, get_components, get_resource,
           new_entity!, remove_entity!, set_components!
using CellularAutomataCore
using CellularAutomataCore: _cell_index, _cell_type, _convert_cell, _linear_index, _read
using Random: Xoshiro

const CAC = CellularAutomataCore

# `entities` holds the cell entities by linear index in :cells mode and is `nothing` in :grid
# mode, where `work` is the state of record.
struct ArkData{S,W,Sp,E,I,F}
    space::Sp
    work::W
    array::ArrayBackend
    entities::E
    index::Vector{I}
    init!::F
end
function ArkData{S}(space, work, array, entities, index::Vector{I}, init!) where {S,I}
    T = ArkData{S,typeof(work),typeof(space),typeof(entities),I,typeof(init!)}
    return T(space, work, array, entities, index, init!)
end
const GridData = ArkData{<:Any,<:Any,<:Any,Nothing}

struct ArkState{Wd,D}
    world::Wd
    data::D
end

# The resource through which read_cell(world, I) finds the data (one type for every problem).
struct CAResource
    data::Any
end

function CAC.allocate_state(b::ArkBackend, prob::CAProblem, scheme::AbstractScheme)
    array = ArrayBackend(b.threaded)
    work = CAC.allocate_state(array, prob, scheme)
    index = vec(collect(cells(prob.space)))
    S = _cell_type(prob.u0)
    comps = b.mode === :cells ? (CACell{S}, b.components...) : b.components
    world = World(comps...; allow_mutable=any(ismutabletype, comps))
    entities = b.mode === :cells ?
               [new_entity!(world, (CACell{S}(i, _read(prob.u0, I)),))
                for (i, I) in enumerate(index)] : nothing
    data = ArkData{S}(prob.space, work, array, entities, index, b.init!)
    add_resource!(world, CAResource(data))
    add_resource!(world, Xoshiro(prob.seed))
    b.init! === nothing || b.init!(world, prob.p)
    return ArkState(world, data)
end

# Copy the cell entities into the working state, and back, one archetype at a time.
function _gather!(world, d::ArkData{S}) where {S}
    for (es, cs) in Query(world, (CACell{S},))
        for k in eachindex(es)
            c = cs[k]
            CAC.write_cell!(d.array, d.work, d.index[c.index], c.state)
        end
    end
    return nothing
end
function _scatter!(world, d::ArkData{S}) where {S}
    for (es, cs) in Query(world, (CACell{S},))
        for k in eachindex(es)
            i = cs[k].index
            cs[k] = CACell{S}(i, CAC.read_cell(d.array, d.work, d.index[i]))
        end
    end
    return nothing
end
_gather!(world, ::GridData) = nothing
_scatter!(world, ::GridData) = nothing

function CAC.advance!(b::ArkBackend, st::ArkState, prob, scheme, params, t)
    d = st.data
    _gather!(st.world, d)
    CAC.advance!(d.array, d.work, prob, scheme, params, t)
    _scatter!(st.world, d)
    b.update! === nothing || b.update!(st.world, params, t + 1)
    return nothing
end

function CAC.current_state(::ArkBackend, st::ArkState)
    _gather!(st.world, st.data)
    return CAC.current_state(st.data.array, st.data.work)
end

function CAC.set_state!(::ArkBackend, st::ArkState, u)
    d = st.data
    if d isa GridData
        CAC.set_state!(d.array, d.work, u)
    else
        for (i, I) in enumerate(d.index)
            _set_cell!(st.world, d, i, _read(u, I))
        end
    end
    return nothing
end

function _set_cell!(world, d::ArkData{S}, i, v) where {S}
    set_components!(world, d.entities[i], (CACell{S}(i, _convert_cell(d.work.cur, v)),))
    return nothing
end

CAC.read_cell(::ArkBackend, st::ArkState, I) = _read_host(st.world, st.data, I)
CAC.write_cell!(::ArkBackend, st::ArkState, I, v) = _write_host!(st.world, st.data, I, v)

function CAC.read_cell(world::World, I)
    return _read_host(world, get_resource(world, CAResource).data, I)
end
function CAC.write_cell!(world::World, I, v)
    return _write_host!(world, get_resource(world, CAResource).data, I, v)
end

function _read_host(world, d::GridData, I)
    return CAC.read_cell(d.array, d.work, _cell_index(d.space, I))
end
function _read_host(world, d::ArkData{S}, I) where {S}
    e = d.entities[_linear_index(d.space, I)]
    return only(get_components(world, e, (CACell{S},))).state
end
function _write_host!(world, d::GridData, I, v)
    return CAC.write_cell!(d.array, d.work, _cell_index(d.space, I), v)
end
_write_host!(world, d::ArkData, I, v) = _set_cell!(world, d, _linear_index(d.space, I), v)

CAC._host(::ArkBackend, st::ArkState) = st.world

# reinit!: remove every entity that is not a cell, restart the random numbers and rerun init!.
function CAC._restart!(st::ArkState, prob, p, t0)
    world, d = st.world, st.data
    keep = d isa GridData ? Set{Entity}() : Set(d.entities)
    doomed = Entity[]
    for (es,) in Query(world, ())
        append!(doomed, Iterators.filter(∉(keep), es))
    end
    foreach(e -> remove_entity!(world, e), doomed)
    copy!(get_resource(world, Xoshiro), Xoshiro(prob.seed))
    d.init! === nothing || d.init!(world, p)
    return nothing
end

end # module
