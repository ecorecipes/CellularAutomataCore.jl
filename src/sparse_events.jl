# Gillespie on the SparseBackend: the next-reaction method with the cell states in the sparse
# backend's tiles and a clock only for the cells that can have a positive rate (cells that
# differ from the background, and the cells whose neighbourhoods read them; a quiescent rule
# gives every other cell rate zero). A cell gets its clock the first time it is reached and keeps
# it, drawing its waiting times from the same counter-based streams as the dense engine, so a run
# equals the ArrayBackend's run event for event.

mutable struct _SparseEventState{S<:_SparseState,P,N}
    tiles::S
    index::Dict{Int,Int}     # cell id -> clock
    ids::Vector{Int}         # clock -> cell id
    rates::Vector{Float64}
    next::Vector{Float64}
    held::Vector{Float64}
    fired::Vector{Int}
    heap::Vector{Int}        # clocks, by next event time
    pos::Vector{Int}
    stale::Bool
    p::P
    slot::Int
    cart::CartesianIndices{N,NTuple{N,Base.OneTo{Int}}}
    lin::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
end

function allocate_state(b::SparseBackend, prob, alg::Gillespie)
    tiles = allocate_state(b, prob, Synchronous())
    dims = size(prob.space)
    return _SparseEventState(tiles, Dict{Int,Int}(), Int[], Float64[], Float64[], Float64[],
                             Int[], Int[], Int[], true, prob.p, 0, CartesianIndices(dims),
                             LinearIndices(dims))
end

current_state(b::SparseBackend, st::_SparseEventState) = current_state(b, st.tiles)
function set_state!(b::SparseBackend, st::_SparseEventState, u)
    set_state!(b, st.tiles, u)
    st.stale = true
    return nothing
end
read_cell(b::SparseBackend, st::_SparseEventState, I) = read_cell(b, st.tiles, I)
function write_cell!(b::SparseBackend, st::_SparseEventState, I, v)
    write_cell!(b, st.tiles, I, v)
    st.stale = true
    return nothing
end

function _restart!(st::_SparseEventState, prob, p, t0)
    empty!(st.index)
    foreach(empty!, (st.ids, st.rates, st.next, st.held, st.fired, st.heap, st.pos))
    st.stale = true
    return nothing
end

# The clock of cell `id`, created at rest (rate zero, its first waiting time held) if new.
function _clock!(st::_SparseEventState, prob, id)
    k = get(st.index, id, 0)
    k == 0 || return k
    push!(st.ids, id)
    push!(st.rates, 0.0)
    push!(st.next, Inf)
    push!(st.held, _exponential(_uniform(prob.seed, id, 0, 0)))
    push!(st.fired, 0)
    k = length(st.ids)
    st.index[id] = k
    push!(st.heap, k)
    push!(st.pos, length(st.heap))
    _sift_up!(st.heap, st.pos, st.next, length(st.heap))
    return k
end

# Call f on the id of every cell whose rate depends on cell J: J and the cells reading it.
function _foreach_dependent(f, st::_SparseEventState, prob, J)
    f(st.lin[J])
    for o in _offsets_of(prob.neighbours)
        _foreach_reader(I -> f(st.lin[I]), prob.space, J, o)
    end
    return nothing
end

function _sparse_cell_rate(st::_SparseEventState, prob, env, id)
    I = st.cart[id]
    _masked(env.mask, I) && return 0.0
    space = prob.space
    nbrs = map(o -> _sparse_read(st.tiles, space, I + o), _offsets_of(prob.neighbours))
    trs = prob.rule.f(_sparse_read(st.tiles, space, I), nbrs, _context(env, I, st.slot, id))
    a = Float64(_total_rate(trs))
    (a >= 0 && isfinite(a)) ||
        throw(ArgumentError("EventRule: total rate $a at cell $I; rates must be nonnegative and finite"))
    return a
end

function _set_rate!(st::_SparseEventState, k, a, t)
    old = st.rates[k]
    a > 0 && a == old && return nothing
    left = old > 0 ? max(old * (st.next[k] - t), 0.0) : st.held[k]
    st.rates[k] = a
    if a > 0
        st.next[k] = t + left / a
    else
        st.held[k] = left
        st.next[k] = Inf
    end
    j = _sift_up!(st.heap, st.pos, st.next, st.pos[k])
    _sift_down!(st.heap, st.pos, st.next, j)
    return nothing
end

function _update_dependents!(st::_SparseEventState, prob, env, J, t)
    _foreach_dependent(st, prob, J) do id
        k = _clock!(st, prob, id)
        _set_rate!(st, k, _sparse_cell_rate(st, prob, env, id), t)
        return nothing
    end
    return nothing
end

function _refresh!(st::_SparseEventState, prob, p, t)
    st.p = p
    st.slot = floor(Int, t)
    env = _env(prob, p, (;))
    tiles = st.tiles
    tc = CartesianIndices(tiles.ntiles)
    N = length(tiles.dims)
    # clocks for the cells off the background and their readers; then every clock's rate
    for (k, tile) in enumerate(tiles.cur)
        tile === nothing && continue
        origin = _tile_origin(tiles, tc[k])
        for L in CartesianIndices(ntuple(_ -> tiles.ts, N))
            J = origin + L
            all(Tuple(J) .<= tiles.dims) || continue
            _read(tile, L) == tiles.background && continue
            _foreach_dependent(id -> (_clock!(st, prob, id); nothing), st, prob, J)
        end
    end
    for k in eachindex(st.ids)
        _set_rate!(st, k, _sparse_cell_rate(st, prob, env, st.ids[k]), t)
    end
    st.stale = false
    return nothing
end

function _next_event(st::_SparseEventState, prob, t)
    isempty(st.heap) && return Inf, 0
    k = st.heap[1]
    return st.next[k], k
end

function _fire!(st::_SparseEventState, prob, p, k, t)
    env = _env(prob, p, (;))
    id = st.ids[k]
    I = st.cart[id]
    space = prob.space
    nbrs = map(o -> _sparse_read(st.tiles, space, I + o), _offsets_of(prob.neighbours))
    cell = _sparse_read(st.tiles, space, I)
    trs = prob.rule.f(cell, nbrs, _context(env, I, st.slot, id))
    u = _uniform_hi(prob.seed, id, st.fired[k])
    write_cell!(SparseBackend(st.tiles.background), st.tiles, I,
                _pick(trs, u * _total_rate(trs), cell))
    st.fired[k] += 1
    st.rates[k] = 0.0
    st.held[k] = _exponential(_uniform(prob.seed, id, st.fired[k], 0))
    _update_dependents!(st, prob, env, I, t)
    return nothing
end
