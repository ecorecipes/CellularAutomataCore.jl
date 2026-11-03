# Sparse backend: the lattice is cut into square tiles and only tiles holding a cell that
# differs from the background state are stored. Each step updates the stored tiles and the
# tiles within reach of them; tiles that end up all-background are dropped. For invasions,
# epidemics and fires spreading across large, mostly quiescent landscapes.

"""
    SparseBackend(background; tilesize=32, threaded=Threads.nthreads() > 1)

Store and update only the parts of a lattice that differ from the `background`
state (for example `0`, `:S` or a layered `NamedTuple`). The lattice is cut
into `tilesize`-wide square tiles; each step updates the stored tiles and their
neighbours within reach of the neighbourhood, and drops tiles that end up
entirely background. The cost per step is proportional to the active area, not
to the size of the lattice.

The rule must be **quiescent**: a background cell whose neighbours are all
background must stay background, at every cell and time. Rules with spontaneous
events (a lightning strike in an empty forest) are not, and are rejected when
the problem is set up if a sample of cells shows it. Runs
[`Synchronous`](@ref) and [`Stochastic`](@ref) on lattices without kernel
fields; masks and covariates are supported. It also runs
[`Gillespie`](@ref)`(:next_reaction)` for an [`EventRule`](@ref) whose
background cells with background neighbours have rate zero: only those cells
and the cells near them get clocks, and a run equals the
[`ArrayBackend`](@ref)'s run event for event.
"""
struct SparseBackend{T} <: AbstractBackend
    background::T
    tilesize::Int
    threaded::Bool
end
function SparseBackend(background; tilesize::Integer=32,
                       threaded::Bool=Threads.nthreads() > 1)
    tilesize >= 1 || throw(ArgumentError("tilesize must be at least 1, got $tilesize"))
    return SparseBackend(background, Int(tilesize), threaded)
end

supports(::SparseBackend, ::Union{Synchronous,Stochastic}) = true
supports(::SparseBackend, g::Gillespie) = g.method === :next_reaction

function check_problem(b::SparseBackend, prob, scheme)
    prob.space isa Lattice ||
        throw(IncompatibleSpaceError("SparseBackend", _describe(prob.space),
                                     "needs a lattice"))
    isempty(prob.fields) ||
        throw(ArgumentError("SparseBackend does not support kernel fields; use ArrayBackend"))
    bg = _fill_value(prob.u0, b.background)
    nbrs = map(_ -> bg, prob.neighbours.offsets)
    env = _Env(prob.p, prob.covariates, (;), prob.mask, prob.seed)
    lin = LinearIndices(size(prob.space))
    t0 = floor(Int, prob.tspan[1])
    for I in Iterators.take(cells(prob.space), 64), t in t0:(t0 + 3)
        ctx = _context(env, I, t, lin[I])
        if prob.rule isa EventRule
            a = _total_rate(prob.rule.f(bg, nbrs, ctx))
            iszero(a) ||
                throw(ArgumentError("SparseBackend needs a quiescent rule: a background cell with background neighbours has rate $a at cell $I, t = $t"))
        else
            v = local_rule(prob.rule, bg, nbrs, ctx)
            v == bg ||
                throw(ArgumentError("SparseBackend needs a quiescent rule: a background cell with background neighbours became $v at cell $I, t = $t"))
        end
    end
    return nothing
end

mutable struct _SparseState{T,N,U}
    background::T
    ts::Int
    dims::NTuple{N,Int}
    ntiles::NTuple{N,Int}
    cur::Vector{Union{Nothing,U}}
    next::Vector{Union{Nothing,U}}
    pool::Vector{U}
    reach::Int
    edge_tiles::Vector{Int}
    prototype::U
end

_map_layers(f, u::AbstractArray) = f(u)
_map_layers(f, u::LayeredState) = map(f, u)

function _new_tile(st::_SparseState)
    isempty(st.pool) || return pop!(st.pool)
    return _map_layers(similar, st.prototype)
end

function _fill_tile!(tile, bg)
    for L in CartesianIndices(_state_size(tile))
        _write!(tile, L, bg)
    end
    return tile
end

_tile_origin(st, tcoord) = CartesianIndex((Tuple(tcoord) .- 1) .* st.ts)
_tile_of(st, I) = LinearIndices(st.ntiles)[((Tuple(I) .- 1) .÷ st.ts .+ 1)...]
_within_tile(st, I) = CartesianIndex((Tuple(I) .- 1) .% st.ts .+ 1)

function allocate_state(b::SparseBackend, prob, scheme)
    dims = size(prob.space)
    N = length(dims)
    ts = b.tilesize
    ntiles = cld.(dims, ts)
    bg = _fill_value(prob.u0, b.background)
    prototype = _padded_similar(prob.u0, ntuple(_ -> ts, N))
    _fill_tile!(prototype, bg)
    U = typeof(prototype)
    nt = prod(ntiles)
    reach = cld(max(_halo(prob.neighbours), 1), ts)
    # tiles on a Constant edge whose value differs from the background are always active
    edge = Int[]
    tl = LinearIndices(ntiles)
    for (d, bd) in enumerate(boundaries(prob.space))
        bd isa Constant && _fill_value(prob.u0, bd.value) != bg || continue
        for T in CartesianIndices(ntiles)
            (T[d] == 1 || T[d] == ntiles[d]) && push!(edge, tl[T])
        end
    end
    st = _SparseState(bg, ts, dims, ntiles, Vector{Union{Nothing,U}}(nothing, nt),
                      Vector{Union{Nothing,U}}(nothing, nt), U[], reach, unique!(edge),
                      prototype)
    set_state!(b, st, prob.u0)
    return st
end

@inline function _sparse_read(st::_SparseState, space, K)
    J, out = _boundary_lookup(space, K)
    out == 0 || return _constant_read(space.boundaries, out, st.prototype)
    tile = @inbounds st.cur[_tile_of(st, J)]
    tile === nothing && return st.background
    return _read(tile, _within_tile(st, J))
end

function _candidate_tiles(st::_SparseState{T,N}, space) where {T,N}
    tl = LinearIndices(st.ntiles)
    tc = CartesianIndices(st.ntiles)
    cand = falses(length(st.cur))
    r = st.reach
    for (k, tile) in enumerate(st.cur)
        tile === nothing && continue
        T0 = tc[k]
        for dT in CartesianIndices(ntuple(_ -> (-r):r, Val(N)))
            coord = ntuple(Val(N)) do d
                c = T0[d] + dT[d]
                return space.boundaries[d] isa Periodic ? mod1(c, st.ntiles[d]) : c
            end
            all(d -> 1 <= coord[d] <= st.ntiles[d], 1:N) || continue
            cand[tl[coord...]] = true
        end
    end
    foreach(k -> cand[k] = true, st.edge_tiles)
    return findall(cand)
end

function _sweep_tile!(tile, st::_SparseState{T,N}, k, prob, env, t, gate) where {T,N}
    space = prob.space
    origin = _tile_origin(st, CartesianIndices(st.ntiles)[k])
    lin = LinearIndices(st.dims)
    offs = prob.neighbours.offsets
    allbg = true
    for L in CartesianIndices(ntuple(_ -> st.ts, Val(N)))
        I = origin + L
        if any(Tuple(I) .> st.dims)
            _write!(tile, L, st.background)
            continue
        end
        c = _sparse_read(st, space, I)
        ctx = _context(env, I, t, lin[I])
        v = if _masked(env.mask, I) || !_gate_open(gate, ctx, I)
            c
        else
            nbrs = map(o -> _sparse_read(st, space, I + o), offs)
            _convert_cell(st.prototype, local_rule(prob.rule, c, nbrs, ctx))
        end
        _write!(tile, L, v)
        allbg &= v == st.background
    end
    return allbg
end

function advance!(b::SparseBackend, st::_SparseState, prob,
                  scheme::Union{Synchronous,Stochastic},
                  params, t)
    env = _Env(params, prob.covariates, (;), prob.mask, prob.seed)
    gate = only(_gates(scheme))
    cand = _candidate_tiles(st, prob.space)
    tiles = [_new_tile(st) for _ in cand]
    empty = falses(length(cand))
    if b.threaded
        Threads.@threads for j in eachindex(cand)
            empty[j] = _sweep_tile!(tiles[j], st, cand[j], prob, env, t, gate)
        end
    else
        for j in eachindex(cand)
            empty[j] = _sweep_tile!(tiles[j], st, cand[j], prob, env, t, gate)
        end
    end
    fill!(st.next, nothing)
    for j in eachindex(cand)
        if empty[j]
            push!(st.pool, tiles[j])
        else
            st.next[cand[j]] = tiles[j]
        end
    end
    for tile in st.cur
        tile === nothing || push!(st.pool, tile)
    end
    st.cur, st.next = st.next, st.cur
    return nothing
end

function current_state(::SparseBackend, st::_SparseState{T,N}) where {T,N}
    u = _map_layers(a -> similar(a, st.dims), st.prototype)
    for I in CartesianIndices(st.dims)
        _write!(u, I, st.background)
    end
    tc = CartesianIndices(st.ntiles)
    for (k, tile) in enumerate(st.cur)
        tile === nothing && continue
        origin = _tile_origin(st, tc[k])
        for L in CartesianIndices(ntuple(_ -> st.ts, Val(N)))
            I = origin + L
            all(Tuple(I) .<= st.dims) && _write!(u, I, _read(tile, L))
        end
    end
    return u
end

function set_state!(b::SparseBackend, st::_SparseState, u)
    for k in eachindex(st.cur)
        st.cur[k] === nothing || push!(st.pool, st.cur[k])
        st.cur[k] = nothing
    end
    for I in CartesianIndices(st.dims)
        v = _read(u, I)
        v == st.background || write_cell!(b, st, I, v)
    end
    return nothing
end

function read_cell(::SparseBackend, st::_SparseState, I)
    I = CartesianIndex(I)
    tile = st.cur[_tile_of(st, I)]
    tile === nothing && return st.background
    return _read(tile, _within_tile(st, I))
end
function write_cell!(::SparseBackend, st::_SparseState, I, v)
    I = CartesianIndex(I)
    k = _tile_of(st, I)
    st.cur[k] === nothing && (st.cur[k] = _fill_tile!(_new_tile(st), st.background))
    _write!(st.cur[k], _within_tile(st, I), _convert_cell(st.prototype, v))
    return nothing
end

"""
    stored_tiles(integ::CAIntegrator) -> Int

The number of tiles a [`SparseBackend`](@ref) integrator currently stores; each
holds `tilesize^N` cells.
"""
function stored_tiles(integ)
    integ.backend isa SparseBackend ||
        throw(ArgumentError("stored_tiles needs an integrator on a SparseBackend"))
    st = integ.state
    tiles = hasproperty(st, :tiles) ? st.tiles : st
    return count(!isnothing, tiles.cur)
end
