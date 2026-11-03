# Backends: where the state lives and how cells are visited. A model is specified once; the
# scheme says in what order cells update.
#
# Backend contract (every backend implements these for the schemes it supports):
#
#   supports(backend, scheme) -> Bool
#   check_problem(backend, prob, scheme)              raise early if the problem cannot run
#   allocate_state(backend, prob, scheme) -> state    the current state (and buffers)
#   advance!(backend, state, prob, scheme, params, t) one full step of the scheme, t -> t + 1
#   current_state(backend, state) -> copy in the user's layout
#   set_state!(backend, state, u)                     replace the whole state
#   read_cell(backend, state, I), write_cell!(backend, state, I, v)
#
# Every backend must reproduce the ReferenceBackend exactly (test/backend_conformance.jl).

"""
    AbstractBackend

Supertype of backends, which say where the state lives and how cells are
visited: [`ReferenceBackend`](@ref), [`ArrayBackend`](@ref),
[`SparseBackend`](@ref), [`GPUBackend`](@ref), [`AgentsBackend`](@ref) and
[`ArkBackend`](@ref). Every backend gives exactly the same result for the same
problem, including stochastic rules (see [`uniform`](@ref)).

A new backend implements [`supports`](@ref),
[`CellularAutomataCore.check_problem`](@ref),
[`CellularAutomataCore.allocate_state`](@ref),
[`CellularAutomataCore.advance!`](@ref),
[`CellularAutomataCore.current_state`](@ref),
[`CellularAutomataCore.set_state!`](@ref), [`read_cell`](@ref) and
[`write_cell!`](@ref) for its type.

Plots.jl exports a different `AbstractBackend`; with both packages loaded,
write `CellularAutomataCore.AbstractBackend` or import it explicitly.
"""
abstract type AbstractBackend end

"""
    ReferenceBackend()

The slow, simple backend that the others are tested against: it applies the
boundary conditions on every neighbour access and visits cells serially.
"""
struct ReferenceBackend <: AbstractBackend end

"""
    ArrayBackend(; threaded=Threads.nthreads() > 1)

Dense arrays on the CPU. On a lattice the state is padded with a halo as wide
as the widest neighbourhood or kernel field, refreshed from the boundary
conditions once per sweep, so the inner loop does no boundary arithmetic; with
`threaded=true` the lattice is split into slabs along its last axis, one task
per slab. The result does not depend on the thread count. Runs every scheme.
"""
struct ArrayBackend <: AbstractBackend
    threaded::Bool
end
ArrayBackend(; threaded::Bool=Threads.nthreads() > 1) = ArrayBackend(threaded)

"""
    GPUBackend(device; workgroupsize=256)
    GPUBackend(array)

Run on a GPU (or any other KernelAbstractions.jl device) with one kernel per
sweep. `device` is a KernelAbstractions backend such as `CUDABackend()`,
`ROCBackend()`, `MetalBackend()`, `JLBackend()` (JLArrays, for testing without a
GPU) or `KernelAbstractions.CPU()`; the second form takes it from an array of
the device's type (via `KernelAbstractions.get_backend`). Needs
`using KernelAbstractions`. The initial state, covariates and mask are copied
to the device; rules, parameters and kernel-field sources must be usable inside
a kernel (isbits values, no allocation). Runs [`Synchronous`](@ref),
[`Stochastic`](@ref) and [`Checkerboard`](@ref) on lattices.

Apple GPUs (`using Metal`; `GPUBackend(MetalBackend())`) have no double
precision: write rules with `Float32` parameters and literals (`0.5f0`), so that
[`bernoulli`](@ref) draws in single precision, and use `Stochastic(0.5f0)`,
`KernelField(...; precision=Float32)` and `Lenia(; precision=Float32)`. Integer
and Boolean models then agree with the CPU backends exactly; models that call
transcendental functions on floating-point states (Lenia) agree to rounding,
because GPU implementations of `exp` and fused multiply-adds differ in the last
bits.
"""
struct GPUBackend{D} <: AbstractBackend
    device::D
    workgroupsize::Int
end
GPUBackend(device; workgroupsize::Integer=256) = GPUBackend(device, Int(workgroupsize))

"""
    supports(backend, scheme) -> Bool

Whether `backend` can run update scheme `scheme`. [`solve`](@ref) raises
[`UnsupportedCombinationError`](@ref) for an unsupported combination.
"""
supports(::AbstractBackend, ::AbstractScheme) = false

"""
    CellularAutomataCore.check_problem(backend, prob, scheme)

Raise an error, when a problem is set up, if `backend` cannot run `prob` with
`scheme` (for example a space or kernel field it does not handle); return
`nothing` otherwise. Part of the backend interface; the fallback accepts every
problem.
"""
check_problem(::AbstractBackend, prob, scheme) = nothing

"""
    CellularAutomataCore.allocate_state(backend, prob, scheme) -> state

Create the backend's state for a run of `prob` with `scheme`, holding the
initial state `prob.u0` and any buffers. Called once per integrator, after
[`supports`](@ref) and [`CellularAutomataCore.check_problem`](@ref). Part of
the backend interface, with [`CellularAutomataCore.advance!`](@ref),
[`CellularAutomataCore.current_state`](@ref),
[`CellularAutomataCore.set_state!`](@ref), [`read_cell`](@ref) and
[`write_cell!`](@ref). A backend implements them for its own type, the type of
its state and the schemes it supports.
"""
function allocate_state end

"""
    CellularAutomataCore.advance!(backend, state, prob, scheme, p, t)

Advance `state` by one step of `scheme`, from time `t` to `t + 1`, with
parameters `p` (the integrator's, which a callback may have changed from
`prob.p`). Stochastic rules draw their random numbers from the time, the cell
and the seed (see [`uniform`](@ref)), so the order in which a backend visits
the cells does not change the result. Part of the backend interface.
"""
function advance! end

"""
    CellularAutomataCore.current_state(backend, state) -> u

A copy of the current state in the layout of `prob.u0` (an array the size of
the space, or a `NamedTuple` of them), which the integrator may keep. Part of
the backend interface.
"""
function current_state end

"""
    CellularAutomataCore.set_state!(backend, state, u)

Replace the whole current state with `u`, given in the layout of `prob.u0`.
Used by `reinit!` and to apply changes callbacks make to `integ.u`. Part of the
backend interface.
"""
function set_state! end
function supports(::Union{ArrayBackend,ReferenceBackend},
                  ::Union{_GatedScheme,RandomSequential,Margolus})
    return true
end
supports(::GPUBackend, ::_GatedScheme) = true

function allocate_state(b::GPUBackend, prob, scheme)
    return throw(ArgumentError("GPUBackend needs KernelAbstractions.jl: run `using KernelAbstractions` (and a GPU package such as CUDA, AMDGPU or Metal)"))
end

# ------------------------------------------------------------------ shared per-cell helpers

# Convert a rule's result to the element type(s) of the state.
@inline _convert_cell(u::AbstractArray, v) = convert(eltype(u), v)
@inline function _convert_cell(u::LayeredState, v::NamedTuple)
    return map((a, x) -> convert(eltype(a), x), u, _ordered(u, v))
end

# The value read beyond a Constant boundary, in the layout of the state.
@inline _fill_value(u::AbstractArray, v) = convert(eltype(u), v)
@inline _fill_value(u::LayeredState, v::NamedTuple) = _convert_cell(u, v)
@inline _fill_value(u::LayeredState, v) = map(a -> convert(eltype(a), v), u)

# Map an unpadded index that may lie outside the lattice. Returns (index, d) where d is the
# highest-numbered Constant axis the index lies beyond, or 0.
@inline function _boundary_lookup(space::Lattice{N}, K::CartesianIndex{N}) where {N}
    mapped = ntuple(d -> _map_axis(space.boundaries[d], K[d], space.dims[d]), Val(N))
    out = 0
    for d in 1:N
        mapped[d][2] && (out = d)
    end
    return CartesianIndex(map(first, mapped)), out
end

# The value of the Constant boundary on axis d, unrolled over the boundary tuple so the result
# type is the cell type (needed inside GPU kernels).
@inline _constant_read(::Tuple{}, d, u) = throw(ArgumentError("no Constant boundary on that axis"))
@inline function _constant_read(bs::Tuple, d, u)
    d == 1 && return _boundary_value(first(bs), u)
    return _constant_read(Base.tail(bs), d - 1, u)
end
@inline _boundary_value(b::Constant, u) = _fill_value(u, b.value)
@inline _boundary_value(b::AbstractBoundary, u) = throw(ArgumentError("not a Constant boundary"))

@inline function _boundary_read(space::Lattice, u, K)
    J, out = _boundary_lookup(space, K)
    out == 0 && return _read(u, J)
    return _constant_read(space.boundaries, out, u)
end

# The same for a padded state: `K` is unpadded, the state is offset by `shift`.
@inline function _boundary_read_padded(space::Lattice, u, K, shift)
    J, out = _boundary_lookup(space, K)
    out == 0 && return _read(u, J + shift)
    return _constant_read(space.boundaries, out, u)
end

"""
    GraphNeighbourStates

The vector of neighbour states a rule receives on a [`GraphSpace`](@ref), in
the order of the graph's neighbour list for the cell. It does not copy the
state.
"""
struct GraphNeighbourStates{T,U} <: AbstractVector{T}
    u::U
    idx::Vector{Int}
end
GraphNeighbourStates(u, idx) = GraphNeighbourStates{typeof(_read(u, 1)),typeof(u)}(u, idx)
Base.size(n::GraphNeighbourStates) = (length(n.idx),)
Base.@propagate_inbounds Base.getindex(n::GraphNeighbourStates, k::Int) = _read(n.u,
                                                                                n.idx[k])

function _neighbour_states(space::GraphSpace, u, i, offs)
    return GraphNeighbourStates(u, space.neighbours[i])
end
@inline function _neighbour_states(space::Lattice, u, I, offs)
    return map(o -> _boundary_read(space, u, I + o), offs)
end
_offsets_of(n::Offsets) = n.offsets
_offsets_of(::GraphNeighbours) = ()

# ------------------------------------------------------------------ plain (unpadded) states

mutable struct _PlainState{U,F}
    cur::U
    next::U
    fieldvals::F
end

function _field_buffers(prob)
    return map(rf -> zeros(eltype(rf.weights), size(prob.space)), prob.fields)
end

function _plain_state(prob)
    return _PlainState(_copy_state(prob.u0), _copy_state(prob.u0), _field_buffers(prob))
end

allocate_state(::ReferenceBackend, prob, scheme) = _plain_state(prob)

function _reference_fields!(st, prob)
    foreach(values(prob.fields), values(st.fieldvals)) do rf, out
        return _compute_field_reference!(out, rf, prob.space, st.cur)
    end
    return nothing
end

function _env(prob, params, fieldvals)
    return _Env(params, prob.covariates, fieldvals, prob.mask, prob.seed)
end

# One gated synchronous sweep from `cur` into `next`, with boundary semantics on every access.
function _plain_sweep!(next, cur, prob, env, t, gate)
    space = prob.space
    offs = _offsets_of(prob.neighbours)
    for (id, I) in enumerate(cells(space))
        ctx = _context(env, I, t, id)
        c = _read(cur, I)
        if _masked(env.mask, I) || !_gate_open(gate, ctx, I)
            _write!(next, I, c)
        else
            nbrs = _neighbour_states(space, cur, I, offs)
            _write!(next, I, _convert_cell(next, local_rule(prob.rule, c, nbrs, ctx)))
        end
    end
    return nothing
end

"""
    CellularAutomataCore.update_cell(prob, u, I, t, p=prob.p)

The new state of cell `I` when the whole state `u` (in the layout of
`prob.u0`) is updated synchronously at time `t` with parameters `p`, without
changing `u`: the rule applied to the cell and its neighbours, read with the
boundary conditions of the space, with the covariates of the problem and the
same random numbers as every backend. A masked cell keeps its state. `I` is a
`CartesianIndex`, a tuple or a linear index on a lattice, and a vertex on a
[`GraphSpace`](@ref).

A building block for backends that run their own engine. Problems with kernel
fields, [`BlockRule`](@ref)s and [`EventRule`](@ref)s are not supported.
"""
function update_cell(prob, u, I, t::Integer, p=prob.p)
    isempty(prob.fields) ||
        throw(ArgumentError("update_cell does not compute kernel fields; use ArrayBackend"))
    prob.rule isa Union{BlockRule,EventRule} &&
        throw(ArgumentError("update_cell needs a local rule, not $(nameof(typeof(prob.rule)))"))
    space = prob.space
    J = _cell_index(space, I)
    c = _read(u, J)
    _masked(prob.mask, J) && return c
    env = _Env(p, prob.covariates, (;), prob.mask, prob.seed)
    ctx = _context(env, J, t, _linear_index(space, J))
    nbrs = _neighbour_states(space, u, J, _offsets_of(prob.neighbours))
    return _convert_cell(u, local_rule(prob.rule, c, nbrs, ctx))
end

function swap!(st::_PlainState)
    st.cur, st.next = st.next, st.cur
    return nothing
end

function _advance_plain!(st::_PlainState, prob, scheme::_GatedScheme, params, t)
    _reference_fields!(st, prob)
    env = _env(prob, params, st.fieldvals)
    for gate in _gates(scheme)
        _plain_sweep!(st.next, st.cur, prob, env, t, gate)
        swap!(st)
    end
    return nothing
end

# Random-sequential: every cell once, in an order drawn from the seed, updating in place.
function _advance_plain!(st::_PlainState, prob, ::RandomSequential, params, t)
    _reference_fields!(st, prob)
    env = _env(prob, params, st.fieldvals)
    space = prob.space
    all_cells = vec(collect(cells(space)))
    keys = [_uniform(prob.seed, id, t, 0xffffffff) for id in eachindex(all_cells)]
    offs = _offsets_of(prob.neighbours)
    u = st.cur
    for id in sortperm(keys)
        I = all_cells[id]
        _masked(env.mask, I) && continue
        ctx = _context(env, I, t, id)
        nbrs = _neighbour_states(space, u, I, offs)
        _write!(u, I, _convert_cell(u, local_rule(prob.rule, _read(u, I), nbrs, ctx)))
    end
    return nothing
end

# Margolus: 2 × 2 blocks whose partition shifts by one cell between even and odd steps.
function _advance_plain!(st::_PlainState, prob, ::Margolus, params, t)
    _reference_fields!(st, prob)
    env = _env(prob, params, st.fieldvals)
    _margolus_sweep!(st.cur, prob.rule.f, env, size(prob.space), t)
    return nothing
end

# The block loop, behind a function barrier so that it compiles for the concrete state and rule types.
function _margolus_sweep!(u, f::F, env, (n1, n2), t) where {F}
    lin = LinearIndices((n1, n2))
    off = isodd(t) ? 1 : 0
    for bj in 0:(n2 ÷ 2 - 1), bi in 0:(n1 ÷ 2 - 1)
        i1, j1 = mod1(2bi + 1 + off, n1), mod1(2bj + 1 + off, n2)
        i2, j2 = mod1(i1 + 1, n1), mod1(j1 + 1, n2)
        idx = (CartesianIndex(i1, j1), CartesianIndex(i2, j1), CartesianIndex(i1, j2),
               CartesianIndex(i2, j2))
        (_masked(env.mask, idx[1]) || _masked(env.mask, idx[2]) ||
         _masked(env.mask, idx[3]) || _masked(env.mask, idx[4])) && continue
        block = (_read(u, idx[1]), _read(u, idx[2]), _read(u, idx[3]), _read(u, idx[4]))
        new = f(block, _context(env, idx[1], t, lin[idx[1]]))
        for k in 1:4                                   # unrolled by the compiler: a 4-tuple
            _write!(u, idx[k], _convert_cell(u, new[k]))
        end
    end
    return nothing
end

function advance!(::ReferenceBackend, st::_PlainState, prob, scheme, params, t)
    return _advance_plain!(st, prob, scheme, params, t)
end

current_state(::AbstractBackend, st::_PlainState) = _copy_state(st.cur)
function set_state!(::AbstractBackend, st::_PlainState, u)
    _copy_into!(st.cur, u)
    return nothing
end
read_cell(::AbstractBackend, st::_PlainState, I) = _read(st.cur, I)
function write_cell!(::AbstractBackend, st::_PlainState, I, v)
    return _write!(st.cur, I, _convert_cell(st.cur, v))
end

_copy_into!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
function _copy_into!(dst::LayeredState, src::LayeredState)
    return foreach(copyto!, values(dst), values(_ordered(dst, src)))
end

# ------------------------------------------------------------------ array backend (halo)

mutable struct _HaloState{U,N,F,C}
    cur::U
    next::U
    halo::Int
    dims::NTuple{N,Int}
    shift::CartesianIndex{N}
    fieldvals::F
    fftcaches::C
end

_padded_similar(u::AbstractArray, dims) = similar(u, dims)
_padded_similar(u::LayeredState, dims) = map(a -> similar(a, dims), u)

_halo_width(prob) = max(_halo(prob.neighbours), _halo(prob.fields), 1)
_halo(::GraphNeighbours) = 0

function allocate_state(b::ArrayBackend, prob, scheme)
    (prob.space isa GraphSpace || !(scheme isa _GatedScheme)) && return _plain_state(prob)
    h = _halo_width(prob)
    dims = size(prob.space)
    N = length(dims)
    shift = CartesianIndex(ntuple(_ -> h, N))
    cur = _padded_similar(prob.u0, dims .+ 2h)
    next = _padded_similar(prob.u0, dims .+ 2h)
    for I in CartesianIndices(dims)
        _write!(cur, I + shift, _read(prob.u0, I))
    end
    fieldvals = _field_buffers(prob)
    caches = map(rf -> rf.fft ? _fft_cache(rf, dims) : nothing, prob.fields)
    return _HaloState(cur, next, h, dims, shift, fieldvals, caches)
end

# Refresh the halo of the current state from the boundary conditions.
function _fill_halo!(st::_HaloState{U,N}, space::Lattice{N}) where {U,N}
    h = st.halo
    full = ntuple(i -> 1:(st.dims[i] + 2h), Val(N))
    for d in 1:N, side in (1, 2)
        r = side == 1 ? (1:h) : ((st.dims[d] + h + 1):(st.dims[d] + 2h))
        _fill_halo_slab!(st.cur, space,
                         CartesianIndices(ntuple(i -> i == d ? r : full[i],
                                                 Val(N))), st.shift)
    end
    return nothing
end
# A function barrier, so the loop over a slab of the halo is compiled for its index type.
function _fill_halo_slab!(cur, space, slab, shift)
    for P in slab
        _write!(cur, P, _boundary_read_padded(space, cur, P - shift, shift))
    end
    return nothing
end

@inline function _update_cell!(next, cur, rule, offs::Tuple, shift, lin, env, t, gate, I)
    P = I + shift
    ctx = _context(env, I, t, @inbounds(lin[I]))
    c = _read(cur, P)
    if _masked(env.mask, I) || !_gate_open(gate, ctx, I)
        _write!(next, P, c)
    else
        nbrs = map(o -> _read(cur, P + o), offs)
        _write!(next, P, _convert_cell(next, local_rule(rule, c, nbrs, ctx)))
    end
    return nothing
end

function _sweep_lattice!(next, cur, rule, offs, shift, dims, env, t, gate, threaded)
    lin = LinearIndices(dims)
    if threaded
        # One task per contiguous slab along the last axis: each thread works through
        # memory-contiguous blocks instead of a flat split of the index space.
        nlast = last(dims)
        nchunks = min(nlast, 4 * Threads.nthreads())
        Threads.@threads for c in 1:nchunks
            lo = div((c - 1) * nlast, nchunks) + 1
            hi = div(c * nlast, nchunks)
            for I in
                CartesianIndices(ntuple(d -> d == length(dims) ? (lo:hi) : (1:dims[d]),
                                        length(dims)))
                _update_cell!(next, cur, rule, offs, shift, lin, env, t, gate, I)
            end
        end
    else
        for I in CartesianIndices(dims)
            _update_cell!(next, cur, rule, offs, shift, lin, env, t, gate, I)
        end
    end
    return nothing
end

function _array_fields!(st::_HaloState, prob, threaded)
    foreach(values(prob.fields), values(st.fieldvals),
            values(st.fftcaches)) do rf, out, cache
        if rf.fft
            _compute_field_fft!(out, rf, _source_array(rf, st.cur, st.shift, st.dims),
                                cache)
        else
            _compute_field_padded!(out, rf, st.cur, st.shift, st.dims, threaded)
        end
    end
    return nothing
end

function advance!(b::ArrayBackend, st::_HaloState, prob, scheme::_GatedScheme, params, t)
    _fill_halo!(st, prob.space)
    _array_fields!(st, prob, b.threaded)
    env = _env(prob, params, st.fieldvals)
    offs = prob.neighbours.offsets
    for (k, gate) in enumerate(_gates(scheme))
        k > 1 && _fill_halo!(st, prob.space)
        _sweep_lattice!(st.next, st.cur, prob.rule, offs, st.shift, st.dims, env, t, gate,
                        b.threaded)
        swap!(st)
    end
    return nothing
end
function advance!(::ArrayBackend, st::_PlainState, prob, scheme, params, t)
    return _advance_plain!(st, prob, scheme,
                           params, t)
end

function swap!(st::_HaloState)
    st.cur, st.next = st.next, st.cur
    return nothing
end

function _interior(st::_HaloState{U,N}) where {U,N}
    return ntuple(i -> (st.halo + 1):(st.halo + st.dims[i]), N)
end

function current_state(::ArrayBackend, st::_HaloState)
    inner = _interior(st)
    return st.cur isa AbstractArray ? st.cur[inner...] : map(a -> a[inner...], st.cur)
end
function set_state!(::ArrayBackend, st::_HaloState, u)
    for I in CartesianIndices(st.dims)
        _write!(st.cur, I + st.shift, _read(u, I))
    end
    return nothing
end
read_cell(::ArrayBackend, st::_HaloState, I) = _read(st.cur, CartesianIndex(I) + st.shift)
function write_cell!(::ArrayBackend, st::_HaloState, I, v)
    return _write!(st.cur, CartesianIndex(I) + st.shift, _convert_cell(st.cur, v))
end
