# The Gillespie engine: exact continuous-time simulation of an EventRule on the CPU backends.
#
# :next_reaction is the modified next-reaction method (Anderson 2007) in the Gibson–Bruck form:
# every cell carries the absolute time of its next event in an indexed min-heap. When a cell's
# rate changes from a to a' at time t without the cell firing, its remaining internal time
# a (τ - t) is kept and τ becomes t + a (τ - t) / a' (held aside while a' = 0). After an event
# only the cell and its dependents (the cells whose neighbourhoods read it, from the reversed
# neighbourhood, precomputed once) are recomputed. Cell i's n-th unit-exponential waiting
# time and its n-th choice of transition come from the counter-based draw (seed, i, n, k = 0),
# so nothing depends on the order in which events are processed.
#
# :direct sums every rate at each event and scans for the cell that fires, drawing from the
# reserved cell id 0 and the event count.

mutable struct _EventState{U,I,P}
    u::U
    cells::Vector{I}
    dep_ptr::Vector{Int}      # dependents of cell j: deps[dep_ptr[j]:(dep_ptr[j + 1] - 1)]
    deps::Vector{Int}
    rates::Vector{Float64}
    next::Vector{Float64}     # next-reaction: absolute time of the next event (Inf if rate 0)
    held::Vector{Float64}     # next-reaction: remaining internal time while the rate is 0
    fired::Vector{Int}        # next-reaction: events so far, the counter of the cell's draws
    heap::Vector{Int}
    pos::Vector{Int}
    ndraws::Int               # direct: draws so far
    stale::Bool               # rates must be recomputed before the next event
    p::P                      # the parameters the rates were computed with
    slot::Int                 # the whole time the rates were computed at
end

_unit_draws(seed, id, n) = (_uniform(seed, id, n, 0), _uniform_hi(seed, id, n))
# The second uniform of a draw, from the other half of the same Philox output.
@inline function _uniform_hi(seed::UInt64, id::Integer, n::Integer)
    r = _philox_draw(seed, id, n, 0)
    bits = (UInt64(r[3]) << 32) | UInt64(r[4])
    return (bits >> 11) * 0x1.0p-53
end
@inline _exponential(u) = -log1p(-u)

# Cells whose rates depend on each cell (itself included), in compressed sparse form.
function _dependents(prob, cs)
    space = prob.space
    n = length(cs)
    lists = [Int[j] for j in 1:n]
    if space isa GraphSpace
        for i in 1:n, j in space.neighbours[i]
            push!(lists[j], i)
        end
    else
        lin = LinearIndices(size(space))
        offs = _offsets_of(prob.neighbours)
        for (i, I) in enumerate(cs), o in offs
            J, out = _boundary_lookup(space, I + o)
            out == 0 && push!(lists[lin[J]], i)
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    for j in 1:n
        unique!(sort!(lists[j]))
        ptr[j + 1] = ptr[j] + length(lists[j])
    end
    deps = Vector{Int}(undef, ptr[end] - 1)
    for j in 1:n
        copyto!(deps, ptr[j], lists[j], 1, length(lists[j]))
    end
    return ptr, deps
end

function _event_state(prob, alg::Gillespie)
    isempty(prob.fields) ||
        throw(ArgumentError("Gillespie() does not support kernel fields yet; the problem has $(keys(prob.fields))"))
    cs = vec(collect(cells(prob.space)))
    n = length(cs)
    ptr, deps = _dependents(prob, cs)
    nrm = alg.method === :next_reaction
    held = nrm ? [_exponential(_uniform(prob.seed, i, 0, 0)) for i in 1:n] : Float64[]
    return _EventState(_copy_state(prob.u0), cs, ptr, deps, zeros(n),
                       nrm ? fill(Inf, n) : Float64[], held, zeros(Int, n),
                       nrm ? collect(1:n) : Int[], nrm ? collect(1:n) : Int[], 0, true,
                       prob.p, 0)
end

allocate_state(::ArrayBackend, prob, alg::Gillespie) = _event_state(prob, alg)
allocate_state(::ReferenceBackend, prob, alg::Gillespie) = _event_state(prob, alg)
supports(::Union{ArrayBackend,ReferenceBackend}, ::Gillespie) = true

current_state(::AbstractBackend, st::_EventState) = _copy_state(st.u)
function set_state!(::AbstractBackend, st::_EventState, u)
    _copy_into!(st.u, u)
    st.stale = true
    return nothing
end
read_cell(::AbstractBackend, st::_EventState, I) = _read(st.u, I)
function write_cell!(::AbstractBackend, st::_EventState, I, v)
    _write!(st.u, I, _convert_cell(st.u, v))
    st.stale = true
    return nothing
end

# Back to the initial random streams, so that reinit! replays a run exactly.
function _restart!(st::_EventState, prob, p, t0)
    fill!(st.rates, 0.0)
    fill!(st.fired, 0)
    st.ndraws = 0
    if !isempty(st.heap)
        fill!(st.next, Inf)
        for i in eachindex(st.held)
            st.held[i] = _exponential(_uniform(prob.seed, i, 0, 0))
        end
    end
    st.stale = true
    return nothing
end

# ------------------------------------------------------------------ indexed min-heap

@inline _hless(key, a, b) = @inbounds key[a] < key[b] || (key[a] == key[b] && a < b)
@inline function _hswap!(heap, pos, i, j)
    @inbounds heap[i], heap[j] = heap[j], heap[i]
    @inbounds pos[heap[i]], pos[heap[j]] = i, j
    return nothing
end
function _sift_up!(heap, pos, key, k)
    @inbounds while k > 1
        parent = k >> 1
        _hless(key, heap[k], heap[parent]) || break
        _hswap!(heap, pos, k, parent)
        k = parent
    end
    return k
end
function _sift_down!(heap, pos, key, k)
    n = length(heap)
    @inbounds while true
        l = 2k
        l > n && break
        c = l < n && _hless(key, heap[l + 1], heap[l]) ? l + 1 : l
        _hless(key, heap[c], heap[k]) || break
        _hswap!(heap, pos, k, c)
        k = c
    end
    return k
end
function _heap_update!(st::_EventState, id)
    k = _sift_up!(st.heap, st.pos, st.next, st.pos[id])
    _sift_down!(st.heap, st.pos, st.next, k)
    return nothing
end

# ------------------------------------------------------------------ rates and events

@inline function _transitions(st, prob, env, id)
    I = st.cells[id]
    nbrs = _neighbour_states(prob.space, st.u, I, _offsets_of(prob.neighbours))
    return prob.rule.f(_read(st.u, I), nbrs, _context(env, I, st.slot, id))
end

function _cell_rate(st, prob, env, id)
    _masked(env.mask, st.cells[id]) && return 0.0
    a = Float64(_total_rate(_transitions(st, prob, env, id)))
    (a >= 0 && isfinite(a)) ||
        throw(ArgumentError("EventRule: total rate $a at cell $(st.cells[id]); rates must be nonnegative and finite"))
    return a
end

# Change cell id's rate at time t, keeping its remaining internal time (next-reaction).
function _set_rate!(st::_EventState, id, a, t)
    if isempty(st.heap)
        st.rates[id] = a
        return nothing
    end
    old = st.rates[id]
    left = old > 0 ? max(old * (st.next[id] - t), 0.0) : st.held[id]
    st.rates[id] = a
    if a > 0
        st.next[id] = t + left / a
    else
        st.held[id] = left
        st.next[id] = Inf
    end
    _heap_update!(st, id)
    return nothing
end

function _refresh!(st::_EventState, prob, p, t)
    st.p = p
    st.slot = floor(Int, t)
    env = _env(prob, p, (;))
    for id in eachindex(st.cells)
        _set_rate!(st, id, _cell_rate(st, prob, env, id), t)
    end
    st.stale = false
    return nothing
end

# The time of the next event and the cell it happens in (Inf, 0 if nothing can happen).
function _next_event(st::_EventState, prob, t)
    if !isempty(st.heap)
        id = st.heap[1]
        return st.next[id], id
    end
    total = sum(st.rates; init=0.0)
    total > 0 || return Inf, 0
    u1, u2 = _unit_draws(prob.seed, 0, st.ndraws)
    st.ndraws += 1
    x = u2 * total
    id = 0
    for i in eachindex(st.rates)
        r = st.rates[i]
        r > 0 || continue
        id = i
        x < r && break
        x -= r
    end
    return t + _exponential(u1) / total, id
end

function _fire!(st::_EventState, prob, p, id, t)
    env = _env(prob, p, (;))
    I = st.cells[id]
    trs = _transitions(st, prob, env, id)
    u = if isempty(st.heap)
        _uniform(prob.seed, 0, st.ndraws - 1, 1)
    else
        _uniform_hi(prob.seed, id, st.fired[id])
    end
    cell = _read(st.u, I)
    _write!(st.u, I, _convert_cell(st.u, _pick(trs, u * _total_rate(trs), cell)))
    if !isempty(st.heap)
        # a fresh waiting time for the cell that fired
        st.fired[id] += 1
        st.rates[id] = 0.0
        st.held[id] = _exponential(_uniform(prob.seed, id, st.fired[id], 0))
    end
    for k in st.dep_ptr[id]:(st.dep_ptr[id + 1] - 1)
        j = st.deps[k]
        _set_rate!(st, j, _cell_rate(st, prob, env, j), t)
    end
    return nothing
end

# Covariates with a time axis change the rates at every whole time.
function _time_varying(prob)
    nd = prob.space isa GraphSpace ? 1 : length(size(prob.space))
    return any(a -> ndims(a) > nd, values(prob.covariates))
end

# ------------------------------------------------------------------ stepping

function _flush_saves!(integ, upto, inclusive)
    saves = getfield(integ, :_saves)
    while !isempty(saves) && (first(saves) < upto || (inclusive && first(saves) == upto))
        s = popfirst!(saves)
        (isempty(integ.sol_t) || last(integ.sol_t) != s) || continue
        push!(integ.sol_u, _copy_state(integ.u))
        push!(integ.sol_t, s)
    end
    return nothing
end

# One event, or a stop without one at a tstop, a whole time (time-varying covariates) or the
# end of tspan, whichever comes first.
function _event_step!(integ::CAIntegrator)
    st, prob = integ.state, integ.prob
    _sync!(integ)
    (st.stale || integ.p !== st.p) && _refresh!(st, prob, integ.p, integ.t)
    τ, id = _next_event(st, prob, integ.t)
    tstops = getfield(integ, :_tstops)
    bound = Float64(prob.tspan[2])
    isempty(tstops) || (bound = min(bound, first(tstops)))
    _time_varying(prob) && (bound = min(bound, floor(integ.t) + 1))
    fire = τ <= bound
    tnew = fire ? τ : bound
    _flush_saves!(integ, tnew, false)
    if fire
        _fire!(st, prob, integ.p, id, τ)
    else
        while !isempty(tstops) && first(tstops) <= tnew
            popfirst!(tstops)
        end
        floor(Int, tnew) != st.slot && (st.stale = true)
    end
    setfield!(integ, :t, tnew)
    setfield!(integ, :_fresh, false)
    _apply_callbacks!(integ)
    _flush_saves!(integ, tnew, true)
    if (fire && integ.save_everystep) || integ.retcode == ReturnCode.Terminated
        (isempty(integ.sol_t) || last(integ.sol_t) != tnew) && _save!(integ)
    end
    return nothing
end

function CommonSolve.step!(integ::CAIntegrator{<:CAProblem,<:Gillespie}, n::Integer=1)
    for _ in 1:n
        _finished(integ) && break
        _event_step!(integ)
    end
    return integ
end

function _run!(integ::CAIntegrator{<:CAProblem,<:Gillespie})
    while !_finished(integ)
        _event_step!(integ)
    end
    return nothing
end

function SciMLBase.add_tstop!(integ::CAIntegrator{<:CAProblem,<:Gillespie}, t)
    t > integ.t || return nothing
    tstops = getfield(integ, :_tstops)
    s = Float64(t)
    k = searchsortedfirst(tstops, s)
    (k <= length(tstops) && tstops[k] == s) || insert!(tstops, k, s)
    return nothing
end
