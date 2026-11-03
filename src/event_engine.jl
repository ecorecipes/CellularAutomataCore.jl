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
#
# :composition_rejection (Slepoy, Thompson & Plimpton 2008) keeps cells with a positive rate in
# groups by the binary exponent of their rate (rates in [2^e, 2^(e+1))) with each group's total;
# an event picks a group by its total, then a member uniformly, accepted with probability
# rate / 2^(e+1), so the cost of an event does not grow with the lattice. It draws from cell id
# 0 like :direct: k = 0 for the time and the group, k = 1 for the transition, k >= 2 for the
# rejection tries.
#
# Kernel fields are kept up to date event by event: when a cell's source value changes by Δ,
# every cell that reads it through offset o (found axis by axis from the boundary conditions)
# gains w_o Δ, and those cells' rates are recomputed with the cell's dependents. Each field also
# counts, at every cell, the nonzero source values it reads (kernel weights are positive), and a
# field whose count falls to zero is set to exactly zero, so rounding never leaves a rate
# slightly negative or positive where nothing is left; the fields are recomputed from the state
# after every n events on n cells.

# Composition–rejection groups: group g holds the cells whose rates have exponent g + _CR_LO.
const _CR_LO = exponent(nextfloat(0.0)) - 1
const _CR_GROUPS = exponent(floatmax(Float64)) - _CR_LO + 1
mutable struct _Groups
    sums::Vector{Float64}           # total rate of each group
    members::Vector{Vector{Int}}
    grp::Vector{Int}                # group of each cell (0 while its rate is 0)
    gpos::Vector{Int}               # position of each cell in its group
    active::Vector{Int}             # the groups with members
    apos::Vector{Int}
    since::Int                      # events since the totals were summed afresh
end
function _Groups(n)
    return _Groups(zeros(_CR_GROUPS), [Int[] for _ in 1:_CR_GROUPS], zeros(Int, n),
                   zeros(Int, n), Int[], zeros(Int, _CR_GROUPS), 0)
end

function _group_remove!(G::_Groups, id, old)
    g = G.grp[id]
    g == 0 && return nothing
    m = G.members[g]
    k = G.gpos[id]
    last = m[end]
    m[k] = last
    G.gpos[last] = k
    pop!(m)
    G.grp[id] = 0
    if isempty(m)
        G.sums[g] = 0.0
        a = G.apos[g]
        lastg = G.active[end]
        G.active[a] = lastg
        G.apos[lastg] = a
        pop!(G.active)
        G.apos[g] = 0
    else
        G.sums[g] -= old
    end
    return nothing
end

function _group_add!(G::_Groups, id, a)
    g = exponent(a) - _CR_LO + 1
    m = G.members[g]
    if isempty(m)
        push!(G.active, g)
        G.apos[g] = length(G.active)
    end
    push!(m, id)
    G.grp[id] = g
    G.gpos[id] = length(m)
    G.sums[g] += a
    return nothing
end

function _clear_groups!(G::_Groups)
    for g in G.active
        empty!(G.members[g])
        G.sums[g] = 0.0
        G.apos[g] = 0
    end
    empty!(G.active)
    fill!(G.grp, 0)
    G.since = 0
    return nothing
end

# The group totals summed afresh from the rates, so that rounding does not accumulate.
function _resum_groups!(G::_Groups, rates)
    for g in G.active
        G.sums[g] = sum(i -> rates[i], G.members[g]; init=0.0)
    end
    G.since = 0
    return nothing
end

mutable struct _EventState{U,I,P,F,C,Z,ZC,G}
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
    fieldvals::F              # kernel field values, kept up to date event by event
    caches::C                 # FFT plans of the fields computed by FFT (nothing otherwise)
    counts::Z                 # nonzero source values each cell's field reads
    count_caches::ZC          # FFT plans for those counts (unit weights)
    mark::Vector{Int}         # stamp of the last event whose rate update included the cell
    stamp::Int
    touched::Vector{Int}      # cells whose rates the current event changes
    since::Int                # events since the fields were last computed from the state
    groups::G                 # composition–rejection groups (nothing for the other methods)
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
    cs = vec(collect(cells(prob.space)))
    n = length(cs)
    ptr, deps = _dependents(prob, cs)
    nrm = alg.method === :next_reaction
    held = nrm ? [_exponential(_uniform(prob.seed, i, 0, 0)) for i in 1:n] : Float64[]
    dims = size(prob.space)
    caches = map(rf -> rf.fft ? _fft_cache(rf, dims) : nothing, prob.fields)
    counts = map(_ -> zeros(Int32, dims), prob.fields)
    count_caches = map(rf -> rf.fft ? _fft_cache(_unit_weights(rf), dims) : nothing,
                       prob.fields)
    return _EventState(_copy_state(prob.u0), cs, ptr, deps, zeros(n),
                       nrm ? fill(Inf, n) : Float64[], held, zeros(Int, n),
                       nrm ? collect(1:n) : Int[], nrm ? collect(1:n) : Int[], 0, true,
                       prob.p, 0, _field_buffers(prob), caches, counts, count_caches,
                       zeros(Int, n), 0, Int[], 0,
                       alg.method === :composition_rejection ? _Groups(n) : nothing)
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
    st.groups === nothing || _clear_groups!(st.groups)
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
        G = st.groups
        if G !== nothing && a != st.rates[id]
            _group_remove!(G, id, st.rates[id])
            a > 0 && _group_add!(G, id, a)
        end
        st.rates[id] = a
        return nothing
    end
    old = st.rates[id]
    a > 0 && a == old && return nothing      # unchanged: keep the clock exactly as it is
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

# ------------------------------------------------------------------ kernel fields

function _unit_weights(rf::_ResolvedField)
    return _ResolvedField(rf.source, rf.offsets, ones(Float64, length(rf.weights)), rf.fft,
                          rf.radius)
end

function _compute_fields!(st::_EventState, prob)
    foreach(values(prob.fields), values(st.fieldvals), values(st.caches), values(st.counts),
            values(st.count_caches)) do rf, out, cache, cnt, ccache
        dims = size(prob.space)
        if rf.fft
            T = eltype(rf.weights)
            src = [T(rf.source(_read(st.u, I))) for I in CartesianIndices(dims)]
            _compute_field_fft!(out, rf, src, cache)
            nz = [Float64(!iszero(x)) for x in src]
            c = _compute_field_fft!(similar(nz), _unit_weights(rf), nz, ccache)
            cnt .= round.(Int32, c)
        else
            _compute_field_reference!(out, rf, prob.space, st.u)
            for I in cells(prob.space)
                k = 0
                for o in rf.offsets
                    k += !iszero(rf.source(_boundary_read(prob.space, st.u, I + o)))
                end
                cnt[I] = k
            end
        end
        for I in eachindex(out)
            iszero(cnt[I]) && (out[I] = zero(eltype(out)))
        end
    end
    st.since = 0
    return nothing
end

# The cells on one axis of length n whose reading position, at offset o, maps to index j:
# a count and up to two indices.
@inline function _axis_readers(::Periodic, j, o, n)
    return (1, mod1(j - o, n), 0)
end
@inline function _axis_readers(::Constant, j, o, n)
    i = j - o
    return 1 <= i <= n ? (1, i, 0) : (0, 0, 0)
end
@inline function _axis_readers(::Reflecting, j, o, n)
    # positions reading j are j + 2nm and 1 - j + 2nm; at most one of each family is in range
    c, a1, a2 = 0, 0, 0
    for x0 in (j - o, 1 - j - o)
        i = x0 + 2n * cld(1 - x0, 2n)
        if i <= n
            c += 1
            c == 1 ? (a1 = i) : (a2 = i)
        end
    end
    return (c, a1, a2)
end

_foreach_combination(f, ::Tuple{}, acc) = f(CartesianIndex(acc))
function _foreach_combination(f, cands::Tuple, acc)
    c, a1, a2 = first(cands)
    c >= 1 && _foreach_combination(f, Base.tail(cands), (acc..., a1))
    c >= 2 && _foreach_combination(f, Base.tail(cands), (acc..., a2))
    return nothing
end

# Call f(I) for every cell I that reads cell J through the offset o (J = I + o after the
# boundary conditions).
function _foreach_reader(f, space::Lattice{N}, J::CartesianIndex{N}, o) where {N}
    cands = ntuple(d -> _axis_readers(space.boundaries[d], J[d], o[d], space.dims[d]),
                   Val(N))
    return _foreach_combination(f, cands, ())
end

# Add a cell's change in source value to every cell that reads it, noting those cells.
function _update_fields!(st::_EventState, prob, J, old, new)
    lin = LinearIndices(size(prob.space))
    foreach(values(prob.fields), values(st.fieldvals), values(st.counts)) do rf, out, cnt
        T = eltype(rf.weights)
        s0, s1 = T(rf.source(old)), T(rf.source(new))
        Δ = s1 - s0
        iszero(Δ) && return nothing
        dn = Int32(!iszero(s1)) - Int32(!iszero(s0))
        for k in eachindex(rf.weights)
            wΔ = rf.weights[k] * Δ
            _foreach_reader(prob.space, J, rf.offsets[k]) do I
                c = cnt[I] + dn
                cnt[I] = c
                out[I] = iszero(c) ? zero(T) : out[I] + wΔ
                _touch!(st, lin[I])
                return nothing
            end
        end
        return nothing
    end
    return nothing
end

@inline function _touch!(st::_EventState, j)
    if st.mark[j] != st.stamp
        st.mark[j] = st.stamp
        push!(st.touched, j)
    end
    return nothing
end

function _refresh!(st::_EventState, prob, p, t)
    st.p = p
    st.slot = floor(Int, t)
    isempty(prob.fields) || _compute_fields!(st, prob)
    env = _env(prob, p, st.fieldvals)
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
    st.groups === nothing || return _next_event_cr(st, st.groups, prob, t)
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

function _next_event_cr(st::_EventState, G::_Groups, prob, t)
    G.since >= length(st.cells) && _resum_groups!(G, st.rates)
    total = 0.0
    for g in G.active
        total += G.sums[g]
    end
    total > 0 || return Inf, 0
    n = st.ndraws
    u1, u2 = _unit_draws(prob.seed, 0, n)
    st.ndraws += 1
    G.since += 1
    # the group, by its total (rounding past the end picks the last group)
    x = u2 * total
    g = G.active[end]
    for h in G.active
        x < G.sums[h] && (g=h; break)
        x -= G.sums[h]
    end
    # a member, uniformly, accepted with probability rate / 2^(e+1)
    m = G.members[g]
    bound = 2.0^(g + _CR_LO)
    j = 1
    while true
        r = _uniform(prob.seed, 0, n, 2j)
        id = m[min(floor(Int, r * length(m)) + 1, length(m))]
        _uniform(prob.seed, 0, n, 2j + 1) * bound < st.rates[id] &&
            return t + _exponential(u1) / total, id
        j += 1
    end
end

function _fire!(st::_EventState, prob, p, id, t)
    env = _env(prob, p, st.fieldvals)
    I = st.cells[id]
    trs = _transitions(st, prob, env, id)
    u = if isempty(st.heap)
        _uniform(prob.seed, 0, st.ndraws - 1, 1)
    else
        _uniform_hi(prob.seed, id, st.fired[id])
    end
    cell = _read(st.u, I)
    new = _convert_cell(st.u, _pick(trs, u * _total_rate(trs), cell))
    _write!(st.u, I, new)
    if !isempty(st.heap)
        # a fresh waiting time for the cell that fired
        st.fired[id] += 1
        st.rates[id] = 0.0
        st.held[id] = _exponential(_uniform(prob.seed, id, st.fired[id], 0))
    end
    if isempty(prob.fields)
        for k in st.dep_ptr[id]:(st.dep_ptr[id + 1] - 1)
            j = st.deps[k]
            _set_rate!(st, j, _cell_rate(st, prob, env, j), t)
        end
        return nothing
    end
    # the cell's dependents and the readers of its source values, each once, in id order
    st.stamp += 1
    empty!(st.touched)
    for k in st.dep_ptr[id]:(st.dep_ptr[id + 1] - 1)
        _touch!(st, st.deps[k])
    end
    _update_fields!(st, prob, I, cell, new)
    sort!(st.touched)
    for j in st.touched
        _set_rate!(st, j, _cell_rate(st, prob, env, j), t)
    end
    st.since += 1
    st.since >= length(st.cells) && (st.stale = true)
    return nothing
end

# Covariates with a time axis change the rates at every whole time.
function _time_varying(prob)
    nd = prob.space isa GraphSpace ? 1 : length(size(prob.space))
    return any(a -> ndims(a) > nd, values(prob.covariates))
end

# ------------------------------------------------------------------ backends holding agents

# Agent backends keep the event state inside their model or world. The engine finds it with
# `_events`; after an event, `_after_fire!` lets the backend copy the cell that changed to its
# state of record; and when the backend has mobile agents, the run stops at every whole time and
# `_agents_act!` runs them there, as after each step of a discrete-time scheme.
_events(backend, st) = st
_after_fire!(backend, st, id) = nothing
_mobile_agents(backend) = false
_agents_act!(backend, st, prob, p, t) = nothing

# The array whose cell type values written into the state are converted to.
_template(st) = st.cur
_template(st::_EventState) = st.u

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
    prob, backend = integ.prob, integ.backend
    _sync!(integ)
    st = _events(backend, integ.state)
    (st.stale || integ.p !== st.p) && _refresh!(st, prob, integ.p, integ.t)
    τ, id = _next_event(st, prob, integ.t)
    tstops = getfield(integ, :_tstops)
    bound = Float64(prob.tspan[2])
    isempty(tstops) || (bound = min(bound, first(tstops)))
    whole = floor(integ.t) + 1
    agents = _mobile_agents(backend)
    (agents || _time_varying(prob)) && (bound = min(bound, whole))
    fire = τ <= bound
    tnew = fire ? τ : bound
    _flush_saves!(integ, tnew, false)
    if fire
        _fire!(st, prob, integ.p, id, τ)
        _after_fire!(backend, integ.state, id)
    else
        while !isempty(tstops) && first(tstops) <= tnew
            popfirst!(tstops)
        end
        if agents && tnew == whole
            _agents_act!(backend, integ.state, prob, integ.p, tnew)
            st.stale = true
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
