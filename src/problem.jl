# Problems, integrators and solutions, following the SciML interface: `CAProblem(rule, u0,
# tspan, p)`, `solve(prob, alg; saveat, callback, ...)`, `init`/`step!`/`solve!`, integrators
# with `u`, `t` and `p`, SciMLBase callbacks and ensembles. The update scheme is the
# algorithm.

"""
    CAProblem(rule, u0, tspan, p=NullParameters(); space=nothing, seed=0,
              covariates=(;), fields=(;), mask=nothing, kwargs...)

A cellular-automaton problem, in the style of SciML problems: `rule` applied to
the initial state `u0` over `tspan` (a number of steps `n`, meaning `(0, n)`, or
a tuple `(t0, t1)` of integer times; real times for an [`EventRule`](@ref)
solved in continuous time with [`Gillespie`](@ref)) with parameters `p`, which
rules read as `ctx.p`.

- `space`: an [`AbstractSpace`](@ref); by default a [`Torus`](@ref) of the size
  of `u0`.
- `seed`: the seed of the counter-based random numbers ([`uniform`](@ref)).
- `covariates`: a `NamedTuple` of arrays the size of the space, or with one
  more (time) axis, read with [`covariate`](@ref).
- `fields`: a `NamedTuple` of [`KernelField`](@ref)s, read with
  [`field`](@ref); merged with the rule's own [`fields`](@ref).
- `mask`: an array of `Bool`s the size of the space; cells where it is `false`
  never update (no-data cells, water, roads), though their neighbours still read
  them.
- `kwargs`: solver keywords stored with the problem, as SciML problems store
  them (`callback`, `saveat`, `save_everystep`, `save_start`, `save_end`,
  `backend`; see [`init`](@ref)). `solve` and `init` use them unless given the
  same keyword, except that callbacks from both are combined, the problem's
  first.

`u0` is an array the size of the space, or a `NamedTuple` of such arrays (one
per layer). The problem checks, once, that everything fits the space
([`StateShapeError`](@ref), [`IncompatibleSpaceError`](@ref)) and that the rule
can read every value of `u0` ([`InvalidStateError`](@ref)). Change a problem
with `remake`.
"""
struct CAProblem{R<:AbstractRule,U,S<:AbstractSpace,P,N,C,FS,F,M,K<:NamedTuple,T<:Real} <:
       SciMLBase.AbstractSciMLProblem
    rule::R
    u0::U
    space::S
    tspan::Tuple{T,T}
    p::P
    seed::UInt64
    neighbours::N
    covariates::C
    field_specs::FS
    fields::F
    mask::M
    kwargs::K
end

_tspan(n::Integer) = (0, Int(n))
_tspan(t::Tuple{Integer,Integer}) = (Int(t[1]), Int(t[2]))
_tspan(n::Real) = (0.0, Float64(n))
_tspan(t::Tuple{Real,Real}) = (Float64(t[1]), Float64(t[2]))
function _tspan(t)
    return throw(ArgumentError("tspan must be a number of steps or a tuple (t0, t1), got $t"))
end

_default_space(u0) = Torus(_state_size(u0))

const SOLVER_KEYWORDS = (:callback, :saveat, :save_everystep, :save_start, :save_end,
                         :backend)

function CAProblem(rule::AbstractRule, u0, tspan, p=NullParameters(); space=nothing,
                   seed::Integer=0, covariates=(;), fields=(;), mask=nothing, kwargs...)
    for k in keys(kwargs)
        k in SOLVER_KEYWORDS ||
            throw(ArgumentError("unknown keyword $k; a problem stores the solver keywords $SOLVER_KEYWORDS"))
    end
    space = space === nothing ? _default_space(u0) : space
    ts = _tspan(tspan)
    ts[2] >= ts[1] || throw(ArgumentError("tspan must not run backwards, got $ts"))
    ts isa Tuple{Int,Int} || rule isa EventRule ||
        throw(ArgumentError("a real-valued tspan $ts needs an EventRule; other rules advance in whole steps"))
    _check_shape(u0, space)
    neighbours = _resolve(neighbourhood(rule), space)
    validate_state(rule, u0)
    _check_covariates(covariates, space)
    _check_mask(mask, space)
    fields isa NamedTuple ||
        throw(ArgumentError("fields must be a NamedTuple of KernelFields"))
    specs = merge(CellularAutomataCore.fields(rule), fields)
    resolved = NamedTuple{keys(specs)}(map((k, f) -> _resolve_field(f, space, k),
                                           keys(specs),
                                           values(specs)))
    return CAProblem(rule, u0, space, ts, p, seed % UInt64, neighbours, covariates, specs,
                     resolved, mask, NamedTuple(kwargs))
end

"""
    remake(prob::CAProblem; rule, u0, tspan, p, space, seed, covariates, fields, mask,
           kwargs)

A copy of `prob` with some of its parts replaced (SciMLBase's `remake`): for
example `remake(prob; seed=2)` for another replicate, or
`remake(prob; p=(β=0.4,))` for other parameters. `kwargs` replaces the stored
solver keywords as a whole.
"""
function SciMLBase.remake(prob::CAProblem; rule=prob.rule, u0=prob.u0, tspan=prob.tspan,
                          p=prob.p, space=prob.space, seed=prob.seed,
                          covariates=prob.covariates, fields=nothing, mask=prob.mask,
                          kwargs=prob.kwargs)
    user_fields = fields === nothing ?
                  Base.structdiff(prob.field_specs,
                                  CellularAutomataCore.fields(prob.rule)) :
                  fields
    return CAProblem(rule, u0, tspan, p; space, seed, covariates, fields=user_fields, mask,
                     kwargs...)
end

function Base.show(io::IO, p::CAProblem)
    return print(io, "CAProblem(", p.rule, " on ", p.space, ", tspan=", p.tspan, ")")
end

# ------------------------------------------------------------------ solutions

"""
    CASolution

The result of `solve`, with the fields of a SciML solution: the saved states
`sol.u` at times `sol.t`, `sol.prob`, `sol.alg` (the update scheme),
`sol.retcode` (`ReturnCode.Success`, or `ReturnCode.Terminated` after
`terminate!`) and `sol.backend`. Times are integers, or `Float64` in continuous
time ([`Gillespie`](@ref)). `sol(t)` is the state saved at time `t` and
`sol[i]` is `sol.u[i]`; [`final_state`](@ref) and [`trajectory`](@ref) are
conveniences.
"""
struct CASolution{U,P,A,B,T<:Real}
    u::Vector{U}
    t::Vector{T}
    prob::P
    alg::A
    backend::B
    retcode::ReturnCode.T
end

Base.length(sol::CASolution) = length(sol.u)
Base.getindex(sol::CASolution, i) = sol.u[i]
Base.firstindex(::CASolution) = 1
Base.lastindex(sol::CASolution) = length(sol.u)
Base.iterate(sol::CASolution, i=1) = i > length(sol.u) ? nothing : (sol.u[i], i + 1)
function (sol::CASolution)(t::Real)
    i = findfirst(==(t), sol.t)
    i === nothing &&
        throw(ArgumentError("no state was saved at t = $t; saved times are $(sol.t)"))
    return sol.u[i]
end
function Base.show(io::IO, sol::CASolution)
    return print(io, "CASolution with ", length(sol.u), " saved states at t = ",
                 isempty(sol.t) ? "[]" : "$(first(sol.t)):…:$(last(sol.t))", ", retcode ",
                 sol.retcode, " (", sol.prob.rule, ")")
end

"""
    final_state(sol)

The last saved state of a solution.
"""
final_state(sol::CASolution) = sol.u[end]

"""
    trajectory(sol)

The saved states stacked into one array with time as the **last** axis: an
`n`-dimensional state gives an `(n + 1)`-dimensional array. For a layered state,
a `NamedTuple` with one such array per layer.
"""
trajectory(sol::CASolution{<:AbstractArray}) = stack(sol.u)
function trajectory(sol::CASolution{<:LayeredState})
    ks = keys(first(sol.u))
    return NamedTuple{ks}(map(k -> stack(getfield.(sol.u, k)), ks))
end

# ------------------------------------------------------------------ integrators

"""
    CAIntegrator

The stepping state of a problem, created by `init(prob, alg; ...)`, with the
interface of a SciML integrator:

- `integ.u`: the current state, in the layout of `u0`. Inside a callback's
  `affect!`, change it in place and the change is applied; elsewhere, call
  `u_modified!(integ, true)` after changing it.
- `integ.t`: the current time; `integ.p`: the parameters, which may be
  reassigned (`integ.p = ...`) to change rates from the next step on.
- `step!(integ)`, `step!(integ, n)`, `solve!(integ)`, `reinit!(integ)`,
  `terminate!(integ)`. In continuous time ([`Gillespie`](@ref)) a step is one
  event, or a stop at a time given to `add_tstop!`.
"""
mutable struct CAIntegrator{Pr<:CAProblem,A<:AbstractScheme,B<:AbstractBackend,S,V,U,C,Pa,
                            T<:Real}
    prob::Pr
    alg::A
    backend::B
    state::S
    t::T
    p::Pa
    saveat::V
    save_everystep::Bool
    save_end::Bool
    sol_u::Vector{U}
    sol_t::Vector{T}
    callbacks::C
    retcode::ReturnCode.T
    _u::U
    _fresh::Bool
    _modified::Bool
    _tstops::Vector{T}   # continuous time: pending stops, sorted
    _saves::Vector{T}    # continuous time: pending saving times, sorted
end

function Base.getproperty(integ::CAIntegrator, name::Symbol)
    name === :u || return getfield(integ, name)
    if !getfield(integ, :_fresh)
        setfield!(integ, :_u,
                  current_state(getfield(integ, :backend), getfield(integ, :state)))
        setfield!(integ, :_fresh, true)
    end
    return getfield(integ, :_u)
end
function Base.setproperty!(integ::CAIntegrator, name::Symbol, v)
    if name === :u
        _check_shape(v, integ.prob.space)
        setfield!(integ, :_u, v)
        setfield!(integ, :_fresh, true)
        setfield!(integ, :_modified, true)
        return v
    end
    return setfield!(integ, name, convert(fieldtype(typeof(integ), name), v))
end
Base.propertynames(::CAIntegrator) = (:u, :t, :p, :prob, :alg, :backend, :retcode)

function Base.show(io::IO, integ::CAIntegrator)
    return print(io, "CAIntegrator at t = ", integ.t, " (", integ.prob.rule, ", ",
                 integ.alg, ")")
end

"""
    u_modified!(integ::CAIntegrator, modified::Bool)

Tell the integrator whether `integ.u` was changed, so that the change is (or is
not) applied before the next step. Callbacks set it automatically.
"""
function SciMLBase.u_modified!(integ::CAIntegrator, modified::Bool)
    setfield!(integ, :_modified, modified)
    return nothing
end

# SciMLBase 3 calls this (its default callback initialiser does); same meaning here.
function SciMLBase.derivative_discontinuity!(integ::CAIntegrator, modified::Bool)
    return SciMLBase.u_modified!(integ, modified)
end

# Apply pending changes to integ.u to the backend. A copy that is not fresh is stale, not
# modified (the backend changed after it was taken), and is never applied.
function _sync!(integ::CAIntegrator)
    if getfield(integ, :_modified) && getfield(integ, :_fresh)
        set_state!(integ.backend, integ.state, getfield(integ, :_u))
    end
    setfield!(integ, :_modified, false)
    return nothing
end

"""
    terminate!(integ::CAIntegrator, retcode=ReturnCode.Terminated)

Stop the run after the current step, for example from a callback when an
epidemic dies out; the solution's `retcode` records it.
"""
function SciMLBase.terminate!(integ::CAIntegrator,
                              retcode::ReturnCode.T=ReturnCode.Terminated)
    integ.retcode = retcode
    return nothing
end

"""
    add_tstop!(integ::CAIntegrator, t)

For a continuous-time ([`Gillespie`](@ref)) integrator, make the run stop at
time `t` even if no event happens then, so that callbacks can act at exactly
`t` ([`PresetTimeCallback`](@ref) and [`PeriodicCallback`](@ref) do this). For
discrete schemes every whole time is already a stop, and it does nothing.
"""
SciMLBase.add_tstop!(::CAIntegrator, t) = nothing

_discrete_callbacks(::Nothing) = ()
_discrete_callbacks(cb::DiscreteCallback) = (cb,)
function _discrete_callbacks(cs::CallbackSet)
    isempty(cs.continuous_callbacks) ||
        throw(ArgumentError("continuous callbacks are not supported: cellular automata have discrete time"))
    return Tuple(cs.discrete_callbacks)
end
function _discrete_callbacks(cbs::Union{Tuple,AbstractVector})
    return Tuple(Iterators.flatten(map(_discrete_callbacks, cbs)))
end
function _discrete_callbacks(cb)
    return throw(ArgumentError("callback must be a DiscreteCallback or a CallbackSet, got $(typeof(cb))"))
end

function _should_save(integ::CAIntegrator)
    t, (t0, t1) = integ.t, integ.prob.tspan
    integ.save_end && t == t1 && return true
    s = integ.saveat
    s === nothing && return integ.save_everystep
    return s isa Real ? _on_grid(t - t0, s) : t in s
end

# Whether time offset d is a multiple of Δ (to rounding, in continuous time).
_on_grid(d::Integer, Δ::Integer) = d % Δ == 0
function _on_grid(d, Δ)
    k = round(d / Δ)
    return abs(d - k * Δ) <= 64 * eps(max(abs(float(d)), 1.0))
end

# Continuous time: the saving times after t0, in order, ending with t1 if save_end.
function _save_times(alg, saveat, (t0, t1), save_end)
    _continuous(alg) || return Int[]
    ts = if saveat === nothing
        Float64[]
    elseif saveat isa Real
        Float64[t0 + k * saveat for k in 1:floor(Int, (t1 - t0) / saveat + 1e-9)]
    else
        sort!(Float64[s for s in saveat])
    end
    filter!(s -> t0 < s <= t1, ts)
    save_end && (isempty(ts) || last(ts) != t1) && push!(ts, t1)
    return ts
end

function _save!(integ::CAIntegrator)
    push!(integ.sol_u, _copy_state(integ.u))
    push!(integ.sol_t, integ.t)
    return nothing
end

"""
    init(prob::CAProblem, alg=Synchronous(); backend=ArrayBackend(), saveat=nothing,
         save_everystep=saveat === nothing, save_start=true, save_end=true,
         callback=nothing)

Set up a problem for stepping, with SciML's saving and callback keywords:

- `saveat`: an integer `k` (save every `k` steps) or a collection of times;
  without it every step is saved unless `save_everystep=false`. The start and
  end states are saved unless `save_start`/`save_end` are `false`.
- `callback`: a `DiscreteCallback`, a `CallbackSet`, or a collection of them,
  checked in order after every step ([`PeriodicCallback`](@ref),
  [`PresetTimeCallback`](@ref) and [`SavingCallback`](@ref) build common ones).
- `backend`: the [`AbstractBackend`](@ref).

Solver keywords stored in the problem apply unless given here; callbacks from
both are combined, the problem's first.

Raises [`UnsupportedCombinationError`](@ref) if the backend cannot run the
scheme.
"""
function CommonSolve.init(prob::CAProblem, alg::AbstractScheme=Synchronous(); kwargs...)
    stored = prob.kwargs
    if haskey(stored, :callback) && haskey(kwargs, :callback) &&
       kwargs[:callback] !== nothing
        cb = (stored.callback, kwargs[:callback])
        return _init(prob, alg; stored..., kwargs..., callback=cb)
    end
    return _init(prob, alg; stored..., kwargs...)
end

function _init(prob::CAProblem, alg::AbstractScheme;
               backend::AbstractBackend=ArrayBackend(), saveat=nothing,
               save_everystep::Bool=saveat === nothing && !_continuous(alg),
               save_start::Bool=true, save_end::Bool=true, callback=nothing)
    _continuous(alg) || prob.tspan isa Tuple{Int,Int} ||
        throw(ArgumentError("$alg advances in whole steps, but tspan is $(prob.tspan); give integer times or solve with Gillespie()"))
    supports(backend, alg) ||
        throw(UnsupportedCombinationError(sprint(show, backend), sprint(show, alg)))
    _check_rule_scheme(prob.rule, alg)
    _check_scheme_space(alg, prob.space)
    check_problem(backend, prob, alg)
    saveat isa Real && saveat <= 0 &&
        throw(ArgumentError("saveat must be positive, got $saveat"))
    st = allocate_state(backend, prob, alg)
    u = current_state(backend, st)
    cbs = _discrete_callbacks(callback)
    T = _continuous(alg) ? Float64 : Int
    integ = CAIntegrator(prob, alg, backend, st, T(prob.tspan[1]), prob.p, saveat,
                         save_everystep, save_end, typeof(u)[], T[], cbs,
                         ReturnCode.Default, u, true, false, T[],
                         _save_times(alg, saveat, prob.tspan, save_end))
    for cb in cbs
        cb.initialize(cb, integ.u, integ.t, integ)
    end
    _sync!(integ)
    save_start && _save!(integ)
    return integ
end

function _finished(integ::CAIntegrator)
    return integ.t >= integ.prob.tspan[2] || integ.retcode == ReturnCode.Terminated
end

function _apply_callbacks!(integ::CAIntegrator)
    for cb in integ.callbacks
        if cb.condition(integ.u, integ.t, integ)
            setfield!(integ, :_modified, true)
            cb.affect!(integ)
            _sync!(integ)
        end
    end
    return nothing
end

function CommonSolve.step!(integ::CAIntegrator, n::Integer=1)
    for _ in 1:n
        _finished(integ) && break
        _sync!(integ)
        advance!(integ.backend, integ.state, integ.prob, integ.alg, integ.p, integ.t)
        setfield!(integ, :t, integ.t + 1)
        setfield!(integ, :_fresh, false)
        _apply_callbacks!(integ)
        (_should_save(integ) || integ.retcode == ReturnCode.Terminated) && _save!(integ)
    end
    return integ
end

_run!(integ::CAIntegrator) = step!(integ, integ.prob.tspan[2] - integ.t)

function CommonSolve.solve!(integ::CAIntegrator)
    _run!(integ)
    for cb in integ.callbacks
        cb.finalize(cb, integ.u, integ.t, integ)
    end
    integ.retcode == ReturnCode.Default && (integ.retcode = ReturnCode.Success)
    return CASolution(integ.sol_u, integ.sol_t, integ.prob, integ.alg, integ.backend,
                      integ.retcode)
end

"""
    reinit!(integ::CAIntegrator, u0=integ.prob.u0; t0=integ.prob.tspan[1])

Restart an integrator from `u0` at time `t0`, discarding its saved states and
rerunning the callbacks' `initialize`. The random numbers restart too, so
reinitialising with the same `u0` and `t0` replays the run.
"""
function SciMLBase.reinit!(integ::CAIntegrator, u0=integ.prob.u0;
                           t0::Real=integ.prob.tspan[1])
    _check_shape(u0, integ.prob.space)
    set_state!(integ.backend, integ.state, u0)
    setfield!(integ, :t, convert(typeof(integ.t), t0))
    _restart!(integ.state, integ.prob, integ.p, integ.t)
    empty!(getfield(integ, :_tstops))
    saves = getfield(integ, :_saves)
    empty!(saves)
    append!(saves,
            _save_times(integ.alg, integ.saveat, (integ.t, integ.prob.tspan[2]),
                        integ.save_end))
    setfield!(integ, :_fresh, false)
    setfield!(integ, :_modified, false)
    integ.retcode = ReturnCode.Default
    empty!(integ.sol_u)
    empty!(integ.sol_t)
    for cb in integ.callbacks
        cb.initialize(cb, integ.u, integ.t, integ)
    end
    _sync!(integ)
    _save!(integ)
    return integ
end

# Restart a backend's own streams of randomness and agents, once the state is set.
_restart!(st, prob, p, t0) = nothing

"""
    solve(prob::CAProblem, alg=Synchronous(); backend=ArrayBackend(), saveat=nothing,
          save_everystep=saveat === nothing, save_start=true, save_end=true,
          callback=nothing) -> CASolution

Run a problem to the end of its `tspan`; the keywords are those of `init`.
Ensembles use SciMLBase's `EnsembleProblem`, whose default `prob_func` here
gives each trajectory its own seed ([`seed_trajectory`](@ref)); a `prob_func`
of your own receives the problem and SciMLBase's `EnsembleContext`, as in
`(prob, ctx) -> remake(prob; seed=ctx.sim_id)`.

```jldoctest
julia> sol = solve(CAProblem(Life(), zeros(Int8, 8, 8), 10); saveat=5);

julia> sol.t
3-element Vector{Int64}:
  0
  5
 10
```
"""
function CommonSolve.solve(prob::CAProblem, alg::AbstractScheme=Synchronous(); kwargs...)
    return solve!(init(prob, alg; kwargs...))
end

# ------------------------------------------------------------------ ensembles

"""
    seed_trajectory(prob, ctx)

The default `prob_func` of an `EnsembleProblem` built on a [`CAProblem`](@ref):
trajectory `i` is `remake(prob; seed=prob.seed + i)`, or uses the per-trajectory
seed SciMLBase derives when the ensemble is solved with a `seed` keyword. Without
it every trajectory would repeat the same counter-based random numbers.
"""
function seed_trajectory(prob::CAProblem, ctx)
    seed = ctx.sim_seed === nothing ? prob.seed + UInt64(ctx.sim_id) : UInt64(ctx.sim_seed)
    return remake(prob; seed)
end

"""
    EnsembleProblem(prob::CAProblem; prob_func=seed_trajectory, kwargs...)

SciMLBase's `EnsembleProblem`, with a default `prob_func` that gives each
trajectory its own seed ([`seed_trajectory`](@ref)). Solve it with
`solve(eprob, alg, EnsembleThreads(); trajectories=n, backend=...)`; the
`prob_func(prob, ctx)` and `output_func(sol, ctx)` functions receive SciMLBase's
`EnsembleContext` (`ctx.sim_id`, `ctx.sim_seed`).
"""
function SciMLBase.EnsembleProblem(prob::CAProblem; prob_func=seed_trajectory,
                                   safetycopy::Bool=false, kwargs...)
    return invoke(SciMLBase.EnsembleProblem, Tuple{Any}, prob; prob_func, safetycopy,
                  kwargs...)
end

# ------------------------------------------------------------------ callbacks

"""
    PeriodicCallback(affect!, Δt; initial_affect=false, phase=0)

A `DiscreteCallback` running `affect!(integ)` after every step that brings the
time to `t0 + phase + k Δt` (`k ≥ 1`), and also at the start if
`initial_affect`; as in DiffEqCallbacks.jl. In continuous time
([`Gillespie`](@ref)) `Δt` and `phase` may be real, and the run stops at exactly
those times.
"""
function PeriodicCallback(affect!, Δt::Real; initial_affect::Bool=false,
                          phase::Real=0)
    Δt > 0 || throw(ArgumentError("Δt must be positive, got $Δt"))
    condition = function (u, t, integ)
        d = t - integ.prob.tspan[1] - phase
        return d > 0 && _on_grid(d, Δt)
    end
    initialize = function (c, u, t, integ)
        t0, t1 = integ.prob.tspan
        if integ.t isa AbstractFloat
            for k in 1:floor(Int, (t1 - t0 - phase) / Δt + 1e-9)
                SciMLBase.add_tstop!(integ, t0 + phase + k * Δt)
            end
        end
        if initial_affect
            u_modified!(integ, true)  # as around every other affect!, so the change is applied
            affect!(integ)
        else
            u_modified!(integ, false)
        end
        return nothing
    end
    return DiscreteCallback(condition, affect!; initialize, save_positions=(false, false))
end

"""
    PresetTimeCallback(tstops, affect!)

A `DiscreteCallback` running `affect!(integ)` after the steps that bring the
time to one of `tstops`; as in DiffEqCallbacks.jl. In continuous time
([`Gillespie`](@ref)) the run stops at exactly those times.
"""
function PresetTimeCallback(tstops, affect!)
    times = Set{Float64}(tstops)
    initialize = function (c, u, t, integ)
        foreach(s -> SciMLBase.add_tstop!(integ, s), times)
        u_modified!(integ, false)
        return nothing
    end
    return DiscreteCallback((u, t, integ) -> t in times, affect!; initialize,
                            save_positions=(false, false))
end

"""
    SavedValues(tType, sType)

Storage for [`SavingCallback`](@ref): times `t::Vector{tType}` and values
`saveval::Vector{sType}`; as in DiffEqCallbacks.jl.
"""
struct SavedValues{T,S}
    t::Vector{T}
    saveval::Vector{S}
end
SavedValues(::Type{T}, ::Type{S}) where {T,S} = SavedValues{T,S}(T[], S[])

"""
    SavingCallback(save_func, saved_values::SavedValues; saveat=nothing, save_start=true)

A `DiscreteCallback` that pushes `save_func(u, t, integ)` into `saved_values`
at the start and after every step (or every `saveat` steps, or at the times in
`saveat`): a cheap way to record a summary, such as the number of infected
cells, without saving whole states. As in DiffEqCallbacks.jl.
"""
function SavingCallback(save_func, saved::SavedValues; saveat=nothing,
                        save_start::Bool=true)
    function record!(integ)
        push!(saved.t, integ.t)
        push!(saved.saveval, save_func(integ.u, integ.t, integ))
        u_modified!(integ, false)
        return nothing
    end
    condition = function (u, t, integ)
        saveat === nothing && return true
        return saveat isa Real ? _on_grid(t - integ.prob.tspan[1], saveat) : t in saveat
    end
    initialize = function (c, u, t, integ)
        if integ.t isa AbstractFloat && saveat !== nothing
            t0, t1 = integ.prob.tspan
            times = saveat isa Real ?
                    (t0 + k * saveat for k in 1:floor(Int, (t1 - t0) / saveat + 1e-9)) :
                    saveat
            foreach(s -> SciMLBase.add_tstop!(integ, s), times)
        end
        save_start && record!(integ)
        u_modified!(integ, false)
        return nothing
    end
    return DiscreteCallback(condition, record!; initialize, save_positions=(false, false))
end
