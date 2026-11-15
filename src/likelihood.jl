# Exact likelihoods of observed paths. In a synchronous step every cell draws its next state
# independently given the current state, so the probability of a whole step is the product over
# cells of each cell's local transition probability, and the probability of a path is the
# product over its steps. A ProbabilisticRule states those local probabilities once; the same
# outcomes drive simulation (one draw per cell, by inversion) and the likelihood, so the two
# cannot disagree.

"""
    ProbabilisticRule(f, neighbourhood=nothing; states=nothing)

A stochastic rule given by its transition probabilities. `f(cell, nbrs, ctx)`
returns the possible next states of a cell with their probabilities, as a tuple
(or vector) of pairs `state => probability`, for example
`(0 => 1 - q, 1 => q)`. The probabilities must be nonnegative and sum to one;
a state may appear more than once, and its probabilities add.

Simulation draws one number per cell and step, [`uniform`](@ref)`(ctx, 1)`,
and picks the outcome whose cumulative probability first exceeds it (with
`Float32` probabilities the draw is `uniform(Float32, ctx, 1)`). `f` must not
draw random numbers itself: the probabilities are a function of the states,
covariates, fields, parameters and time only. The same outcomes give
[`transition_probability`](@ref), and so the exact likelihood of an observed
path, [`path_logpdf`](@ref).

`neighbourhood` and `states` are as for [`LocalRule`](@ref).

```jldoctest
julia> si = ProbabilisticRule(VonNeumann(1)) do cell, nbrs, ctx
           cell == 1 && return (1 => 1.0,)
           q = 1 - (1 - ctx.p.beta)^count(==(1), nbrs)
           return (0 => 1 - q, 1 => q)
       end;

julia> u0 = zeros(Int8, 5, 5); u0[3, 3] = 1;

julia> prob = CAProblem(si, u0, 3, (beta=0.3,); seed=1);

julia> path_logpdf(prob, solve(prob).u) < 0
true
```
"""
struct ProbabilisticRule{F,N,S} <: AbstractRule
    f::F
    neighbourhood::N
    states::S
end
function ProbabilisticRule(f, neighbourhood::Union{Nothing,AbstractNeighbourhood}=nothing;
                           states=nothing)
    return ProbabilisticRule(f, neighbourhood, states)
end

function Adapt.adapt_structure(to, r::ProbabilisticRule)
    return ProbabilisticRule(Adapt.adapt(to, r.f), r.neighbourhood, nothing)
end
neighbourhood(r::ProbabilisticRule) = r.neighbourhood
function validate_state(r::ProbabilisticRule, u0)
    r.states === nothing && return nothing
    return _check_values(r, u0, r.states, "allowed states are $(collect(r.states))")
end
function Base.show(io::IO, r::ProbabilisticRule)
    name = string(nameof(r.f))
    print(io, "ProbabilisticRule(", startswith(name, '#') ? "<anonymous function>" : name)
    r.neighbourhood === nothing || print(io, ", ", r.neighbourhood)
    return print(io, ")")
end

const _OUTCOME_DRAW = 1

@inline _outcome_draw(ctx, ::Type{Float32}) = uniform(Float32, ctx, _OUTCOME_DRAW)
@inline _outcome_draw(ctx, ::Type) = uniform(ctx, _OUTCOME_DRAW)

# Inversion over a tuple of outcomes, unrolled so that it runs inside GPU kernels. The fallback
# is the last outcome with positive probability, taken when rounding leaves the draw above the
# cumulative sum: an outcome of probability zero is never chosen.
@inline _pick(::Tuple{}, u, acc, fallback) = fallback
@inline function _pick(outs::Tuple, u, acc, fallback)
    o = first(outs)
    p = last(o)
    acc += p
    p > 0 && u < acc && return first(o)
    return _pick(Base.tail(outs), u, acc, p > 0 ? first(o) : fallback)
end

@inline function _sample_outcome(outs::Tuple, ctx)
    T = typeof(last(first(outs)))
    return _pick(outs, _outcome_draw(ctx, T), zero(T), first(first(outs)))
end
function _sample_outcome(outs::AbstractVector{<:Pair}, ctx)
    isempty(outs) && throw(ArgumentError("a ProbabilisticRule returned no outcomes"))
    T = typeof(last(first(outs)))
    u = _outcome_draw(ctx, T)
    acc = zero(T)
    fallback = first(first(outs))
    for o in outs
        p = last(o)
        acc += p
        p > 0 && u < acc && return first(o)
        p > 0 && (fallback = first(o))
    end
    return fallback
end

@inline local_rule(r::ProbabilisticRule, cell, nbrs, ctx) = _sample_outcome(r.f(cell, nbrs,
                                                                                ctx), ctx)

"""
    transition_probability(rule, next, cell, nbrs, ctx) -> Real

The probability that `rule` gives a cell the state `next`, when the cell is in
state `cell`, its neighbours in states `nbrs` and its context is `ctx` (the
arguments of [`local_rule`](@ref)). Defined for [`ProbabilisticRule`](@ref)
(the sum of the probabilities of the outcomes equal to `next`) and for the
deterministic rules [`Wolfram`](@ref), [`Totalistic`](@ref), [`Life`](@ref)
and [`Generations`](@ref) (one or zero). A [`LocalRule`](@ref) draws from an
unstated distribution and has none: write it as a `ProbabilisticRule`. Other
rule types may add a method; [`path_logpdf`](@ref) uses it.
"""
function transition_probability(rule::AbstractRule, next, cell, nbrs, ctx)
    return throw(NoLikelihoodError(_rule_name(rule),
                                   "the rule does not state its transition probabilities; " *
                                   "write it as a ProbabilisticRule, or add a method to transition_probability"))
end
function transition_probability(rule::ProbabilisticRule, next, cell, nbrs, ctx)
    return _outcome_probability(rule.f(cell, nbrs, ctx), next, identity)
end

const _DeterministicRule = Union{Wolfram,Totalistic,Life,Generations}

function transition_probability(rule::_DeterministicRule, next, cell, nbrs, ctx)
    return local_rule(rule, cell, nbrs, ctx) == next ? 1.0 : 0.0
end

# Internal form: outcomes are converted to the state's element type(s) before comparison.
function _transition_probability(rule::AbstractRule, conv, next, cell, nbrs, ctx)
    return transition_probability(rule, next, cell, nbrs, ctx)
end
function _transition_probability(rule::ProbabilisticRule, conv, next, cell, nbrs, ctx)
    return _outcome_probability(rule.f(cell, nbrs, ctx), next, conv)
end
function _transition_probability(rule::_DeterministicRule, conv, next, cell, nbrs, ctx)
    return conv(local_rule(rule, cell, nbrs, ctx)) == next ? 1.0 : 0.0
end

_probability_type(p::AbstractFloat) = typeof(p)
_probability_type(p) = Float64

function _outcome_probability(outs, next, conv)
    ok = (outs isa Tuple || outs isa AbstractVector) && !isempty(outs) &&
         all(o -> o isa Pair, outs)
    ok ||
        throw(ArgumentError("a ProbabilisticRule must return a nonempty tuple or vector of state => probability pairs, got $outs"))
    total = 0.0
    q = 0.0
    for o in outs
        p = Float64(last(o))
        p >= 0 ||
            throw(ArgumentError("a ProbabilisticRule returned a negative probability: $outs"))
        total += p
        conv(first(o)) == next && (q += p)
    end
    tol = sqrt(eps(_probability_type(last(first(outs)))))
    abs(total - 1) <= tol ||
        throw(ArgumentError("the probabilities a ProbabilisticRule returned sum to $total, not 1: $outs"))
    return q
end

# ------------------------------------------------------------------ path likelihoods

_scheme_probability(::Synchronous, q, unchanged::Bool) = q
_scheme_probability(s::Stochastic, q, unchanged::Bool) = (1 - s.p) * unchanged + s.p * q

function _check_likelihood(prob, alg)
    if !(alg isa Union{Synchronous,Stochastic})
        why = alg isa Checkerboard ? "its half-steps are not observed" :
              "cells do not update independently given the state"
        throw(NoLikelihoodError(sprint(show, alg),
                                why *
                                "; only Synchronous() and Stochastic(p) give a product over cells"))
    end
    prob.rule isa Union{BlockRule,EventRule} &&
        throw(NoLikelihoodError(_rule_name(prob.rule),
                                "only rules with a local rule have transition probabilities per cell"))
    haskey(prob.kwargs, :callback) &&
        throw(NoLikelihoodError("a problem with callbacks",
                                "callbacks change the state outside the rule, and the likelihood covers the rule only"))
    return nothing
end

function _check_path_state(prob, u, what)
    _state_size(u) == _state_size(prob.u0) ||
        throw(StateShapeError(what, _state_size(prob.u0), _state_size(u)))
    return nothing
end

function _likelihood_fields(prob, u)
    return map(prob.fields) do rf
        out = zeros(eltype(rf.weights), size(prob.space))
        return _compute_field_reference!(out, rf, prob.space, u)
    end
end

"""
    transition_logpdf(prob, u, unext, t=prob.tspan[1]; p=prob.p, alg=Synchronous()) -> Float64

The log-probability that one step of `prob`'s rule, from the state `u` at time
`t` with parameters `p`, gives the state `unext`: the sum over cells of the log
of [`transition_probability`](@ref), with the boundaries, covariates, kernel
fields and mask of the problem. A masked cell must keep its state. With
`alg = Stochastic(q)` each cell updates with probability `q` and otherwise keeps
its state, so its probability is `(1 - q) [unext = u] + q P`. Returns `-Inf`
for an impossible step.

Only [`Synchronous`](@ref) and [`Stochastic`](@ref) steps factorise over cells;
other schemes, [`BlockRule`](@ref)s, [`EventRule`](@ref)s and problems with
callbacks raise [`NoLikelihoodError`](@ref).
"""
function transition_logpdf(prob::CAProblem, u, unext, t::Integer=prob.tspan[1];
                           p=prob.p, alg::AbstractScheme=Synchronous())
    _check_likelihood(prob, alg)
    _check_path_state(prob, u, "the state")
    _check_path_state(prob, unext, "the next state")
    return _transition_logpdf(prob, u, unext, Int(t), p, alg)
end

function _transition_logpdf(prob, u, unext, t, p, alg)
    space = prob.space
    env = _Env(p, prob.covariates, _likelihood_fields(prob, u), prob.mask, prob.seed)
    offs = _offsets_of(prob.neighbours)
    conv = v -> _convert_cell(u, v)
    ll = 0.0
    for (id, I) in enumerate(cells(space))
        c = _read(u, I)
        n = _read(unext, I)
        if _masked(env.mask, I)
            c == n || return -Inf
            continue
        end
        ctx = _context(env, I, t, id)
        nbrs = _neighbour_states(space, u, I, offs)
        P = _transition_probability(prob.rule, conv, n, c, nbrs, ctx)
        q = Float64(_scheme_probability(alg, P, c == n))
        q > 0 || return -Inf
        ll += log(q)
    end
    return ll
end

"""
    path_logpdf(prob, states; t0=prob.tspan[1], p=prob.p, alg=Synchronous()) -> Float64
    path_logpdf(prob, sol::CASolution; p=prob.p, alg=sol.alg) -> Float64

The exact log-likelihood of a fully observed path: the log-probability that
`prob`'s rule, started in `states[1]` at time `t0`, passes through
`states[2]`, `states[3]`, ... at the following steps. It is the sum of
[`transition_logpdf`](@ref) over consecutive pairs, conditional on the first
state. A solution must have saved every step.

The path must be observed completely, without error, at every step: with gaps or
noisy observations the states in between are latent and their probability has
to be summed over (by data augmentation or a particle filter), which this
function does not do.
"""
function path_logpdf(prob::CAProblem, states::AbstractVector; t0::Integer=prob.tspan[1],
                     p=prob.p, alg::AbstractScheme=Synchronous())
    _check_likelihood(prob, alg)
    isempty(states) && throw(ArgumentError("path_logpdf needs at least one state"))
    foreach(u -> _check_path_state(prob, u, "a state of the path"), states)
    ll = 0.0
    for k in 1:(length(states) - 1)
        ll += _transition_logpdf(prob, states[k], states[k + 1], Int(t0) + k - 1, p, alg)
        ll == -Inf && return ll
    end
    return ll
end

function path_logpdf(prob::CAProblem, sol::CASolution; p=prob.p, alg=sol.alg)
    t = sol.t
    all(==(1), diff(t)) ||
        throw(ArgumentError("path_logpdf needs every step saved; the solution was saved at $t"))
    return path_logpdf(prob, sol.u; t0=Int(first(t)), p, alg)
end
