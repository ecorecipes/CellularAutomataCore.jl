# Event rules: a cell's possible transitions and their rates, the continuous-time Markov
# chain on the lattice that plant-disease and pest models are often written as. Solved
# exactly, event by event, with the Gillespie scheme (src/event_engine.jl), or in discrete
# time with any synchronous scheme, each cell firing with probability 1 - exp(-rate).

"""
    EventRule(f, neighbourhood=nothing; states=nothing)

A rule given as rates: `f(cell, nbrs, ctx)` returns the cell's possible
transitions as a tuple (or vector) of `rate => new_state` pairs, an empty tuple
when it cannot change. The cell leaves its state at the total rate and moves to
each `new_state` with probability proportional to its rate: a continuous-time
Markov chain on the lattice. Rates must be nonnegative and finite and may read
`ctx.p`, covariates and the neighbours' states; they must not draw random
numbers.

Solve it exactly with the [`Gillespie`](@ref) scheme, with real-valued time, or
in discrete time with [`Synchronous`](@ref) (or [`Stochastic`](@ref) or
[`Checkerboard`](@ref)), where in each step a cell with total rate `a` changes
with probability `1 - exp(-a)`, using draws `k = 1` and `k = 2` of
[`uniform`](@ref). `states`, if given, are the allowed values of the initial
state.

```jldoctest
julia> sis = EventRule(VonNeumann(1)) do cell, nbrs, ctx
           cell == 0 ? (0.5 * count(==(1), nbrs) => 1,) : (1.0 => 0,)
       end;

julia> sis.f(0, (1, 1, 0, 0), nothing)
(1.0 => 1,)
```
"""
struct EventRule{F,N,S} <: AbstractRule
    f::F
    neighbourhood::N
    states::S
end
function EventRule(f, neighbourhood::Union{Nothing,AbstractNeighbourhood}=nothing;
                   states=nothing)
    return EventRule(f, neighbourhood, states)
end

neighbourhood(r::EventRule) = r.neighbourhood
function Adapt.adapt_structure(to, r::EventRule)
    return EventRule(Adapt.adapt(to, r.f), r.neighbourhood, nothing)
end
function validate_state(r::EventRule, u0)
    r.states === nothing && return nothing
    return _check_values(r, u0, r.states, "allowed states are $(collect(r.states))")
end
function Base.show(io::IO, r::EventRule)
    name = string(nameof(r.f))
    print(io, "EventRule(", startswith(name, '#') ? "<anonymous function>" : name)
    r.neighbourhood === nothing || print(io, ", ", r.neighbourhood)
    return print(io, ")")
end

# The total rate, in the precision of the rates (`false` is the additive identity of every
# number type, so Float32 rates stay Float32 on devices without double precision).
@inline _total_rate(trs) = foldl((s, tr) -> s + first(tr), trs; init=false)

# The transition that the point `x` in [0, total) falls in; rounding past the end picks the
# last transition with a positive rate.
@inline _pick(::Tuple{}, x, fallback) = fallback
@inline function _pick(trs::Tuple, x, fallback)
    r, s = first(trs)
    r > 0 && x < r && return s
    return _pick(Base.tail(trs), x - r, r > 0 ? s : fallback)
end
function _pick(trs, x, fallback)
    for (r, s) in trs
        r > 0 || continue
        x < r && return s
        x -= r
        fallback = s
    end
    return fallback
end

@inline _draw_like(::Float32, ctx, k) = uniform(Float32, ctx, k)
@inline _draw_like(_, ctx, k) = uniform(ctx, k)

# Discrete time: fire with probability 1 - exp(-a) in one step.
@inline function local_rule(r::EventRule, cell, nbrs, ctx)
    trs = r.f(cell, nbrs, ctx)
    a = _total_rate(trs)
    a > 0 || return cell
    _draw_like(a, ctx, 1) < -expm1(-a) || return cell
    return _pick(trs, _draw_like(a, ctx, 2) * a, cell)
end

"""
    Gillespie(method=:next_reaction)

Exact, event-by-event simulation of an [`EventRule`](@ref) in continuous time
(the stochastic simulation algorithm), on the CPU backends
([`ArrayBackend`](@ref), [`ReferenceBackend`](@ref)). Time is real: `tspan` may
be `(0.0, 12.5)`, `saveat` takes real times, `step!` advances one event, and
the solution's `t` is a `Vector{Float64}`.

- `:next_reaction`: each cell has its own exponential clock, kept in an indexed
  heap; after an event only the cell and the cells whose neighbourhoods contain
  it are recomputed (the modified next-reaction method, with Gibson–Bruck reuse
  of waiting times). The cost of an event does not grow with the lattice.
- `:direct`: the direct method, summing all rates at every event; slow, kept as
  a reference.
- `:composition_rejection`: the direct method with composition–rejection
  selection [Slepoy2008](@cite): cells are grouped by the binary order of
  magnitude of their rates, an event picks a group by its total rate and then a
  member by rejection, so its cost does not grow with the lattice either.

Random numbers are counter-based, as everywhere in the package: the
next-reaction method draws cell `i`'s `n`-th waiting time from `(seed, i, n)`,
so runs are reproducible. The methods sample the same process but consume draws
differently, so they agree in distribution, not path by path.

Rates are recomputed when a callback changes the state or `integ.p`, and at
every whole time when a covariate has a time axis (the rule sees `ctx.t`, the
integer part of the time). [`KernelField`](@ref)s are kept up to date event by
event: when a cell changes, the cells that read it gain its weighted change and
their rates are recomputed, so an event costs time in proportion to the
kernel's size; the fields are recomputed from the state after every `n` events
on `n` cells, so rounding does not accumulate.
"""
struct Gillespie <: AbstractScheme
    method::Symbol
    function Gillespie(method::Symbol=:next_reaction)
        method in (:next_reaction, :direct, :composition_rejection) ||
            throw(ArgumentError("Gillespie method must be :next_reaction, :direct or :composition_rejection, got :$method"))
        return new(method)
    end
end
Base.show(io::IO, g::Gillespie) = print(io, "Gillespie(:", g.method, ")")

# Whether a scheme runs in continuous time (real-valued tspan and saving times).
_continuous(::AbstractScheme) = false
_continuous(::Gillespie) = true
