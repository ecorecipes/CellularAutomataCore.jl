# Rules. A rule is a local rule in the sense of Ceccherini-Silberstein and Coornaert: a
# neighbourhood S and a map from the states of the centre and of S to the next state of the
# centre. Every rule implements `local_rule(rule, cell, nbrs, ctx)` and `neighbourhood(rule)`.

"""
    AbstractRule

Supertype of rules. A rule implements

- `local_rule(rule, cell, nbrs, ctx)`: the next state of a cell, given its state
  `cell`, the states `nbrs` of its neighbours (a tuple on a lattice, in the order
  of [`neighbourhood_offsets`](@ref); a vector on a [`GraphSpace`](@ref)) and a
  [`CellContext`](@ref) `ctx`;
- `neighbourhood(rule)`: the [`AbstractNeighbourhood`](@ref) it reads, or
  `nothing` for the space's [`default_neighbourhood`](@ref);

and optionally `validate_state(rule, u0)`, which checks an initial state once.
"""
abstract type AbstractRule end

"""
    local_rule(rule, cell, nbrs, ctx)

The next state of a cell under `rule`; see [`AbstractRule`](@ref). The result is
converted to the element type of the state (for a layered state, it is a
`NamedTuple` with one value per layer).
"""
function local_rule end

"""
    neighbourhood(rule)

The neighbourhood `rule` reads, or `nothing` for the default neighbourhood of
the space it runs on.
"""
function neighbourhood end

"""
    validate_state(rule, u0)

Check that every value of the initial state `u0` can be read by `rule`, raising
[`InvalidStateError`](@ref) otherwise. Called once by [`CAProblem`](@ref); the
default accepts every state.
"""
validate_state(::AbstractRule, u0) = nothing

_rule_name(r) = sprint(show, r)

function _check_values(rule, u0, allowed, msg)
    u0 isa AbstractArray ||
        throw(InvalidStateError(_rule_name(rule), nothing, nothing,
                                "this rule needs a single-layer state, not a NamedTuple of layers"))
    for I in eachindex(IndexCartesian(), u0)
        v = u0[I]
        v in allowed || throw(InvalidStateError(_rule_name(rule), I, v, msg))
    end
    return nothing
end

# ------------------------------------------------------------------ user-defined local rules

"""
    LocalRule(f, neighbourhood=nothing; states=nothing)

A rule given by a function `f(cell, nbrs, ctx)` returning the next state of a
cell (see [`AbstractRule`](@ref) for the arguments). `neighbourhood` is any
[`AbstractNeighbourhood`](@ref), or `nothing` for the space's default. If
`states` is a collection, every value of an initial state must be in it.

States can be integers, `Bool`s, `Symbol`s, isbits structs, or (for a layered
state) `NamedTuple`s with one value per layer. Random numbers come from
[`uniform`](@ref)`(ctx, k)` and [`bernoulli`](@ref)`(ctx, p, k)`.

```jldoctest
julia> majority = LocalRule(VonNeumann(1)) do cell, nbrs, ctx
           s = cell + sum(nbrs)
           s >= 3 ? one(cell) : zero(cell)
       end;

julia> u = solve(CAProblem(majority, Int8[0 1 0; 1 1 0; 0 0 0], 1)).u[end];

julia> u
3×3 Matrix{Int8}:
 0  0  0
 0  1  0
 0  0  0
```
"""
struct LocalRule{F,N,S} <: AbstractRule
    f::F
    neighbourhood::N
    states::S
end
function LocalRule(f, neighbourhood::Union{Nothing,AbstractNeighbourhood}=nothing;
                   states=nothing)
    return LocalRule(f, neighbourhood, states)
end

@inline local_rule(r::LocalRule, cell, nbrs, ctx) = r.f(cell, nbrs, ctx)
# On a device a LocalRule keeps its function and neighbourhood; `states` is only for
# validating initial states, which happens on the host.
function Adapt.adapt_structure(to, r::LocalRule)
    return LocalRule(Adapt.adapt(to, r.f), r.neighbourhood, nothing)
end
neighbourhood(r::LocalRule) = r.neighbourhood
function validate_state(r::LocalRule, u0)
    r.states === nothing && return nothing
    return _check_values(r, u0, r.states, "allowed states are $(collect(r.states))")
end
function Base.show(io::IO, r::LocalRule)
    name = string(nameof(r.f))
    print(io, "LocalRule(", startswith(name, '#') ? "<anonymous function>" : name)
    r.neighbourhood === nothing || print(io, ", ", r.neighbourhood)
    return print(io, ")")
end

# ------------------------------------------------------------------ Wolfram codes

function _digits_table(code::Integer, k::Int, n::Int, what)
    code >= 0 || throw(ArgumentError("$what code must be nonnegative, got $code"))
    big(code) < big(k)^n ||
        throw(ArgumentError("$what code $code is out of range for $k states: it must be below $k^$n"))
    k <= 256 || throw(ArgumentError("at most 256 states are supported, got $k"))
    return UInt8.(digits(big(code); base=k, pad=n))
end

"""
    Wolfram(code; states=2, radius=1)

A one-dimensional rule with Wolfram's numbering: the new state of a cell is the
digit of `code` (in base `states`) whose position is the window of states, read
left to right as a base-`states` number. `radius` is `r` for the window
`-r:r`, or a tuple `(left, right)` for an asymmetric window. `Wolfram(30)` and
`Wolfram(110)` are the elementary rules; `Wolfram(code; states=3)` is a
three-colour rule. Runs on one-dimensional lattices.

```jldoctest
julia> sol = solve(CAProblem(Wolfram(90), single_seed(7), 2));

julia> sol.u[end]'
1×7 adjoint(::Vector{Int8}) with eltype Int8:
 0  1  0  0  0  1  0
```
"""
struct Wolfram{V<:AbstractVector{UInt8}} <: AbstractRule
    states::Int
    left::Int
    right::Int
    table::V
end
Adapt.@adapt_structure Wolfram

# The code number is the table read as a base-`states` number (digit i = table[i + 1]).
_code(table, k) = foldr((d, acc) -> acc * k + d, big.(table); init=big(0))
function Wolfram(code::Integer; states::Integer=2, radius=1)
    states >= 2 || throw(ArgumentError("a rule needs at least 2 states, got $states"))
    l, r = radius isa Integer ? (radius, radius) : Tuple(radius)
    (l >= 0 && r >= 0 && l + r >= 1) ||
        throw(ArgumentError("radius must be nonnegative with at least one neighbour, got $radius"))
    table = _digits_table(code, states, states^(l + r + 1), "Wolfram")
    return Wolfram(states, l, r, table)
end

function neighbourhood(r::Wolfram)
    return Offsets(Tuple(CartesianIndex(i) for i in vcat((-r.left):-1, 1:(r.right))))
end

@inline function local_rule(r::Wolfram, cell, nbrs, ctx)
    k = r.states
    idx = 0
    for i in 1:(r.left)
        idx = idx * k + Int(nbrs[i])
    end
    idx = idx * k + Int(cell)
    for i in (r.left + 1):(r.left + r.right)
        idx = idx * k + Int(nbrs[i])
    end
    return oftype(cell, @inbounds r.table[idx + 1])
end

function validate_state(r::Wolfram, u0)
    return _check_values(r, u0, 0:(r.states - 1), "states must be in 0:$(r.states - 1)")
end

function Base.show(io::IO, r::Wolfram)
    print(io, "Wolfram(", _code(Vector(r.table), r.states))
    r.states == 2 || print(io, "; states=", r.states)
    r.left == r.right == 1 ||
        print(io, r.states == 2 ? "; " : ", ", "radius=",
              r.left == r.right ? r.left : (r.left, r.right))
    return print(io, ")")
end

"""
    Totalistic(code; states=2, radius=1)

A one-dimensional totalistic rule with Wolfram's numbering: the new state of a
cell is the digit of `code` (in base `states`) at the position given by the sum
of the states in the window `-radius:radius`, centre included. Runs on
one-dimensional lattices.
"""
struct Totalistic{V<:AbstractVector{UInt8}} <: AbstractRule
    states::Int
    radius::Int
    table::V
end
Adapt.@adapt_structure Totalistic
function Totalistic(code::Integer; states::Integer=2, radius::Integer=1)
    states >= 2 || throw(ArgumentError("a rule needs at least 2 states, got $states"))
    radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
    n = (2radius + 1) * (states - 1) + 1
    return Totalistic(states, radius,
                      _digits_table(code, states, n, "totalistic"))
end

function neighbourhood(r::Totalistic)
    return Offsets(Tuple(CartesianIndex(i) for i in (-r.radius):(r.radius) if i != 0))
end

@inline function local_rule(r::Totalistic, cell, nbrs, ctx)
    s = Int(cell)
    for x in nbrs
        s += Int(x)
    end
    return oftype(cell, @inbounds r.table[s + 1])
end

function validate_state(r::Totalistic, u0)
    return _check_values(r, u0, 0:(r.states - 1), "states must be in 0:$(r.states - 1)")
end

function Base.show(io::IO, r::Totalistic)
    return print(io, "Totalistic(", _code(Vector(r.table), r.states), "; states=", r.states,
                 ", radius=", r.radius,
                 ")")
end

# ------------------------------------------------------------------ Life-like and Generations

"""
    LIFE_PRESETS

Named [`Life`](@ref) rules: `:conway` (B3/S23), `:highlife` (B36/S23), `:seeds`
(B2/S), `:day_and_night` (B3678/S34678), `:morley` (B368/S245), `:two_by_two`
(B36/S125), `:replicator` (B1357/S1357), `:diamoeba` (B35678/S5678),
`:thirty_four` (B34/S34), `:life_without_death` (B3/S012345678) and `:maze`
(B3/S12345).
"""
const LIFE_PRESETS = Dict{Symbol,String}(:conway => "B3/S23", :highlife => "B36/S23",
                                         :seeds => "B2/S", :day_and_night => "B3678/S34678",
                                         :morley => "B368/S245", :two_by_two => "B36/S125",
                                         :replicator => "B1357/S1357",
                                         :diamoeba => "B35678/S5678",
                                         :thirty_four => "B34/S34",
                                         :life_without_death => "B3/S012345678",
                                         :maze => "B3/S12345")

"""
    GENERATIONS_PRESETS

Named [`Generations`](@ref) rules: `:brians_brain` (B2/S/C3) and `:star_wars`
(B2/S345/C4).
"""
const GENERATIONS_PRESETS = Dict{Symbol,String}(:brians_brain => "B2/S/C3",
                                                :star_wars => "B2/S345/C4")

function _mask(counts, rule)
    m = UInt64(0)
    for c in counts
        0 <= c <= 63 ||
            throw(ArgumentError("neighbour counts must be in 0:63, got $c in $rule"))
        m |= UInt64(1) << c
    end
    return m
end

_counts(m::UInt64) = [c for c in 0:63 if (m >> c) & 1 == 1]

# Counts as rule-string digits ("236"), or as comma-separated ranges ("34..45,50") when a
# count exceeds 8 and single digits would be ambiguous.
function _digitstring(m::UInt64)
    cs = _counts(m)
    all(<=(8), cs) && return join(cs)
    runs = String[]
    i = 1
    while i <= length(cs)
        j = i
        while j < length(cs) && cs[j + 1] == cs[j] + 1
            j += 1
        end
        push!(runs, j > i ? "$(cs[i])..$(cs[j])" : string(cs[i]))
        i = j + 1
    end
    return join(runs, ",")
end

_small_counts(r) = all(<=(8), _counts(r.birth)) && all(<=(8), _counts(r.survive))
# Counts as Julia source that `Life` accepts: `34:45`, or `[34:45; 50]` for several runs.
function _countsexpr(m::UInt64)
    parts = split(replace(_digitstring(m), ".." => ":"), ",")
    return length(parts) == 1 ?
           (occursin(':', parts[1]) ? parts[1] : "[" * parts[1] * "]") :
           "[" * join(parts, "; ") * "]"
end

function _parse_digits(s::AbstractString, rule::AbstractString)
    for ch in s
        '0' <= ch <= '8' ||
            throw(RuleParseError(String(rule),
                                 "neighbour counts in a rule string are digits 0 to 8, got '$ch'"))
    end
    return [ch - '0' for ch in s]
end

# Returns (birth digits, survive digits, C or nothing).
function _parse_rule(str::AbstractString)
    s = uppercase(replace(str, r"\s" => ""))
    m = match(r"^B([0-9]*)/?S([0-9]*)(?:/C?([0-9]+))?$", s)
    if m !== nothing
        c = m.captures[3] === nothing ? nothing : parse(Int, m.captures[3])
        return _parse_digits(m.captures[1], str), _parse_digits(m.captures[2], str), c
    end
    m = match(r"^([0-9]*)/([0-9]*)(?:/([0-9]+))?$", s)  # S/B or S/B/C
    if m !== nothing
        c = m.captures[3] === nothing ? nothing : parse(Int, m.captures[3])
        return _parse_digits(m.captures[2], str), _parse_digits(m.captures[1], str), c
    end
    return throw(RuleParseError(String(str),
                                "expected B/S notation such as \"B3/S23\" or S/B notation such as \"23/3\""))
end

"""
    Life(rule="B3/S23"; neighbourhood=nothing)
    Life(preset::Symbol; neighbourhood=nothing)
    Life(birth, survive; neighbourhood=nothing)

A two-state outer-totalistic ("Life-like") rule. A dead cell (`0`) is born when
the number of live neighbours (`1`) is in `birth`; a live cell survives when it
is in `survive`. The rule can be given in B/S notation (`"B36/S23"`), S/B
notation (`"23/36"`), as a name from [`LIFE_PRESETS`](@ref), or as two
collections of counts (needed for counts above 8 in large neighbourhoods, for
example `Life(34:45, 34:58; neighbourhood=Moore(5))`). The
neighbourhood defaults to the space's ([`Moore`](@ref)`(1)` on a square
lattice, [`Hexagonal`](@ref)`(1)` on a hexagonal one).

```jldoctest
julia> Life(:highlife)
Life("B36/S23")

julia> Life("23/3")
Life("B3/S23")
```
"""
struct Life{N} <: AbstractRule
    birth::UInt64
    survive::UInt64
    neighbourhood::N
end
function Life(rule::AbstractString="B3/S23"; neighbourhood=nothing)
    b, s, c = _parse_rule(rule)
    c === nothing || c == 2 ||
        throw(RuleParseError(String(rule),
                             "a Life rule has two states; use Generations for C$c"))
    return Life(_mask(b, rule), _mask(s, rule), neighbourhood)
end
function Life(preset::Symbol; neighbourhood=nothing)
    haskey(LIFE_PRESETS, preset) ||
        throw(ArgumentError("unknown Life preset :$preset; known presets are $(sort(collect(keys(LIFE_PRESETS))))"))
    return Life(LIFE_PRESETS[preset]; neighbourhood)
end
function Life(birth, survive; neighbourhood=nothing)
    return Life(_mask(birth, "Life"), _mask(survive, "Life"), neighbourhood)
end

neighbourhood(r::Life) = r.neighbourhood

@inline function _live_count(nbrs)
    n = 0
    for x in nbrs
        n += (x == 1)
    end
    return n
end

@inline function local_rule(r::Life, cell, nbrs, ctx)
    n = _live_count(nbrs)
    mask = cell == 1 ? r.survive : r.birth
    return oftype(cell, (mask >> n) & 1)
end

validate_state(r::Life, u0) = _check_values(r, u0, (0, 1), "a Life rule has states 0 and 1")

"""
    rulestring(rule)

The B/S (or B/S/C) notation of a [`Life`](@ref) or [`Generations`](@ref) rule.
Counts above 8 are written as comma-separated ranges (`"B34..45/S34..58"`),
which [`Life`](@ref)'s string parser does not read back.
"""
rulestring(r::Life) = "B" * _digitstring(r.birth) * "/S" * _digitstring(r.survive)

function Base.show(io::IO, r::Life)
    if _small_counts(r)
        print(io, "Life(", repr(rulestring(r)))
    else
        print(io, "Life(", _countsexpr(r.birth), ", ", _countsexpr(r.survive))
    end
    r.neighbourhood === nothing || print(io, "; neighbourhood=", r.neighbourhood)
    return print(io, ")")
end

"""
    Generations(rule; neighbourhood=nothing)
    Generations(preset::Symbol; neighbourhood=nothing)

A Generations rule with `C` states: `0` is dead, `1` alive and `2:C-1` dying. A
dead cell is born when its number of live neighbours is in `birth`; a live cell
stays alive when the number is in `survive` and otherwise starts dying; a dying
cell advances one state per step and dies after state `C - 1`. Written
`"B2/S/C3"` (Brian's Brain), `"B2/S345/C4"` or in S/B/C notation `"345/2/4"`;
named rules are in [`GENERATIONS_PRESETS`](@ref). With `C = 2` it is the
[`Life`](@ref) rule of the same B/S.
"""
struct Generations{N} <: AbstractRule
    birth::UInt64
    survive::UInt64
    states::Int
    neighbourhood::N
end
function Generations(rule::AbstractString; neighbourhood=nothing)
    b, s, c = _parse_rule(rule)
    c === nothing && throw(RuleParseError(String(rule),
                                          "a Generations rule needs a state count, as in \"B2/S/C3\""))
    c >= 2 ||
        throw(RuleParseError(String(rule), "a Generations rule needs at least 2 states"))
    return Generations(_mask(b, rule), _mask(s, rule), c, neighbourhood)
end
function Generations(preset::Symbol; neighbourhood=nothing)
    haskey(GENERATIONS_PRESETS, preset) ||
        throw(ArgumentError("unknown Generations preset :$preset; known presets are $(sort(collect(keys(GENERATIONS_PRESETS))))"))
    return Generations(GENERATIONS_PRESETS[preset]; neighbourhood)
end

neighbourhood(r::Generations) = r.neighbourhood

@inline function local_rule(r::Generations, cell, nbrs, ctx)
    s = Int(cell)
    if s == 0
        return oftype(cell, (r.birth >> _live_count(nbrs)) & 1)
    elseif s == 1
        (r.survive >> _live_count(nbrs)) & 1 == 1 && return cell
        return oftype(cell, r.states > 2 ? 2 : 0)
    else
        return oftype(cell, s + 1 < r.states ? s + 1 : 0)
    end
end

function validate_state(r::Generations, u0)
    return _check_values(r, u0, 0:(r.states - 1), "states must be in 0:$(r.states - 1)")
end

function rulestring(r::Generations)
    return "B" * _digitstring(r.birth) * "/S" * _digitstring(r.survive) * "/C" *
           string(r.states)
end

function Base.show(io::IO, r::Generations)
    print(io, "Generations(", repr(rulestring(r)))
    r.neighbourhood === nothing || print(io, "; neighbourhood=", r.neighbourhood)
    return print(io, ")")
end
