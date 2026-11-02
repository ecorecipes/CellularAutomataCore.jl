# Models: a declarative description of a cellular automaton (named state variables with
# defaults and domains, named parameters with defaults, a rule, kernel fields and the
# covariates it needs), converted into a CAProblem by mapping names to values, in the style
# of ModelingToolkit's `ODEProblem(sys, u0map, tspan, pmap)`.

"""
    StateVariable(name, default; domain=nothing, description="")

A state variable of a [`CAModel`](@ref): a layer of the state, whose element type
is the type of `default`. If `domain` is a collection, every initial value must
lie in it.
"""
struct StateVariable{T}
    name::Symbol
    default::T
    domain::Any
    description::String
end
function StateVariable(name::Symbol, default; domain=nothing,
                       description::AbstractString="")
    domain === nothing || default in domain ||
        throw(ArgumentError("default $(repr(default)) of state $name is not in its domain"))
    return StateVariable(name, default, domain, String(description))
end

"""
    Parameter(name, default; description="")

A parameter of a [`CAModel`](@ref), with its default value.
"""
struct Parameter{T}
    name::Symbol
    default::T
    description::String
end
function Parameter(name::Symbol, default; description::AbstractString="")
    return Parameter(name, default,
                     String(description))
end

"""
    CAModel(rule; name=:model, states, parameters=(;), fields=(;), covariates=(),
            description="")

A declarative cellular-automaton model, turned into a problem by
`CAProblem(model, u0map, tspan, pmap; space)`.

- `states`: a `NamedTuple` of default values, one per state variable
  (`(host=Int8(1), pest=0.0f0)`), or a vector of [`StateVariable`](@ref)s. A
  model with one state variable has an array state; with several, a layered
  `NamedTuple` state.
- `parameters`: a `NamedTuple` of default values, or a vector of
  [`Parameter`](@ref)s; the rule reads them as `ctx.p`.
- `fields`: [`KernelField`](@ref)s by name; `covariates`: the names of the
  covariate layers the rule reads, which every problem must supply.

[`@camodel`](@ref) builds a model from a block of declarations.
"""
struct CAModel{R<:AbstractRule,F<:NamedTuple}
    name::Symbol
    states::Vector{StateVariable}
    parameters::Vector{Parameter}
    rule::R
    fields::F
    covariates::Vector{Symbol}
    description::String
end

_state_variables(s::NamedTuple) = StateVariable[StateVariable(k, v) for (k, v) in pairs(s)]
_state_variables(s::AbstractVector) = StateVariable[s...]
_parameters(p::NamedTuple) = Parameter[Parameter(k, v) for (k, v) in pairs(p)]
_parameters(p::AbstractVector) = Parameter[p...]

function CAModel(rule::AbstractRule; name::Symbol=:model, states, parameters=(;),
                 fields=(;),
                 covariates=(), description::AbstractString="")
    sv = _state_variables(states)
    isempty(sv) && throw(ArgumentError("a model needs at least one state variable"))
    pv = _parameters(parameters)
    allunique(s.name for s in sv) || throw(ArgumentError("state names must be unique"))
    allunique(p.name for p in pv) || throw(ArgumentError("parameter names must be unique"))
    fields isa NamedTuple ||
        throw(ArgumentError("fields must be a NamedTuple of KernelFields"))
    return CAModel(name, sv, pv, rule, fields, Symbol[covariates...], String(description))
end

"""
    states(model) -> Vector{StateVariable}
    parameters(model) -> Vector{Parameter}

The state variables and parameters of a [`CAModel`](@ref).
"""
states(m::CAModel) = m.states
@doc (@doc states) parameters(m::CAModel) = m.parameters

"""
    defaults(model) -> NamedTuple

The default values of a model's parameters, the `p` of a problem built with an
empty `pmap`.
"""
function defaults(m::CAModel)
    return NamedTuple{Tuple(p.name for p in m.parameters)}(Tuple(p.default
                                                                 for p in
                                                                     m.parameters))
end

function Base.show(io::IO, m::CAModel)
    print(io, "CAModel ", m.name, " with states (", join((s.name for s in m.states), ", "),
          ")")
    isempty(m.parameters) ||
        print(io, " and parameters (", join((p.name for p in m.parameters), ", "), ")")
    return nothing
end
function Base.show(io::IO, ::MIME"text/plain", m::CAModel)
    println(io, "CAModel ", m.name)
    isempty(m.description) || println(io, "  ", m.description)
    println(io, "  states:")
    for s in m.states
        dom = s.domain === nothing ? "" : " ∈ $(s.domain)"
        println(io, "    ", s.name, "::", typeof(s.default), " = ", repr(s.default), dom,
                isempty(s.description) ? "" : "  # $(s.description)")
    end
    if !isempty(m.parameters)
        println(io, "  parameters:")
        for p in m.parameters
            println(io, "    ", p.name, " = ", repr(p.default),
                    isempty(p.description) ? "" : "  # $(p.description)")
        end
    end
    isempty(m.fields) || println(io, "  fields: ", join(keys(m.fields), ", "))
    isempty(m.covariates) || println(io, "  covariates: ", join(m.covariates, ", "))
    return print(io, "  rule: ", m.rule)
end

_pairs(x::NamedTuple) = pairs(x)
_pairs(x::AbstractDict) = pairs(x)
_pairs(x::Union{Tuple,AbstractVector}) = x
_pairs(::Nothing) = ()

function _layer(s::StateVariable, given, dims)
    T = typeof(s.default)
    given === nothing && return fill(s.default, dims)
    given isa AbstractArray && return convert(Array{T}, given)
    given isa Function && return T[given(I) for I in CartesianIndices(dims)]
    return fill(convert(T, given), dims)
end

"""
    CAProblem(model::CAModel, u0map, tspan, pmap=(); space, seed=0, covariates=(;),
              fields=(;), mask=nothing, kwargs...)

The problem of a [`CAModel`](@ref) on `space`. `u0map` maps state names to an
initial value for every cell, an array the size of the space, or a function of
the cell index (`I -> ...`); states it leaves out take their defaults. `pmap`
maps parameter names to values; parameters it leaves out take their defaults.
Both may be `NamedTuple`s, `Dict`s or collections of `name => value` pairs.
Initial values outside a state's domain raise [`InvalidStateError`](@ref).
Solver keywords (`callback`, `saveat`, ...) are stored with the problem, as for
a problem built from a rule.
"""
function CAProblem(model::CAModel, u0map, tspan, pmap=(); space::AbstractSpace,
                   seed::Integer=0, covariates=(;), fields=(;), mask=nothing, kwargs...)
    dims = size(space)
    given = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in _pairs(u0map))
    names = Set(s.name for s in model.states)
    for k in keys(given)
        k in names ||
            throw(ArgumentError("model $(model.name) has no state $k; states are $(sort(collect(names)))"))
    end
    layers = map(s -> _layer(s, get(given, s.name, nothing), dims), model.states)
    for (s, a) in zip(model.states, layers)
        s.domain === nothing && continue
        i = findfirst(x -> !(x in s.domain), a)
        i === nothing ||
            throw(InvalidStateError("model $(model.name)", i, a[i],
                                    "state $(s.name) must lie in $(s.domain)"))
    end
    u0 = length(layers) == 1 ? only(layers) :
         NamedTuple{Tuple(s.name for s in model.states)}(Tuple(layers))
    p = merge(defaults(model), _parameter_values(model, pmap))
    for c in model.covariates
        haskey(covariates, c) ||
            throw(ArgumentError("model $(model.name) needs covariate $c; give it with covariates=(; $c=...)"))
    end
    return CAProblem(model.rule, u0, tspan, isempty(p) ? NullParameters() : p; space, seed,
                     covariates, fields=merge(model.fields, fields), mask, kwargs...)
end

function _parameter_values(model, pmap)
    known = Set(p.name for p in model.parameters)
    out = Pair{Symbol,Any}[]
    for (k, v) in _pairs(pmap)
        Symbol(k) in known ||
            throw(ArgumentError("model $(model.name) has no parameter $k; parameters are $(sort(collect(known)))"))
        push!(out, Symbol(k) => v)
    end
    return (; out...)
end

# ------------------------------------------------------------------ the @camodel DSL

"""
    @camodel name begin
        @description "..."
        @states x::T = default [∈ domain] ...
        @parameters a = 1.0 b = 2.0 ...
        @neighbourhood Moore(1)
        @fields name = KernelField(...) ...
        @covariates c1 c2 ...
        @rule begin
            ...  # uses cell, nbrs, ctx and the parameters by name
        end
    end

Build a [`CAModel`](@ref) from declarations, in the spirit of ModelingToolkit's
`@mtkmodel`. Inside `@rule`, `cell` is the cell's state (a `NamedTuple` if there
are several state variables), `nbrs` its neighbours' states, `ctx` its
[`CellContext`](@ref), and every parameter is available by name. The rule's
value is the cell's next state. A model in continuous time declares
`@events begin ... end` instead of `@rule`, whose value is the cell's
transitions as `rate => new_state` pairs (an [`EventRule`](@ref)), and is
solved with [`Gillespie`](@ref) or in discrete time.

```jldoctest
julia> sir = @camodel SIR begin
           @states state::Symbol = :S ∈ (:S, :I, :R)
           @parameters β = 0.3 γ = 0.1
           @neighbourhood VonNeumann(1)
           @rule begin
               if cell == :S
                   bernoulli(ctx, 1 - (1 - β)^count(==(:I), nbrs), 1) ? :I : :S
               elseif cell == :I
                   bernoulli(ctx, γ, 2) ? :R : :I
               else
                   cell
               end
           end
       end;

julia> println(sir)
CAModel SIR with states (state) and parameters (β, γ)
```
"""
macro camodel(name, block)
    block isa Expr && block.head == :block ||
        throw(ArgumentError("@camodel needs a begin ... end block"))
    states = Expr[]
    params = Expr[]
    pnames = Symbol[]
    nbhd = :nothing
    fields = Expr[]
    covs = Symbol[]
    desc = ""
    rulebody = nothing
    ruletype = LocalRule
    for stmt in block.args
        stmt isa LineNumberNode && continue
        (stmt isa Expr && stmt.head == :macrocall) ||
            throw(ArgumentError("@camodel: expected @states, @parameters, @neighbourhood, @fields, @covariates, @description, @rule or @events, got $stmt"))
        tag = Symbol(lstrip(string(stmt.args[1]), '@'))
        args = filter(a -> !(a isa LineNumberNode), stmt.args[2:end])
        if tag == :states
            append!(states, map(_state_decl, args))
        elseif tag == :parameters
            for a in args
                (a isa Expr && a.head == :(=)) ||
                    throw(ArgumentError("@parameters: expected name = default, got $a"))
                push!(pnames, a.args[1])
                push!(params, :($Parameter($(QuoteNode(a.args[1])), $(esc(a.args[2])))))
            end
        elseif tag in (:neighbourhood, :neighborhood)
            nbhd = esc(only(args))
        elseif tag == :fields
            for a in args
                push!(fields, :($(esc(a.args[1])) = $(esc(a.args[2]))))
            end
        elseif tag == :covariates
            append!(covs, args)
        elseif tag == :description
            desc = only(args)
        elseif tag in (:rule, :events)
            rulebody === nothing ||
                throw(ArgumentError("@camodel: give one @rule or one @events block"))
            rulebody = only(args)
            ruletype = tag == :rule ? LocalRule : EventRule
        else
            throw(ArgumentError("@camodel: unknown declaration @$tag"))
        end
    end
    rulebody === nothing &&
        throw(ArgumentError("@camodel: a @rule or @events block is required"))
    isempty(states) && throw(ArgumentError("@camodel: @states is required"))
    cell, nbrs, ctx = esc(:cell), esc(:nbrs), esc(:ctx)
    unpack = isempty(pnames) ? nothing :
             Expr(:(=), Expr(:tuple, Expr(:parameters, map(esc, pnames)...)),
                  :($ctx.p))
    f = :(($cell, $nbrs, $ctx) -> begin
              $unpack
              $(esc(rulebody))
          end)
    return quote
        $CAModel($ruletype($f, $nbhd); name=($(QuoteNode(name))),
                 states=($StateVariable[$(states...)]),
                 parameters=($Parameter[$(params...)]),
                 fields=(; $(fields...)),
                 covariates=($(QuoteNode(covs))), description=($desc))
    end
end

# `x::T = default` or `x::T = default ∈ domain` (also `in`).
function _state_decl(a)
    (a isa Expr && a.head == :(=)) ||
        throw(ArgumentError("@states: expected name::Type = default, got $a"))
    lhs, rhs = a.args
    (lhs isa Expr && lhs.head == :(::)) ||
        throw(ArgumentError("@states: expected name::Type, got $lhs"))
    name, T = lhs.args
    domain = :nothing
    if rhs isa Expr && rhs.head == :call && rhs.args[1] in (:∈, :in)
        rhs, domain = rhs.args[2], esc(rhs.args[3])
    end
    return :($StateVariable($(QuoteNode(name)), convert($(esc(T)), $(esc(rhs)));
                            domain=($domain)))
end
