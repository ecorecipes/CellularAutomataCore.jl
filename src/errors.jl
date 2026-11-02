# Exceptions of CellularAutomataCore. Every exception the package defines lives in this
# file, with its `showerror` method, and subtypes the root `CellularAutomataCoreError`.
#
# - Carry the offending names (layer, rule, backend, model) in the fields and the message.
#   Library code never calls a bare `error("...")`.
# - Invalid arguments and keywords raise `ArgumentError`, not a type from this file.
# - A docstring names another package's type as a code span, never with `@ref`.

"""
    CellularAutomataCoreError

Abstract supertype of every exception that CellularAutomataCore defines:
[`RuleParseError`](@ref), [`RLEParseError`](@ref), [`StateShapeError`](@ref),
[`InvalidStateError`](@ref), [`IncompatibleSpaceError`](@ref) and
[`UnsupportedCombinationError`](@ref). It marks the package that introduced an
error, not a kind of failure. Invalid arguments and keywords raise Base's
`ArgumentError` instead, which is outside this root.

```jldoctest
julia> try
           Life("B9/S23")
       catch e
           e isa CellularAutomataCoreError
       end
true
```
"""
abstract type CellularAutomataCoreError <: Exception end

"""
    RuleParseError(rule, msg)

Thrown when a rule string such as `"B3/S23"` or `"B2/S/C3"` cannot be parsed.
`rule` is the offending string.
"""
struct RuleParseError <: CellularAutomataCoreError
    rule::String
    msg::String
end

function Base.showerror(io::IO, e::RuleParseError)
    return print(io, "RuleParseError: rule ", repr(e.rule), ": ", e.msg)
end

"""
    RLEParseError(line, msg)

Thrown when run-length-encoded (RLE) pattern text cannot be parsed. `line` is the
1-based line number of the offending text, or `0` when the problem is not tied
to one line.
"""
struct RLEParseError <: CellularAutomataCoreError
    line::Int
    msg::String
end

function Base.showerror(io::IO, e::RLEParseError)
    where = e.line > 0 ? " at line $(e.line)" : ""
    return print(io, "RLEParseError", where, ": ", e.msg)
end

"""
    StateShapeError(what, expected, got)

Thrown when a state does not have the size of its space. `what` names the
offending state or layer; `expected` and `got` are the sizes.
"""
struct StateShapeError <: CellularAutomataCoreError
    what::String
    expected::Tuple
    got::Tuple
end

function Base.showerror(io::IO, e::StateShapeError)
    return print(io, "StateShapeError: ", e.what, " has size ", e.got,
                 " but the space has size ",
                 e.expected)
end

"""
    InvalidStateError(rule, index, value, msg)

Thrown when an initial state holds a value that the rule cannot read, for
example a `2` in a two-state rule. `rule` names the rule, `index` is the first
offending cell and `value` its value. States are checked once, when a problem is
constructed, never during a run.
"""
struct InvalidStateError <: CellularAutomataCoreError
    rule::String
    index::Any
    value::Any
    msg::String
end

function Base.showerror(io::IO, e::InvalidStateError)
    return print(io, "InvalidStateError in ", e.rule, ": cell ", e.index, " has value ",
                 repr(e.value), "; ", e.msg)
end

"""
    IncompatibleSpaceError(what, space, msg)

Thrown when a neighbourhood, rule or state cannot be used on a space, for
example a Moore neighbourhood on a hexagonal grid or a one-dimensional rule on a
two-dimensional lattice. `what` names the offending object and `space`
describes the space.
"""
struct IncompatibleSpaceError <: CellularAutomataCoreError
    what::String
    space::String
    msg::String
end

function Base.showerror(io::IO, e::IncompatibleSpaceError)
    return print(io, "IncompatibleSpaceError: ", e.what, " on ", e.space, ": ", e.msg)
end

"""
    UnsupportedCombinationError(backend, scheme)

Thrown when a backend does not support an update scheme. `backend` and `scheme`
name them; [`supports`](@ref) answers the same question without throwing.
"""
struct UnsupportedCombinationError <: CellularAutomataCoreError
    backend::String
    scheme::String
end

function Base.showerror(io::IO, e::UnsupportedCombinationError)
    return print(io, "UnsupportedCombinationError: backend ", e.backend,
                 " does not support the update scheme ", e.scheme)
end
