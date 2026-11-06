# States. A state is an array the size of its space, or a NamedTuple of such arrays (one per
# layer, structure-of-arrays). A cell of a layered state is the NamedTuple of its layer values.

const LayeredState = NamedTuple{<:Any,<:Tuple{Vararg{AbstractArray}}}

@inline _read(u::AbstractArray, I) = @inbounds u[I]
@inline function _read(u::LayeredState, I)
    return NamedTuple{keys(u)}(_read_layers(values(u), I))
end
# Unrolled over the layers: `map` over a long tuple of arrays of different element types is not
# inferred and allocates on every cell.
@inline _read_layers(::Tuple{}, I) = ()
@inline _read_layers(as::Tuple, I) = ((@inbounds first(as)[I]),
                                      _read_layers(Base.tail(as), I)...)

@inline _write!(u::AbstractArray, I, v) = (@inbounds u[I] = v; nothing)
@inline function _write!(u::LayeredState, I, v::NamedTuple)
    return _write_layers!(values(u), values(_ordered(u, v)), I)
end
# Unrolled over the layers: a multi-argument `foreach` over tuples of different element types
# goes through `zip` and allocates on every cell.
@inline _write_layers!(::Tuple{}, ::Tuple{}, I) = nothing
@inline function _write_layers!(as::Tuple, xs::Tuple, I)
    @inbounds first(as)[I] = first(xs)
    return _write_layers!(Base.tail(as), Base.tail(xs), I)
end

# Reorder a cell value into the layer order of the state, so rules may build their result
# NamedTuple in any order.
@inline _ordered(::NamedTuple{K}, v::NamedTuple) where {K} = NamedTuple{K}(v)

_state_size(u::AbstractArray) = size(u)
_state_size(u::LayeredState) = size(first(values(u)))

function _check_shape(u::AbstractArray, space)
    size(u) == size(space) ||
        throw(StateShapeError("the initial state", size(space), size(u)))
    return nothing
end
function _check_shape(u::LayeredState, space)
    isempty(u) && throw(ArgumentError("a layered state needs at least one layer"))
    for (k, a) in pairs(u)
        size(a) == size(space) ||
            throw(StateShapeError("layer $(repr(k)) of the initial state", size(space),
                                  size(a)))
    end
    return nothing
end

# A Constant boundary is read mid-run, so a value the state cannot hold is caught when the problem
# is built. Only numbers are checked: a rule that reads no neighbours may never read the boundary of
# a state of tuples or symbols, whose default Constant(0) must stay legal.
_check_boundaries(u, space) = nothing
function _check_boundaries(u, space::Lattice)
    foreach(b -> _check_boundary(u, b, space), boundaries(space))
    return nothing
end
_check_boundary(u, ::AbstractBoundary, space) = nothing
_check_boundary(u, b::Constant, space) = _check_fill(u, b.value, space)

_holds(T, x) = true
function _holds(::Type{T}, x::Number) where {T<:Number}
    try
        convert(T, x)
        return true
    catch e
        e isa InexactError || rethrow()
        return false
    end
end

function _boundary_mismatch(v, space, msg)
    return IncompatibleSpaceError("the boundary Constant($(repr(v)))", _describe(space),
                                  msg)
end

function _check_fill(u::AbstractArray, v, space)
    v isa NamedTuple &&
        throw(_boundary_mismatch(v, space, "it names layers, and the state has none"))
    _holds(eltype(u), v) ||
        throw(_boundary_mismatch(v, space, "a state of $(eltype(u)) cannot hold it"))
    return nothing
end
function _check_fill(u::LayeredState, v::NamedTuple, space)
    unknown = Tuple(k for k in keys(v) if !haskey(u, k))
    isempty(unknown) ||
        throw(_boundary_mismatch(v, space,
                                 "the state has no layer $(join(map(repr, unknown), ", ")); its layers are $(keys(u))"))
    for (k, x) in pairs(v)
        _holds(eltype(u[k]), x) ||
            throw(_boundary_mismatch(v, space,
                                     "layer $(repr(k)) of $(eltype(u[k])) cannot hold $x"))
    end
    return nothing
end
function _check_fill(u::LayeredState, v, space)
    for (k, a) in pairs(u)
        _holds(eltype(a), v) ||
            throw(_boundary_mismatch(v, space,
                                     "a number fills every layer, and layer $(repr(k)) of $(eltype(a)) cannot hold it; name the layers to give them their own values, as in Constant((; $(k)=...))"))
    end
    return nothing
end

_copy_state(u::AbstractArray) = copy(u)
_copy_state(u::LayeredState) = map(copy, u)

"""
    single_seed(dims...; value=1, T=Int8)

A state of zeros of element type `T` with a single cell set to `value` at the
centre (rounded towards the first index): the classic starting point of
elementary cellular automata.

```jldoctest
julia> single_seed(5)
5-element Vector{Int8}:
 0
 0
 1
 0
 0
```
"""
function single_seed(dims::Integer...; value=1, T::Type=Int8)
    u = zeros(T, dims...)
    u[CartesianIndex(map(n -> cld(n, 2), dims))] = value
    return u
end

"""
    place!(u, pattern, corner=nothing)

Copy the array `pattern` into the state `u` with its first cell at the index
`corner` (a tuple or `CartesianIndex`), or centred in `u` when `corner` is
`nothing`. Returns `u`. Useful with patterns read by [`parse_rle`](@ref).
"""
function place!(u::AbstractArray{T,N}, pattern::AbstractArray{<:Any,N},
                corner=nothing) where {T,N}
    any(size(pattern) .> size(u)) &&
        throw(ArgumentError("pattern of size $(size(pattern)) does not fit in a state of size $(size(u))"))
    c = corner === nothing ? (size(u) .- size(pattern)) .÷ 2 .+ 1 : Tuple(corner)
    all(c .>= 1) && all(c .+ size(pattern) .- 1 .<= size(u)) ||
        throw(ArgumentError("pattern placed at $c does not fit in a state of size $(size(u))"))
    for I in CartesianIndices(pattern)
        u[CartesianIndex(Tuple(I) .+ c .- 1)] = pattern[I]
    end
    return u
end
