# Neighbourhoods: which cells a local rule reads. On a lattice every neighbourhood is
# resolved, once, into an `Offsets` tuple in a fixed order; the engines and rules see only
# that, so a rule's neighbour tuple has a length known to the compiler.

"""
    AbstractNeighbourhood

Supertype of neighbourhoods: [`Moore`](@ref), [`VonNeumann`](@ref),
[`Hexagonal`](@ref), [`Offsets`](@ref) and [`GraphNeighbours`](@ref). The
centre cell is never part of a neighbourhood: a local rule receives it
separately.
"""
abstract type AbstractNeighbourhood end

"""
    Moore(radius=1)

All cells within Chebyshev distance `radius` on a square lattice of any
dimension: 8 cells in two dimensions at radius 1, 24 at radius 2.
"""
struct Moore <: AbstractNeighbourhood
    radius::Int
    function Moore(radius::Integer=1)
        radius >= 1 || throw(ArgumentError("Moore radius must be at least 1, got $radius"))
        return new(radius)
    end
end

"""
    VonNeumann(radius=1)

All cells within Manhattan distance `radius` on a square lattice of any
dimension: 4 cells in two dimensions at radius 1, 12 at radius 2.
"""
struct VonNeumann <: AbstractNeighbourhood
    radius::Int
    function VonNeumann(radius::Integer=1)
        radius >= 1 ||
            throw(ArgumentError("von Neumann radius must be at least 1, got $radius"))
        return new(radius)
    end
end

"""
    Hexagonal(radius=1)

All cells within hexagonal distance `radius` on a [`HexGrid`](@ref): 6 cells at
radius 1, 18 at radius 2.
"""
struct Hexagonal <: AbstractNeighbourhood
    radius::Int
    function Hexagonal(radius::Integer=1)
        radius >= 1 ||
            throw(ArgumentError("hexagonal radius must be at least 1, got $radius"))
        return new(radius)
    end
end

"""
    Offsets(offsets...)

An explicit neighbourhood on a lattice: the cells at the given offsets from the
centre, in the given order. Offsets are `CartesianIndex`es or tuples of
integers; the zero offset (the centre) is not allowed and duplicates are
rejected. Every other lattice neighbourhood is resolved into an `Offsets`, see
[`neighbourhood_offsets`](@ref). Iterating over an `Offsets` gives the offsets
as `CartesianIndex`es, in order.

```jldoctest
julia> Offsets((-1, 0), (1, 0))  # the cells above and below
Offsets((-1, 0), (1, 0))

julia> collect(Offsets((-1, 0), (1, 0)))
2-element Vector{CartesianIndex{2}}:
 CartesianIndex(-1, 0)
 CartesianIndex(1, 0)
```
"""
struct Offsets{N,M} <: AbstractNeighbourhood
    offsets::NTuple{M,CartesianIndex{N}}
    function Offsets(offsets::Tuple{CartesianIndex{N},Vararg{CartesianIndex{N}}}) where {N}
        any(o -> all(iszero, Tuple(o)), offsets) &&
            throw(ArgumentError("the zero offset (the centre cell) cannot be a neighbour"))
        allunique(offsets) || throw(ArgumentError("offsets must be unique"))
        return new{N,length(offsets)}(offsets)
    end
end
Offsets(o::CartesianIndex, os::CartesianIndex...) = Offsets((o, os...))
function Offsets(o::Tuple{Integer,Vararg{Integer}}, os::Tuple{Integer,Vararg{Integer}}...)
    return Offsets(map(CartesianIndex, (o, os...)))
end
function Offsets(offsets::AbstractVector)
    isempty(offsets) && return Offsets()
    return Offsets(Tuple(map(o -> CartesianIndex(Tuple(o)), offsets)))
end
Offsets() = throw(ArgumentError("a neighbourhood needs at least one offset"))
Offsets(::Tuple{}) = Offsets()

Base.length(::Offsets{N,M}) where {N,M} = M
Base.eltype(::Type{<:Offsets{N}}) where {N} = CartesianIndex{N}
Base.iterate(o::Offsets, i...) = iterate(o.offsets, i...)
Base.show(io::IO, o::Offsets) = print(io, "Offsets", map(Tuple, o.offsets))

"""
    GraphNeighbours()

The neighbourhood of a [`GraphSpace`](@ref): each cell reads the cells listed
for it in the graph. It is the default neighbourhood of a graph space and is
meaningless on a lattice.
"""
struct GraphNeighbours <: AbstractNeighbourhood end

"""
    default_neighbourhood(space)

The neighbourhood a rule uses when it does not name one: `Moore(1)` on a square
lattice, `Hexagonal(1)` on a hexagonal one and `GraphNeighbours()` on a
[`GraphSpace`](@ref).
"""
default_neighbourhood(::Lattice{N,Square}) where {N} = Moore(1)
default_neighbourhood(::Lattice{2,Hex}) = Hexagonal(1)
default_neighbourhood(::GraphSpace) = GraphNeighbours()

function _offsets_where(pred, N, r)
    return Offsets(Tuple(I
                         for I in CartesianIndices(ntuple(_ -> (-r):r, N))
                         if !all(iszero, Tuple(I)) && pred(Tuple(I))))
end

function _incompatible(nbhd, space, msg)
    return throw(IncompatibleSpaceError(sprint(show, nbhd), _describe(space), msg))
end

"""
    neighbourhood_offsets(neighbourhood, space) -> Offsets

Resolve a neighbourhood on a lattice into explicit [`Offsets`](@ref), in
column-major order of the offsets (first axis fastest). For a one-dimensional
`Moore(r)` this is `-r, …, -1, 1, …, r`, left to right. Raises
[`IncompatibleSpaceError`](@ref) for a neighbourhood that does not fit the
space, such as `Moore` on a hexagonal grid or offsets of the wrong dimension.

```jldoctest
julia> neighbourhood_offsets(VonNeumann(1), Torus(8, 8))
Offsets((0, -1), (-1, 0), (1, 0), (0, 1))
```
"""
function neighbourhood_offsets(n::Moore, s::Lattice{N,Square}) where {N}
    return _offsets_where(_ -> true, N, n.radius)
end
function neighbourhood_offsets(n::VonNeumann, s::Lattice{N,Square}) where {N}
    return _offsets_where(o -> sum(abs, o) <= n.radius, N, n.radius)
end
function neighbourhood_offsets(n::Hexagonal, s::Lattice{2,Hex})
    return _offsets_where(o -> max(abs(o[1]), abs(o[2]), abs(o[1] + o[2])) <= n.radius, 2,
                          n.radius)
end
neighbourhood_offsets(n::Offsets{N}, ::Lattice{N}) where {N} = n
function neighbourhood_offsets(n::Offsets{M}, s::Lattice{N}) where {M,N}
    return _incompatible(n, s,
                         "offsets are $M-dimensional but the lattice is $N-dimensional")
end
function neighbourhood_offsets(n::Union{Moore,VonNeumann}, s::Lattice{2,Hex})
    return _incompatible(n, s, "use Hexagonal(r) or Offsets on a hexagonal grid")
end
function neighbourhood_offsets(n::Hexagonal, s::Lattice)
    return _incompatible(n, s, "Hexagonal(r) needs a HexGrid")
end
function neighbourhood_offsets(n::GraphNeighbours, s::Lattice)
    return _incompatible(n, s, "GraphNeighbours() needs a GraphSpace")
end
function neighbourhood_offsets(n::AbstractNeighbourhood, s::GraphSpace)
    return _incompatible(n, s, "a graph space has no offsets; use GraphNeighbours()")
end

# Halo width needed by a set of offsets: the largest absolute offset on any axis.
_halo(o::Offsets) = maximum(I -> maximum(abs, Tuple(I)), o.offsets)

# A neighbourhood that is `nothing` means "the space's default".
_resolve(::Nothing, space) = _resolve(default_neighbourhood(space), space)
_resolve(n::AbstractNeighbourhood, space::Lattice) = neighbourhood_offsets(n, space)
_resolve(::GraphNeighbours, ::GraphSpace) = GraphNeighbours()
_resolve(n::AbstractNeighbourhood, space::GraphSpace) = neighbourhood_offsets(n, space)
