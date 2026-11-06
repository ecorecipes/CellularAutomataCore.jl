# Spaces: where the cells are and which cell lies beyond an edge.

"""
    AbstractBoundary

Supertype of the boundary conditions of a [`Lattice`](@ref) axis:
[`Periodic`](@ref), [`Reflecting`](@ref) and [`Constant`](@ref).
"""
abstract type AbstractBoundary end

"""
    Periodic()

The axis wraps around: the cell beyond the last is the first.
"""
struct Periodic <: AbstractBoundary end

"""
    Reflecting()

The axis is mirrored at its edges, with the edge cell repeated: on an axis of
length `n`, index `0` reads cell `1`, index `-1` reads cell `2`, and index
`n + 1` reads cell `n`.
"""
struct Reflecting <: AbstractBoundary end

"""
    Constant(value)

Every cell beyond the edge holds `value`, for example `Constant(0)` for a dead
or empty margin. When cells lie beyond the edges of several `Constant` axes at
once, the value of the highest-numbered of those axes is used.

A number fills every layer of a layered state, so it must suit them all:
`Constant(-1)` cannot fill a layer of `Bool`. A `NamedTuple` gives layers their
own values and the layers it does not name read zero, so `Constant((; depth=-1.0))`
marks the edge in one layer whatever the others hold. Naming a layer the state
does not have is an [`IncompatibleSpaceError`](@ref) when the problem is built.
"""
struct Constant{T} <: AbstractBoundary
    value::T
end

# Map an index on one axis of length n. Returns (index, outside) where `outside` is true
# only for a Constant boundary and an out-of-range index.
@inline _map_axis(::Periodic, j::Int, n::Int) = (mod1(j, n), false)
@inline function _map_axis(::Reflecting, j::Int, n::Int)
    m = mod(j - 1, 2n)
    return (m < n ? m + 1 : 2n - m, false)
end
@inline _map_axis(::Constant, j::Int, n::Int) = (j, !(1 <= j <= n))

"""
    AbstractSpace

Supertype of the spaces cells live in: [`Lattice`](@ref) (square or hexagonal,
any number of dimensions, built by [`Torus`](@ref), [`Grid`](@ref) and
[`HexGrid`](@ref)) and [`GraphSpace`](@ref).
"""
abstract type AbstractSpace end

"""
    AbstractGeometry

The shape of the cells of a [`Lattice`](@ref): [`Square`](@ref) or [`Hex`](@ref).
"""
abstract type AbstractGeometry end

"""
    Square()

Square (hypercubic) cells: the default geometry of a [`Lattice`](@ref).
"""
struct Square <: AbstractGeometry end

"""
    Hex()

Hexagonal cells in axial coordinates: cell `(q, r)` is stored at index `[q, r]`
and its six nearest neighbours are at offsets `(±1, 0)`, `(0, ±1)`, `(1, -1)`
and `(-1, 1)`. Pointy-top hexagons with centres at
`x = √3 (q + r / 2)`, `y = 1.5 r`.
"""
struct Hex <: AbstractGeometry end

"""
    Lattice{N,G,B}

A regular `N`-dimensional lattice of `prod(dims)` cells with geometry `G`
([`Square`](@ref) or [`Hex`](@ref)) and one boundary condition per axis. A state
on it is an `Array` of size `dims`, or a `NamedTuple` of such arrays (one per
layer). Build one with [`Torus`](@ref), [`Grid`](@ref) or [`HexGrid`](@ref).
"""
struct Lattice{N,G<:AbstractGeometry,B<:Tuple} <: AbstractSpace
    dims::NTuple{N,Int}
    boundaries::B
    geometry::G
    function Lattice(dims::NTuple{N,Int}, boundaries::Tuple,
                     geometry::AbstractGeometry) where {N}
        all(>(0), dims) ||
            throw(ArgumentError("lattice dimensions must be positive, got $dims"))
        length(boundaries) == N ||
            throw(ArgumentError("need one boundary per axis ($N), got $(length(boundaries))"))
        all(b -> b isa AbstractBoundary, boundaries) ||
            throw(ArgumentError("boundaries must be AbstractBoundary values"))
        geometry isa Hex && N != 2 &&
            throw(ArgumentError("a hexagonal lattice is two-dimensional, got $N dimensions"))
        return new{N,typeof(geometry),typeof(boundaries)}(dims, boundaries, geometry)
    end
end

_boundaries(b::AbstractBoundary, N) = ntuple(_ -> b, N)
_boundaries(b::Tuple, N) = b
function _boundaries(b, N)
    return throw(ArgumentError("boundary must be an AbstractBoundary or a tuple of them"))
end

"""
    Torus(dims...)
    Torus(dims::Tuple)

A square lattice that is periodic on every axis: a ring in one dimension, a
torus in two.

```jldoctest
julia> Torus(5, 4)
Torus((5, 4))
```
"""
Torus(dims::Integer...) = Torus(Tuple(dims))
Torus(dims::Tuple) = Lattice(Int.(dims), _boundaries(Periodic(), length(dims)), Square())

"""
    Grid(dims...; boundary=Constant(0))

A square lattice with edges. `boundary` is one [`AbstractBoundary`](@ref) for
every axis or a tuple with one per axis; `Grid(64, 64; boundary=(Periodic(),
Constant(0)))` is a cylinder.
"""
Grid(dims::Integer...; boundary=Constant(0)) = Grid(Tuple(dims); boundary)
function Grid(dims::Tuple; boundary=Constant(0))
    return Lattice(Int.(dims), _boundaries(boundary, length(dims)), Square())
end

"""
    HexGrid(nq, nr; boundary=Periodic())

A two-dimensional lattice of hexagonal cells in axial coordinates (see
[`Hex`](@ref)): a parallelogram of `nq × nr` cells, periodic by default.
"""
function HexGrid(nq::Integer, nr::Integer; boundary=Periodic())
    return Lattice((Int(nq), Int(nr)), _boundaries(boundary, 2), Hex())
end

Base.size(s::Lattice) = s.dims
Base.ndims(::Lattice{N}) where {N} = N
Base.length(s::Lattice) = prod(s.dims)

"""
    cells(space)

The indices of the cells of `space`: `CartesianIndices(size(space))` for a
[`Lattice`](@ref), `1:n` for a [`GraphSpace`](@ref).
"""
cells(s::Lattice) = CartesianIndices(s.dims)

"""
    boundaries(space::Lattice)

The tuple of per-axis boundary conditions.
"""
boundaries(s::Lattice) = s.boundaries

"""
    geometry(space::Lattice)

The cell geometry, [`Square`](@ref) or [`Hex`](@ref).
"""
geometry(s::Lattice) = s.geometry

function Base.show(io::IO, s::Lattice{N,Square}) where {N}
    if all(b -> b isa Periodic, s.boundaries)
        print(io, "Torus(", s.dims, ")")
    else
        print(io, "Grid(", s.dims, "; boundary=", s.boundaries, ")")
    end
end
function Base.show(io::IO, s::Lattice{2,Hex})
    return print(io, "HexGrid(", s.dims, "; boundary=", s.boundaries, ")")
end

"""
    GraphSpace(neighbours::Vector{Vector{Int}})
    GraphSpace(n, edges; directed=false)

An irregular space of `n` cells in which cell `i` reads the cells
`neighbours[i]`, in that order: patches of a metapopulation, trees at measured
positions, fields connected by roads. A state on it is a `Vector` of length `n`
(or a `NamedTuple` of them). The second form builds the lists from an edge list
of `(i, j)` pairs; undirected edges are added in both directions.

```jldoctest
julia> ring = GraphSpace(4, [(1, 2), (2, 3), (3, 4), (4, 1)]);

julia> ring.neighbours[1]
2-element Vector{Int64}:
 2
 4
```
"""
struct GraphSpace <: AbstractSpace
    neighbours::Vector{Vector{Int}}
    function GraphSpace(neighbours::Vector{Vector{Int}})
        n = length(neighbours)
        for (i, ns) in enumerate(neighbours), j in ns
            1 <= j <= n ||
                throw(ArgumentError("cell $i has neighbour $j outside 1:$n"))
        end
        return new(neighbours)
    end
end

function GraphSpace(n::Integer, edges; directed::Bool=false)
    ns = [Int[] for _ in 1:n]
    for (i, j) in edges
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError("edge ($i, $j) outside 1:$n"))
        push!(ns[i], j)
        directed || push!(ns[j], i)
    end
    foreach(sort!, ns)
    return GraphSpace(ns)
end

Base.size(s::GraphSpace) = (length(s.neighbours),)
Base.ndims(::GraphSpace) = 1
Base.length(s::GraphSpace) = length(s.neighbours)
cells(s::GraphSpace) = 1:length(s.neighbours)
Base.show(io::IO, s::GraphSpace) = print(io, "GraphSpace(", length(s), " cells)")

# Short description for error messages.
_describe(s::AbstractSpace) = sprint(show, s)
