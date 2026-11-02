# Summaries of states: counts, clusters and spatial autocorrelation, for saving callbacks and
# analysis.

"""
    state_counts(u) -> Dict

How many cells hold each value in the state `u` (a single-layer array).

```jldoctest
julia> state_counts([:S :I; :S :S])[:S]
3
```
"""
function state_counts(u::AbstractArray)
    d = Dict{eltype(u),Int}()
    for x in u
        d[x] = get(d, x, 0) + 1
    end
    return d
end

# Neighbours of a cell for measures: lattice offsets with boundary semantics, cells beyond a
# Constant edge excluded.
function _measure_neighbours(space::Lattice, offsets, I)
    out = CartesianIndex{ndims(space)}[]
    for o in offsets
        J, beyond = _boundary_lookup(space, I + o)
        beyond == 0 && push!(out, J)
    end
    return out
end
_measure_neighbours(space::GraphSpace, offsets, i) = space.neighbours[i]
_measure_offsets(space::Lattice, nbhd) = neighbourhood_offsets(nbhd, space).offsets
_measure_offsets(space::GraphSpace, nbhd) = ()

"""
    cluster_sizes(u, pred=!iszero; space=Torus(size(u)...), neighbourhood=VonNeumann(1))

The sizes, largest first, of the connected clusters of cells whose state
satisfies `pred`, connected through `neighbourhood` on `space` (periodic edges
wrap; nothing lies beyond other edges). For a [`GraphSpace`](@ref), clusters
follow the graph's edges and `neighbourhood` is ignored.

```jldoctest
julia> cluster_sizes([1 1 0 0; 0 0 0 1; 0 0 0 1]; space=Grid(3, 4))
2-element Vector{Int64}:
 2
 2
```
"""
function cluster_sizes(u::AbstractArray, pred=(!iszero); space=_default_space(u),
                       neighbourhood=VonNeumann(1))
    _check_shape(u, space)
    offsets = _measure_offsets(space, neighbourhood)
    seen = falses(size(u))
    sizes = Int[]
    stack = eltype(cells(space))[]
    for I in cells(space)
        (seen[I] || !pred(u[I])) && continue
        seen[I] = true
        push!(stack, I)
        n = 0
        while !isempty(stack)
            J = pop!(stack)
            n += 1
            for K in _measure_neighbours(space, offsets, J)
                if !seen[K] && pred(u[K])
                    seen[K] = true
                    push!(stack, K)
                end
            end
        end
        push!(sizes, n)
    end
    return sort!(sizes; rev=true)
end

"""
    morans_i(x; space=Torus(size(x)...), neighbourhood=VonNeumann(1)) -> Float64

Moran's I spatial autocorrelation of the numbers `x`, with weight 1 between
each cell and each of its neighbours (cells beyond a non-periodic edge are not
neighbours): near 1 for smooth patches, near `-1/(n - 1)` for no pattern, and
negative for checkerboards.

```jldoctest
julia> morans_i([isodd(i + j) ? 1.0 : 0.0 for i in 1:6, j in 1:6])
-1.0
```
"""
function morans_i(x::AbstractArray{<:Real}; space=_default_space(x),
                  neighbourhood=VonNeumann(1))
    _check_shape(x, space)
    offsets = _measure_offsets(space, neighbourhood)
    m = sum(x) / length(x)
    num = 0.0
    W = 0
    for I in cells(space)
        for J in _measure_neighbours(space, offsets, I)
            num += (x[I] - m) * (x[J] - m)
            W += 1
        end
    end
    den = sum(v -> (v - m)^2, x)
    den == 0 && throw(ArgumentError("Moran's I is undefined for a constant field"))
    return length(x) / W * num / den
end
