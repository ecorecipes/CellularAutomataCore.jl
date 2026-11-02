# Plot recipes (RecipesBase), so that Plots.jl draws spaces, states and solutions without this
# package depending on Plots.
#
#   plot(space, u)    a state: heatmap on a square lattice, hexagons on a hexagonal one
#   plot(sol)         one-dimensional: space-time diagram of all saved states;
#                     otherwise: the state at `snapshot` (default the last)
#
# Keywords: `snapshot` (index into sol.u) and `layer` (a layer name for layered states).
# Not `frame`: Plots reads that as an alias of its own `framestyle` attribute.

_layer(u::AbstractArray, ::Nothing) = u
_layer(u::LayeredState, ::Nothing) = first(values(u))
_layer(u::LayeredState, k::Symbol) = getfield(u, k)

@recipe function f(space::Lattice{1}, u::AbstractVector; layer=nothing)
    seriestype := :heatmap
    yticks := false
    xlabel --> "cell"
    return 1:length(u), [1], reshape(_layer(u, layer), 1, :)
end

@recipe function f(space::Lattice{1}, u::LayeredState; layer=nothing)
    return space, _layer(u, layer)
end

@recipe function f(space::Lattice{2,Square}, u::Union{AbstractMatrix,LayeredState};
                   layer=nothing)
    z = _layer(u, layer)
    seriestype := :heatmap
    aspect_ratio --> :equal
    yflip --> true
    framestyle --> :box
    xlims --> (0.5, size(z, 2) + 0.5)
    ylims --> (0.5, size(z, 1) + 0.5)
    return 1:size(z, 2), 1:size(z, 1), z
end

"""
    hexagon_vertices(q, r) -> (xs, ys)

The seven vertices (the first repeated to close it) of the pointy-top hexagon of
the cell at axial coordinates `(q, r)` of a [`HexGrid`](@ref), with unit
circumradius.
"""
function hexagon_vertices(q::Integer, r::Integer)
    cx, cy = sqrt(3) * (q + r / 2), 1.5r
    angles = ((30 + 60k) * pi / 180 for k in 0:6)
    return [cx + cos(a) for a in angles], [cy + sin(a) for a in angles]
end

@recipe function f(space::Lattice{2,Hex}, u::Union{AbstractMatrix,LayeredState};
                   layer=nothing)
    z = _layer(u, layer)
    xs, ys = Float64[], Float64[]
    for I in CartesianIndices(z)
        hx, hy = hexagon_vertices(I[1], I[2])
        append!(xs, hx)
        append!(ys, hy)
        push!(xs, NaN)
        push!(ys, NaN)
    end
    seriestype := :shape
    fill_z := vec(Float64.(z))
    linewidth --> 0.5
    aspect_ratio --> :equal
    legend --> false
    framestyle --> :none
    return xs, ys
end

# The @recipe macro does not support an early `return`, so both cases end in one if/else.
@recipe function f(sol::CASolution; snapshot=nothing, layer=nothing)
    space = sol.prob.space
    if space isa Lattice{1} && snapshot === nothing
        z = stack(_layer(u, layer) for u in sol.u)
        seriestype := :heatmap
        yflip --> true
        xlabel --> "cell"
        ylabel --> "t"
        (1:size(z, 1), sol.t, permutedims(z))
    else
        i = snapshot === nothing ? lastindex(sol.u) : snapshot
        title --> "t = $(sol.t[i])"
        if layer !== nothing
            layer := layer
        end
        (space, sol.u[i])
    end
end

# plot(res::ABCResult): one histogram per parameter, proposals behind accepted samples, both
# normalised so that a narrow posterior and a broad prior share an axis.
@recipe function f(res::ABCResult{<:NamedTuple})
    names = keys(first(res.proposals))
    layout --> length(names)
    xguide --> permutedims([string(name) for name in names])  # one label per subplot
    for (k, name) in enumerate(names)
        @series begin
            seriestype := :histogram
            subplot := k
            normalize --> :pdf
            fillalpha --> 0.3
            linewidth --> 0
            label --> (k == 1 ? "proposals" : "")
            [getfield(q, name) for q in res.proposals]
        end
        @series begin
            seriestype := :histogram
            subplot := k
            normalize --> :pdf
            fillalpha --> 0.7
            linewidth --> 0
            label --> (k == 1 ? "accepted" : "")
            [getfield(q, name) for q in res.samples]
        end
    end
end
