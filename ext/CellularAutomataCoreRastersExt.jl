# Spaces, masks and covariates from Rasters.jl rasters, and states written back as rasters. The
# raster's X and Y dimensions are the lattice's first and second axes; a Ti dimension is the time
# axis of a covariate. File-backed rasters are read into memory first.
module CellularAutomataCoreRastersExt

using CellularAutomataCore
using Rasters: Rasters, AbstractRaster, Raster, RasterStack, X, Y, Ti, dims, hasdim

const CAC = CellularAutomataCore

function _checked(r::AbstractRaster)
    (hasdim(r, X) && hasdim(r, Y)) ||
        throw(ArgumentError("the raster needs X and Y dimensions, got $(map(Rasters.name, dims(r)))"))
    expected = hasdim(r, Ti) ? 3 : 2
    ndims(r) == expected ||
        throw(ArgumentError("the raster may have only X, Y and Ti dimensions, got $(map(Rasters.name, dims(r)))"))
    order = hasdim(r, Ti) ? (X, Y, Ti) : (X, Y)
    return Rasters.read(permutedims(r, order))
end

# Whether a value is data rather than the raster's missing value.
_isdata(v, mv) = !ismissing(v) && (mv === nothing || ismissing(mv) || !isequal(v, mv))

function CAC.raster_space(r::AbstractRaster; boundary=CAC.Constant(0))
    a = _checked(r)
    return CAC.Grid(size(a, 1), size(a, 2); boundary)
end

function CAC.raster_mask(r::AbstractRaster)
    a = _checked(r)
    data = parent(a)
    first_time = ndims(data) == 3 ? view(data, :, :, 1) : data
    mv = Rasters.missingval(a)
    return Bool[_isdata(v, mv) for v in first_time]
end

function CAC.raster_covariate(r::AbstractRaster; missing_as=0, T::Type=Float64)
    a = _checked(r)
    mv = Rasters.missingval(a)
    return T[_isdata(v, mv) ? T(v) : T(missing_as) for v in parent(a)]
end

function CAC.state_raster(u::AbstractArray, template::AbstractRaster; missingval=nothing,
                          mask=nothing, name=:state)
    t = _checked(template)
    d = (dims(t, X), dims(t, Y))
    size(u) == (length(d[1]), length(d[2])) ||
        throw(CAC.StateShapeError("a state written on the raster",
                                  (length(d[1]), length(d[2])),
                                  size(u)))
    if mask === nothing
        return Raster(copy(u), d; missingval, name)
    end
    size(mask) == size(u) ||
        throw(CAC.StateShapeError("the mask", size(u), size(mask)))
    missingval === nothing &&
        throw(ArgumentError("a mask needs a missingval for the cells outside it"))
    data = [mask[I] ? u[I] : convert(eltype(u), missingval) for I in CartesianIndices(u)]
    return Raster(data, d; missingval=convert(eltype(u), missingval), name)
end

function CAC.state_raster(u::NamedTuple, template::AbstractRaster; kwargs...)
    layers = map((k, a) -> CAC.state_raster(a, template; name=k, kwargs...), keys(u),
                 values(u))
    return RasterStack(NamedTuple{keys(u)}(layers))
end

end # module
