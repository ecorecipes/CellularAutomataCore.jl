# Landscapes from rasters. The functions are declared here, with fallbacks that explain what to
# load; their methods for Rasters.jl's rasters live in the Rasters extension. A raster's X and Y
# dimensions become the first and second axes of the lattice, in that order, and a time (Ti)
# dimension becomes the time axis of a covariate.

const _NEEDS_RASTERS = "needs Rasters.jl: run `using Rasters` first (and ArchGDAL or NCDatasets to read files)"

"""
    raster_space(r; boundary=Constant(0)) -> Lattice

A [`Grid`](@ref) the size of the raster `r`'s X and Y dimensions, with the
given boundary (nothing lies beyond a landscape by default). Cell `(i, j)` of
the lattice is the raster's cell at `X = i`, `Y = j`. Needs Rasters.jl.
"""
raster_space(r; kwargs...) = throw(ArgumentError("raster_space $_NEEDS_RASTERS"))

"""
    raster_mask(r) -> Matrix{Bool}

The cells of the raster `r` that hold data (`true`) rather than its missing
value, as a mask for [`CAProblem`](@ref): cells outside it never update. For
a raster with a time dimension, a cell holds data if it does at the first
time. Needs Rasters.jl.
"""
raster_mask(r) = throw(ArgumentError("raster_mask $_NEEDS_RASTERS"))

"""
    raster_covariate(r; missing_as=0, T=Float64) -> Array{T}

The values of the raster `r` as a covariate layer (see [`covariate`](@ref)):
a matrix for a two-dimensional raster, or an array with time last for a
raster with a time dimension, whose slices the covariate cycles through. Cells
holding the raster's missing value get `missing_as`. Needs Rasters.jl.
"""
raster_covariate(r; kwargs...) = throw(ArgumentError("raster_covariate $_NEEDS_RASTERS"))

"""
    state_raster(u, template; missingval=nothing, mask=nothing)

A state (an array, or a `NamedTuple` of layers) as a raster on the X and Y
dimensions of `template`: a `Raster` for an array and a `RasterStack` for a
layered state, ready to write with `Rasters.write`. Cells where `mask` is
`false` get `missingval`. Needs Rasters.jl.
"""
state_raster(u, template; kwargs...) = throw(ArgumentError("state_raster $_NEEDS_RASTERS"))
