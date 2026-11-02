# Dispersal kernels: weights over offsets, for nonlocal interactions (spores, seeds, insects)
# read through kernel fields. Distances are measured between cell centres: in cell widths on
# a square lattice, in centre-to-centre spacings on a hexagonal one.

"""
    DispersalKernel

Supertype of dispersal kernels: [`GaussianKernel`](@ref),
[`ExponentialKernel`](@ref), [`PowerLawKernel`](@ref), [`UniformKernel`](@ref),
[`LeniaKernel`](@ref), [`AnisotropicKernel`](@ref) and
[`EmpiricalKernel`](@ref). A kernel gives an unnormalised weight to every
offset within its radius; [`kernel_weights`](@ref) lists and normalises them.
"""
abstract type DispersalKernel end

"""
    GaussianKernel(σ; radius=ceil(Int, 3σ))

Weight `exp(-d² / 2σ²)` at distance `d`, cut off at `radius`.
"""
struct GaussianKernel <: DispersalKernel
    σ::Float64
    radius::Int
end
function GaussianKernel(σ::Real; radius::Integer=ceil(Int, 3σ))
    σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
    radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
    return GaussianKernel(Float64(σ), Int(radius))
end
_kernel_value(k::GaussianKernel, d, o) = exp(-d^2 / (2 * k.σ^2))

"""
    ExponentialKernel(scale; radius=ceil(Int, 5scale))

Weight `exp(-d / scale)` at distance `d`, cut off at `radius`: thinner-tailed
than a power law, fatter than a Gaussian.
"""
struct ExponentialKernel <: DispersalKernel
    scale::Float64
    radius::Int
end
function ExponentialKernel(scale::Real; radius::Integer=ceil(Int, 5scale))
    scale > 0 || throw(ArgumentError("scale must be positive, got $scale"))
    radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
    return ExponentialKernel(Float64(scale), Int(radius))
end
_kernel_value(k::ExponentialKernel, d, o) = exp(-d / k.scale)

"""
    PowerLawKernel(α; radius=10)

Weight `(1 + d)^(-α)` at distance `d`, cut off at `radius`: a fat-tailed
(Lévy-like) kernel for occasional long-distance dispersal.
"""
struct PowerLawKernel <: DispersalKernel
    α::Float64
    radius::Int
end
function PowerLawKernel(α::Real; radius::Integer=10)
    α > 0 || throw(ArgumentError("α must be positive, got $α"))
    radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
    return PowerLawKernel(Float64(α), Int(radius))
end
_kernel_value(k::PowerLawKernel, d, o) = (1 + d)^(-k.α)

"""
    UniformKernel(radius)

Equal weight on every cell within distance `radius`: a disc.
"""
struct UniformKernel <: DispersalKernel
    radius::Int
    function UniformKernel(radius::Integer)
        radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
        return new(radius)
    end
end
_kernel_value(::UniformKernel, d, o) = 1.0

"""
    LeniaKernel(R; peaks=(1.0,))

The smooth shell kernel of Lenia [Chan2019](@cite): with `r = d / R`, the shell
`r ∈ (0, 1)` is divided into `length(peaks)` rings, ring `i` having height
`peaks[i]` and profile `exp(4 - 1 / (x (1 - x)))` across it.
"""
struct LeniaKernel <: DispersalKernel
    radius::Int
    peaks::Vector{Float64}
end
function LeniaKernel(R::Integer; peaks=(1.0,))
    R >= 1 || throw(ArgumentError("R must be at least 1, got $R"))
    isempty(peaks) && throw(ArgumentError("at least one peak is needed"))
    return LeniaKernel(Int(R), collect(Float64, peaks))
end
function _kernel_value(k::LeniaKernel, d, o)
    r = d / k.radius
    (r <= 0 || r >= 1) && return 0.0
    B = length(k.peaks)
    x = B * r
    i = min(floor(Int, x), B - 1)
    f = x - i
    return (0 < f < 1) ? k.peaks[i + 1] * exp(4 - 1 / (f * (1 - f))) : 0.0
end

"""
    AnisotropicKernel(base, direction, strength)

A `base` kernel tilted towards `direction` (radians; `0` points along
increasing column index, "east", and `π/2` along decreasing row index,
"north", as drawn by `plot`): the weight at an offset making angle `θ` with
`direction` is multiplied by `exp(strength * cos(θ))`, a von Mises tilt. Use it
for wind- or slope-driven dispersal.
"""
struct AnisotropicKernel{K<:DispersalKernel} <: DispersalKernel
    base::K
    direction::Float64
    strength::Float64
end
function AnisotropicKernel(base::DispersalKernel, direction::Real, strength::Real)
    strength >= 0 || throw(ArgumentError("strength must be nonnegative, got $strength"))
    return AnisotropicKernel(base, Float64(direction), Float64(strength))
end
_kernel_radius(k::DispersalKernel) = k.radius
_kernel_radius(k::AnisotropicKernel) = _kernel_radius(k.base)
function _kernel_value(k::AnisotropicKernel, d, o)
    w = _kernel_value(k.base, d, o)
    d == 0 && return w
    # offsets are (row, column); east is +column, north is -row
    θ = atan(-o[1], o[2])
    return w * exp(k.strength * cos(θ - k.direction))
end

"""
    EmpiricalKernel(weights::AbstractMatrix)

A two-dimensional kernel given as a matrix of weights with odd side lengths,
centred on the middle entry (the cell itself): `weights[i, j]` is the weight of
offset `(i - ci, j - cj)`.
"""
struct EmpiricalKernel <: DispersalKernel
    weights::Matrix{Float64}
    function EmpiricalKernel(w::AbstractMatrix{<:Real})
        all(isodd, size(w)) ||
            throw(ArgumentError("kernel matrix sides must be odd, got $(size(w))"))
        all(x -> x >= 0 && isfinite(x), w) ||
            throw(ArgumentError("kernel weights must be finite and nonnegative"))
        sum(w) > 0 || throw(ArgumentError("kernel weights must not all be zero"))
        return new(Matrix{Float64}(w))
    end
end
_kernel_radius(k::EmpiricalKernel) = maximum(size(k.weights) .÷ 2)

_distance(::Lattice{N,Square}, o) where {N} = sqrt(sum(abs2, Tuple(o)))
function _distance(::Lattice{2,Hex}, o)
    q, r = Tuple(o)
    return hypot(sqrt(3) * (q + r / 2), 1.5r) / sqrt(3)
end

"""
    kernel_weights(kernel, space; include_centre=true) -> (offsets, weights)

The offsets of a kernel on a lattice and their weights, normalised to sum to
one. Offsets farther than the kernel's radius, or with zero weight, are left
out; so is the zero offset (the cell itself) when `include_centre` is `false`.

```jldoctest
julia> offsets, weights = kernel_weights(UniformKernel(1), Torus(5, 5));

julia> length(offsets), sum(weights)
(5, 1.0)
```
"""
function kernel_weights(kernel::DispersalKernel, space::Lattice{N};
                        include_centre::Bool=true) where {N}
    r = _kernel_radius(kernel)
    offsets = CartesianIndex{N}[]
    weights = Float64[]
    for o in CartesianIndices(ntuple(_ -> (-r):r, N))
        all(iszero, Tuple(o)) && !include_centre && continue
        d = _distance(space, o)
        d <= r + 1e-9 || continue
        w = Float64(_kernel_value(kernel, d, Tuple(o)))
        w > 0 && (push!(offsets, o); push!(weights, w))
    end
    isempty(weights) &&
        throw(ArgumentError("kernel $kernel has no positive weight on $space"))
    return offsets, weights ./ sum(weights)
end
function kernel_weights(kernel::EmpiricalKernel, space::Lattice{2};
                        include_centre::Bool=true)
    c = (size(kernel.weights) .+ 1) .÷ 2
    offsets = CartesianIndex{2}[]
    weights = Float64[]
    for I in CartesianIndices(kernel.weights)
        o = CartesianIndex(Tuple(I) .- c)
        all(iszero, Tuple(o)) && !include_centre && continue
        w = kernel.weights[I]
        w > 0 && (push!(offsets, o); push!(weights, w))
    end
    isempty(weights) && throw(ArgumentError("the empirical kernel has no positive weight"))
    return offsets, weights ./ sum(weights)
end
function kernel_weights(kernel::EmpiricalKernel, space::Lattice; include_centre::Bool=true)
    return throw(IncompatibleSpaceError("EmpiricalKernel", _describe(space),
                                        "an empirical kernel matrix is two-dimensional"))
end
function kernel_weights(kernel::DispersalKernel, space::GraphSpace;
                        include_centre::Bool=true)
    return throw(IncompatibleSpaceError(string(nameof(typeof(kernel))), _describe(space),
                                        "dispersal kernels need a lattice"))
end
