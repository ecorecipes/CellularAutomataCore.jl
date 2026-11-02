# Lenia: a continuous cellular automaton with a smooth shell kernel and a Gaussian growth
# function (Chan 2019), built on kernel fields.

"""
    Lenia(; R=13, μ=0.15, σ=0.015, dt=0.1, peaks=(1.0,), method=:direct, precision=Float64)

The continuous Lenia rule [Chan2019](@cite) on states in `[0, 1]`. At every
step each cell computes the potential `U`, the [`LeniaKernel`](@ref)`(R; peaks)`
weighted mean of the states around it, and moves by
`dt * (2 exp(-(U - μ)² / 2σ²) - 1)`, clipped to `[0, 1]`. The potential is the
rule's own [`KernelField`](@ref) named `:potential`; `method = :fft` computes
it by FFT (needs `using FFTW` and a periodic lattice), which is much faster for
the usual radius of 13. `precision = Float32` runs the rule and its field in
single precision (for Apple GPUs through Metal; use a `Float32` state).
[`ORBIUM`](@ref) is a glider for the default parameters.
"""
struct Lenia{T<:AbstractFloat,K} <: AbstractRule
    μ::T
    σ::T
    dt::T
    kernel::K
    fft::Bool
end
function Lenia(; R::Integer=13, μ::Real=0.15, σ::Real=0.015, dt::Real=0.1, peaks=(1.0,),
               method::Symbol=:direct, precision::Type{<:AbstractFloat}=Float64)
    σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
    0 < dt <= 1 || throw(ArgumentError("dt must be in (0, 1], got $dt"))
    method in (:direct, :fft) ||
        throw(ArgumentError("method must be :direct or :fft, got :$method"))
    T = precision
    return Lenia(T(μ), T(σ), T(dt), LeniaKernel(R; peaks), method == :fft)
end
# On a device the rule needs only its numbers; the kernel lives in the resolved field.
Adapt.adapt_structure(to, r::Lenia) = Lenia(r.μ, r.σ, r.dt, nothing, r.fft)

function fields(r::Lenia{T}) where {T}
    return (potential=KernelField(r.kernel, identity; method=r.fft ? :fft : :direct,
                                  precision=T),)
end
neighbourhood(::Lenia) = nothing

@inline _lenia_growth(r::Lenia, u) = 2 * exp(-(u - r.μ)^2 / (2 * r.σ^2)) - 1

@inline function local_rule(r::Lenia, cell, nbrs, ctx)
    return clamp(cell + r.dt * _lenia_growth(r, field(ctx, :potential)), zero(cell),
                 one(cell))
end

function validate_state(r::Lenia, u0)
    u0 isa AbstractArray{<:AbstractFloat} ||
        throw(InvalidStateError(_rule_name(r), nothing, nothing,
                                "Lenia needs a floating-point state"))
    for I in eachindex(IndexCartesian(), u0)
        0 <= u0[I] <= 1 ||
            throw(InvalidStateError(_rule_name(r), I, u0[I], "states must be in [0, 1]"))
    end
    return nothing
end

function Base.show(io::IO, r::Lenia)
    return print(io, "Lenia(R=", r.kernel === nothing ? "?" : r.kernel.radius, ", μ=", r.μ,
                 ", σ=", r.σ, ", dt=", r.dt, ")")
end

const _ORBIUM_ROWS = [[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0,
                       0, 0],
                      [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15,
                       0.09,
                       0.2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13,
                       0.19,
                       0.18, 0.45, 0, 0, 0],
                      [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02,
                       0.16,
                       0.68, 0, 0, 0],
                      [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0,
                       0, 0.18,
                       0.42, 0, 0],
                      [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0,
                       0, 0,
                       0, 0.82, 0, 0],
                      [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0,
                       0, 0,
                       0.22, 0.17, 0],
                      [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0,
                       0, 0,
                       0.49, 0],
                      [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0,
                       0, 0, 0,
                       0.36, 0],
                      [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0,
                       0, 0, 0,
                       0.22, 0.07],
                      [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0,
                       0.18, 0.11],
                      [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0,
                       0.19,
                       0.1],
                      [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14,
                       0.04,
                       0.1, 0.21, 0.05],
                      [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31,
                       0.18,
                       0.19, 0.2, 0.01],
                      [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6,
                       0.39,
                       0.28, 0.24, 0.13, 0],
                      [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6,
                       0.51,
                       0.39, 0.29, 0.19, 0.04, 0],
                      [0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46,
                       0.4,
                       0.31, 0.21, 0.08, 0, 0],
                      [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33,
                       0.26,
                       0.18, 0.09, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18,
                       0.13, 0.05,
                       0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0,
                       0, 0, 0,
                       0]]

"""
    ORBIUM

A 20 × 20 seed of *Orbium unicaudatus*, the glider of Lenia [Chan2019](@cite)
for `R = 13`, `μ = 0.15`, `σ = 0.015`, `dt = 0.1` (the pattern as distributed
with CellIA.jl, MIT licence). Place it on an empty grid with [`place!`](@ref).
"""
const ORBIUM = Float64.(permutedims(reduce(hcat, _ORBIUM_ROWS)))
