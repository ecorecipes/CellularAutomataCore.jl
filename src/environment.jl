# The environment of a rule beyond its neighbourhood: covariate layers (fixed or cycling in
# time), kernel fields (weighted sums of a cell-level quantity over a dispersal kernel,
# recomputed from the state at the start of every step) and masks (cells that never update).

"""
    KernelField(kernel, source=identity; include_centre=true, method=:direct)

A field recomputed at the start of every step: at each cell `I`, the sum over
the displacements `o` of `kernel` of `weight(o) * source(state at I - o)`, with
the state read beyond the edges according to the space's boundary conditions.
The kernel is a dispersal kernel: `weight(o)` is the share of propagules that
travel displacement `o` from their source, so an [`AnisotropicKernel`](@ref)
pointing east spreads infection eastwards. Rules read it with
[`field`](@ref)`(ctx, name)`. `source` maps a cell state to a number, for
example `cell -> cell == :I` for the density of infectious cells around each
cell. Weights are normalised to sum to one ([`kernel_weights`](@ref)).

`method = :fft` computes the sum by fast Fourier transform, which is much
faster for wide kernels; it needs `using FFTW`, periodic boundaries on every
axis, and agrees with `:direct` to floating-point rounding rather than exactly.
`precision = Float32` stores the weights and the field in single precision, as
devices without double precision (Apple GPUs through Metal) need.
"""
struct KernelField{K<:DispersalKernel,F,T<:AbstractFloat}
    kernel::K
    source::F
    include_centre::Bool
    method::Symbol
    precision::Type{T}
end
function KernelField(kernel::DispersalKernel, source=identity; include_centre::Bool=true,
                     method::Symbol=:direct, precision::Type{<:AbstractFloat}=Float64)
    method in (:direct, :fft) ||
        throw(ArgumentError("method must be :direct or :fft, got :$method"))
    return KernelField(kernel, source, include_centre, method, precision)
end

"""
    fields(rule) -> NamedTuple

The [`KernelField`](@ref)s a rule needs, by name; [`CAProblem`](@ref) merges
them with the fields it is given. The default is none. [`Lenia`](@ref) declares
its potential field this way.
"""
fields(::AbstractRule) = (;)

# A field resolved on a space: offsets and weights in vectors, so wide kernels do not become
# huge tuples. `radius` is the halo it needs. Every field is a bits type (no Symbol), so it can
# be passed into GPU kernels.
struct _ResolvedField{F,O,W}
    source::F
    offsets::O
    weights::W
    fft::Bool
    radius::Int
end
Adapt.@adapt_structure _ResolvedField

function _resolve_field(f::KernelField, space::Lattice, name)
    offsets, weights = kernel_weights(f.kernel, space; include_centre=f.include_centre)
    # a cell receives from the sources at I - o: store the offsets to read, -o
    offsets = map(o -> -o, offsets)
    weights = convert(Vector{f.precision}, weights)
    if f.method == :fft && !all(b -> b isa Periodic, boundaries(space))
        throw(ArgumentError("field :$name uses method=:fft, which needs periodic boundaries on every axis"))
    end
    radius = maximum(o -> maximum(abs, Tuple(o)), offsets)
    return _ResolvedField(f.source, offsets, weights, f.method == :fft, radius)
end
function _resolve_field(f::KernelField, space::GraphSpace, name)
    return throw(IncompatibleSpaceError("kernel field :$name", _describe(space),
                                        "kernel fields need a lattice"))
end
function _resolve_field(f, space, name)
    return throw(ArgumentError("field :$name must be a KernelField, got $(typeof(f))"))
end

_halo(fields::NamedTuple) = isempty(fields) ? 0 : maximum(f -> f.radius, values(fields))

# Covariates: arrays the size of the space (fixed), or with one more axis for time, read at
# time t from slice mod1(t + 1, nt) so a covariate with nt slices repeats with period nt.
function _check_covariates(covs::NamedTuple, space)
    for (k, a) in pairs(covs)
        a isa AbstractArray ||
            throw(ArgumentError("covariate :$k must be an array, got $(typeof(a))"))
        N = ndims(space)
        (ndims(a) == N || ndims(a) == N + 1) && size(a)[1:N] == size(space) ||
            throw(StateShapeError("covariate :$k", size(space), size(a)))
    end
    return nothing
end
function _check_covariates(covs, space)
    return throw(ArgumentError("covariates must be a NamedTuple of arrays"))
end

@inline _covariate_value(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}, t) where {N} = @inbounds a[I]
@inline function _covariate_value(a::AbstractArray{<:Any,M}, I::CartesianIndex{N},
                                  t) where {M,N}
    return @inbounds a[I, mod1(t + 1, size(a, M))]
end
@inline _covariate_value(a::AbstractVector, i::Integer, t) = @inbounds a[i]
@inline _covariate_value(a::AbstractMatrix, i::Integer, t) = @inbounds a[i,
                                                                         mod1(t + 1,
                                                                              size(a, 2))]

"""
    covariate(ctx::CellContext, name::Symbol)

The value of covariate `name` at the cell: a fixed layer if the covariate array
has the size of the space, or the slice for the current time if it has one more
axis (slice `mod1(t + 1, nt)` of `nt`, so it repeats every `nt` steps).
Covariates are given to [`CAProblem`](@ref) as a `NamedTuple` of arrays.
"""
@inline covariate(ctx::CellContext, name::Symbol) = _covariate_value(getfield(ctx.covariates,
                                                                              name),
                                                                     ctx.index, ctx.t)

"""
    field(ctx::CellContext, name::Symbol)

The value at the cell of the [`KernelField`](@ref) `name`, computed from the
state at the start of the current step.
"""
@inline field(ctx::CellContext, name::Symbol) = @inbounds getfield(ctx.fields, name)[ctx.index]

function _check_mask(mask, space)
    mask === nothing && return nothing
    mask isa AbstractArray{Bool} || throw(ArgumentError("mask must be an array of Bools"))
    size(mask) == size(space) || throw(StateShapeError("the mask", size(space), size(mask)))
    return nothing
end

# Everything a sweep needs besides the states: passed whole into engines and GPU kernels.
struct _Env{P,C,F,M}
    p::P
    covariates::C
    fields::F
    mask::M
    seed::UInt64
end
Adapt.@adapt_structure _Env

@inline function _context(env::_Env, I, t, id)
    return CellContext(env.p, I, t, id, env.seed, env.covariates, env.fields)
end

@inline _masked(::Nothing, I) = false
@inline _masked(mask, I) = @inbounds !mask[I]

# Field values from the reference definition: boundary semantics on every access.
function _compute_field_reference!(out, rf::_ResolvedField, space::Lattice, u)
    for I in cells(space)
        s = zero(eltype(rf.weights))
        for k in eachindex(rf.weights)
            s += rf.weights[k] * rf.source(_boundary_read(space, u, I + rf.offsets[k]))
        end
        out[I] = s
    end
    return out
end

# Field values from a padded state whose halo already holds the boundary values.
@inline function _field_at(cur, rf, P)
    s = zero(eltype(rf.weights))
    @inbounds for k in eachindex(rf.weights)
        s += rf.weights[k] * rf.source(_read(cur, P + rf.offsets[k]))
    end
    return s
end

function _compute_field_padded!(out, rf::_ResolvedField, cur, shift, dims, threaded)
    inner = CartesianIndices(dims)
    if threaded
        Threads.@threads for I in inner
            @inbounds out[I] = _field_at(cur, rf, I + shift)
        end
    else
        for I in inner
            @inbounds out[I] = _field_at(cur, rf, I + shift)
        end
    end
    return out
end

# FFT fields are provided by the FFTW extension.
function _compute_field_fft!(out, rf, src, cache)
    return throw(ArgumentError("a KernelField with method=:fft needs FFTW: run `using FFTW` first"))
end
_fft_cache(rf, dims) = nothing

function _source_array(rf, cur, shift, dims)
    T = eltype(rf.weights)
    return [T(rf.source(_read(cur, I + shift))) for I in CartesianIndices(dims)]
end
