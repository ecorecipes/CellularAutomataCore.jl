# Observation models: what an instrument or a survey would report from a simulated state. A
# footprint maps model cells to observation units (pixels, plots, inspected plants) by the share
# of each unit that each cell covers; a link turns the aggregated signal into a distribution
# over what is reported. Both simulate and give a log-density, so the same model serves
# simulation-based fitting and likelihood-based fitting. Masked or missing observations are
# skipped: that assumes missingness does not depend on the unobserved state.

"""
    MapGrid(dims; origin=0, spacing=1)

A regular grid of axis-aligned boxes in map coordinates, for building
[`footprint`](@ref)s: box `I` spans `origin[d] + (I[d] - 1) * spacing[d]` to
`origin[d] + I[d] * spacing[d]` along axis `d`. A negative spacing runs the
axis backwards, as the rows of most rasters do. `origin` and `spacing` are
numbers (the same on every axis) or tuples. Box `I` of a `MapGrid` with the
lattice's `dims` is cell `I` of the lattice; the linear order of boxes is the
order of `cells`.
"""
struct MapGrid{N}
    dims::NTuple{N,Int}
    origin::NTuple{N,Float64}
    spacing::NTuple{N,Float64}
    function MapGrid(dims::NTuple{N,Integer}; origin=0, spacing=1) where {N}
        o = origin isa Real ? ntuple(_ -> Float64(origin), N) : Float64.(Tuple(origin))
        s = spacing isa Real ? ntuple(_ -> Float64(spacing), N) : Float64.(Tuple(spacing))
        length(o) == N && length(s) == N ||
            throw(ArgumentError("origin and spacing need one value per axis ($N), got $origin and $spacing"))
        all(>(0), dims) ||
            throw(ArgumentError("grid dimensions must be positive, got $dims"))
        all(x -> x != 0 && isfinite(x), s) ||
            throw(ArgumentError("spacing must be finite and nonzero, got $spacing"))
        return new{N}(Int.(dims), o, s)
    end
end
MapGrid(dims::Integer...; kwargs...) = MapGrid(dims; kwargs...)
Base.size(g::MapGrid) = g.dims

# The extent of box i along axis d, as (low, high).
@inline function _box(g::MapGrid, d, i)
    a = g.origin[d] + (i - 1) * g.spacing[d]
    b = a + g.spacing[d]
    return min(a, b), max(a, b)
end

"""
    Footprint

The weights `W[j, i]` with which model cell `i` contributes to observation unit
`j`, stored sparsely by unit. Build one with [`footprint`](@ref) from two
[`MapGrid`](@ref)s (exact overlap of boxes), from a matrix with one row per unit
and one column per cell (for example point-spread weights computed elsewhere),
or from triplets with `Footprint(units, cells, weights; nunits, ncells)`.
[`aggregate`](@ref) applies it; [`coverage`](@ref) gives each unit's total
weight.
"""
struct Footprint{T<:Real}
    ncells::Int
    offsets::Vector{Int}
    cells::Vector{Int}
    weights::Vector{T}
end

function Footprint(units::AbstractVector{<:Integer}, cellidx::AbstractVector{<:Integer},
                   weights::AbstractVector{<:Real}; nunits::Integer, ncells::Integer)
    length(units) == length(cellidx) == length(weights) ||
        throw(ArgumentError("units, cells and weights must have the same length"))
    all(j -> 1 <= j <= nunits, units) ||
        throw(ArgumentError("unit indices must be in 1:$nunits"))
    all(i -> 1 <= i <= ncells, cellidx) ||
        throw(ArgumentError("cell indices must be in 1:$ncells"))
    all(w -> w >= 0 && isfinite(w), weights) ||
        throw(ArgumentError("footprint weights must be finite and nonnegative"))
    order = sortperm(collect(zip(units, cellidx)))
    counts = zeros(Int, nunits)
    foreach(j -> counts[j] += 1, units)
    offsets = cumsum([1; counts])
    return Footprint(Int(ncells), offsets, Int.(cellidx[order]),
                     float.(collect(weights[order])))
end

function Footprint(W::AbstractMatrix{<:Real})
    J, I, w = Int[], Int[], eltype(float(W))[]
    for j in axes(W, 1), i in axes(W, 2)
        x = W[j, i]
        if x != 0
            push!(J, j)
            push!(I, i)
            push!(w, x)
        end
    end
    return Footprint(J, I, w; nunits=size(W, 1), ncells=size(W, 2))
end

_nunits(F::Footprint) = length(F.offsets) - 1
Base.size(F::Footprint) = (_nunits(F), F.ncells)
function Base.Matrix(F::Footprint)
    W = zeros(eltype(F.weights), size(F))
    for j in 1:_nunits(F), k in F.offsets[j]:(F.offsets[j + 1] - 1)
        W[j, F.cells[k]] += F.weights[k]
    end
    return W
end
function Base.show(io::IO, F::Footprint)
    return print(io, "Footprint(", _nunits(F), " units, ", F.ncells, " cells, ",
                 length(F.weights), " weights)")
end

"""
    footprint(cells::MapGrid, units::MapGrid) -> Footprint

The exact overlap of two grids of boxes in the same map coordinates: the weight
of cell `i` in unit `j` is the volume (area in two dimensions) of their
intersection divided by the volume of the unit, so a unit's weights sum to the
share of it that the cells cover ([`coverage`](@ref)). Weights are not
renormalised where a unit extends beyond the cells: simulate a margin, or leave
those units out. The grids need not be nested or aligned, and either may be the
finer.
"""
function footprint(cg::MapGrid{N}, ug::MapGrid{N}) where {N}
    # per axis, the overlapping (cell index, unit index, length) along that axis
    axis_overlaps = ntuple(Val(N)) do d
        ov = Tuple{Int,Int,Float64}[]
        for j in 1:ug.dims[d]
            ulo, uhi = _box(ug, d, j)
            for i in 1:cg.dims[d]
                clo, chi = _box(cg, d, i)
                len = min(uhi, chi) - max(ulo, clo)
                len > 0 && push!(ov, (i, j, len))
            end
        end
        return ov
    end
    unit_volume = prod(abs, ug.spacing)
    clin = LinearIndices(cg.dims)
    ulin = LinearIndices(ug.dims)
    J, I, w = Int[], Int[], Float64[]
    for combo in Iterators.product(axis_overlaps...)
        ci = CartesianIndex(map(first, combo))
        uj = CartesianIndex(map(c -> c[2], combo))
        push!(J, ulin[uj])
        push!(I, clin[ci])
        push!(w, prod(map(last, combo)) / unit_volume)
    end
    return Footprint(J, I, w; nunits=prod(ug.dims), ncells=prod(cg.dims))
end

"""
    coverage(F::Footprint) -> Vector

Each unit's total weight: one for a unit wholly inside the model cells of an
overlap footprint, less for one that extends beyond them.
"""
function coverage(F::Footprint)
    return [sum(@view(F.weights[F.offsets[j]:(F.offsets[j + 1] - 1)]);
                init=zero(eltype(F.weights)))
            for j in 1:_nunits(F)]
end

"""
    aggregate(F::Footprint, v) -> Vector
    aggregate(F::Footprint, v, w) -> Vector

The signal each unit receives: `sum(W[j, i] * v[i] for i)`, for values `v`
with one entry per cell (an array of the lattice's size, read in linear order).
With cell weights `w` (for example host canopy), the weighted mean
`sum(W[j, i] * w[i] * v[i]) / sum(W[j, i] * w[i])`, which is `missing` for a
unit whose cells have no weight: the share of represented host that shows the
signal, not of the unit's whole area.
"""
function aggregate(F::Footprint, v::AbstractArray)
    length(v) == F.ncells ||
        throw(ArgumentError("aggregate needs one value per cell ($(F.ncells)), got $(length(v))"))
    return [sum((F.weights[k] * v[F.cells[k]] for k in F.offsets[j]:(F.offsets[j + 1] - 1));
                init=zero(float(eltype(v)))) for j in 1:_nunits(F)]
end
function aggregate(F::Footprint, v::AbstractArray, w::AbstractArray)
    length(v) == length(w) == F.ncells ||
        throw(ArgumentError("aggregate needs one value and one weight per cell ($(F.ncells))"))
    return map(1:_nunits(F)) do j
        num = 0.0
        den = 0.0
        for k in F.offsets[j]:(F.offsets[j + 1] - 1)
            a = F.weights[k] * w[F.cells[k]]
            num += a * v[F.cells[k]]
            den += a
        end
        return den > 0 ? num / den : missing
    end
end

# ------------------------------------------------------------------ links

# Observation noise draws from its own stream of the counter-based generator: the high word
# of the cell id carries a tag that no lattice cell id reaches, so observation draws never
# coincide with the draws of the dynamics under the same seed.
const _OBSERVATION_STREAM = UInt64(0x4f425356) << 32
@inline _obs_uniform(seed, j, t, k) = _uniform(UInt64(seed),
                                               _OBSERVATION_STREAM | UInt64(j),
                                               t, k)

"""
    Detection(false_positive, sensitivity)

A binary report (`true`/`1` positive) whose probability of being positive rises
linearly with the signal `r` in `[0, 1]`:
`false_positive + (sensitivity - false_positive) * r`. With `r` the share of a
unit showing symptoms, `false_positive` is the rate on units with none and
`sensitivity` the rate on units wholly symptomatic. The linear form assumes a
mixed unit's report mixes those of pure ones; a detector triggered by any
symptomatic plant needs another link. A unit with no represented host (signal
`missing`) is positive with the false-positive probability.
"""
struct Detection
    false_positive::Float64
    sensitivity::Float64
    function Detection(f::Real, s::Real)
        0 <= f <= 1 && 0 <= s <= 1 ||
            throw(ArgumentError("detection probabilities must be in [0, 1], got $f and $s"))
        return new(Float64(f), Float64(s))
    end
end

@inline function _positive_probability(d::Detection, r)
    x = r === missing ? 0.0 : Float64(r)
    0 <= x <= 1 || throw(ArgumentError("a Detection signal must be in [0, 1], got $r"))
    return d.false_positive + (d.sensitivity - d.false_positive) * x
end

"""
    Misclassification(M; classes=1:size(M, 1))

A class label reported for each cell with error: `M[a, b]` is the probability
that a cell of true class `a` is reported as class `b` (each row sums to one).
`classes[a]` is the state, or the value of the signal, that class `a` stands
for, and reports are class numbers `b`. Labels of mixed units need their own
rule, so a `Misclassification` takes no footprint.
"""
struct Misclassification{C}
    M::Matrix{Float64}
    classes::C
    function Misclassification(M::AbstractMatrix{<:Real}; classes=1:size(M, 1))
        size(M, 1) == length(classes) ||
            throw(ArgumentError("a confusion matrix needs one row per class ($(length(classes))), got $(size(M, 1))"))
        all(>=(0), M) || throw(ArgumentError("confusion probabilities must be nonnegative"))
        all(r -> isapprox(sum(r), 1; atol=1e-8), eachrow(M)) ||
            throw(ArgumentError("each row of a confusion matrix must sum to one"))
        return new{typeof(classes)}(Float64.(M), classes)
    end
end

function _class_row(m::Misclassification, r)
    a = findfirst(==(r), m.classes)
    a === nothing &&
        throw(ArgumentError("signal $r is not one of the classes $(m.classes)"))
    return a
end

"""
    GaussianError(sd)

A continuous report equal to the signal plus independent normal error with
standard deviation `sd`, for fractions or reflectances on a scale where that is
a fair approximation.
"""
struct GaussianError
    sd::Float64
    function GaussianError(sd::Real)
        sd > 0 || throw(ArgumentError("sd must be positive, got $sd"))
        return new(Float64(sd))
    end
end

function _unit_logpdf(d::Detection, y, r)
    q = _positive_probability(d, r)
    return log(_is_positive(y) ? q : 1 - q)
end
_is_positive(y::Bool) = y
function _is_positive(y::Real)
    y == 0 || y == 1 || throw(ArgumentError("a Detection report must be 0 or 1, got $y"))
    return y == 1
end
function _unit_logpdf(m::Misclassification, y, r)
    1 <= y <= size(m.M, 2) ||
        throw(ArgumentError("a reported class must be in 1:$(size(m.M, 2)), got $y"))
    return log(m.M[_class_row(m, r), y])
end
function _unit_logpdf(g::GaussianError, y, r)
    r === missing &&
        throw(ArgumentError("a GaussianError report needs a signal; got missing"))
    z = (y - r) / g.sd
    return -z^2 / 2 - log(g.sd) - log(2π) / 2
end

function _unit_draw(d::Detection, r, seed, j, t)
    return _obs_uniform(seed, j, t, 1) < _positive_probability(d, r)
end
function _unit_draw(m::Misclassification, r, seed, j, t)
    row = @view m.M[_class_row(m, r), :]
    u = _obs_uniform(seed, j, t, 1)
    acc = 0.0
    last_positive = findlast(>(0), row)
    for b in eachindex(row)
        acc += row[b]
        row[b] > 0 && u < acc && return b
    end
    return last_positive
end
function _unit_draw(g::GaussianError, r, seed, j, t)
    r === missing &&
        throw(ArgumentError("a GaussianError report needs a signal; got missing"))
    u1 = 1 - _obs_uniform(seed, j, t, 1)
    u2 = _obs_uniform(seed, j, t, 2)
    return r + g.sd * sqrt(-2 * log(u1)) * cospi(2 * u2)
end

# ------------------------------------------------------------------ observation models

"""
    Observation(link; signal=identity, weight=nothing, footprint=nothing)

What is reported about a state: each cell's state gives a `signal` (for example
`s -> s == 2` for symptomatic, or a layer's value), the [`Footprint`](@ref)
aggregates the signals to observation units (with cell `weight`s, such as host
canopy, the weighted mean; see [`aggregate`](@ref)), and the `link`
([`Detection`](@ref), [`Misclassification`](@ref) or [`GaussianError`](@ref))
gives the distribution of each unit's report. Without a footprint each cell is
its own unit. `weight` is a function of a cell's state or an array with one
weight per cell. Units' reports are independent given the state.

[`observation_logpdf`](@ref) scores reports and [`simulate_observation`](@ref)
draws them.
"""
struct Observation{L,S,W,F}
    link::L
    signal::S
    weight::W
    footprint::F
    function Observation(link; signal=identity, weight=nothing, footprint=nothing)
        link isa Union{Detection,Misclassification,GaussianError} ||
            throw(ArgumentError("link must be a Detection, Misclassification or GaussianError, got $(typeof(link))"))
        link isa Misclassification && footprint !== nothing &&
            throw(ArgumentError("a Misclassification reports a class per cell and takes no footprint"))
        footprint === nothing || footprint isa Footprint ||
            throw(ArgumentError("footprint must be a Footprint, got $(typeof(footprint))"))
        return new{typeof(link),typeof(signal),typeof(weight),typeof(footprint)}(link,
                                                                                 signal,
                                                                                 weight,
                                                                                 footprint)
    end
end

function _cell_values(f, u)
    n = prod(_state_size(u))
    return [f(_read(u, i)) for i in 1:n]
end
_cell_weights(::Nothing, u) = nothing
_cell_weights(w::AbstractArray, u) = vec(w)
_cell_weights(w, u) = _cell_values(w, u)

function _signals(obs::Observation, u)
    s = _cell_values(obs.signal, u)
    w = _cell_weights(obs.weight, u)
    if obs.footprint === nothing
        w === nothing && return s
        length(w) == length(s) ||
            throw(ArgumentError("weight needs one value per cell ($(length(s))), got $(length(w))"))
        return [w[i] > 0 ? s[i] : missing for i in eachindex(s)]
    end
    obs.footprint.ncells == length(s) ||
        throw(StateShapeError("the state for the footprint", (obs.footprint.ncells,),
                              (length(s),)))
    return w === nothing ? aggregate(obs.footprint, s) : aggregate(obs.footprint, s, w)
end

"""
    observation_logpdf(obs::Observation, y, u) -> Float64

The log-probability of the reports `y` (one per observation unit, in unit
order; `missing` for a masked or cloudy unit, which is skipped) given the state
`u`. Sum it over observation times, with the states at those times, for a
whole series.
"""
function observation_logpdf(obs::Observation, y::AbstractArray, u)
    r = _signals(obs, u)
    length(y) == length(r) ||
        throw(ArgumentError("observation_logpdf needs one report per unit ($(length(r))), got $(length(y))"))
    ll = 0.0
    for (yj, rj) in zip(y, r)
        yj === missing && continue
        ll += _unit_logpdf(obs.link, yj, rj)
        ll == -Inf && return ll
    end
    return ll
end

"""
    simulate_observation(obs::Observation, u, t=0; seed=0, mask=nothing) -> Vector

Reports drawn for the state `u` at time `t`: one per unit, `missing` where
`mask` (one `Bool` per unit, `true` for observed) is `false`. The draws come
from the counter-based generator, in a stream kept apart from the dynamics'
draws, keyed by `seed`, unit and time, so they are reproducible on any machine.
"""
function simulate_observation(obs::Observation, u, t::Integer=0; seed::Integer=0,
                              mask=nothing)
    r = _signals(obs, u)
    mask === nothing || length(mask) == length(r) ||
        throw(ArgumentError("mask needs one value per unit ($(length(r))), got $(length(mask))"))
    out = [_unit_draw(obs.link, r[j], seed, j, t) for j in eachindex(r)]
    mask === nothing && return out
    return [mask[j] ? out[j] : missing for j in eachindex(out)]
end
