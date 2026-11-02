# Counter-based random numbers. A draw is a pure function of (seed, cell, time, draw
# number), so a synchronous stochastic rule gives the same result whatever order the
# cells are visited in: serially, on several threads, on a GPU or on an agent backend.
# Philox4x32-10 (Salmon et al. 2011), the generator of the Random123 library.

const _PHILOX_M0 = 0xD2511F53
const _PHILOX_M1 = 0xCD9E8D57
const _PHILOX_W0 = 0x9E3779B9
const _PHILOX_W1 = 0xBB67AE85

@inline function _mulhilo(a::UInt32, b::UInt32)
    p = UInt64(a) * UInt64(b)
    return (p >> 32) % UInt32, p % UInt32
end

"""
    CellularAutomataCore.philox4x32(counter, key) -> NTuple{4,UInt32}

The Philox4x32-10 bijection of the Random123 library [Salmon2011](@cite): ten
rounds mapping a 128-bit `counter::NTuple{4,UInt32}` under a 64-bit
`key::NTuple{2,UInt32}` to 128 pseudo-random bits. Allocation-free and usable
inside GPU kernels. [`uniform`](@ref) builds on it.
"""
@inline function philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for round in 1:10
        hi0, lo0 = _mulhilo(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(_PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ⊻ c1 ⊻ k0, lo1, hi0 ⊻ c3 ⊻ k1, lo0
        if round < 10
            k0 += _PHILOX_W0
            k1 += _PHILOX_W1
        end
    end
    return (c0, c1, c2, c3)
end

"""
    CellContext

What a local rule knows besides the states of its cell and neighbours. Fields:

- `p`: the problem's parameters (`NullParameters()` if none were given); a
  callback may reassign `integ.p` to change them during a run;
- `index`: the cell's index in its space (a `CartesianIndex` on a lattice, an
  `Int` on a [`GraphSpace`](@ref));
- `t`: the time of the state being read; the rule computes the state at `t + 1`.

Random numbers for the cell come from [`uniform`](@ref) and
[`bernoulli`](@ref), never from a global generator. Covariate layers and kernel
fields are read with [`covariate`](@ref) and [`field`](@ref).
"""
struct CellContext{P,I,C,F}
    p::P
    index::I
    t::Int
    cellid::Int
    seed::UInt64
    covariates::C
    fields::F
end
function CellContext(p, index, t::Integer, cellid::Integer, seed::Integer)
    return CellContext(p, index, Int(t), Int(cellid), UInt64(seed), (;), (;))
end

"""
    uniform(ctx::CellContext, k::Integer=1) -> Float64

The `k`-th uniform random number in `[0, 1)` for the cell and time step of `ctx`.
The value is a pure function of the problem's `seed`, the cell, the time and `k`:
the same model gives the same numbers on every backend and thread count. Use a
different `k ≥ 1` for every independent draw a rule makes in one step; `k = 0`
is reserved for the update schemes (see [`Stochastic`](@ref)).

```jldoctest
julia> ctx = CellularAutomataCore.CellContext(nothing, CartesianIndex(1), 0, 1, 42);

julia> uniform(ctx) == uniform(ctx)
true

julia> uniform(ctx, 1) == uniform(ctx, 2)
false
```
"""
@inline function uniform(ctx::CellContext, k::Integer=1)
    return _uniform(ctx.seed, ctx.cellid, ctx.t, k)
end

"""
    uniform(Float32, ctx::CellContext, k::Integer=1) -> Float32

The `k`-th draw as a `Float32` in `[0, 1)`, with 24 random bits: for rules that
run on devices without double precision, such as Apple GPUs through Metal.
[`bernoulli`](@ref) uses it when the probability is a `Float32`.
"""
@inline uniform(::Type{Float32}, ctx::CellContext, k::Integer=1) = _uniform32(ctx.seed,
                                                                              ctx.cellid,
                                                                              ctx.t, k)
@inline uniform(::Type{Float64}, ctx::CellContext, k::Integer=1) = uniform(ctx, k)

@inline function _philox_draw(seed::UInt64, cellid::Integer, t::Integer, k::Integer)
    id = UInt64(cellid)
    ctr = (id % UInt32, (id >> 32) % UInt32, UInt64(t) % UInt32, k % UInt32)
    key = (seed % UInt32, (seed >> 32) % UInt32)
    return philox4x32(ctr, key)
end

@inline function _uniform32(seed::UInt64, cellid::Integer, t::Integer, k::Integer)
    r = _philox_draw(seed, cellid, t, k)
    return Float32(r[1] >> 8) * _TWO_TO_MINUS_24
end
# A Float32 constant: hex float literals are Float64, which Metal cannot use.
const _TWO_TO_MINUS_24 = 1.0f0 / 16777216

@inline function _uniform(seed::UInt64, cellid::Integer, t::Integer, k::Integer)
    r = _philox_draw(seed, cellid, t, k)
    bits = (UInt64(r[1]) << 32) | UInt64(r[2])
    return (bits >> 11) * 0x1.0p-53
end

"""
    bernoulli(ctx::CellContext, p, k::Integer=1) -> Bool

`true` with probability `p`, using the `k`-th draw of [`uniform`](@ref). The draw
has the precision of `p`: a `Float32` probability uses `uniform(Float32, ctx, k)`,
so a rule written with `Float32` parameters needs no double precision.
"""
@inline bernoulli(ctx::CellContext, p, k::Integer=1) = uniform(ctx, k) < p
@inline bernoulli(ctx::CellContext, p::Float32, k::Integer=1) = uniform(Float32, ctx, k) < p
