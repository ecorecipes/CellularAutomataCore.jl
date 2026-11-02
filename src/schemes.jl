# Update schemes: in what order cells update. Synchronous, Stochastic and Checkerboard are
# "gated" synchronous sweeps (a gate says which cells update in a sweep; the others keep their
# state), so every backend that can sweep can run them. RandomSequential and Margolus are
# inherently sequential or block-structured and run on the CPU backends.

"""
    AbstractScheme

Supertype of update schemes, the algorithms passed to `solve` (a subtype of
SciMLBase's `AbstractSciMLAlgorithm`): [`Synchronous`](@ref),
[`Stochastic`](@ref), [`Checkerboard`](@ref), [`RandomSequential`](@ref),
[`Margolus`](@ref) and, in continuous time, [`Gillespie`](@ref).
"""
abstract type AbstractScheme <: SciMLBase.AbstractSciMLAlgorithm end

"""
    Synchronous()

Every cell computes its next state from the current states of its neighbours,
then all cells update at once (double buffering).
"""
struct Synchronous <: AbstractScheme end

"""
    Stochastic(p)

A synchronous step in which each cell updates with probability `p` and
otherwise keeps its state: an asynchronous scheme that still runs in parallel.
Whether a cell updates is decided by its draw `k = 0`
([`uniform`](@ref)), so it is reproducible on every backend; the draw has the
precision of `p`, so `Stochastic(0.5f0)` needs no double precision.
"""
struct Stochastic{T<:AbstractFloat} <: AbstractScheme
    p::T
    function Stochastic(p::Real)
        0 < p <= 1 || throw(ArgumentError("update probability must be in (0, 1], got $p"))
        q = p isa AbstractFloat ? p : Float64(p)
        return new{typeof(q)}(q)
    end
end

"""
    Checkerboard()

Two synchronous half-steps per step on a lattice: first the cells whose indices
sum to an even number, then the others, each half reading the state left by the
previous one. On a square lattice with a von Neumann neighbourhood the cells
updated together are never neighbours. Kernel fields are computed once, at the
start of the step.
"""
struct Checkerboard <: AbstractScheme end

"""
    RandomSequential()

Cells update one at a time, in a fresh random order every step, each reading
the current (partly updated) state: the classic asynchronous scheme. The order
is drawn from the problem's seed, so runs are reproducible. Kernel fields are
computed once, at the start of the step. Sequential by nature, so it runs on
the CPU backends only.
"""
struct RandomSequential <: AbstractScheme end

"""
    Margolus()

The Margolus neighbourhood scheme for [`BlockRule`](@ref)s: the lattice is cut
into 2 × 2 blocks, each block is replaced by the rule's function of it, and the
partition shifts by one cell diagonally between even and odd steps. Needs a
two-dimensional periodic lattice with even side lengths.
"""
struct Margolus <: AbstractScheme end

const _GatedScheme = Union{Synchronous,Stochastic,Checkerboard}

# Gates: which cells update in one synchronous sweep.
struct _AllCells end
struct _Probability{T}
    p::T
end
struct _Parity
    odd::Bool
end
@inline _gate_open(::_AllCells, ctx, I) = true
@inline _gate_open(g::_Probability, ctx, I) = _uniform(ctx.seed, ctx.cellid, ctx.t, 0) < g.p
@inline function _gate_open(g::_Probability{Float32}, ctx, I)
    return _uniform32(ctx.seed, ctx.cellid, ctx.t, 0) < g.p
end
@inline _gate_open(g::_Parity, ctx, I::CartesianIndex) = isodd(sum(Tuple(I))) == g.odd

_gates(::Synchronous) = (_AllCells(),)
_gates(s::Stochastic) = (_Probability(s.p),)
_gates(::Checkerboard) = (_Parity(false), _Parity(true))

"""
    BlockRule(f)

A rule for the [`Margolus`](@ref) scheme. `f(block, ctx)` receives the states of
a 2 × 2 block as a tuple `(top_left, bottom_left, top_right, bottom_right)`
(column-major order) and a [`CellContext`](@ref) whose `index` is the block's
top-left cell, and returns the block's new states in the same order. Block rules
model conserved, reversible or particle-like dynamics: lattice gases, sand.
"""
struct BlockRule{F} <: AbstractRule
    f::F
end
neighbourhood(::BlockRule) = nothing
function local_rule(r::BlockRule, cell, nbrs, ctx)
    return throw(ArgumentError("a BlockRule has no local rule; solve it with the Margolus() scheme"))
end
function Base.show(io::IO, r::BlockRule)
    name = string(nameof(r.f))
    return print(io, "BlockRule(", startswith(name, '#') ? "<anonymous function>" : name,
                 ")")
end

function _check_rule_scheme(rule, scheme)
    if rule isa BlockRule && !(scheme isa Margolus)
        throw(ArgumentError("a BlockRule needs the Margolus() scheme, not $scheme"))
    elseif scheme isa Margolus && !(rule isa BlockRule)
        throw(ArgumentError("the Margolus() scheme needs a BlockRule, not $(typeof(rule))"))
    elseif scheme isa Gillespie && !(rule isa EventRule)
        throw(ArgumentError("the Gillespie() scheme needs an EventRule, not $(typeof(rule))"))
    end
    return nothing
end

function _check_scheme_space(::Margolus, space)
    ok = space isa Lattice{2,Square} && all(b -> b isa Periodic, boundaries(space)) &&
         all(iseven, size(space))
    ok || throw(IncompatibleSpaceError("Margolus()", _describe(space),
                                       "needs a two-dimensional periodic square lattice with even sides"))
    return nothing
end
function _check_scheme_space(::Checkerboard, space::GraphSpace)
    return throw(IncompatibleSpaceError("Checkerboard()", _describe(space),
                                        "needs a lattice"))
end
_check_scheme_space(::AbstractScheme, space) = nothing
