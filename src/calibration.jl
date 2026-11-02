# Calibration by approximate Bayesian computation (ABC): propose parameter sets, simulate each
# once as one trajectory of an EnsembleProblem, and keep the proposals whose simulations lie
# closest to the observations. Proposal i runs with the seed seed_trajectory gives trajectory i,
# so a result is reproducible and does not depend on the ensemble algorithm.

"""
    ABCResult

The result of [`abc_rejection`](@ref), with fields

- `samples`: the accepted parameter sets, closest first, an approximate sample
  from the posterior;
- `distances`: their distances from the observations;
- `tolerance`: the largest accepted distance (the requested tolerance if
  nothing was accepted);
- `proposals`, `proposal_distances`: every proposal, in the order proposed,
  and its distance.

With Plots.jl loaded, `plot(res)` draws a histogram of each parameter of the
proposals and of the accepted samples, when the parameter sets are
`NamedTuple`s of numbers.
"""
struct ABCResult{P}
    samples::Vector{P}
    distances::Vector{Float64}
    tolerance::Float64
    proposals::Vector{P}
    proposal_distances::Vector{Float64}
end

function Base.show(io::IO, res::ABCResult)
    n, k = length(res.proposals), length(res.samples)
    print(io, "ABCResult: ", k, " of ", n, " proposals accepted")
    k > 0 && print(io, ", distances ", round(first(res.distances); sigdigits=3), " to ",
                   round(res.tolerance; sigdigits=3))
    return nothing
end

"""
    abc_rejection(prob, prior, distance, observed; n, rng, keep=max(1, round(Int, n / 20)),
                  tolerance=Inf, alg=Synchronous(), ensemblealg=EnsembleThreads(),
                  kwargs...) -> ABCResult
    abc_rejection(prob, proposals::AbstractVector, distance, observed; keep, tolerance,
                  alg, ensemblealg, kwargs...) -> ABCResult

Calibrate the parameters of `prob` against `observed` data by rejection
approximate Bayesian computation [Beaumont2002](@cite).
`n` parameter sets are drawn from the prior, `prior(rng)` with the random
number generator `rng` (for example `Xoshiro(1)` from Random), or are given as
the vector `proposals`. Proposal `i` is simulated once, as
`remake(prob; p=proposals[i])` with the seed that [`seed_trajectory`](@ref)
gives trajectory `i`, and scored by `distance(sol, observed)`, a real number
that is small when the simulation `sol` resembles the observations. The `keep`
closest proposals whose distance is at most `tolerance` are accepted.

Useful distances compare maps, as `1 - cohens_kappa(final_state(sol), observed)`
(see [`cohens_kappa`](@ref)), or summaries, such as the difference in the
number of occupied cells or of clusters ([`cluster_sizes`](@ref)).

The simulations run as an `EnsembleProblem` solved with `alg` and
`ensemblealg`; other keywords (`backend`, `saveat`, `save_everystep`,
`callback`, ...) are passed to `solve`. Only the distances are kept, not the
solutions. Pass `save_everystep=false` when the distance needs only the final
state.
"""
function abc_rejection(prob::CAProblem, proposals::AbstractVector, distance, observed;
                       keep::Integer=max(1, round(Int, length(proposals) / 20)),
                       tolerance::Real=Inf, alg::AbstractScheme=Synchronous(),
                       ensemblealg=EnsembleThreads(), kwargs...)
    n = length(proposals)
    n >= 1 || throw(ArgumentError("abc_rejection needs at least one proposal"))
    1 <= keep <= n ||
        throw(ArgumentError("keep must be between 1 and the number of proposals ($n), got $keep"))
    ps = collect(proposals)
    prob_func = (pr, ctx) -> remake(seed_trajectory(pr, ctx); p=ps[ctx.sim_id])
    output_func = (sol, ctx) -> (Float64(distance(sol, observed)), false)
    ens = EnsembleProblem(prob; prob_func, output_func)
    d = Float64.(solve(ens, alg, ensemblealg; trajectories=n, kwargs...).u)
    order = sortperm(d)  # NaN distances sort last and are never accepted
    accepted = first(filter(i -> d[i] <= tolerance, order), keep)
    tol = isempty(accepted) ? Float64(tolerance) : d[last(accepted)]
    return ABCResult(ps[accepted], d[accepted], tol, ps, d)
end

function abc_rejection(prob::CAProblem, prior, distance, observed; n::Integer,
                       rng=nothing, kwargs...)
    rng === nothing &&
        throw(ArgumentError("abc_rejection needs `rng`, a random number generator for the prior (for example `Xoshiro(1)` from Random)"))
    n >= 1 || throw(ArgumentError("n must be at least 1, got $n"))
    return abc_rejection(prob, [prior(rng) for _ in 1:n], distance, observed; kwargs...)
end
