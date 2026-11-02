# Rejection ABC: every proposal's distance must equal that of a direct solve with the seed its
# trajectory gets, acceptance must follow the distances, and the true parameter of a synthetic
# observation must be recovered.

@testset "calibration" begin
    si = LocalRule(VonNeumann(1)) do c, n, ctx
        c == 1 && return Int8(1)
        return bernoulli(ctx, 1 - (1 - ctx.p.β)^count(==(1), n)) ? Int8(1) : Int8(0)
    end
    u0 = zeros(Int8, 30, 30)
    u0[15, 15] = 1
    prob = CAProblem(si, u0, 20, (β=0.3,); seed=3)
    observed = final_state(solve(remake(prob; seed=100); save_everystep=false))
    occupied(sol) = count(==(1), final_state(sol))
    dist(sol, obs) = abs(occupied(sol) - count(==(1), obs))

    proposals = [(β=b,) for b in 0.05:0.05:0.95]
    res = abc_rejection(prob, proposals, dist, observed; keep=4, save_everystep=false)
    @test res isa ABCResult{@NamedTuple{β::Float64}}
    @test res.proposals == proposals && length(res.samples) == 4
    for i in (1, 6, 19)
        s = solve(remake(prob; p=proposals[i], seed=prob.seed + i); save_everystep=false)
        @test res.proposal_distances[i] == dist(s, observed)
    end
    @test issorted(res.distances) && res.tolerance == last(res.distances)
    @test res.distances == sort(res.proposal_distances)[1:4]
    @test all(s -> 0.15 <= s.β <= 0.45, res.samples)
    @test startswith(sprint(show, res), "ABCResult: 4 of 19 proposals accepted, distances")

    # serial and threaded ensembles agree; the default keeps one in twenty
    serial = abc_rejection(prob, proposals, dist, observed; ensemblealg=EnsembleSerial(),
                           save_everystep=false)
    @test serial.proposal_distances == res.proposal_distances
    @test length(serial.samples) == 1

    # a tolerance can accept fewer than `keep`, or none
    tight = abc_rejection(prob, proposals, dist, observed; keep=10, tolerance=res.tolerance,
                          save_everystep=false)
    @test tight.samples == res.samples[1:length(tight.samples)]
    none = abc_rejection(prob, proposals, (s, o) -> 1.0, observed; tolerance=0.5)
    @test isempty(none.samples) && none.tolerance == 0.5
    @test sprint(show, none) == "ABCResult: 0 of 19 proposals accepted"

    # a prior drawn with an RNG is the same as proposals drawn from a copy of it
    prior = rng -> (β=rand(rng),)
    drawn = abc_rejection(prob, prior, dist, observed; n=30, rng=Xoshiro(7), keep=3,
                          save_everystep=false)
    rng = Xoshiro(7)
    @test drawn.proposals == [prior(rng) for _ in 1:30]
    @test drawn.samples ==
          abc_rejection(prob, drawn.proposals, dist, observed; keep=3,
                        save_everystep=false).samples

    # a map distance (Cohen's kappa) works on the same machinery
    kappa = abc_rejection(prob, proposals, (s, o) -> 1 - cohens_kappa(final_state(s), o),
                          observed; keep=3, save_everystep=false)
    @test all(d -> 0 <= d <= 1, kappa.proposal_distances)

    # plot(res): proposals then accepted samples, one subplot per parameter
    RB = CellularAutomataCore.RecipesBase
    RB.is_key_supported(::Symbol) = true
    two = ABCResult([(a=1.0, b=2)], [0.0], 0.0, [(a=1.0, b=2), (a=3.0, b=4)], [0.0, 1.0])
    series = RB.apply_recipe(Dict{Symbol,Any}(), two)
    @test length(series) == 4
    @test [s.plotattributes[:subplot] for s in series] == [1, 1, 2, 2]
    @test [only(s.args) for s in series] == [[1.0, 3.0], [1.0], [2, 4], [2]]
    @test series[1].plotattributes[:seriestype] == :histogram

    @test_throws ArgumentError abc_rejection(prob, proposals, dist, observed; keep=0)
    @test_throws ArgumentError abc_rejection(prob, proposals, dist, observed; keep=20)
    @test_throws ArgumentError abc_rejection(prob, empty(proposals), dist, observed)
    @test_throws ArgumentError abc_rejection(prob, prior, dist, observed; n=10)
    @test_throws ArgumentError abc_rejection(prob, prior, dist, observed; n=0,
                                             rng=Xoshiro(1))
end
