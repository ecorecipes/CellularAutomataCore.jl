# Likelihoods of observed paths. Sampling is checked against inversion written from its
# definition; step probabilities against an SI step written out by hand from the source
# equations; and the whole construction against simulation: the probabilities of every possible
# next state of a small lattice sum to one, and match the frequencies of simulated steps.

function all_states(dims, values)
    n = prod(dims)
    return [reshape([values[1 + (k ÷ length(values)^(i - 1)) % length(values)] for i in 1:n],
                    dims) for k in 0:(length(values) ^ n - 1)]
end

@testset "likelihood" begin
    @testset "sampling by inversion" begin
        # duplicate state 3 and a zero-probability outcome 2, which must never be drawn
        outs(c, n, ctx) = (Int8(1) => 0.2, Int8(2) => 0.0, Int8(3) => 0.5, Int8(3) => 0.3)
        r = ProbabilisticRule(outs)
        rv = ProbabilisticRule((c, n, ctx) -> collect(outs(c, n, ctx)))
        counts = Dict{Int8,Int}()
        N = 20_000
        for t in 1:N
            ctx = CellularAutomataCore.CellContext(nothing, CartesianIndex(1), t, 7, 99)
            u = uniform(ctx, 1)
            expected = u < 0.2 ? Int8(1) : Int8(3)
            got = local_rule(r, Int8(0), (), ctx)
            @test got == expected
            @test local_rule(rv, Int8(0), (), ctx) == got
            counts[got] = get(counts, got, 0) + 1
        end
        @test !haskey(counts, Int8(2))
        @test abs(counts[Int8(1)] / N - 0.2) < 4 * sqrt(0.2 * 0.8 / N)
        # Float32 probabilities use the Float32 draw
        r32 = ProbabilisticRule((c, n, ctx) -> (0 => 0.6f0, 1 => 0.4f0))
        for t in 1:200
            ctx = CellularAutomataCore.CellContext(nothing, CartesianIndex(1), t, 3, 5)
            @test local_rule(r32, 0, (), ctx) == (uniform(Float32, ctx, 1) < 0.6f0 ? 0 : 1)
        end
        # rounding never yields a zero-probability last outcome
        rr = ProbabilisticRule((c, n, ctx) -> (0 => 0.1, 1 => 0.9 - 1e-17, 2 => 0.0))
        @test all(t -> local_rule(rr, 0, (),
                                  CellularAutomataCore.CellContext(nothing,
                                                                   CartesianIndex(1), t,
                                                                   1, 1)) != 2, 1:2000)
        @test sprint(show, r) == "ProbabilisticRule(outs)"
        @test sprint(show, ProbabilisticRule((c, n, ctx) -> (c => 1.0,), Moore(1))) ==
              "ProbabilisticRule(<anonymous function>, Moore(1))"
    end

    si = ProbabilisticRule(VonNeumann(1); states=(0, 1)) do c, n, ctx
        c == 1 && return (Int8(1) => 1.0,)
        q = 1 - (1 - ctx.p.β)^count(==(1), n)
        return (Int8(0) => 1 - q, Int8(1) => q)
    end

    @testset "transition_probability" begin
        ctx = CellularAutomataCore.CellContext((β=0.3,), CartesianIndex(1, 1), 0, 1, 0)
        @test transition_probability(si, 1, 0, (1, 1, 0, 0), ctx) ≈ 1 - 0.7^2
        @test transition_probability(si, 0, 0, (1, 1, 0, 0), ctx) ≈ 0.7^2
        @test transition_probability(si, 0, 1, (0, 0, 0, 0), ctx) == 0
        life = Life()
        nb = (1, 1, 1, 0, 0, 0, 0, 0)
        @test transition_probability(life, 1, 0, nb, ctx) == 1
        @test transition_probability(life, 0, 0, nb, ctx) == 0
        @test transition_probability(Wolfram(30), 1, 0, (1, 0), ctx) == 1
        @test_throws NoLikelihoodError transition_probability(LocalRule((c, n, x) -> c), 0,
                                                              0,
                                                              (), ctx)
        @test_throws InvalidStateError CAProblem(si, [0 2; 0 0], 1)
    end

    rng = Xoshiro(41)
    u0 = Int8.(rand(rng, 0:1, 12, 10) .* (rand(rng, 12, 10) .< 0.3))
    prob = CAProblem(si, u0, 6, (β=0.35,); seed=8, space=Grid(12, 10))

    # The SI step written out from its definition: a susceptible cell with k infected von
    # Neumann neighbours inside the grid (outside reads 0) is infected with 1 - (1 - β)^k.
    function hand_si(u, v, β; stochastic=nothing)
        ll = 0.0
        for I in CartesianIndices(u)
            k = 0
            for o in ((-1, 0), (1, 0), (0, -1), (0, 1))
                J = I + CartesianIndex(o)
                checkbounds(Bool, u, J) && u[J] == 1 && (k += 1)
            end
            q = u[I] == 1 ? Float64(v[I] == 1) :
                (v[I] == 1 ? 1 - (1 - β)^k : (1 - β)^k)
            stochastic === nothing ||
                (q = (1 - stochastic) * (u[I] == v[I]) + stochastic * q)
            ll += log(q)
        end
        return ll
    end

    @testset "one step and a path against the SI equations" begin
        sol = solve(prob)
        for k in 1:6
            @test transition_logpdf(prob, sol.u[k], sol.u[k + 1], k - 1) ≈
                  hand_si(sol.u[k], sol.u[k + 1], 0.35)
        end
        total = sum(hand_si(sol.u[k], sol.u[k + 1], 0.35) for k in 1:6)
        @test path_logpdf(prob, sol.u) ≈ total
        @test path_logpdf(prob, sol) ≈ total
        @test path_logpdf(prob, sol; p=(β=0.2,)) ≈
              sum(hand_si(sol.u[k], sol.u[k + 1], 0.2) for k in 1:6)
        @test path_logpdf(prob, sol.u[1:1]) == 0
        # recovery is impossible under SI
        bad = copy(sol.u[2])
        bad[findfirst(==(1), sol.u[1])] = 0
        @test transition_logpdf(prob, sol.u[1], bad) == -Inf
        @test path_logpdf(prob, [sol.u[1], bad, bad]) == -Inf
        # Stochastic(q): each cell updates with probability q
        sols = solve(prob, Stochastic(0.5))
        @test path_logpdf(prob, sols) ≈
              sum(hand_si(sols.u[k], sols.u[k + 1], 0.35; stochastic=0.5) for k in 1:6)
        @test_throws ArgumentError path_logpdf(prob, solve(prob; saveat=2))
        @test_throws ArgumentError path_logpdf(prob, typeof(u0)[])
    end

    @testset "deterministic rules" begin
        lp = CAProblem(Life(), Int8.(rand(Xoshiro(3), 0:1, 9, 9)), 5)
        sol = solve(lp)
        @test path_logpdf(lp, sol) == 0
        v = copy(sol.u[2])
        v[1] = 1 - v[1]
        @test transition_logpdf(lp, sol.u[1], v) == -Inf
    end

    # A rule reading a covariate, a kernel field, its neighbours and the boundary, on a 2 x 2
    # grid, so that all 16 next states can be listed.
    fr = ProbabilisticRule(VonNeumann(1)) do c, n, ctx
        c == 1 && return (Int8(1) => 0.8, Int8(0) => 0.2)
        λ = ctx.p.β * covariate(ctx, :s) * (field(ctx, :P) + 0.1) + 0.3 * count(==(1), n)
        q = 1 - exp(-λ)
        return (Int8(0) => 1 - q, Int8(1) => q)
    end
    fprob = CAProblem(fr, Int8[1 0; 0 0], 1, (β=0.9,); seed=1,
                      space=Grid(2, 2; boundary=(Reflecting(), Constant(1))),
                      covariates=(s=[0.5 1.0; 2.0 0.2],),
                      fields=(P=KernelField(ExponentialKernel(1.0), ==(1)),))
    nexts = all_states((2, 2), Int8[0, 1])

    @testset "normalisation over every next state" begin
        for alg in (Synchronous(), Stochastic(0.3))
            @test sum(exp(transition_logpdf(fprob, fprob.u0, v; alg)) for v in nexts) ≈ 1
        end
        # a masked cell keeps its state
        mprob = remake(fprob; mask=Bool[1 0; 1 1])
        lps = [transition_logpdf(mprob, mprob.u0, v) for v in nexts]
        @test sum(exp, lps) ≈ 1
        @test all(lps[k] == -Inf for k in eachindex(nexts) if nexts[k][1, 2] != 0)
        # on a graph
        g = GraphSpace(4, [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        sis = ProbabilisticRule() do c, n, ctx
            c == 1 && return (0 => 0.3, 1 => 0.7)
            q = 1 - 0.6^count(==(1), n)
            return (0 => 1 - q, 1 => q)
        end
        gp = CAProblem(sis, [1, 0, 0, 1], 1; space=g)
        @test sum(exp(transition_logpdf(gp, gp.u0, v)) for v in all_states((4,), [0, 1])) ≈
              1
        # layered states, with outcomes naming the layers in another order
        lay = ProbabilisticRule() do c, n, ctx
            c.a == 1 && return ((b=c.b, a=Int8(1)) => 1.0,)
            return ((b=Int8(0), a=Int8(0)) => 0.5, (b=Int8(1), a=Int8(1)) => 0.25,
                    (b=Int8(1), a=Int8(0)) => 0.25)
        end
        lp = CAProblem(lay, (a=Int8[1 0], b=Int8[0 0]), 1; space=Grid(1, 2))
        layered_nexts = [(a=a, b=b) for a in all_states((1, 2), Int8[0, 1])
                         for b in all_states((1, 2), Int8[0, 1])]
        @test sum(exp(transition_logpdf(lp, lp.u0, v)) for v in layered_nexts) ≈ 1
        @test transition_logpdf(lp, lp.u0, (a=Int8[1 1], b=Int8[0 1])) ≈ log(0.25)
    end

    @testset "simulated frequencies" begin
        for alg in (Synchronous(), Stochastic(0.6))
            N = 20_000
            freq = Dict{Matrix{Int8},Int}()
            for s in 1:N
                v = final_state(solve(remake(fprob; seed=s), alg; save_everystep=false))
                freq[v] = get(freq, v, 0) + 1
            end
            for v in nexts
                P = exp(transition_logpdf(fprob, fprob.u0, v; alg))
                @test abs(get(freq, v, 0) / N - P) <= 4 * sqrt(P * (1 - P) / N) + 1e-12
            end
        end
    end

    @testset "errors" begin
        sol = solve(prob)
        @test_throws NoLikelihoodError path_logpdf(prob, sol.u; alg=RandomSequential())
        @test_throws NoLikelihoodError path_logpdf(prob, sol.u; alg=Checkerboard())
        @test_throws NoLikelihoodError path_logpdf(CAProblem(LocalRule((c, n, x) -> c), u0,
                                                             1), [u0, u0])
        rot = BlockRule((b, ctx) -> b)
        @test_throws NoLikelihoodError path_logpdf(CAProblem(rot, zeros(Int8, 4, 4), 1),
                                                   [zeros(Int8, 4, 4), zeros(Int8, 4, 4)])
        cb = DiscreteCallback((u, t, integ) -> false, integ -> nothing)
        @test_throws NoLikelihoodError path_logpdf(CAProblem(si, u0, 6, (β=0.35,);
                                                             callback=cb), sol.u)
        @test_throws StateShapeError transition_logpdf(prob, u0, zeros(Int8, 3, 3))
        e = try
            path_logpdf(prob, sol.u; alg=RandomSequential())
        catch err
            err
        end
        @test occursin("RandomSequential()", sprint(showerror, e))
        badsum = ProbabilisticRule((c, n, ctx) -> (0 => 0.5, 1 => 0.4))
        bp = CAProblem(badsum, [0 0], 1)
        @test_throws ArgumentError transition_logpdf(bp, [0 0], [0 1])
        neg = ProbabilisticRule((c, n, ctx) -> (0 => 1.5, 1 => -0.5))
        @test_throws ArgumentError transition_logpdf(CAProblem(neg, [0 0], 1), [0 0], [0 1])
        notpairs = ProbabilisticRule((c, n, ctx) -> (0, 1))
        @test_throws ArgumentError transition_logpdf(CAProblem(notpairs, [0 0], 1), [0 0],
                                                     [0 1])
    end
end
