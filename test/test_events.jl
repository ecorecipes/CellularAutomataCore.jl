# Event rules: continuous time with Gillespie (next-reaction and direct methods) and discrete
# time with the synchronous schemes. Stochastic checks use fixed seeds and tolerances of at
# least four standard errors.
@testset "events" begin
    decay = EventRule(VonNeumann(1)) do c, n, ctx
        return c == 1 ? (ctx.p.λ => 0,) : ()
    end
    sis = EventRule(VonNeumann(1)) do c, n, ctx
        return c == 0 ? (ctx.p.β * count(==(1), n) => 1,) : (ctx.p.γ => 0,)
    end
    within(x, μ, se) = abs(x - μ) <= 4se
    mean(x) = sum(x) / length(x)
    var(x) = sum(abs2, x .- mean(x)) / (length(x) - 1)

    @testset "EventRule" begin
        @test neighbourhood(decay) == VonNeumann(1)
        @test startswith(sprint(show, decay), "EventRule(<anonymous function>")
        pick = CellularAutomataCore._pick
        trs = (0.0 => :a, 1.0 => :b, 2.0 => :c)
        @test pick(trs, 0.5, :x) == :b && pick(trs, 2.9, :x) == :c
        @test pick(trs, 3.5, :x) == :c && pick((0.0 => :a,), 0.0, :x) == :x
        @test pick([1.0 => :b, 2.0 => :c], 1.5, :x) == :c
        @test CellularAutomataCore._total_rate((1.0f0 => 1, 2.0f0 => 2)) === 3.0f0
        @test CellularAutomataCore._total_rate(()) == 0
        @test_throws InvalidStateError CAProblem(EventRule((c, n, ctx) -> ();
                                                           states=(0, 1)),
                                                 fill(2, 3, 3), 1.0)
    end

    @testset "problems and schemes" begin
        u0 = ones(Int8, 4, 4)
        @test CAProblem(decay, u0, 2.5, (λ=1.0,)).tspan == (0.0, 2.5)
        @test CAProblem(decay, u0, 3, (λ=1.0,)).tspan == (0, 3)
        @test CAProblem(decay, u0, (1, 2.5), (λ=1.0,)).tspan == (1.0, 2.5)
        @test_throws ArgumentError CAProblem(Life(), zeros(Int8, 4, 4), 2.5)
        @test_throws ArgumentError solve(CAProblem(decay, u0, 2.5, (λ=1.0,)))
        @test_throws ArgumentError solve(CAProblem(Life(), zeros(Int8, 4, 4), 2),
                                         Gillespie())
        @test_throws ArgumentError Gillespie(:tau_leap)
        @test sprint(show, Gillespie()) == "Gillespie(:next_reaction)"
        p = CAProblem(decay, u0, 1.0, (λ=1.0,))
        @test_throws UnsupportedCombinationError solve(p, Gillespie(:direct);
                                                       backend=SparseBackend(0))
        @test_throws UnsupportedCombinationError solve(p, Gillespie(:composition_rejection);
                                                       backend=SparseBackend(0))
        @test_throws UnsupportedCombinationError solve(p, Gillespie();
                                                       backend=GPUBackend(KernelAbstractions.CPU()))
        bad = EventRule((c, n, ctx) -> (-1.0 => 0,), VonNeumann(1))
        @test_throws ArgumentError solve(CAProblem(bad, u0, 1.0), Gillespie())
        @test_throws ArgumentError solve(p, Gillespie(); saveat=0)
    end

    @testset "exponential lifetimes ($m)" for m in (:next_reaction, :direct,
                                                    :composition_rejection)
        # independent cells decaying at rate λ: a fraction exp(-λ t) survives to t
        n = 100 * 100
        for (λ, t) in ((1.0, 1.0), (2.0, 0.25))
            sol = solve(CAProblem(decay, ones(Int8, 100, 100), t, (λ=λ,); seed=11),
                        Gillespie(m))
            q = exp(-λ * t)
            @test within(mean(final_state(sol)), q, sqrt(q * (1 - q) / n))
        end
        # the first of n events comes after an exponential time of rate n λ
        times = map(1:300) do s
            integ = init(CAProblem(decay, ones(Int8, 10, 10), 10.0, (λ=1.0,); seed=s),
                         Gillespie(m))
            step!(integ)
            return integ.t
        end
        @test within(mean(times), 1 / 100, (1 / 100) / sqrt(300))
    end

    @testset "reproducibility" begin
        u0 = zeros(Int8, 20, 20)
        u0[10, 10] = 1
        prob = CAProblem(sis, u0, 5.0, (β=0.6, γ=1.0); seed=3)
        a = solve(prob, Gillespie(); saveat=1.0)
        @test a.u == solve(prob, Gillespie(); saveat=1.0).u
        @test a.u == solve(prob, Gillespie(); saveat=1.0, backend=ReferenceBackend()).u
        @test a.u != solve(remake(prob; seed=4), Gillespie(); saveat=1.0).u
        d = solve(prob, Gillespie(:direct); saveat=1.0)
        @test d.u == solve(prob, Gillespie(:direct); saveat=1.0).u
        c = solve(prob, Gillespie(:composition_rejection); saveat=1.0)
        @test c.u == solve(prob, Gillespie(:composition_rejection); saveat=1.0).u
        @test c.u == solve(prob, Gillespie(:composition_rejection); saveat=1.0,
                backend=ReferenceBackend()).u
        integ = init(prob, Gillespie(:composition_rejection); saveat=1.0)
        solve!(integ)
        reinit!(integ)
        solve!(integ)
        @test integ.sol_u == c.u
    end

    @testset "next-reaction and direct methods agree in distribution" begin
        # two-state switching, 0 -> 1 at rate 1 and 1 -> 0 at rate 3: the fraction in
        # state 1 at time t from all zeros is (1 - exp(-4t)) / 4
        flip = EventRule(VonNeumann(1)) do c, n, ctx
            return c == 0 ? (1.0 => 1,) : (3.0 => 0,)
        end
        q = (1 - exp(-4 * 0.3)) / 4
        for m in (:next_reaction, :direct, :composition_rejection)
            sol = solve(CAProblem(flip, zeros(Int8, 60, 60), 0.3; seed=5), Gillespie(m))
            @test within(mean(final_state(sol)), q, sqrt(q * (1 - q) / 3600))
        end
        # an interacting model: the mean prevalence of the SIS contact process
        u0 = zeros(Int8, 16, 16)
        u0[8:9, 8:9] .= 1
        prevalence(m) = [mean(final_state(solve(CAProblem(sis, u0, 3.0, (β=0.7, γ=1.0);
                                                          seed=s), Gillespie(m))))
                         for s in 1:150]
        a, b = prevalence(:next_reaction), prevalence(:direct)
        @test within(mean(a) - mean(b), 0, sqrt(var(a) / length(a) + var(b) / length(b)))
        c = prevalence(:composition_rejection)
        @test within(mean(a) - mean(c), 0, sqrt(var(a) / length(a) + var(c) / length(c)))
        # rates over six orders of magnitude, so composition–rejection uses many groups: cells
        # of rate r decay by time t with probability 1 - exp(-r t)
        ladder = EventRule() do c, n, ctx
            return c == 1 ? (covariate(ctx, :r) => 0,) : ()
        end
        rs = [10.0^(k - 4) for k in 0:6]
        r = [rs[mod1(i + j, 7)] * (1 + 0.3 * isodd(i)) for i in 1:70, j in 1:70]
        for m in (:next_reaction, :composition_rejection)
            sol = solve(CAProblem(ladder, ones(Int8, 70, 70), 0.5; seed=9,
                                  covariates=(; r)),
                        Gillespie(m))
            for v in unique(r)
                cellsv = findall(==(v), r)
                p = 1 - exp(-v * 0.5)
                left = count(I -> final_state(sol)[I] == 0, cellsv) / length(cellsv)
                @test within(left, p, sqrt(max(p * (1 - p), 1e-4) / length(cellsv)))
            end
        end
    end

    @testset "saving and stepping" begin
        u0 = zeros(Int8, 12, 12)
        u0[6, 6] = 1
        prob = CAProblem(sis, u0, 2.0, (β=0.8, γ=0.5); seed=2)
        sol = solve(prob, Gillespie(); saveat=0.5)
        @test sol.t == [0.0, 0.5, 1.0, 1.5, 2.0] && sol.t isa Vector{Float64}
        @test sol(1.0) == sol.u[3] && final_state(sol) == sol.u[end]
        @test solve(prob, Gillespie(); saveat=[1.75, 0.25]).t == [0.0, 0.25, 1.75, 2.0]
        @test solve(prob, Gillespie(); saveat=0.5, save_start=false).t ==
              [0.5, 1.0, 1.5, 2.0]
        @test solve(prob, Gillespie(); saveat=[0.5], save_end=false).t == [0.0, 0.5]
        @test solve(prob, Gillespie()).t == [0.0, 2.0]
        every = solve(prob, Gillespie(); save_everystep=true)
        @test issorted(every.t) && allunique(every.t) && length(every.t) > 3
        # consecutive saved states differ in exactly one cell, except at the end
        @test all(count(every.u[k] .!= every.u[k + 1]) == 1
                  for k in 1:(length(every.u) - 2))
        @test every.u[end] == sol.u[end]
        integ = init(prob, Gillespie())
        u = copy(integ.u)
        step!(integ)
        @test 0 < integ.t < 2 && count(integ.u .!= u) == 1
        step!(integ, 3)
        @test solve!(integ).retcode == ReturnCode.Success && integ.t == 2.0
    end

    @testset "callbacks and tstops" begin
        u0 = ones(Int8, 10, 10)
        prob = CAProblem(decay, u0, 2.0, (λ=1.0,); seed=6)
        # refill every cell at t = 0.5, exactly
        refill = PresetTimeCallback([0.5], integ -> (integ.u .= 1))
        sol = solve(prob, Gillespie(); callback=refill, saveat=[0.5])
        @test sol.t == [0.0, 0.5, 2.0] && all(==(1), sol(0.5))
        # a periodic callback stops the run at every multiple of 0.25
        hits = Float64[]
        record = PeriodicCallback(integ -> (push!(hits, integ.t); u_modified!(integ, false)),
                                  0.25)
        solve(prob, Gillespie(); callback=record)
        @test hits ≈ 0.25:0.25:2.0
        # changing the parameters stops the decay
        stop = PresetTimeCallback([0.3], integ -> (integ.p = (λ=0.0,)))
        s = solve(prob, Gillespie(); callback=stop, saveat=[0.3])
        @test s(0.3) == final_state(s) && any(==(0), final_state(s))
        # terminate when everything has decayed
        done = DiscreteCallback((u, t, integ) -> all(==(0), u), terminate!)
        s = solve(CAProblem(decay, u0, 50.0, (λ=1.0,); seed=6), Gillespie(); callback=done)
        @test s.retcode == ReturnCode.Terminated && last(s.t) < 50 &&
              all(==(0), final_state(s))
        # a saving callback records summaries at real times
        saved = SavedValues(Float64, Int)
        solve(prob, Gillespie();
              callback=SavingCallback((u, t, i) -> count(==(1), u), saved;
                                      saveat=0.5))
        @test saved.t ≈ [0.0, 0.5, 1.0, 1.5, 2.0] && issorted(saved.saveval; rev=true)
    end

    @testset "reinit! replays the run" begin
        u0 = zeros(Int8, 10, 10)
        u0[5, 5] = 1
        integ = init(CAProblem(sis, u0, 3.0, (β=0.9, γ=0.5); seed=8), Gillespie();
                     saveat=1.0)
        sol1 = solve!(integ)
        u1, t1 = copy(sol1.u), copy(sol1.t)
        reinit!(integ)
        sol2 = solve!(integ)
        @test sol2.u == u1 && sol2.t == t1
    end

    @testset "time-varying covariates" begin
        # no decay during [0, 1), fast decay during [1, 2)
        k = cat(zeros(6, 6), fill(50.0, 6, 6); dims=3)
        rule = EventRule((c, n, ctx) -> c == 1 ? (covariate(ctx, :k) => 0,) : (),
                         VonNeumann(1))
        for m in (:next_reaction, :direct)
            sol = solve(CAProblem(rule, ones(Int8, 6, 6), 2.0; covariates=(k=k,), seed=1),
                        Gillespie(m); saveat=[0.999])
            @test all(==(1), sol(0.999)) && all(==(0), final_state(sol))
        end
    end

    @testset "graphs, masks and layers" begin
        g = GraphSpace([[2], [1, 3], [2]])
        spread = EventRule(GraphNeighbours()) do c, n, ctx
            return c == 0 ? (10.0 * count(==(1), n) => 1,) : ()
        end
        sol = solve(CAProblem(spread, [1, 0, 0], 5.0; space=g, seed=1), Gillespie())
        @test final_state(sol) == [1, 1, 1]
        mask = trues(5, 5)
        mask[3, 3] = false
        sol = solve(CAProblem(decay, ones(Int8, 5, 5), 20.0, (λ=1.0,); mask, seed=1),
                    Gillespie())
        @test final_state(sol)[3, 3] == 1 && count(==(1), final_state(sol)) == 1
        layered = EventRule(VonNeumann(1)) do c, n, ctx
            return c.x == 0 ? (1.0 => (x=1, y=c.y + 1.0),) : ()
        end
        u0 = (x=zeros(Int, 4, 4), y=zeros(4, 4))
        s = final_state(solve(CAProblem(layered, u0, 30.0; seed=1), Gillespie()))
        @test all(==(1), s.x) && all(==(1.0), s.y)
    end

    @testset "kernel fields" begin
        # infection from the four neighbours as a kernel field gives the same process, and with
        # weights of 1/4 exactly the same path, as the neighbourhood rule
        byfield = EventRule() do c, n, ctx
            return c == 0 ? (ctx.p.β * field(ctx, :inf) => 1,) : (ctx.p.γ => 0,)
        end
        bynbrs = EventRule(VonNeumann(1)) do c, n, ctx
            return c == 0 ? (ctx.p.β * (count(==(1), n) / 4) => 1,) : (ctx.p.γ => 0,)
        end
        inf = KernelField(UniformKernel(1), c -> c == 1 ? 1.0 : 0.0; include_centre=false)
        u0 = zeros(Int, 12, 9)
        u0[3, 4] = u0[10, 2] = u0[1, 1] = 1
        θ = (β=2.0, γ=1.0)
        for bd in (Periodic(), Reflecting(), Constant(0)),
            m in (:next_reaction, :direct, :composition_rejection)
            space = Grid(12, 9; boundary=bd)
            a = solve(CAProblem(byfield, u0, 30.0, θ; space, seed=5, fields=(; inf)),
                      Gillespie(m); saveat=0:30)
            b = solve(CAProblem(bynbrs, u0, 30.0, θ; space, seed=5), Gillespie(m);
                      saveat=0:30)
            @test a.u == b.u
        end
        # fields kept up to date event by event equal fields computed from the state, also when
        # the kernel is wider than the lattice (a cell read through several offsets and mirrors)
        tilted = AnisotropicKernel(ExponentialKernel(1.5; radius=5), 0.8, 1.2)
        load = EventRule() do c, n, ctx
            return c == 0 ? (3.0 * field(ctx, :f) => 1,) : (0.7 => 0,)
        end
        for bd in (Reflecting(), Periodic(), (Reflecting(), Constant(0)))
            space = Grid(3, 4; boundary=bd)
            v0 = zeros(Int, 3, 4)
            v0[2, 2] = 1
            prob = CAProblem(load, v0, 40.0; space, seed=3,
                             fields=(f=KernelField(tilted, c -> Float64(c)),))
            integ = init(prob, Gillespie())
            worst = 0.0
            while !(integ.t >= 40.0 || integ.retcode == ReturnCode.Terminated)
                step!(integ)
                st = integ.state
                fresh = CellularAutomataCore._compute_field_reference!(similar(st.fieldvals.f),
                                                                       prob.fields.f, space,
                                                                       st.u)
                worst = max(worst, maximum(abs, fresh .- st.fieldvals.f))
                all(>=(0), st.fieldvals.f) || break
            end
            @test worst < 1e-12 && all(>=(0), integ.state.fieldvals.f)
        end
        # a field whose sources all vanish is exactly zero (no rate left by rounding)
        dieoff = EventRule() do c, n, ctx
            return c == 0 ? (0.1 * field(ctx, :f) => 1,) : (5.0 => 0,)
        end
        prob = CAProblem(dieoff, fill(1, 6, 6), 200.0; space=Torus(6, 6), seed=2,
                         fields=(f=KernelField(tilted, c -> Float64(c)),))
        integ = init(prob, Gillespie())
        solve!(integ)
        @test all(==(0), integ.u) && all(iszero, integ.state.fieldvals.f)
        # FFT fields: the same field, to rounding
        spread = EventRule() do c, n, ctx
            return c == 0 ? (30.0 * field(ctx, :f) => 1,) : (0.7 => 0,)
        end
        big = zeros(Int, 40, 40)
        big[20, 20] = big[5, 30] = 1
        fft_field = KernelField(GaussianKernel(2.0), c -> Float64(c); method=:fft)
        prob = CAProblem(spread, big, 5.0; space=Torus(40, 40), seed=1,
                         fields=(f=fft_field,))
        integ = init(prob, Gillespie())
        solve!(integ)
        fresh = CellularAutomataCore._compute_field_reference!(similar(integ.state.fieldvals.f),
                                                               prob.fields.f, prob.space,
                                                               integ.u)
        @test sum(integ.u) > 2 && maximum(abs, fresh .- integ.state.fieldvals.f) < 1e-12
        # a callback that changes the state recomputes the fields
        cb = PresetTimeCallback([2.5], integ -> (integ.u.=0; nothing))
        sol = solve(CAProblem(spread, big, 6.0; space=Torus(40, 40), seed=1,
                              fields=(f=KernelField(GaussianKernel(2.0), c -> Float64(c)),)),
                    Gillespie(); callback=cb, saveat=[2.0, 3.0, 6.0])
        @test sol.t == [0.0, 2.0, 3.0, 6.0]
        @test sum(sol.u[2]) > 0 && sum(sol.u[3]) == 0 && sum(sol.u[4]) == 0
        # the exponential lifetime of a single infection's neighbour: rate β w for weight w
        hit = 0
        runs = 4000
        w = 0.25
        for seed in 1:runs
            v0 = zeros(Int, 5, 5)
            v0[3, 3] = 1
            p1 = CAProblem(byfield, v0, 0.5, (β=2.0, γ=0.0); space=Torus(5, 5), seed,
                           fields=(; inf))
            hit += solve(p1, Gillespie()).u[end][3, 4] == 1
        end
        @test within(hit / runs, 1 - exp(-2.0 * w * 0.5),
                     sqrt((1 - exp(-0.25)) * exp(-0.25) / runs))
    end

    @testset "sparse backend" begin
        # the next-reaction method on the sparse backend equals the array backend event for event
        sir = EventRule(Moore(1)) do c, n, ctx
            return c == 0 ? (0.3 * count(==(1), n) => 1,) : c == 1 ? (1.0 => 2,) : ()
        end
        u0 = zeros(Int8, 40, 30)
        u0[20, 15] = u0[1, 1] = u0[40, 7] = 1
        for bd in (Periodic(), Reflecting(), Constant(0), (Periodic(), Constant(0))),
            ts in (3, 32)
            prob = CAProblem(sir, u0, 25.0; space=Grid(40, 30; boundary=bd), seed=7)
            a = solve(prob, Gillespie(); saveat=0:25)
            b = solve(prob, Gillespie(); saveat=0:25,
                      backend=SparseBackend(Int8(0); tilesize=ts))
            @test a.u == b.u && count(==(2), a.u[end]) > 3
        end
        # masks, covariates, callbacks that edit the state, reinit!
        mask = trues(40, 30)
        mask[:, 10] .= false
        hot = [1.0 + (j > 15) for i in 1:40, j in 1:30]
        sir2 = EventRule(Moore(1)) do c, n, ctx
            return c == 0 ? (0.3 * covariate(ctx, :hot) * count(==(1), n) => 1,) :
                   c == 1 ? (1.0 => 2,) : ()
        end
        seedmore = PresetTimeCallback([5.0], integ -> (integ.u[5, 25]=1; nothing))
        prob = CAProblem(sir2, u0, 20.0; seed=2, mask, covariates=(; hot))
        a = solve(prob, Gillespie(); saveat=0:20, callback=seedmore)
        integ = init(prob, Gillespie(); saveat=0:20, callback=seedmore,
                     backend=SparseBackend(Int8(0); tilesize=8))
        solve!(integ)
        @test integ.sol_u == a.u && all(==(0), a.u[end][:, 10])
        reinit!(integ)
        solve!(integ)
        @test integ.sol_u == a.u
        reinit!(integ)
        @test stored_tiles(integ) == 3                  # the three seeded cells' tiles
        # a rule that is not quiescent under the background is rejected
        spont = EventRule(VonNeumann(1)) do c, n, ctx
            return c == 0 ? (0.01 => 1,) : ()
        end
        @test_throws ArgumentError solve(CAProblem(spont, u0, 1.0), Gillespie();
                                         backend=SparseBackend(Int8(0)))
    end

    @testset "discrete time" begin
        # one synchronous step: a cell of total rate λ changes with probability 1 - exp(-λ)
        prob = CAProblem(decay, ones(Int8, 100, 100), 1, (λ=1.0,); seed=9)
        q = exp(-1)
        @test within(mean(final_state(solve(prob))), q, sqrt(q * (1 - q) / 10^4))
        u0 = zeros(Int8, 30, 30)
        u0[15, 15] = 1
        p = CAProblem(sis, u0, 10, (β=0.4, γ=0.3); seed=2)
        @test solve(p).u == solve(p; backend=ReferenceBackend()).u ==
              solve(p; backend=GPUBackend(KernelAbstractions.CPU())).u
        # Float32 rates draw in single precision
        decay32 = EventRule((c, n, ctx) -> c == 1 ? (1.0f0 => 0,) : (), VonNeumann(1))
        s32 = solve(CAProblem(decay32, ones(Int8, 100, 100), 1; seed=9))
        @test within(mean(final_state(s32)), q, sqrt(q * (1 - q) / 10^4))
    end

    @testset "@camodel @events" begin
        m = @camodel SIS begin
            @states state::Int8 = 0 ∈ (0, 1)
            @parameters β = 0.6 γ = 1.0
            @neighbourhood VonNeumann(1)
            @events begin
                cell == 0 ? (β * count(==(1), nbrs) => 1,) : (γ => 0,)
            end
        end
        @test m.rule isa EventRule
        u0 = zeros(Int8, 20, 20)
        u0[10, 10] = 1
        a = solve(CAProblem(m, (state=u0,), 5.0; space=Torus(20, 20), seed=3), Gillespie();
                  saveat=1.0)
        b = solve(CAProblem(sis, u0, 5.0, (β=0.6, γ=1.0); seed=3), Gillespie(); saveat=1.0)
        @test a.u == b.u
        @test_throws LoadError @eval @camodel Bad begin
            @states x::Int = 0
            @rule cell
            @events ()
        end
    end
end
