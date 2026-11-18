# The SciML interface: problems with p, remake, integrators, callbacks, saving and ensembles.
@testset "SciML interface" begin
    sir = LocalRule(VonNeumann(1)) do c, n, ctx
        return c == :S ? (bernoulli(ctx, 1 - (1 - ctx.p.β)^count(==(:I), n), 1) ? :I : :S) :
               c == :I ? (bernoulli(ctx, ctx.p.γ, 2) ? :R : :I) : c
    end
    v = fill(:S, 30, 30)
    v[15, 15] = :I
    prob = CAProblem(sir, v, 60, (β=0.4, γ=0.2); seed=1)

    @testset "problems" begin
        @test prob.p == (β=0.4, γ=0.2) &&
              prob isa CellularAutomataCore.SciMLBase.AbstractSciMLProblem
        @test CAProblem(Life(), zeros(Int8, 3, 3), 1).p === NullParameters()
        r = remake(prob; p=(β=0.1, γ=0.5), seed=7)
        @test r.p == (β=0.1, γ=0.5) && r.seed == 7 && r.u0 === prob.u0
        @test remake(prob; tspan=5).tspan == (0, 5)
        lp = CAProblem(Lenia(), zeros(32, 32), 1)
        @test keys(remake(lp; seed=2).fields) == (:potential,)
    end

    @testset "solver keywords stored in the problem" begin
        sp = CAProblem(sir, v, 60, (β=0.4, γ=0.2); seed=1, saveat=20)
        @test sp.kwargs == (saveat=20,)
        @test solve(sp).t == [0, 20, 40, 60]
        @test solve(sp; saveat=30).t == [0, 30, 60]          # the call wins
        @test remake(sp; seed=2).kwargs == (saveat=20,)
        @test remake(sp; kwargs=(;)).kwargs == (;)
        @test_throws ArgumentError CAProblem(sir, v, 60, (β=0.4, γ=0.2); colour=:red)
        # callbacks from the problem and the call are combined, the problem's first
        order = Int[]
        first_cb = DiscreteCallback((u, t, integ) -> t == 1, integ -> push!(order, 1))
        second_cb = DiscreteCallback((u, t, integ) -> t == 1, integ -> push!(order, 2))
        cp = CAProblem(sir, v, 3, (β=0.4, γ=0.2); callback=first_cb)
        solve(cp; callback=second_cb)
        @test order == [1, 2]
        empty!(order)
        solve(cp)
        @test order == [1]
        # and the model layer passes them through
        m = CAModel(sir; states=(state=:S,), parameters=(β=0.4, γ=0.2))
        @test CAProblem(m, (state=v,), 60; space=Torus(30, 30), saveat=30).kwargs ==
              (saveat=30,)
    end

    @testset "solutions" begin
        sol = solve(prob; saveat=20)
        @test sol.retcode == ReturnCode.Success && sol.alg == Synchronous()
        @test sol(40) == sol.u[3] && sol.t == [0, 20, 40, 60]
        @test_throws ArgumentError sol(41)
        @test solve(prob; save_everystep=false).t == [0, 60]
        @test solve(prob).t == collect(0:60)
        @test occursin("retcode", sprint(show, sol))
    end

    @testset "integrator" begin
        integ = init(prob; saveat=1000)
        @test integ.t == 0 && integ.u == v && integ.p == prob.p
        step!(integ, 5)
        @test integ.t == 5
        # editing u outside a callback needs u_modified!
        u = integ.u
        u[1, 1] = :R
        u_modified!(integ, true)
        step!(integ)
        @test integ.u[1, 1] == :R
        # changing p changes the dynamics from the next step
        integ.p = (β=0.0, γ=1.0)
        step!(integ)
        @test !any(==(:I), integ.u)
        reinit!(integ)
        @test integ.t == 0 && integ.u == v
        @test occursin("CAIntegrator", sprint(show, integ))
        @test :u in propertynames(integ)
    end

    @testset "callbacks" begin
        cull = PeriodicCallback(10) do integ
            return integ.u[integ.u .== :I] .= :R
        end
        stop = DiscreteCallback((u, t, integ) -> !any(==(:I), u),
                                integ -> terminate!(integ))
        sol = solve(prob; callback=CallbackSet(cull, stop), saveat=5)
        @test sol.retcode == ReturnCode.Terminated && last(sol.t) == 10
        @test !any(==(:I), final_state(sol))
        # a preset-time callback changing p, and a saving callback
        saved = SavedValues(Int, Int)
        cbs = (PresetTimeCallback([3], integ -> (integ.p = (β=0.0, γ=0.2))),
               SavingCallback((u, t, integ) -> count(==(:I), u), saved; saveat=2))
        s = solve(prob; callback=cbs, saveat=1000)
        @test saved.t == collect(0:2:60)
        @test length(saved.saveval) == 31 && saved.saveval[1] == 1
        # after t = 3 no new infections: susceptibles never decrease again
        nS = SavedValues(Int, Int)
        solve(prob; callback=(cbs[1], SavingCallback((u, t, i) -> count(==(:S), u), nS)))
        @test all(==(nS.saveval[4]), nS.saveval[5:end])
        # initial_affect and phase
        hits = Int[]
        solve(prob;
              callback=PeriodicCallback(i -> push!(hits, i.t), 20; initial_affect=true,
                                        phase=5))
        @test hits == [0, 25, 45]
        # an initial affect changes the state the first step runs from, not only the saved copy
        counting = CAProblem(LocalRule((c, nb, ctx) -> c + 1, VonNeumann(1)),
                             zeros(Int, 3, 3), 2)
        start = PeriodicCallback(i -> (i.u[1, 1]=10; nothing), 100; initial_affect=true)
        @test [u[1, 1] for u in solve(counting; callback=start).u] == [10, 11, 12]
        @test_throws ArgumentError PeriodicCallback(identity, 0)
        @test_throws ArgumentError solve(prob; callback=1)
        # callbacks give the same result on every backend
        a = solve(prob; callback=cull, backend=ReferenceBackend())
        b = solve(prob; callback=cull, backend=SparseBackend(:S; tilesize=4))
        @test a.u == b.u
    end

    @testset "ensembles" begin
        ep = EnsembleProblem(prob)
        es = solve(ep, Synchronous(), EnsembleSerial(); trajectories=4, saveat=60)
        @test length(es.u) == 4 && [s.prob.seed for s in es.u] == UInt64.(2:5)
        @test length(unique(final_state(s) for s in es.u)) > 1
        et = solve(ep, Synchronous(), EnsembleThreads(); trajectories=4, saveat=60)
        @test [final_state(s) for s in et.u] == [final_state(s) for s in es.u]
        out = solve(EnsembleProblem(prob;
                                    output_func=(sol, ctx) -> (count(==(:R),
                                                                     final_state(sol)),
                                                               false)),
                    Synchronous(), EnsembleSerial(); trajectories=3, saveat=60)
        @test out.u isa Vector{Int}
        s1 = solve(ep, Synchronous(), EnsembleThreads(); trajectories=3, seed=42, saveat=60)
        s2 = solve(ep, Synchronous(), EnsembleSerial(); trajectories=3, seed=42, saveat=60)
        @test [final_state(s) for s in s1.u] == [final_state(s) for s in s2.u]
    end

    # The parts of SciML's integrator interface that simulation-based resilience metrics
    # (DynamicalResilience.jl) drive: timed steps, set_u!, check_error, a seed at init,
    # exact replay with reinit!, successful_retcode on solutions.
    @testset "integrator contract for simulation-based metrics" begin
        SB = CellularAutomataCore.SciMLBase
        integ = init(prob; save_everystep=false)
        step!(integ, 7, true)
        @test integ.t == 7
        step!(integ, 2.5, false)                      # at least 2.5: three whole steps
        @test integ.t == 10
        @test_throws ArgumentError step!(integ, 2.5, true)
        @test_throws ArgumentError step!(integ, -1, true)
        @test SB.check_error(integ) == ReturnCode.Success
        step!(integ, 1000, true)                      # stops at the end of tspan
        @test integ.t == 60
        terminate!(integ)
        @test SB.check_error(integ) == ReturnCode.Terminated

        # a seed given to init is remake(prob; seed)
        a = init(prob; seed=9, save_everystep=false)
        b = init(remake(prob; seed=9); save_everystep=false)
        step!(a, 30, true)
        step!(b, 30, true)
        @test a.u == b.u && a.prob.seed == 9

        # reinit! replays exactly, also after a pulse applied with set_u!, which copies
        r = init(prob; save_everystep=false)
        step!(r, 20, true)
        before = copy(r.u)
        pulse = ifelse.(r.u .== :I, :R, r.u)
        SB.set_u!(r, pulse)
        pulse[1] = :I                                 # the integrator holds a copy
        @test r.u[1] != :I || before[1] == :I
        step!(r, 10, true)
        after = copy(r.u)
        reinit!(r)
        step!(r, 20, true)
        @test r.u == before
        SB.set_u!(r, ifelse.(r.u .== :I, :R, r.u))
        step!(r, 10, true)
        @test r.u == after
        @test count(==(:I), after) == 0               # the pulse cured every infection

        # layered states
        layered = CAProblem(LocalRule((c, n, ctx) -> c, VonNeumann(1)),
                            (a=zeros(Int, 4, 4), b=ones(Int, 4, 4)), 3)
        li = init(layered; save_everystep=false)
        SB.set_u!(li, (a=fill(2, 4, 4), b=zeros(Int, 4, 4)))
        step!(li, 1, true)
        @test all(==(2), li.u.a) && all(iszero, li.u.b)

        # solutions: the generic successful_retcode would compare the solution itself
        @test SB.successful_retcode(solve(prob; saveat=60))
        tp = CAProblem(sir, v, 60, (β=0.4, γ=0.2); seed=1,
                       callback=DiscreteCallback((u, t, i) -> t == 3, terminate!))
        @test SB.successful_retcode(solve(tp))       # Terminated counts as success

        # continuous time: a timed step lands exactly on t + dt
        sis = EventRule(VonNeumann(1)) do c, n, ctx
            return c == 0 ? (ctx.p.β * count(==(1), n) => 1,) : (ctx.p.γ => 0,)
        end
        g0 = zeros(Int, 20, 20)
        g0[10, 10] = 1
        g = init(CAProblem(sis, g0, 5.0, (β=1.0, γ=0.5); seed=3), Gillespie();
                 save_everystep=false)
        step!(g, 1.25, true)
        @test g.t == 1.25
        step!(g, 0.5, true)
        @test g.t == 1.75
        step!(g, 0.1, false)
        @test g.t >= 1.85

        # disturb!: a ready-made pulse
        u = ones(Int8, 200, 200)
        disturb!(u, Xoshiro(1); fraction=0.3)
        @test abs(count(iszero, u) / length(u) - 0.3) <= 4 * sqrt(0.3 * 0.7 / length(u))
        @test disturb!(ones(Int8, 50), Xoshiro(2); fraction=0.5) ==
              disturb!(ones(Int8, 50), Xoshiro(2); fraction=0.5)
        s = fill(:S, 100)
        s[1:50] .= :I
        w = Int8[0, 1, 2, 3, 0, 5]
        @test disturb!(copy(w), Xoshiro(5); fraction=1.0, from=(>(1))) ==
              Int8[0, 1, 0, 0, 0, 0]
        disturb!(s, Xoshiro(3); fraction=1.0, value=:R, from=:I)
        @test count(==(:R), s) == 50 && count(==(:S), s) == 50
        @test disturb!(ones(Int8, 10), Xoshiro(4); fraction=0.0) == ones(Int8, 10)
        @test_throws ArgumentError disturb!(ones(3), Xoshiro(1); fraction=1.5)
    end
end
