# The public backend interface. `CellLoopBackend` is written only against it, as a package
# outside this one would write a backend, and joins the conformance battery; update_cell is
# checked against a synchronous step of the reference backend.

const CAC = CellularAutomataCore

struct CellLoopBackend <: AbstractBackend end
mutable struct CellLoopState{U}
    cur::U
    next::U
end
CAC.supports(::CellLoopBackend, ::Synchronous) = true
function CAC.check_problem(::CellLoopBackend, prob, scheme)
    prob.u0 isa AbstractArray ||
        throw(ArgumentError("CellLoopBackend needs a single array"))
    # update_cell raises for what it cannot evaluate (kernel fields, block and event rules)
    CAC.update_cell(prob, prob.u0, first(cells(prob.space)), first(prob.tspan))
    return nothing
end
function CAC.allocate_state(::CellLoopBackend, prob, scheme)
    return CellLoopState(copy(prob.u0), copy(prob.u0))
end
function CAC.advance!(::CellLoopBackend, st::CellLoopState, prob, scheme, p, t)
    for I in cells(prob.space)
        st.next[I] = CAC.update_cell(prob, st.cur, I, t, p)
    end
    st.cur, st.next = st.next, st.cur
    return nothing
end
CAC.current_state(::CellLoopBackend, st::CellLoopState) = copy(st.cur)
CAC.set_state!(::CellLoopBackend, st::CellLoopState, u) = (copyto!(st.cur, u); nothing)
CAC.read_cell(::CellLoopBackend, st::CellLoopState, I) = st.cur[I]
CAC.write_cell!(::CellLoopBackend, st::CellLoopState, I, v) = (st.cur[I]=v; nothing)

@testset "backend interface" begin
    @test issubset([:check_problem, :allocate_state, :advance!, :current_state, :set_state!,
                    :update_cell], names(CellularAutomataCore))
    @test !Base.isexported(CellularAutomataCore, :advance!)

    # update_cell over every cell is one synchronous step, at any time and with any p
    rng = Random.Xoshiro(3)
    sir = LocalRule(VonNeumann(1)) do c, n, ctx
        c == 0 &&
            return bernoulli(ctx, 1 - (1 - ctx.p.β * covariate(ctx, :s))^count(==(1), n)) ?
                   Int8(1) : Int8(0)
        return c == 1 && bernoulli(ctx, ctx.p.γ, 2) ? Int8(2) : c
    end
    u0 = zeros(Int8, 12, 10)
    u0[6, 5] = 1
    mask = trues(12, 10)
    mask[1:3, :] .= false
    probs = [CAProblem(sir, u0, 8, (β=0.5, γ=0.2); seed=4, mask,
                       covariates=(s=rand(rng, 12, 10),),
                       space=Grid(12, 10; boundary=(Constant(1), Reflecting()))),
             CAProblem(Life(), Int8.(rand(rng, 0:1, 9, 11)), 6; space=HexGrid(9, 11)),
             CAProblem(LocalRule((c, n, ctx) -> (a=n[1].b, b=c.a + uniform(ctx)), Moore(1)),
                       (a=rand(rng, 5, 5), b=rand(rng, 5, 5)), 5; seed=2),
             CAProblem(LocalRule((c, n, ctx) -> Int8(isodd(sum(n) + c))),
                       Int8.(rand(rng, 0:1, 15)), 5;
                       space=GraphSpace(15, [(i, mod1(i + 4, 15)) for i in 1:15]))]
    for prob in probs
        sol = solve(prob; backend=ReferenceBackend())
        for t in (0, 3)
            u = sol.u[t + 1]
            stepped = [CAC.update_cell(prob, u, I, t) for I in cells(prob.space)]
            @test all(I -> CAC._read(sol.u[t + 2], I) == stepped[I], keys(stepped))
        end
    end
    prob = probs[1]
    @test CAC.update_cell(prob, u0, (2, 5), 0) == 0  # masked: unchanged
    @test CAC.update_cell(prob, u0, 66, 0) ==
          CAC.update_cell(prob, u0, CartesianIndex(6, 6), 0)
    # with p = (β=0, γ=0) nothing can change
    @test all(I -> CAC.update_cell(prob, u0, I, 0, (β=0.0, γ=0.0)) == u0[I],
              cells(prob.space))

    @test_throws ArgumentError CAC.update_cell(CAProblem(Lenia(), zeros(20, 20), 1),
                                               zeros(20, 20), (1, 1), 0)
    block = BlockRule((b, ctx) -> b)
    @test_throws ArgumentError CAC.update_cell(CAProblem(block, zeros(Int8, 4, 4), 1),
                                               zeros(Int8, 4, 4), (1, 1), 0)

    # the backend runs through solve, callbacks and reinit!
    life = CAProblem(Life(), Int8.(rand(rng, 0:1, 16, 16)), 10)
    @test solve(life; backend=CellLoopBackend()).u == solve(life).u
    integ = init(life; backend=CellLoopBackend())
    step!(integ, 4)
    integ.u[1, 1] = 1
    step!(integ)
    @test integ.u == let ref = init(life)
        step!(ref, 4)
        ref.u[1, 1] = 1
        step!(ref)
        ref.u
    end
    reinit!(integ)
    @test integ.u == life.u0 && integ.t == 0
    @test_throws UnsupportedCombinationError solve(life, Stochastic(0.5);
                                                   backend=CellLoopBackend())
    @test_throws ArgumentError solve(probs[3]; backend=CellLoopBackend())
end
