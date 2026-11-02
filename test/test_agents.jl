# Agent backends with mobile agents. Without mobile agents every mode is in the conformance
# battery; here sprayers random-walk over an SIS epidemic and clear the infected cells they
# visit. The walkers are written once against read_cell/write_cell! and a seeded RNG, so the
# Agents.jl and Ark.jl versions, in both modes, must agree exactly.

# (`@agent` needs `using Agents`, whose `GraphSpace` would clash with ours in these tests)
mutable struct Sprayer <: Agents.AbstractAgent
    const id::Int
    pos::NTuple{2,Int}
end
Sprayer(; id, pos) = Sprayer(id, pos)
struct Walker
    pos::NTuple{2,Int}
end

const MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
walk(pos, rng, dims) = mod1.(pos .+ MOVES[rand(rng, 1:4)], dims)
function spray!(host, pos)
    read_cell(host, pos) == 1 && write_cell!(host, pos, 0)
    return nothing
end

function agents_sprayers(mode; n=5, dims=(20, 20))
    return AgentsBackend(mode; agent_type=Sprayer,
                         (init!)=m -> foreach(k -> Agents.add_agent!((2k, 3k), Sprayer, m),
                                              1:n),
                         (agent_step!)=(a, m) -> begin
                             Agents.move_agent!(a, walk(a.pos, Agents.abmrng(m), dims), m)
                             spray!(m, a.pos)
                         end)
end
function ark_sprayers(mode; n=5, dims=(20, 20))
    return ArkBackend(mode; components=(Walker,),
                      (init!)=(w, p) -> foreach(k -> Ark.new_entity!(w, (Walker((2k, 3k)),)),
                                                1:n),
                      (update!)=(w, p, t) -> begin
                          rng = Ark.get_resource(w, Random.Xoshiro)
                          for (es, ws) in Ark.Query(w, (Walker,))
                              for i in eachindex(es)
                                  pos = walk(ws[i].pos, rng, dims)
                                  ws[i] = Walker(pos)
                                  spray!(w, pos)
                              end
                          end
                      end)
end
count_walkers(w) = Ark.count_entities(Ark.Query(w, (Walker,)))

@testset "agent backends" begin
    sis = LocalRule(VonNeumann(1)) do c, n, ctx
        c == 1 && return bernoulli(ctx, ctx.p.γ) ? Int8(0) : Int8(1)
        return bernoulli(ctx, 1 - (1 - ctx.p.β)^count(==(1), n)) ? Int8(1) : Int8(0)
    end
    u0 = zeros(Int8, 20, 20)
    u0[8:12, 8:12] .= 1
    prob = CAProblem(sis, u0, 30, (β=0.3, γ=0.1); seed=5)
    plain = solve(prob)

    @testset "construction" begin
        @test_throws ArgumentError AgentsBackend(:agents)
        @test_throws ArgumentError ArkBackend(:entities)
        @test sprint(show, AgentsBackend(:cells)) == "AgentsBackend(:cells)"
        @test sprint(show, agents_sprayers(:grid)) ==
              "AgentsBackend(:grid; agent_type=Sprayer)"
        @test sprint(show, ArkBackend()) == "ArkBackend(:grid)"
        @test supports(AgentsBackend(), Margolus()) && !supports(ArkBackend(), Gillespie())
        ev = EventRule((c, n, ctx) -> ())
        @test_throws UnsupportedCombinationError solve(CAProblem(ev, u0, 1.0), Gillespie();
                                                       backend=AgentsBackend())
        @test_throws ArgumentError agent_model(init(prob))
    end

    @testset "mobile agents agree across packages and modes" begin
        sols = [solve(prob; backend=b)
                for b in (agents_sprayers(:grid), agents_sprayers(:cells),
                          ark_sprayers(:grid), ark_sprayers(:cells))]
        @test all(s -> s.u == sols[1].u, sols)
        @test sols[1].u != plain.u
        @test sum(sols[1].u[end]) < sum(plain.u[end])  # spraying lowers prevalence
        @test solve(prob; backend=agents_sprayers(:cells)).u == sols[1].u  # reproducible
        @test solve(remake(prob; seed=6); backend=ark_sprayers(:grid)).u != sols[1].u
    end

    @testset "Agents.jl model" begin
        times = Int[]
        b = AgentsBackend(:cells; agent_type=Sprayer,
                          (init!)=m -> Agents.add_agent!((1, 1), Sprayer, m),
                          (agent_step!)=(a, m) -> nothing,
                          (model_step!)=m -> push!(times, m.t))
        integ = init(prob; backend=b)
        m = agent_model(integ)
        @test Agents.nagents(m) == 401 && m.p == (β=0.3, γ=0.1)
        @test m[401] isa Sprayer && m[1].pos == (1, 1) && m[21].pos == (1, 2)
        step!(integ, 3)
        @test times == [1, 2, 3] && Agents.abmtime(m) == 3
        @test [m[i].state for i in 1:400] == vec(integ.u)
        I = CartesianIndex(3, 4)
        @test read_cell(m, I) == read_cell(m, (3, 4)) == read_cell(m, 63) == integ.u[I]
        # integrator parameters reach the model
        integ.p = (β=0.0, γ=1.0)
        step!(integ)
        @test agent_model(integ).p == (β=0.0, γ=1.0) && all(iszero, integ.u)
        # reinit! removes the mobile agents and reruns init!
        reinit!(integ)
        @test Agents.nagents(m) == 401 && m[401].pos == (1, 1) && m.t == 0
    end

    @testset "Ark.jl world" begin
        integ = init(prob; backend=ArkBackend(:cells))
        w = agent_model(integ)
        cellsq = Ark.Query(w, (CACell{Int8},))
        @test Ark.count_entities(cellsq) == 400
        step!(integ, 4)
        u = similar(u0)
        for (es, cs) in Ark.Query(w, (CACell{Int8},)), c in cs
            u[c.index] = c.state
        end
        @test u == integ.u
        @test read_cell(w, (5, 6)) == read_cell(w, 105) == integ.u[5, 6]
    end

    @testset "reinit! replays a run with mobile agents" begin
        for b in (agents_sprayers(:cells), ark_sprayers(:grid))
            integ = init(prob; backend=b)
            solve!(integ)
            first = copy(integ.sol_u)
            reinit!(integ)
            solve!(integ)
            @test integ.sol_u == first
            w = agent_model(integ)
            @test (w isa Agents.ABM ? Agents.nagents(w) - 400 : count_walkers(w)) == 5
        end
    end

    @testset "callbacks see a consistent state" begin
        still = LocalRule((c, n, ctx) -> c, VonNeumann(1))
        p0 = CAProblem(still, zeros(Int8, 6, 6), 3)
        for b in (AgentsBackend(:grid), AgentsBackend(:cells), ArkBackend(:grid),
                  ArkBackend(:cells))
            edit! = integ -> begin
                integ.u[1, 1] = 5               # before agent_model: kept
                write_cell!(agent_model(integ), (2, 2), 6)
                integ.u[3, 3] = 7               # after: applied to the fresh state
            end
            s = solve(p0; backend=b, callback=PresetTimeCallback([2], edit!))
            @test s.u[end][1, 1] == 5 && s.u[end][2, 2] == 6 && s.u[end][3, 3] == 7
            @test s.u[2] == zeros(Int8, 6, 6)
        end
        # adding mobile agents from a callback
        add! = integ -> Agents.add_agent!((1, 1), Sprayer, agent_model(integ))
        integ = init(prob; backend=agents_sprayers(:grid; n=0),
                     callback=PresetTimeCallback([5, 10], add!))
        solve!(integ)
        @test Agents.nagents(agent_model(integ)) == 2
    end

    @testset "ensembles" begin
        ens = EnsembleProblem(prob)
        serial = solve(ens, Synchronous(), EnsembleSerial(); trajectories=3,
                       backend=ark_sprayers(:cells))
        threaded = solve(ens, Synchronous(), EnsembleThreads(); trajectories=3,
                         backend=ark_sprayers(:cells))
        @test [s.u for s in serial.u] == [s.u for s in threaded.u]
        @test serial.u[1].u != serial.u[2].u
    end
end
