# Models: CAModel, CAProblem(model, u0map, tspan, pmap) and the @camodel DSL.
@testset "models" begin
    sir_rule = LocalRule(VonNeumann(1)) do c, n, ctx
        return c == :S ? (bernoulli(ctx, 1 - (1 - ctx.p.β)^count(==(:I), n), 1) ? :I : :S) :
               c == :I ? (bernoulli(ctx, ctx.p.γ, 2) ? :R : :I) : c
    end
    m = CAModel(sir_rule; name=:SIR,
                states=[StateVariable(:state, :S; domain=(:S, :I, :R))],
                parameters=(β=0.3, γ=0.1))

    @testset "CAModel" begin
        @test [s.name for s in states(m)] == [:state] &&
              [p.name for p in parameters(m)] == [:β, :γ]
        @test defaults(m) == (β=0.3, γ=0.1)
        @test sprint(show, m) == "CAModel SIR with states (state) and parameters (β, γ)"
        @test occursin("∈", sprint(show, MIME("text/plain"), m))
        @test_throws ArgumentError CAModel(sir_rule; states=(;))
        @test_throws ArgumentError CAModel(sir_rule; states=(a=1,),
                                           parameters=[Parameter(:x, 1), Parameter(:x, 2)])
        @test_throws ArgumentError StateVariable(:s, :X; domain=(:S, :I))
    end

    @testset "problems from models" begin
        seedI = I -> I == CartesianIndex(10, 10) ? :I : :S
        p = CAProblem(m, (state=seedI,), 30, (β=0.5,); space=Torus(20, 20), seed=4)
        @test p.p == (β=0.5, γ=0.1) && p.u0[10, 10] == :I && count(==(:I), p.u0) == 1
        u0 = fill(:S, 20, 20)
        u0[10, 10] = :I
        hand = CAProblem(sir_rule, u0, 30, (β=0.5, γ=0.1); seed=4)
        @test solve(p).u == solve(hand).u
        @test CAProblem(m, [:state => u0], 5; space=Torus(20, 20)).u0 == u0
        @test all(==(:I), CAProblem(m, Dict(:state => :I), 5; space=Torus(4, 4)).u0)
        @test all(==(:S), CAProblem(m, (), 5; space=Torus(4, 4)).u0)
        @test_throws ArgumentError CAProblem(m, (x=1,), 5; space=Torus(4, 4))
        @test_throws ArgumentError CAProblem(m, (), 5, (δ=1,); space=Torus(4, 4))
        @test_throws InvalidStateError CAProblem(m, (state=:Z,), 5; space=Torus(4, 4))
        nop = CAModel(LocalRule((c, n, ctx) -> c); states=(x=0,))
        @test CAProblem(nop, (), 1; space=Torus(3)).p === NullParameters()
    end

    @testset "@camodel" begin
        sir = @camodel SIR begin
            @description "lattice SIR"
            @states state::Symbol = :S ∈ (:S, :I, :R)
            @parameters β = 0.3 γ = 0.1
            @neighbourhood VonNeumann(1)
            @rule begin
                if cell == :S
                    bernoulli(ctx, 1 - (1 - β)^count(==(:I), nbrs), 1) ? :I : :S
                elseif cell == :I
                    bernoulli(ctx, γ, 2) ? :R : :I
                else
                    cell
                end
            end
        end
        @test sir.name == :SIR && sir.description == "lattice SIR" &&
              defaults(sir) == (β=0.3, γ=0.1)
        u0map = (state=I -> I == CartesianIndex(10, 10) ? :I : :S,)
        a = solve(CAProblem(sir, u0map, 30, (β=0.5,); space=Torus(20, 20), seed=4))
        b = solve(CAProblem(m, u0map, 30, (β=0.5,); space=Torus(20, 20), seed=4))
        @test a.u == b.u  # the DSL and the hand-written rule agree

        # several states, a kernel field and a covariate
        hostpest = @camodel HostPest begin
            @states host::Float64 = 1.0 pest::Float64 = 0.0
            @parameters r = 0.5 m = 0.2
            @fields spores = KernelField(GaussianKernel(1.0), c -> c.pest)
            @covariates suitability
            @rule begin
                incoming = field(ctx, :spores)
                pest = clamp(cell.pest +
                             r * covariate(ctx, :suitability) * incoming * cell.host -
                             m * cell.pest, 0.0, 1.0)
                (host=max(cell.host - 0.1 * pest, 0.0), pest=pest)
            end
        end
        @test [s.name for s in states(hostpest)] == [:host, :pest] &&
              hostpest.covariates == [:suitability]
        @test_throws ArgumentError CAProblem(hostpest, (), 3; space=Torus(8, 8))
        hp = CAProblem(hostpest, (pest=I -> I == CartesianIndex(4, 4) ? 0.5 : 0.0,), 3;
                       space=Torus(8, 8), covariates=(suitability=ones(8, 8),))
        @test keys(hp.u0) == (:host, :pest) && keys(hp.fields) == (:spores,)
        s = solve(hp)
        @test sum(final_state(s).pest) != 0.5 && all(0 .<= final_state(s).host .<= 1)
        # models run on every backend; the generated rule closes over nothing
        @test solve(hp; backend=GPUBackend(KernelAbstractions.CPU())).u == s.u

        @test_throws LoadError @eval @camodel Bad begin
            @states x::Int = 0
        end
        @test_throws LoadError @eval @camodel Bad begin
            @rule cell
        end
        @test_throws LoadError @eval @camodel Bad begin
            @states x::Int = 0
            @wibble 1
            @rule cell
        end
    end
end
