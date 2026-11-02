# Problems, integrators and solutions.

struct NoSuchScheme <: AbstractScheme end

@testset "problems and solutions" begin
    u0 = single_seed(21)

    @testset "construction" begin
        p = CAProblem(Wolfram(30), u0, 5)
        @test p.tspan == (0, 5) && p.space == Torus(21) && p.p === NullParameters()
        @test CAProblem(Wolfram(30), u0, (3, 8)).tspan == (3, 8)
        @test_throws ArgumentError CAProblem(Wolfram(30), u0, (5, 3))
        @test_throws ArgumentError CAProblem(Wolfram(30), u0, 2.5)
        @test_throws StateShapeError CAProblem(Wolfram(30), u0, 5; space=Torus(20))
        @test_throws StateShapeError CAProblem(Life(),
                                               (a=zeros(Int8, 3, 3), b=zeros(Int8, 3, 4)),
                                               1)
        @test_throws IncompatibleSpaceError CAProblem(Wolfram(30), zeros(Int8, 4, 4), 1)
        @test_throws IncompatibleSpaceError CAProblem(Life(; neighbourhood=Moore(1)),
                                                      zeros(Int8, 4, 4), 1;
                                                      space=HexGrid(4, 4))
        @test occursin("Wolfram(30)", sprint(show, p))
    end

    @testset "saving" begin
        p = CAProblem(Wolfram(30), u0, 10)
        @test solve(p).t == collect(0:10)
        @test solve(p; saveat=3).t == [0, 3, 6, 9, 10]
        @test solve(p; saveat=[2, 7]).t == [0, 2, 7, 10]
        @test solve(p; saveat=5, save_start=false).t == [5, 10]
        @test solve(p; saveat=4, save_end=false).t == [0, 4, 8]
        @test solve(p; saveat=5, save_end=false).t == [0, 5, 10]
        @test solve(CAProblem(Wolfram(30), u0, (3, 6))).t == [3, 4, 5, 6]
        @test solve(CAProblem(Wolfram(30), u0, 0)).t == [0]
        @test_throws ArgumentError solve(p; saveat=0)
    end

    @testset "integrator" begin
        p = CAProblem(Wolfram(30), u0, 10)
        integ = init(p; saveat=100)
        step!(integ)
        @test integ.t == 1
        step!(integ, 3)
        @test integ.t == 4
        @test integ.u == solve(CAProblem(Wolfram(30), u0, 4)).u[end]
        sol = solve!(integ)
        @test integ.t == 10 && sol.t == [0, 10]
        step!(integ, 5)  # past the end: nothing happens
        @test integ.t == 10
        # the problem's initial state is not modified
        @test u0 == single_seed(21)
    end

    @testset "solutions" begin
        sol = solve(CAProblem(Wolfram(30), u0, 6))
        @test length(sol) == 7 && sol[1] == u0 && sol[end] == final_state(sol)
        @test collect(sol) == sol.u
        @test size(trajectory(sol)) == (21, 7)
        @test trajectory(sol)[:, end] == final_state(sol)
        s2 = solve(CAProblem(Life(), zeros(Int8, 4, 5), 3))
        @test size(trajectory(s2)) == (4, 5, 4)
        layered = (a=zeros(Int8, 4, 4), b=ones(Float64, 4, 4))
        keep = LocalRule((c, n, ctx) -> c)
        tr = trajectory(solve(CAProblem(keep, layered, 2)))
        @test keys(tr) == (:a, :b) && size(tr.b) == (4, 4, 3)
        @test occursin("CASolution", sprint(show, sol))
    end

    @testset "parameters, time and seeds reach the rule" begin
        # each cell records what the context says
        probe = LocalRule((c, n, ctx) -> ctx.p.scale * ctx.t + uniform(ctx), Moore(1))
        u = zeros(Float64, 3, 3)
        a = final_state(solve(CAProblem(probe, u, (4, 5), (scale=10.0,); seed=1)))
        b = final_state(solve(CAProblem(probe, u, (4, 5), (scale=10.0,); seed=1)))
        c = final_state(solve(CAProblem(probe, u, (4, 5), (scale=10.0,); seed=2)))
        @test a == b && a != c
        @test all(x -> 40 <= x < 41, a)
        @test allunique(a)  # different cells, different draws
        idx = LocalRule((c, n, ctx) -> Float64(ctx.index[1] + 10ctx.index[2]))
        @test final_state(solve(CAProblem(idx, zeros(3, 3), 1))) ==
              [i + 10j for i in 1:3, j in 1:3]
    end

    @testset "unsupported combinations" begin
        @test !supports(ArrayBackend(), NoSuchScheme())
        @test supports(ArrayBackend(), Synchronous()) &&
              supports(ReferenceBackend(), Synchronous())
        @test_throws UnsupportedCombinationError solve(CAProblem(Wolfram(30), u0, 1),
                                                       NoSuchScheme())
    end

    @testset "plot recipes" begin
        # Without Plots loaded nothing answers `is_key_supported`; accept every attribute.
        RB = CellularAutomataCore.RecipesBase
        RB.is_key_supported(::Symbol) = true
        recipe(args...) = only(RB.apply_recipe(Dict{Symbol,Any}(), args...))
        sol1 = solve(CAProblem(Wolfram(30), u0, 5))
        r = recipe(sol1)  # space-time diagram: cells × times, one row per time
        @test r.plotattributes[:seriestype] == :heatmap
        @test r.args[1] == 1:21 && r.args[2] == 0:5 &&
              r.args[3] == permutedims(stack(sol1.u))
        sol2 = solve(CAProblem(Life(), zeros(Int8, 4, 5), 2))
        r = recipe(sol2)  # the last state, passed on to the (space, state) recipe
        @test r.args == (Torus(4, 5), sol2.u[end])
        @test r.plotattributes[:title] == "t = 2"
        r = only(RB.apply_recipe(Dict{Symbol,Any}(:snapshot => 1), sol2))
        @test r.args == (Torus(4, 5), sol2.u[1]) && r.plotattributes[:title] == "t = 0"
        r = recipe(Torus(4, 5), sol2.u[end])
        @test r.plotattributes[:seriestype] == :heatmap && r.args[3] == sol2.u[end]
        r = recipe(HexGrid(3, 3), reshape(1.0:9.0, 3, 3))
        @test r.plotattributes[:seriestype] == :shape
        @test r.plotattributes[:fill_z] == 1.0:9.0
        @test length(r.args[1]) == 9 * 8  # seven vertices and a NaN separator per cell
        r = recipe(Torus(4), (a=zeros(4), b=ones(4)))  # layered: first layer by default
        @test r.args == (Torus(4), zeros(4))
        xs, ys = hexagon_vertices(0, 0)
        @test length(xs) == 7 && xs[1] ≈ xs[end] && all(hypot.(xs, ys) .≈ 1)
    end
end
