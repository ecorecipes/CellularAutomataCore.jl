# Update schemes: their defining properties, independent of backends.
@testset "update schemes" begin
    rng = Random.Xoshiro(11)
    u = Int8.(rand(rng, 0:1, 20, 20))
    count_updates = LocalRule((c, n, ctx) -> c + one(c))  # every update adds one

    @testset "Stochastic" begin
        @test final_state(solve(CAProblem(Life(), u, 5), Stochastic(1.0))) ==
              final_state(solve(CAProblem(Life(), u, 5)))
        sol = solve(CAProblem(count_updates, zeros(Int, 100, 100), 10; seed=1),
                    Stochastic(0.3))
        @test abs(sum(final_state(sol)) / (100 * 100 * 10) - 0.3) < 0.01
        @test_throws ArgumentError Stochastic(0.0)
        @test_throws ArgumentError Stochastic(1.5)
    end

    @testset "Checkerboard" begin
        # each cell updates exactly once per step
        @test all(==(3),
                  final_state(solve(CAProblem(count_updates, zeros(Int, 6, 7), 3),
                                    Checkerboard())))
        # the second half reads the first half's results: a cell copying its left neighbour
        copyleft = LocalRule((c, n, ctx) -> n[1], Offsets((0, -1)))
        v = [0 1 0 0]
        w = final_state(solve(CAProblem(copyleft, v, 1), Checkerboard()))
        # (1, j) with j odd has an even index sum and updates first, giving [0 1 1 0]; then
        # j = 2, 4 copy their updated left neighbours. Synchronous updating gives [0 0 1 0].
        @test w == [0 0 1 1]
        @test final_state(solve(CAProblem(copyleft, v, 1))) == [0 0 1 0]
        @test_throws IncompatibleSpaceError solve(CAProblem(count_updates, zeros(Int, 4), 1;
                                                            space=GraphSpace(4, [(1, 2)])),
                                                  Checkerboard())
    end

    @testset "RandomSequential" begin
        @test all(==(4),
                  final_state(solve(CAProblem(count_updates, zeros(Int, 5, 5), 4),
                                    RandomSequential())))
        # in place: information can cross the whole line in one step
        copyleft = LocalRule((c, n, ctx) -> max(c, n[1]), Offsets((-1,)))
        v = zeros(Int, 50)
        v[1] = 1
        w = final_state(solve(CAProblem(copyleft, v, 1; space=Grid(50)),
                              RandomSequential()))
        @test sum(w) > 2  # synchronous would give exactly 2
        @test sum(final_state(solve(CAProblem(copyleft, v, 1; space=Grid(50))))) == 2
        # the order depends on the seed
        a = solve(CAProblem(copyleft, v, 1; space=Grid(50), seed=1), RandomSequential())
        b = solve(CAProblem(copyleft, v, 1; space=Grid(50), seed=2), RandomSequential())
        @test final_state(a) != final_state(b)
        @test !supports(SparseBackend(0), RandomSequential())
    end

    @testset "Margolus" begin
        swap = BlockRule((b, ctx) -> (b[4], b[3], b[2], b[1]))  # rotate by 180°
        v = zeros(Int8, 4, 4)
        v[1, 1] = 1
        s = solve(CAProblem(swap, v, 2), Margolus())
        @test s.u[2][2, 2] == 1 && sum(s.u[2]) == 1          # block at (1,1) on even steps
        @test s.u[3][3, 3] == 1 && sum(s.u[3]) == 1          # block at (2,2) on odd steps
        @test_throws ArgumentError solve(CAProblem(swap, v, 1))
        @test_throws ArgumentError solve(CAProblem(Life(), v, 1), Margolus())
        @test_throws IncompatibleSpaceError solve(CAProblem(swap, zeros(Int8, 5, 4), 1),
                                                  Margolus())
        @test_throws IncompatibleSpaceError solve(CAProblem(swap, v, 1; space=Grid(4, 4)),
                                                  Margolus())
        @test occursin("BlockRule", sprint(show, swap))
    end
end
