# SparseBackend specifics, the GPU backend's constructor and checks, Lenia, and measures.
@testset "sparse backend" begin
    seedgrow = Life("B1/S012345678")
    u = zeros(Int8, 200, 200)
    u[100, 100] = 1
    integ = init(CAProblem(seedgrow, u, 10); backend=SparseBackend(Int8(0); tilesize=16))
    @test stored_tiles(integ) == 1
    step!(integ, 10)
    @test stored_tiles(integ) <= 9 &&
          integ.u == final_state(solve(CAProblem(seedgrow, u, 10)))
    @test_throws ArgumentError stored_tiles(init(CAProblem(seedgrow, u, 1)))
    # not quiescent: a background cell would change on its own
    flip = LocalRule((c, n, ctx) -> one(c) - c)
    @test_throws ArgumentError solve(CAProblem(flip, zeros(Int8, 8, 8), 1);
                                     backend=SparseBackend(Int8(0)))
    @test_throws IncompatibleSpaceError solve(CAProblem(seedgrow, zeros(Int8, 4), 1;
                                                        space=GraphSpace(4, [(1, 2)]));
                                              backend=SparseBackend(Int8(0)))
    @test_throws ArgumentError solve(CAProblem(LocalRule((c, n, ctx) -> c), zeros(8, 8), 1;
                                               fields=(f=KernelField(UniformKernel(1)),));
                                     backend=SparseBackend(0.0))
    # a Constant edge different from the background keeps edge tiles active
    edge = solve(CAProblem(Life("B1/S"), zeros(Int8, 40, 40), 1;
                           space=Grid(40, 40; boundary=Constant(1)));
                 backend=SparseBackend(Int8(0); tilesize=8))
    @test edge.u[end] == final_state(solve(CAProblem(Life("B1/S"), zeros(Int8, 40, 40), 1;
                                                     space=Grid(40, 40; boundary=Constant(1)))))
    @test_throws ArgumentError SparseBackend(0; tilesize=0)
end

@testset "GPU backend construction" begin
    @test GPUBackend(JLArrays.JLArray(zeros(2))).device isa JLArrays.JLBackend
    @test_throws IncompatibleSpaceError solve(CAProblem(Life(), zeros(Int8, 4), 1;
                                                        space=GraphSpace(4, [(1, 2)]));
                                              backend=GPUBackend(KernelAbstractions.CPU()))
    probe = LocalRule((c, n, ctx) -> field(ctx, :f))
    @test_throws ArgumentError solve(CAProblem(probe, zeros(8, 8), 1;
                                               fields=(f=KernelField(UniformKernel(1);
                                                                     method=:fft),));
                                     backend=GPUBackend(KernelAbstractions.CPU()))
    @test !supports(GPUBackend(KernelAbstractions.CPU()), RandomSequential())
end

@testset "Lenia" begin
    u = zeros(64, 64)
    place!(u, ORBIUM, (20, 20))
    sol = solve(CAProblem(Lenia(; method=:fft), u, 200); saveat=100)
    mass = sum.(sol.u)
    @test all(m -> abs(m - mass[1]) / mass[1] < 0.15, mass)  # the glider survives
    centre(x) = (sum(i * x[i, j] for i in axes(x, 1), j in axes(x, 2)) / sum(x))
    @test abs(centre(sol.u[2]) - centre(sol.u[1])) > 3                       # and moves
    d = solve(CAProblem(Lenia(), u, 10); saveat=10)
    f = solve(CAProblem(Lenia(; method=:fft), u, 10); saveat=10)
    @test maximum(abs.(final_state(d) .- final_state(f))) < 1e-10
    @test_throws InvalidStateError CAProblem(Lenia(), fill(2.0, 8, 8), 1)
    @test_throws InvalidStateError CAProblem(Lenia(), zeros(Int, 8, 8), 1)
    @test_throws ArgumentError Lenia(; σ=0)
    @test occursin("Lenia(R=13", sprint(show, Lenia()))
    @test size(ORBIUM) == (20, 20) && maximum(ORBIUM) == 1
end

@testset "measures" begin
    @test state_counts([:a :b; :a :a]) == Dict(:a => 3, :b => 1)
    u = [1 1 0 0; 0 0 0 1; 1 0 0 1]
    @test cluster_sizes(u; space=Grid(3, 4)) == [2, 2, 1]
    @test cluster_sizes(u) == [5]                    # on a torus everything joins up
    @test cluster_sizes(u; space=Grid(3, 4), neighbourhood=Moore(1)) == [2, 2, 1]
    @test cluster_sizes(u, ==(0); space=Grid(3, 4)) == [7]
    g = GraphSpace(4, [(1, 2), (3, 4)])
    @test cluster_sizes([1, 1, 1, 0]; space=g) == [2, 1]
    checker = [isodd(i + j) ? 1.0 : 0.0 for i in 1:6, j in 1:6]
    @test morans_i(checker) ≈ -1
    stripes = [j <= 3 ? 1.0 : 0.0 for i in 1:6, j in 1:6]
    @test morans_i(stripes; space=Grid(6, 6)) > 0.5
    @test_throws ArgumentError morans_i(ones(3, 3))
end
