# Dispersal kernels, kernel fields, covariates and masks.
@testset "environment" begin
    @testset "kernel weights" begin
        for k in
            (GaussianKernel(1.5), ExponentialKernel(1.0), PowerLawKernel(2.0; radius=4),
             UniformKernel(2), LeniaKernel(5), LeniaKernel(6; peaks=(0.5, 1.0)))
            o, w = kernel_weights(k, Torus(20, 20))
            @test sum(w) ≈ 1 && all(>(0), w) && length(o) == length(w)
            @test all(x -> sqrt(sum(abs2, Tuple(x))) <=
                           CellularAutomataCore._kernel_radius(k) + 1e-9, o)
        end
        o, w = kernel_weights(UniformKernel(1), Torus(5, 5))
        @test Set(Tuple.(o)) == Set([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
        o, _ = kernel_weights(UniformKernel(1), Torus(5, 5); include_centre=false)
        @test (0, 0) ∉ Tuple.(o)
        # Gaussian weights fall with distance; Lenia's shell is zero at the centre
        o, w = kernel_weights(GaussianKernel(1.0; radius=3), Torus(9, 9))
        @test w[findfirst(==(CartesianIndex(0, 0)), o)] >
              w[findfirst(==(CartesianIndex(0, 2)), o)]
        o, _ = kernel_weights(LeniaKernel(4), Torus(9, 9))
        @test CartesianIndex(0, 0) ∉ o
        # anisotropy: wind towards the east (+column) favours positive column offsets
        o, w = kernel_weights(AnisotropicKernel(GaussianKernel(1.0; radius=2), 0.0, 2.0),
                              Torus(9, 9))
        east = w[findfirst(==(CartesianIndex(0, 1)), o)]
        west = w[findfirst(==(CartesianIndex(0, -1)), o)]
        north = w[findfirst(==(CartesianIndex(-1, 0)), o)]
        @test east > north > west
        o, w = kernel_weights(AnisotropicKernel(GaussianKernel(1.0; radius=2), π / 2, 2.0),
                              Torus(9, 9))
        @test w[findfirst(==(CartesianIndex(-1, 0)), o)] >
              w[findfirst(==(CartesianIndex(1, 0)), o)]
        # empirical kernels and hexagonal distances
        o, w = kernel_weights(EmpiricalKernel([0 1 0; 2 4 2; 0 1 0]), Torus(5, 5))
        @test w[findfirst(==(CartesianIndex(0, 0)), o)] ≈ 0.4
        o, _ = kernel_weights(UniformKernel(1), HexGrid(6, 6); include_centre=false)
        @test length(o) == 6
        @test_throws ArgumentError GaussianKernel(-1.0)
        @test_throws ArgumentError EmpiricalKernel([1 1; 1 1])
        @test_throws IncompatibleSpaceError kernel_weights(UniformKernel(1),
                                                           GraphSpace([[1]]))
        @test_throws IncompatibleSpaceError kernel_weights(EmpiricalKernel(ones(3, 3)),
                                                           Torus(3, 3, 3))
    end

    @testset "kernel fields" begin
        probe = LocalRule((c, n, ctx) -> field(ctx, :f))
        u = zeros(7, 7)
        u[4, 4] = 1.0
        # a uniform radius-1 field: 1/5 on the cross around the live cell
        v = final_state(solve(CAProblem(probe, u, 1;
                                        fields=(f=KernelField(UniformKernel(1)),))))
        @test v[4, 4] ≈ 0.2 && v[3, 4] ≈ 0.2 && v[4, 5] ≈ 0.2 && v[3, 3] == 0
        # the source function and the edges: beyond a Constant(1) edge cells count as 1
        e = final_state(solve(CAProblem(probe, zeros(5, 5), 1;
                                        fields=(f=KernelField(UniformKernel(1)),),
                                        space=Grid(5, 5; boundary=Constant(1.0)))))
        @test e[1, 1] ≈ 0.4 && e[1, 3] ≈ 0.2 && e[3, 3] == 0
        # FFT agrees with direct summation to rounding, on every backend that computes it
        x = rand(Random.Xoshiro(1), 32, 24)
        fd = (f=KernelField(GaussianKernel(2.0), identity),)
        ff = (f=KernelField(GaussianKernel(2.0), identity; method=:fft),)
        a = final_state(solve(CAProblem(probe, x, 1; fields=fd)))
        b = final_state(solve(CAProblem(probe, x, 1; fields=ff)))
        @test maximum(abs.(a .- b)) < 1e-12
        @test_throws ArgumentError CAProblem(probe, x, 1; fields=ff, space=Grid(32, 24))
        @test_throws ArgumentError KernelField(UniformKernel(1); method=:magic)
        @test_throws ArgumentError CAProblem(probe, x, 1; fields=(f=1,))
        @test_throws IncompatibleSpaceError CAProblem(probe, zeros(3), 1;
                                                      fields=(f=KernelField(UniformKernel(1)),),
                                                      space=GraphSpace(3, [(1, 2)]))
        @test fields(Life()) == (;)
        # dispersal direction: an eastward kernel carries a source's influence east
        src = zeros(9, 9)
        src[5, 5] = 1.0
        east = (f=KernelField(AnisotropicKernel(GaussianKernel(1.0; radius=3), 0.0, 3.0)),)
        w = final_state(solve(CAProblem(probe, src, 1; fields=east)))
        @test w[5, 6] > w[5, 4] && w[4, 5] ≈ w[6, 5]
        nw = (f=KernelField(EmpiricalKernel([1 0 0; 0 0 0; 0 0 0])),)  # all mass at offset (-1, -1)
        w = final_state(solve(CAProblem(probe, src, 1; fields=nw)))
        @test w[4, 4] ≈ 1 && sum(w) ≈ 1
    end

    @testset "covariates" begin
        probe = LocalRule((c, n, ctx) -> covariate(ctx, :c))
        fixed = reshape(Float64.(1:12), 3, 4)
        @test final_state(solve(CAProblem(probe, zeros(3, 4), 2; covariates=(c=fixed,)))) ==
              fixed
        cycling = cat(fill(1.0, 3, 4), fill(2.0, 3, 4); dims=3)
        sol = solve(CAProblem(probe, zeros(3, 4), 4; covariates=(c=cycling,)))
        @test [x[1, 1] for x in sol.u] == [0.0, 1.0, 2.0, 1.0, 2.0]
        g = GraphSpace(3, [(1, 2), (2, 3)])
        @test final_state(solve(CAProblem(probe, zeros(3), 1; space=g,
                                          covariates=(c=[5.0, 6, 7],)))) ==
              [5.0, 6, 7]
        @test_throws StateShapeError CAProblem(probe, zeros(3, 4), 1;
                                               covariates=(c=zeros(4, 3),))
        @test_throws ArgumentError CAProblem(probe, zeros(3, 4), 1; covariates=(c=1.0,))
        @test_throws ArgumentError CAProblem(probe, zeros(3, 4), 1;
                                             covariates=[zeros(3, 4)])
    end

    @testset "masks" begin
        grow = LocalRule((c, n, ctx) -> c + 1)
        mask = trues(4, 4)
        mask[2, 3] = false
        v = final_state(solve(CAProblem(grow, zeros(Int, 4, 4), 3; mask)))
        @test v[2, 3] == 0 && count(==(3), v) == 15
        @test_throws StateShapeError CAProblem(grow, zeros(Int, 4, 4), 1; mask=trues(3, 3))
        @test_throws ArgumentError CAProblem(grow, zeros(Int, 4, 4), 1; mask=ones(4, 4))
    end
end
