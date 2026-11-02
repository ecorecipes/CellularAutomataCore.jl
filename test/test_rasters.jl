# Landscapes from rasters (the Rasters extension): spaces, masks and covariates follow the raster's X
# and Y dimensions whatever their stored order, missing values become masks and fill values, a Ti
# dimension becomes the time axis of a covariate, and states are written back on the raster's grid.
using Rasters: Rasters, Raster, RasterStack, X, Y, Ti, Band

@testset "rasters" begin
    data = reshape(Float64.(1:12), 3, 4)
    data[2, 3] = -9.0
    r = Raster(data, (X(10.0:10.0:30.0), Y(1.0:4.0)); missingval=-9.0)

    space = raster_space(r)
    @test size(space) == (3, 4) && all(b -> b isa Constant, boundaries(space))
    @test boundaries(raster_space(r; boundary=Periodic())) == (Periodic(), Periodic())

    mask = raster_mask(r)
    @test mask == (data .!= -9.0) && count(!, mask) == 1
    cov = raster_covariate(r; missing_as=0)
    @test cov[2, 3] == 0 && cov[1, 1] == 1 && size(cov) == (3, 4)

    # stored with Y first, the raster gives the same lattice
    ry = permutedims(r, (Y, X))
    @test size(raster_space(ry)) == (3, 4) && raster_mask(ry) == mask &&
          raster_covariate(ry) == cov

    # missing values as `missing`
    rm = Raster(Union{Missing,Float64}[1 missing; 3 4], (X(1:2), Y(1:2));
                missingval=missing)
    @test raster_mask(rm) == [true false; true true]
    @test raster_covariate(rm; missing_as=-1, T=Float32) == Float32[1 -1; 3 4]

    # a time dimension becomes the covariate's last axis, which the core cycles through
    rt = Raster(cat(fill(0.0, 3, 4), fill(1.0, 3, 4); dims=3), (X(1:3), Y(1:4), Ti(1:2)))
    ct = raster_covariate(rt)
    @test size(ct) == (3, 4, 2) && all(ct[:, :, 2] .== 1)
    copy_cov = LocalRule((c, n, ctx) -> covariate(ctx, :season), VonNeumann(1))
    sol = solve(CAProblem(copy_cov, zeros(3, 4), 3; space=raster_space(rt),
                          covariates=(; season=ct)))
    # the step from time t reads slice mod1(t + 1, 2): states after steps 0, 1, 2 hold 0, 1, 0
    @test [first(u) for u in sol.u] == [0, 0, 1, 0]

    # a masked run on the raster's landscape, written back on its grid
    grow = LocalRule((c, n, ctx) -> c + 1, VonNeumann(1))
    u = solve(CAProblem(grow, zeros(Int, 3, 4), 2; space, mask)).u[end]
    @test u[2, 3] == 0 && u[1, 1] == 2
    out = state_raster(u, r)
    @test out isa Raster && parent(out) == u &&
          Rasters.lookup(out, X) == Rasters.lookup(r, X)
    masked = state_raster(u, r; mask, missingval=-1)
    @test masked[2, 3] == -1 && Rasters.missingval(masked) == -1
    stack = state_raster((a=u, b=float.(u)), r)
    @test stack isa RasterStack && keys(stack) == (:a, :b) && parent(stack[:b]) == float.(u)

    @test_throws StateShapeError state_raster(zeros(2, 2), r)
    @test_throws ArgumentError state_raster(u, r; mask)
    rb = Raster(zeros(3, 4, 2), (X(1:3), Y(1:4), Band(1:2)))
    @test_throws ArgumentError raster_space(rb)
    @test_throws ArgumentError raster_space(Raster(zeros(3), (X(1:3),)))
    # without a raster, the functions say what to load
    @test_throws ArgumentError raster_space(zeros(3, 3))
    @test_throws ArgumentError raster_mask(zeros(3, 3))
end
