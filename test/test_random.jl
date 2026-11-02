# Counter-based random numbers: Philox4x32-10 against the Random123 known-answer vectors, and
# the per-cell draws built on it.
@testset "random" begin
    @testset "Philox4x32-10 known answers" begin
        @test philox4x32((0x00000000, 0x00000000, 0x00000000, 0x00000000),
                         (0x00000000, 0x00000000)) ==
              (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
        @test philox4x32((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
                         (0xffffffff, 0xffffffff)) ==
              (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
        @test philox4x32((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
                         (0xa4093822, 0x299f31d0)) ==
              (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)
    end

    ctx(id, t, seed) = CellContext(nothing, id, t, id, UInt64(seed))

    @testset "uniform is a pure function of seed, cell, time and draw" begin
        @test uniform(ctx(7, 3, 1)) == uniform(ctx(7, 3, 1))
        @test uniform(ctx(7, 3, 1)) != uniform(ctx(8, 3, 1))
        @test uniform(ctx(7, 3, 1)) != uniform(ctx(7, 4, 1))
        @test uniform(ctx(7, 3, 1)) != uniform(ctx(7, 3, 2))
        @test uniform(ctx(7, 3, 1), 1) != uniform(ctx(7, 3, 1), 2)
        @test uniform(ctx(7, 3, 1)) == uniform(ctx(7, 3, 1), 1)
    end

    @testset "uniform is in [0, 1) and roughly uniform" begin
        xs = [uniform(ctx(i, 0, 2026), k) for i in 1:5000, k in 1:4]
        @test all(x -> 0 <= x < 1, xs)
        @test abs(sum(xs) / length(xs) - 0.5) < 0.01
        @test abs(count(<(0.25), xs) / length(xs) - 0.25) < 0.01
        # draws for different k are not correlated
        a, b = xs[:, 1], xs[:, 2]
        cov = sum((a .- 0.5) .* (b .- 0.5)) / length(a)
        @test abs(cov / (1 / 12)) < 0.05
    end

    @testset "single precision" begin
        xs = [uniform(Float32, ctx(i, 0, 2026), 1) for i in 1:20_000]
        @test eltype(xs) == Float32 && all(x -> 0 <= x < 1, xs)
        @test abs(sum(xs) / length(xs) - 0.5) < 0.01
        @test uniform(Float64, ctx(3, 1, 1), 2) == uniform(ctx(3, 1, 1), 2)
        hits = count(i -> bernoulli(ctx(i, 1, 5), 0.3f0), 1:20_000)
        @test abs(hits / 20_000 - 0.3) < 0.015
        # a Float32 probability uses the Float32 draw
        c = ctx(9, 2, 4)
        @test bernoulli(c, 0.5f0) == (uniform(Float32, c) < 0.5f0)
        @test Stochastic(0.5f0).p isa Float32 && Stochastic(1).p isa Float64
    end

    @testset "bernoulli" begin
        hits = count(i -> bernoulli(ctx(i, 1, 5), 0.3), 1:20_000)
        @test abs(hits / 20_000 - 0.3) < 0.015
        @test !bernoulli(ctx(1, 1, 5), 0.0)
        @test bernoulli(ctx(1, 1, 5), 1.0)
    end
end
