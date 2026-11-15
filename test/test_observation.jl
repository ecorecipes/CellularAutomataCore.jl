# Observation models. Footprint weights are checked against a brute-force overlap (the share of
# a fine set of points in each unit that falls in each cell); aggregation against the dense
# matrix product; links against their densities written out by hand and against the frequencies
# of simulated reports.

# Which box of a MapGrid holds the point x, or nothing.
function box_of(g::MapGrid{N}, x) where {N}
    idx = ntuple(N) do d
        k = floor(Int, (x[d] - g.origin[d]) / g.spacing[d]) + 1
        return 1 <= k <= g.dims[d] ? k : 0
    end
    any(==(0), idx) && return nothing
    return LinearIndices(g.dims)[idx...]
end

@testset "observation" begin
    @testset "footprints" begin
        cg = MapGrid((7, 5); origin=(0.3, -1.0), spacing=(1.3, 0.9))
        ug = MapGrid((4, 3); origin=(0.0, 3.0), spacing=(2.1, -1.7))
        F = footprint(cg, ug)
        @test size(F) == (12, 35)
        W = Matrix(F)
        # brute force: m x m midpoints of each unit
        m = 200
        Wb = zeros(12, 35)
        for J in CartesianIndices(ug.dims)
            j = LinearIndices(ug.dims)[J]
            for a in 1:m, b in 1:m
                x = (ug.origin[1] + (J[1] - 1 + (a - 0.5) / m) * ug.spacing[1],
                     ug.origin[2] + (J[2] - 1 + (b - 0.5) / m) * ug.spacing[2])
                i = box_of(cg, x)
                i === nothing || (Wb[j, i] += 1 / m^2)
            end
        end
        @test maximum(abs.(W .- Wb)) < 0.01
        @test coverage(F) ≈ vec(sum(W; dims=2))
        @test any(<(0.99), coverage(F)) && any(c -> c ≈ 1, coverage(F))
        # nested and aligned: each 2 x 2 unit holds four cells at a quarter each
        Fn = footprint(MapGrid(4, 4), MapGrid((2, 2); spacing=2))
        @test all(row -> sort(filter(>(0), row)) == fill(0.25, 4), eachrow(Matrix(Fn)))
        # a finer sensor than the model: each unit lies in one cell
        Ff = footprint(MapGrid((2, 2); spacing=2), MapGrid(4, 4))
        @test all(row -> count(>(0), row) == 1 && sum(row) ≈ 1, eachrow(Matrix(Ff)))
        @test Matrix(Footprint(W)) ≈ W
        Ft = Footprint([1, 2, 1], [3, 1, 3], [0.2, 0.5, 0.3]; nunits=2, ncells=3)
        @test Matrix(Ft) ≈ [0 0 0.5; 0.5 0 0]
        @test sprint(show, Ft) == "Footprint(2 units, 3 cells, 3 weights)"
        v = rand(Xoshiro(5), 35)
        @test aggregate(F, v) ≈ W * v
        w = rand(Xoshiro(6), 35)
        w[1:10] .= 0
        r = aggregate(F, v, w)
        for j in 1:12
            den = sum(W[j, :] .* w)
            if den > 0
                @test r[j] ≈ sum(W[j, :] .* w .* v) / den
            else
                @test ismissing(r[j])
            end
        end
        @test_throws ArgumentError aggregate(F, rand(3))
        @test_throws ArgumentError MapGrid((2, 2); spacing=0)
        @test_throws ArgumentError MapGrid((2, 2); origin=(0, 0, 0))
        @test_throws ArgumentError Footprint([1], [4], [1.0]; nunits=1, ncells=3)
        @test_throws ArgumentError Footprint([1], [1], [-1.0]; nunits=1, ncells=3)
    end

    @testset "detection" begin
        d = Detection(0.05, 0.9)
        u = Int8[0 2 1; 2 2 0]
        obs = Observation(d; signal=s -> s == 2)
        y = [false, true, true, missing, true, false]
        sig = vec(u .== 2)
        hand = sum(log(yj ? 0.05 + 0.85 * s : 1 - (0.05 + 0.85 * s))
                   for (yj, s) in zip(y, sig) if !ismissing(yj))
        @test observation_logpdf(obs, y, u) ≈ hand
        @test observation_logpdf(obs, [0, 1, 1, missing, 1, 0], u) ≈ hand
        # through a footprint with host weights: a unit with no host reports false positives
        F = footprint(MapGrid(2, 3), MapGrid((1, 3); spacing=(2, 1)))
        host = [1.0 0.0 0.5; 2.0 0.0 1.0]
        obsF = Observation(d; signal=s -> s == 2, weight=host, footprint=F)
        rr = [(1 * 0 + 2 * 1) / 3, missing, (0.5 * 0 + 1 * 0) / 1.5]
        q(r) = 0.05 + 0.85 * (ismissing(r) ? 0.0 : r)
        @test observation_logpdf(obsF, [true, true, false], u) ≈
              log(q(rr[1])) + log(q(rr[2])) + log(1 - q(rr[3]))
        # simulated frequencies at a signal of one half
        half = Observation(Detection(0.1, 0.8))
        ys = simulate_observation(half, fill(0.5, 200, 100), 3; seed=11)
        @test abs(count(ys) / 20_000 - 0.45) < 4 * sqrt(0.45 * 0.55 / 20_000)
        @test simulate_observation(half, fill(0.5, 200, 100), 3; seed=11) == ys
        @test simulate_observation(half, fill(0.5, 200, 100), 4; seed=11) != ys
        masked = simulate_observation(obs, u; seed=1,
                                      mask=[true, false, true, true, true, true])
        @test ismissing(masked[2]) && count(ismissing, masked) == 1
        # the observation stream is not the dynamics' stream
        ctxs = [CellularAutomataCore.CellContext(nothing, CartesianIndex(1), 3, j, 11)
                for j in 1:50]
        @test simulate_observation(Observation(Detection(0, 1)), fill(0.5, 50), 3;
                                   seed=11) !=
              [uniform(c, 1) < 0.5 for c in ctxs]
        @test_throws ArgumentError Detection(1.2, 0.5)
        @test_throws ArgumentError observation_logpdf(Observation(d), [1], [1.5])
        @test_throws ArgumentError observation_logpdf(obs, [2, 0, 0, 0, 0, 0], u)
        @test_throws ArgumentError observation_logpdf(obs, [true], u)
        @test_throws StateShapeError observation_logpdf(obsF, [true, true, false],
                                                        zeros(3, 3))
    end

    @testset "misclassification" begin
        M = [0.9 0.1 0.0; 0.2 0.7 0.1; 0.0 0.05 0.95]
        m = Misclassification(M; classes=(:S, :I, :R))
        obs = Observation(m)
        u = [:S :I; :R :I]
        y = [1, 3, 2, 3]
        @test observation_logpdf(obs, y, u) ≈ log(0.9) + log(0.1) + log(0.7) + log(0.95)
        @test observation_logpdf(obs, [3, 1, 1, 1], u) == -Inf
        ys = simulate_observation(obs, fill(:I, 100, 100); seed=4)
        for b in 1:3
            @test abs(count(==(b), ys) / 10_000 - M[2, b]) <=
                  4 * sqrt(M[2, b] * (1 - M[2, b]) / 10_000) + 1e-12
        end
        @test_throws ArgumentError Misclassification([0.5 0.4; 0 1])
        @test_throws ArgumentError Misclassification(M; classes=(:S, :I))
        @test_throws ArgumentError Observation(m; footprint=Footprint(ones(1, 4)))
        @test_throws ArgumentError observation_logpdf(Observation(Misclassification(M)),
                                                      [1],
                                                      [:S])
    end

    @testset "gaussian error" begin
        g = GaussianError(0.2)
        obs = Observation(g; signal=s -> 0.5 * s)
        u = [0.0 1.0; 2.0 0.4]
        y = [0.1, 1.2, 0.3, missing]
        hand = sum(-((yj - 0.5 * s) / 0.2)^2 / 2 - log(0.2) - log(2π) / 2
                   for (yj, s) in zip(y, vec(u)) if !ismissing(yj))
        @test observation_logpdf(obs, y, u) ≈ hand
        ys = simulate_observation(Observation(g), fill(0.3, 200, 100); seed=9)
        @test abs(sum(ys) / 20_000 - 0.3) < 4 * 0.2 / sqrt(20_000)
        @test abs(sqrt(sum(abs2, ys .- 0.3) / 20_000) - 0.2) < 0.01
        @test_throws ArgumentError GaussianError(0)
        @test_throws ArgumentError Observation(:not_a_link)
    end
end
