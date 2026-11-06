# Backend conformance. Every backend must reproduce the ReferenceBackend exactly, saved state
# by saved state, on a battery of problems covering every space, boundary, neighbourhood,
# state layout, rule family, update scheme and environment feature, deterministic and
# stochastic. Each case lists its scheme, the background state that makes it quiescent (for
# SparseBackend; `nothing` if it is not) and whether its states can live on a device. A new
# backend is added to `backends()` below and runs every case it supports.

struct HostPest
    host::Int8
    pest::Float32
end

struct Case
    name::String
    prob::Any
    alg::AbstractScheme
    background::Any
    device::Bool
end
function Case(name, prob; alg=Synchronous(), background=nothing, device=true)
    return Case(name, prob, alg,
                background, device)
end

function conformance_cases()
    rng = Random.Xoshiro(2026)
    r01(dims...) = Int8.(rand(rng, 0:1, dims...))
    sir = LocalRule(VonNeumann(1)) do cell, nbrs, ctx
        if cell == :S
            k = count(==(:I), nbrs)
            return bernoulli(ctx, 1 - (1 - ctx.p.β)^k, 1) ? :I : :S
        elseif cell == :I
            return bernoulli(ctx, ctx.p.γ, 2) ? :R : :I
        end
        return cell
    end
    sir0 = fill(:S, 24, 24)
    sir0[12, 12] = :I
    layered = LocalRule(Moore(1)) do cell, nbrs, ctx
        pressure = sum(n.pest for n in nbrs) / length(nbrs)
        host = cell.host == 1 && bernoulli(ctx, 0.2 * pressure) ? Int8(0) : cell.host
        return (pest=Float32(0.5 * cell.pest + 0.5 * pressure * cell.host), host=host)
    end
    lay0 = (host=r01(16, 16), pest=Float32.(rand(rng, 16, 16)))
    structrule = LocalRule(VonNeumann(1)) do cell, nbrs, ctx
        return HostPest(cell.host,
                        Float32(maximum(n.pest for n in nbrs) * cell.host * 0.9 +
                                0.1 * uniform(ctx)))
    end
    st0 = [HostPest(Int8(rand(rng, 0:1)), rand(rng, Float32)) for _ in 1:12, _ in 1:12]
    graph = GraphSpace(30, [(i, mod1(i + k, 30)) for i in 1:30 for k in (1, 7)])
    sis = LocalRule() do cell, nbrs, ctx
        return cell == 1 ? (bernoulli(ctx, 0.3) ? Int8(0) : Int8(1)) :
               (bernoulli(ctx, 0.25 * count(==(1), nbrs)) ? Int8(1) : Int8(0))
    end
    # an SI epidemic driven by a dispersal-kernel field and a covariate
    si = LocalRule() do cell, nbrs, ctx
        cell == 1 && return cell
        λ = ctx.p.β * covariate(ctx, :suitability) * field(ctx, :pressure)
        return bernoulli(ctx, 1 - exp(-λ)) ? one(cell) : cell
    end
    si0 = zeros(Int8, 30, 26)
    si0[15, 13] = 1
    suit = rand(rng, 30, 26, 3)
    # (named so that no rule closure above captures it: a closure assigning to an outer
    # local shares one variable across all cells, a data race under parallel backends)
    windfield = (pressure=KernelField(AnisotropicKernel(ExponentialKernel(1.5), π / 4, 1.0),
                                      x -> x == 1),)
    mask = trues(30, 26)
    mask[5:10, 5:20] .= false
    rotate = BlockRule((b, ctx) -> bernoulli(ctx, 0.5) ? (b[3], b[1], b[4], b[2]) : b)
    # single precision throughout, as on Apple GPUs (Metal has no Float64)
    sir32 = LocalRule(VonNeumann(1)) do c, n, ctx
        c == Int8(0) &&
            return bernoulli(ctx, 1.0f0 - (1.0f0 - ctx.p.β)^count(==(Int8(1)), n), 1) ?
                   Int8(1) : Int8(0)
        c == Int8(1) && return bernoulli(ctx, ctx.p.γ, 2) ? Int8(2) : Int8(1)
        return c
    end
    sir32_0 = zeros(Int8, 24, 24)
    sir32_0[12, 12] = 1
    si32 = LocalRule() do c, n, ctx
        c == 1 && return c
        return bernoulli(ctx, 1.0f0 - exp(-ctx.p.β * field(ctx, :P))) ? one(c) : c
    end
    field32 = (P=KernelField(AnisotropicKernel(GaussianKernel(1.5), 1.0, 1.0), ==(1);
                             precision=Float32),)
    orbium32 = place!(zeros(Float32, 40, 40), Float32.(ORBIUM), (10, 10))
    wolfram3 = Wolfram(7_000_000_001; states=3)
    return [Case("elementary rule 30 on a ring",
                 CAProblem(Wolfram(30), single_seed(31), 20);
                 background=Int8(0)),
            Case("3-state Wolfram, reflecting line",
                 CAProblem(wolfram3, Int8.(rand(rng, 0:2, 40)), 15;
                           space=Grid(40; boundary=Reflecting()))),
            Case("asymmetric Wolfram, constant edges",
                 CAProblem(Wolfram(12345; radius=(2, 1)), r01(33), 12;
                           space=Grid(33; boundary=Constant(1)))),
            Case("totalistic radius 2", CAProblem(Totalistic(52; radius=2), r01(33), 12)),
            Case("Life on a torus", CAProblem(Life(), r01(20, 24), 30); background=Int8(0)),
            Case("Life on a cylinder",
                 CAProblem(Life(), r01(20, 24), 30;
                           space=Grid(20, 24; boundary=(Periodic(), Constant(0))));
                 background=Int8(0)),
            Case("radius-2 von Neumann, mixed edges",
                 CAProblem(Life("B3/S2345"; neighbourhood=VonNeumann(2)), r01(15, 17), 20;
                           space=Grid(15, 17; boundary=(Reflecting(), Constant(1))));
                 background=Int8(0)),
            Case("Brian's Brain",
                 CAProblem(Generations(:brians_brain), Int8.(rand(rng, 0:2, 18, 18)), 25);
                 background=Int8(0)),
            Case("hexagonal Life B2/S34",
                 CAProblem(Life("B2/S34"), r01(14, 16), 20; space=HexGrid(14, 16));
                 background=Int8(0)),
            Case("3-D Life-like", CAProblem(Life("B5/S45"), r01(8, 9, 10), 6);
                 background=Int8(0)),
            Case("offsets wider than the torus",
                 CAProblem(Life("B1/S1"; neighbourhood=Offsets((2, 1), (-3, 0), (0, 4))),
                           r01(3, 7), 8)),
            Case("stochastic SIR with Symbol states",
                 CAProblem(sir, sir0, 30, (β=0.3, γ=0.1); seed=11); background=:S,
                 device=false),
            Case("stochastic layered host-pest",
                 CAProblem(layered, lay0, 15; seed=5,
                           space=Grid(16, 16; boundary=Constant((host=0, pest=0.0f0))))),
            Case("stochastic layered host-pest, one layer marked at the edge",
                 CAProblem(layered, lay0, 15; seed=5,
                           space=Grid(16, 16; boundary=Constant((; pest=1.0f0))))),
            Case("struct states with noise",
                 CAProblem(structrule, st0, 10; seed=8,
                           space=Grid(12, 12; boundary=Reflecting()))),
            Case("SIS on a graph",
                 CAProblem(sis, Int8.(rand(rng, 0:1, 30)), 25; space=graph, seed=3);
                 device=false),
            Case("Stochastic(0.4) Life", CAProblem(Life(), r01(20, 20), 15; seed=2);
                 alg=Stochastic(0.4), background=Int8(0)),
            Case("Checkerboard majority",
                 CAProblem(Life("B5678/S45678"; neighbourhood=VonNeumann(1)), r01(18, 18),
                           10);
                 alg=Checkerboard()),
            Case("RandomSequential SIR", CAProblem(sir, sir0, 15, (β=0.3, γ=0.1); seed=4);
                 alg=RandomSequential(), device=false),
            Case("Margolus stochastic rotation", CAProblem(rotate, r01(16, 12), 12; seed=6);
                 alg=Margolus()),
            Case("kernel field, cycling covariate and mask",
                 CAProblem(si, si0, 12, (β=4.0,); seed=9, fields=windfield,
                           covariates=(suitability=suit,), mask=mask,
                           space=Grid(30, 26; boundary=(Reflecting(), Constant(0))))),
            Case("Float32 SIR with Stochastic(0.5f0)",
                 CAProblem(sir32, sir32_0, 25, (β=0.4f0, γ=0.2f0); seed=12);
                 alg=Stochastic(0.5f0), background=Int8(0)),
            Case("Float32 kernel field",
                 CAProblem(si32, si0, 12, (β=2.0f0,); seed=13, fields=field32)),
            Case("Float32 Lenia", CAProblem(Lenia(; precision=Float32), orbium32, 6))]
end

function backends()
    return Any["ArrayBackend (serial)" => (c -> ArrayBackend(; threaded=false)),
               "ArrayBackend (threaded)" => (c -> ArrayBackend(; threaded=true)),
               "SparseBackend" => (c -> c.background === nothing ? nothing :
                                        SparseBackend(c.background; tilesize=5)),
               "GPUBackend (KernelAbstractions CPU)" => (c -> c.device ?
                                                              GPUBackend(KernelAbstractions.CPU();
                                                                         workgroupsize=32) :
                                                              nothing),
               "GPUBackend (JLArrays)" => (c -> c.device ?
                                                GPUBackend(JLArrays.JLBackend()) :
                                                nothing),
               "AgentsBackend (:grid)" => (c -> AgentsBackend(:grid)),
               "AgentsBackend (:cells)" => (c -> AgentsBackend(:cells)),
               "ArkBackend (:grid)" => (c -> ArkBackend(:grid)),
               "ArkBackend (:cells)" => (c -> ArkBackend(:cells)),
               "CellLoopBackend (public interface only)" => (c -> CellLoopBackend())]
end

# Whether a backend can run a case: it supports the scheme and accepts the problem.
function runnable(backend, c::Case)
    backend === nothing && return false
    supports(backend, c.alg) || return false
    try
        CellularAutomataCore.check_problem(backend, c.prob, c.alg)
    catch
        return false
    end
    return true
end

@testset "backend conformance" begin
    cases = conformance_cases()
    references = Dict(c.name => solve(c.prob, c.alg; backend=ReferenceBackend())
                      for c in cases)
    # the battery is not vacuous: stochastic problems actually change state
    for name in ("stochastic SIR with Symbol states", "SIS on a graph",
                 "stochastic layered host-pest", "kernel field, cycling covariate and mask")
        @test references[name].u[end] != references[name].u[1]
    end
    for (bname, make) in backends()
        @testset "$bname" begin
            ran = 0
            for c in cases
                backend = make(c)
                runnable(backend, c) || continue
                sol = solve(c.prob, c.alg; backend)
                ref = references[c.name]
                @test sol.t == ref.t
                @test sol.u == ref.u
                ran += 1
            end
            @test ran >= 5  # every backend runs a real share of the battery
        end
    end
end
