# Conformance on an Apple GPU through Metal. Not part of `Pkg.test()` (CI has no Apple GPU);
# run on a Mac with Metal.jl available, for example:
#
#   julia --project=<env with Metal, KernelAbstractions, Random, Test and this package> test/metal.jl
#
# Metal has no Float64, so the problems use Float32 parameters, `Stochastic(0.5f0)` and
# `precision=Float32` fields. Discrete models must match ReferenceBackend exactly; Float32 Lenia
# agrees to rounding, because Metal's `exp` and fused multiply-adds differ from the CPU's.
using CellularAutomataCore, KernelAbstractions, Metal, Random, Test

Metal.functional() || error("Metal is not functional on this machine")
const DEV = GPUBackend(MetalBackend())
function exact(prob, alg=Synchronous())
    return solve(prob, alg; backend=ReferenceBackend(), saveat=5).u ==
           solve(prob, alg; backend=DEV, saveat=5).u
end

@testset "Metal" begin
    rng = Xoshiro(1)
    @test exact(CAProblem(Life(), Int8.(rand(rng, 0:1, 64, 48)), 20))
    @test exact(CAProblem(Wolfram(110), Int8.(rand(rng, 0:1, 200)), 30))
    @test exact(CAProblem(Life("B5678/S45678"; neighbourhood=VonNeumann(1)),
                          Int8.(rand(rng, 0:1, 40, 40)), 10), Checkerboard())
    sir = LocalRule(VonNeumann(1)) do c, n, ctx
        c == Int8(0) &&
            return bernoulli(ctx, 1.0f0 - (1.0f0 - ctx.p.β)^count(==(Int8(1)), n), 1) ?
                   Int8(1) : Int8(0)
        c == Int8(1) && return bernoulli(ctx, ctx.p.γ, 2) ? Int8(2) : Int8(1)
        return c
    end
    u = zeros(Int8, 64, 64)
    u[32, 32] = 1
    p = CAProblem(sir, u, 40, (β=0.4f0, γ=0.2f0); seed=3)
    @test exact(p)
    @test exact(p, Stochastic(0.5f0))
    layered = LocalRule(Moore(1)) do c, n, ctx
        pr = sum(x.pest for x in n) / 8.0f0
        return (pest=0.5f0 * c.pest + 0.5f0 * pr * c.host,
                host=c.host == Int8(1) && bernoulli(ctx, 0.2f0 * pr) ? Int8(0) : c.host)
    end
    lay = (host=Int8.(rand(rng, 0:1, 32, 32)), pest=rand(rng, Float32, 32, 32))
    @test exact(CAProblem(layered, lay, 15; seed=5,
                          space=Grid(32, 32; boundary=Constant((host=0, pest=0.0f0)))))
    si = LocalRule() do c, n, ctx
        return c == Int8(1) ? c :
               (bernoulli(ctx, 1.0f0 - exp(-ctx.p.β * field(ctx, :P))) ? Int8(1) : c)
    end
    f32 = (P=KernelField(AnisotropicKernel(GaussianKernel(2.0; radius=6), 0.5, 1.0),
                         ==(Int8(1));
                         precision=Float32),)
    @test exact(CAProblem(si, u, 25, (β=1.5f0,); fields=f32, seed=2))
    o = place!(zeros(Float32, 64, 64), Float32.(ORBIUM), (20, 20))
    lp = CAProblem(Lenia(; precision=Float32), o, 50)
    a = final_state(solve(lp; backend=ReferenceBackend()))
    g = final_state(solve(lp; backend=DEV))
    @test maximum(abs.(a .- g)) < 1.0f-3
end
