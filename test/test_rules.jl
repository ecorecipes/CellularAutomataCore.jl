# Rule families, each against an independent naive implementation written from the
# definition, plus classic patterns.

# Naive one step of a Wolfram-numbered rule on a ring: digit `idx` of `code` in base k.
function naive_wolfram(code, k, l, r, u)
    n = length(u)
    map(1:n) do i
        idx = 0
        for j in (i - l):(i + r)
            idx = idx * k + u[mod1(j, n)]
        end
        return div(big(code), big(k)^idx) % k
    end
end

function naive_totalistic(code, k, r, u)
    n = length(u)
    return map(i -> div(big(code), big(k)^sum(u[mod1(j, n)] for j in (i - r):(i + r))) % k,
               1:n)
end

# Naive Generations on a torus with the Moore(1) neighbourhood; C = 2 is Life.
function naive_generations(birth, survive, C, u)
    m, n = size(u)
    map(CartesianIndices(u)) do I
        live = count(u[mod1(I[1] + a, m), mod1(I[2] + b, n)] == 1
                     for a in -1:1, b in -1:1 if (a, b) != (0, 0))
        s = u[I]
        return s == 0 ? (live in birth ? 1 : 0) :
               s == 1 ? (live in survive ? 1 : (C > 2 ? 2 : 0)) : (s + 1 < C ? s + 1 : 0)
    end
end

step1(rule, u; kw...) = final_state(solve(CAProblem(rule, u, 1; kw...)))

@testset "rules" begin
    @testset "Wolfram codes against the definition" begin
        for (code, k, l, r) in [(30, 2, 1, 1), (110, 2, 1, 1), (90, 2, 1, 1),
                                (2_000_000_000, 2, 2, 2), (7_625_597_484_986, 3, 1, 1),
                                (12345, 2, 2, 1), (3, 2, 0, 1)]
            rule = Wolfram(code; states=k, radius=(l, r))
            for seed in 1:3
                u = Int8.(rand(Random.Xoshiro(seed), 0:(k - 1), 23))
                @test step1(rule, u) == naive_wolfram(code, k, l, r, u)
            end
        end
        @test_throws ArgumentError Wolfram(256)
        @test_throws ArgumentError Wolfram(-1)
        @test_throws ArgumentError Wolfram(1; states=1)
        @test_throws ArgumentError Wolfram(1; radius=(0, 0))
        @test sprint(show, Wolfram(30)) == "Wolfram(30)"
        @test sprint(show, Wolfram(5; states=3, radius=(1, 2))) ==
              "Wolfram(5; states=3, radius=(1, 2))"
    end

    @testset "rule 90 from one seed is Pascal's triangle mod 2" begin
        n, T = 129, 40
        sol = solve(CAProblem(Wolfram(90), single_seed(n), T))
        c = cld(n, 2)
        for t in 0:T, x in (-t):t
            expected = iseven(t + x) ? isodd(binomial(big(t), (t + x) ÷ 2)) : false
            @test sol.u[t + 1][c + x] == expected
        end
    end

    @testset "totalistic codes against the definition" begin
        for (code, k, r) in [(912, 3, 1), (1599, 3, 1), (10, 2, 1), (52, 2, 2)]
            rule = Totalistic(code; states=k, radius=r)
            u = Int8.(rand(Random.Xoshiro(code), 0:(k - 1), 31))
            @test step1(rule, u) == naive_totalistic(code, k, r, u)
        end
        @test_throws ArgumentError Totalistic(3^7; states=3)
    end

    @testset "Life and Generations against the definition" begin
        for (rule, b, s, C) in
            [(Life(), [3], [2, 3], 2), (Life(:highlife), [3, 6], [2, 3], 2),
             (Life(:seeds), [2], Int[], 2),
             (Generations(:brians_brain), [2], Int[], 3),
             (Generations(:star_wars), [2], [3, 4, 5], 4),
             (Generations("B3/S23/C2"), [3], [2, 3], 2)]
            for seed in 1:3
                u = Int8.(rand(Random.Xoshiro(seed), 0:(C - 1), 12, 15))
                @test step1(rule, u) == naive_generations(b, s, C, u)
            end
        end
    end

    @testset "rule strings" begin
        @test rulestring(Life("B36/S23")) == "B36/S23"
        @test rulestring(Life("23/36")) == "B36/S23"
        @test rulestring(Life("b3s23")) == "B3/S23"
        @test rulestring(Generations("345/2/4")) == "B2/S345/C4"
        @test rulestring(Generations("B2/S/3")) == "B2/S/C3"
        @test rulestring(Life(:life_without_death)) == "B3/S012345678"
        @test all(p -> rulestring(Life(p)) == LIFE_PRESETS[p], keys(LIFE_PRESETS))
        @test all(p -> rulestring(Generations(p)) == GENERATIONS_PRESETS[p],
                  keys(GENERATIONS_PRESETS))
        @test rulestring(Life(34:45, 34:58)) == "B34..45/S34..58"
        @test_throws RuleParseError Life("B9/S23")
        @test_throws RuleParseError Life("hello")
        @test_throws RuleParseError Life("B3/S23/C3")
        @test_throws RuleParseError Generations("B3/S23")
        @test_throws ArgumentError Life(:nonsense)
        @test_throws ArgumentError Generations(:nonsense)
        @test_throws ArgumentError Life([64], [2])
    end

    @testset "classic Life patterns" begin
        blinker = zeros(Int8, 5, 5)
        blinker[3, 2:4] .= 1
        sol = solve(CAProblem(Life(), blinker, 2))
        @test sol.u[2] == permutedims(blinker)
        @test sol.u[3] == blinker
        block = zeros(Int8, 4, 4)
        block[2:3, 2:3] .= 1
        @test step1(Life(), block) == block
        glider, _ = parse_rle("x = 3, y = 3\nbo\$2bo\$3o!")
        u = place!(zeros(Int8, 10, 10), glider, (2, 2))
        @test final_state(solve(CAProblem(Life(), u, 4))) == circshift(u, (1, 1))
        @test final_state(solve(CAProblem(Life(), u, 40))) == circshift(u, (10, 10))
    end

    @testset "Life on a hexagonal grid and on a graph" begin
        u = zeros(Int8, 6, 6)
        u[3, 3] = u[4, 3] = 1
        # B2 on Hexagonal(1): the cells adjacent to both live cells are born
        v = step1(Life("B2/S"), u; space=HexGrid(6, 6))
        @test Set(Tuple.(findall(==(1), v))) == Set([(3, 4), (4, 2)])
        ring = GraphSpace(6, [(i, mod1(i + 1, 6)) for i in 1:6])
        w = step1(Life("B1/S"), Int8[1, 0, 0, 0, 0, 0]; space=ring)
        @test w == Int8[0, 1, 0, 0, 0, 1]
    end

    @testset "validation of initial states" begin
        @test_throws InvalidStateError CAProblem(Life(), Int8[0 2; 1 0], 1)
        @test_throws InvalidStateError CAProblem(Wolfram(30), Int8[0, 1, 3], 1)
        @test_throws InvalidStateError CAProblem(Generations(:brians_brain), Int8[0, 3, 1],
                                                 1)
        @test_throws InvalidStateError CAProblem(Life(), (a=zeros(Int8, 3, 3),), 1)
        r = LocalRule((c, n, ctx) -> c; states=(:S, :I))
        @test_throws InvalidStateError CAProblem(r, [:S :I; :R :S], 1)
        @test CAProblem(r, [:S :I; :I :S], 1) isa CAProblem
    end
end
