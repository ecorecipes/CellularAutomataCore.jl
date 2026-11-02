# Run-length-encoded patterns.
@testset "RLE" begin
    @testset "parsing" begin
        p, rule = parse_rle("""
                            #N Glider
                            #C a comment
                            x = 3, y = 3, rule = B3/S23
                            bo\$2bo\$
                            3o!
                            """)
        @test p == Int8[0 1 0; 0 0 1; 1 1 1]
        @test rule == "B3/S23"
        # multi-digit runs, blank-row runs, no rule, trailing text after '!'
        p, rule = parse_rle("x = 12, y = 4\n12o2\$o10bo! trailing")
        @test rule === nothing
        @test p[1, :] == ones(Int8, 12)
        @test p[2, :] == zeros(Int8, 12)
        @test p[3, [1, 12]] == [1, 1] && sum(p[3, :]) == 2
        # multi-state tags
        p, _ = parse_rle("x = 4, y = 1, rule = B2/S/C3\n.A2B!")
        @test p == Int8[0 1 2 2]
    end

    @testset "round trips" begin
        rng = Random.Xoshiro(3)
        for (k, dims) in [(2, (7, 9)), (2, (1, 30)), (3, (5, 5)), (25, (4, 6))]
            p = Int8.(rand(rng, 0:(k - 1), dims...))
            q, _ = parse_rle(rle_string(p))
            # trailing all-zero rows and columns are part of the declared size
            @test q == p
        end
        p = zeros(Int8, 4, 4)
        p[3, 2] = 1
        @test parse_rle(rle_string(p))[1] == p
        @test parse_rle(rle_string(p; rule=Life(:highlife)))[2] == "B36/S23"
        long = collect(Int8.(isodd.(1:200))')
        s = rle_string(long)
        @test all(l -> length(l) <= 70, split(s, '\n'))
        @test parse_rle(s)[1] == long
    end

    @testset "files and IO" begin
        p = Int8[1 0; 0 1]
        mktempdir() do dir
            path = joinpath(dir, "p.rle")
            write_rle(path, p; rule="B3/S23")
            @test read_rle(path) == (p, "B3/S23")
            @test read_rle(IOBuffer(read(path, String))) == (p, "B3/S23")
        end
    end

    @testset "errors" begin
        @test_throws RLEParseError parse_rle("")
        @test_throws RLEParseError parse_rle("#C only a comment")
        @test_throws RLEParseError parse_rle("not a header\nbo!")
        @test_throws RLEParseError parse_rle("x = 2, y = 1\n3o!")
        @test_throws RLEParseError parse_rle("x = 2, y = 1\nbz!")
        @test_throws RLEParseError parse_rle("x = 2, y = 1\nbo")
        @test_throws ArgumentError rle_string([0 30])
    end
end
