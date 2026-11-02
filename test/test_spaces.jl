# Spaces, boundary conditions and neighbourhoods.
const CAC = CellularAutomataCore

@testset "spaces" begin
    @testset "constructors" begin
        t = Torus(4, 5)
        @test size(t) == (4, 5) && length(t) == 20 && ndims(t) == 2
        @test all(b -> b isa Periodic, boundaries(t))
        @test geometry(t) isa Square
        @test cells(t) == CartesianIndices((4, 5))
        g = Grid(3, 3; boundary=(Periodic(), Constant(0)))
        @test boundaries(g) == (Periodic(), Constant(0))
        @test geometry(HexGrid(3, 4)) isa Hex
        @test_throws ArgumentError Grid(0, 3)
        @test_throws ArgumentError Lattice((3, 3, 3), (Periodic(), Periodic(), Periodic()),
                                           Hex())
        @test_throws ArgumentError Grid(3, 3; boundary=(Periodic(),))
        @test sprint(show, Torus(5, 4)) == "Torus((5, 4))"
    end

    @testset "graph spaces" begin
        g = GraphSpace(4, [(1, 2), (2, 3), (3, 4), (4, 1)])
        @test g.neighbours == [[2, 4], [1, 3], [2, 4], [1, 3]]
        @test length(g) == 4 && cells(g) == 1:4
        d = GraphSpace(3, [(1, 2), (2, 3)]; directed=true)
        @test d.neighbours == [[2], [3], Int[]]
        @test_throws ArgumentError GraphSpace([[2], [5]])
        @test_throws ArgumentError GraphSpace(3, [(1, 4)])
    end

    @testset "boundary mapping" begin
        @test [CAC._map_axis(Periodic(), j, 4)[1] for j in -1:6] == [3, 4, 1, 2, 3, 4, 1, 2]
        @test [CAC._map_axis(Reflecting(), j, 4)[1] for j in -2:7] ==
              [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]
        @test CAC._map_axis(Constant(0), 0, 4) == (0, true)
        @test CAC._map_axis(Constant(0), 2, 4) == (2, false)
        u = reshape(1:12, 3, 4)
        s = Grid(3, 4; boundary=(Reflecting(), Constant(-1)))
        @test CAC._boundary_read(s, u, CartesianIndex(0, 2)) == u[1, 2]
        @test CAC._boundary_read(s, u, CartesianIndex(4, 4)) == u[3, 4]
        @test CAC._boundary_read(s, u, CartesianIndex(2, 5)) == -1
        # several Constant axes: the highest-numbered one wins
        c = Grid(3, 3; boundary=(Constant(7), Constant(9)))
        @test CAC._boundary_read(c, zeros(Int, 3, 3), CartesianIndex(0, 0)) == 9
        @test CAC._boundary_read(c, zeros(Int, 3, 3), CartesianIndex(0, 2)) == 7
        # layered states read NamedTuples; a scalar Constant fills every layer
        L = (a=fill(1, 3, 3), b=fill(2.0, 3, 3))
        @test CAC._boundary_read(Grid(3, 3), L, CartesianIndex(0, 1)) == (a=0, b=0.0)
        @test CAC._boundary_read(Torus(3, 3), L, CartesianIndex(0, 1)) == (a=1, b=2.0)
    end

    @testset "neighbourhoods" begin
        n(nb, s) = length(neighbourhood_offsets(nb, s))
        @test n(Moore(1), Torus(5, 5)) == 8
        @test n(Moore(2), Torus(5, 5)) == 24
        @test n(Moore(1), Torus(5, 5, 5)) == 26
        @test n(Moore(1), Torus(9)) == 2
        @test n(VonNeumann(1), Torus(5, 5)) == 4
        @test n(VonNeumann(2), Torus(5, 5)) == 12
        @test n(VonNeumann(1), Torus(5, 5, 5)) == 6
        @test n(Hexagonal(1), HexGrid(5, 5)) == 6
        @test n(Hexagonal(2), HexGrid(5, 5)) == 18
        @test neighbourhood_offsets(Moore(2), Torus(9)) ==
              Offsets((-2,), (-1,), (1,), (2,))
        hex1 = Set(Tuple.(neighbourhood_offsets(Hexagonal(1), HexGrid(4, 4)).offsets))
        @test hex1 == Set([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)])
        @test default_neighbourhood(Torus(3, 3)) == Moore(1)
        @test default_neighbourhood(HexGrid(3, 3)) == Hexagonal(1)
        @test default_neighbourhood(GraphSpace([[2], [1]])) == GraphNeighbours()
        @test CAC._halo(neighbourhood_offsets(Moore(3), Torus(9, 9))) == 3
        @test_throws ArgumentError Moore(0)
        @test_throws ArgumentError Offsets((0, 0), (1, 0))
        @test collect(Offsets((1, 0), (0, -2))) ==
              [CartesianIndex(1, 0), CartesianIndex(0, -2)]
        @test eltype(Offsets((1,), (2,))) == CartesianIndex{1}
        @test_throws ArgumentError Offsets((1, 0), (1, 0))
        @test_throws IncompatibleSpaceError neighbourhood_offsets(Moore(1), HexGrid(3, 3))
        @test_throws IncompatibleSpaceError neighbourhood_offsets(Hexagonal(1), Torus(3, 3))
        @test_throws IncompatibleSpaceError neighbourhood_offsets(Offsets((1,)),
                                                                  Torus(3, 3))
        @test_throws IncompatibleSpaceError neighbourhood_offsets(Moore(1),
                                                                  GraphSpace([[1]]))
        @test_throws IncompatibleSpaceError neighbourhood_offsets(GraphNeighbours(),
                                                                  Torus(3))
    end

    @testset "states" begin
        @test single_seed(5) == Int8[0, 0, 1, 0, 0]
        @test single_seed(4) == Int8[0, 1, 0, 0]
        @test single_seed(3, 3; value=2, T=Int)[2, 2] == 2
        u = zeros(Int, 5, 5)
        place!(u, [1 2; 3 4])
        @test u[2:3, 2:3] == [1 2; 3 4]
        v = place!(zeros(Int, 5, 5), [1 2; 3 4], (4, 4))
        @test v[4:5, 4:5] == [1 2; 3 4]
        @test_throws ArgumentError place!(zeros(Int, 5, 5), [1 2; 3 4], (5, 5))
        @test_throws ArgumentError place!(zeros(Int, 1, 1), [1 2; 3 4])
    end

    @testset "layered cell reads and writes do not allocate" begin
        # Layers of different element types once went through a zip that allocated per cell.
        u = (a=zeros(Int8, 4, 4), b=falses(4, 4) |> collect, c=zeros(Int16, 4, 4))
        write_then_read(u, I) = (CellularAutomataCore._write!(u, I,
                                                              (c=Int16(3), a=Int8(1),
                                                               b=true));
                                 CellularAutomataCore._read(u, I))
        I = CartesianIndex(2, 3)
        @test write_then_read(u, I) == (a=Int8(1), b=true, c=Int16(3))
        @test (@allocated write_then_read(u, I)) == 0
    end
end
