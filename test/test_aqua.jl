# Package hygiene: method ambiguities, unbound type parameters, undefined exports, stale
# dependencies, compat bounds and type piracy.
using Aqua

@testset "Aqua" begin
    Aqua.test_all(CellularAutomataCore)
end
