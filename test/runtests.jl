using Test
using Random
using CellularAutomataCore
using CellularAutomataCore: philox4x32
using Agents: Agents
using Ark: Ark
using FFTW: FFTW
using JLArrays: JLArrays
using KernelAbstractions: KernelAbstractions

@testset "CellularAutomataCore" begin
    include("test_random.jl")
    include("test_spaces.jl")
    include("test_rules.jl")
    include("test_rle.jl")
    include("test_problem.jl")
    include("test_schemes.jl")
    include("test_environment.jl")
    include("test_sciml.jl")
    include("test_model.jl")
    include("test_events.jl")
    include("test_agents.jl")
    include("test_extras.jl")
    include("test_evaluation.jl")
    include("test_calibration.jl")
    include("test_rasters.jl")
    include("test_backend_interface.jl")
    include("backend_conformance.jl")
    include("test_errors.jl")
    include("test_docstrings.jl")
    include("test_aqua.jl")
end
