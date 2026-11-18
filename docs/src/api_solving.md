# Solving

Problems, integrators, solutions, callbacks, ensembles and update schemes.

## Problems, integrators and solutions

The interface follows SciML: `CAProblem(rule, u0, tspan, p)`, `solve(prob, alg;
saveat, callback, ...)`, `init`/`step!`/`solve!`, integrators with `u`, `t` and
`p`, and SciMLBase's `remake`, `reinit!`, `terminate!`, `u_modified!`,
`NullParameters`, `ReturnCode`, `DiscreteCallback`, `CallbackSet`,
`EnsembleProblem`, `EnsembleThreads`, `EnsembleSerial` and
`EnsembleDistributed`, re-exported by this package. The update scheme is the
algorithm.

The integrator also has SciML's timed `step!(integ, dt, true)`, `set_u!`,
`check_error` and `derivative_discontinuity!` (SciMLBase 3's name for
`u_modified!`), and solutions answer `successful_retcode`. That is the interface
the simulation-based resilience metrics of
[DynamicalResilience.jl](https://github.com/ecorecipes/DynamicalResilience.jl)
drive (`recovery`, `basin_stability`, early warnings), so they apply to any
`CAProblem` without this package depending on them; [`disturb!`](@ref) is a
ready-made pulse.

```@docs
CAProblem
remake(::CAProblem)
init(::CAProblem)
solve(::CAProblem)
CAIntegrator
step!(::CAIntegrator, ::Real, ::Bool)
u_modified!(::CAIntegrator, ::Bool)
CellularAutomataCore.SciMLBase.set_u!(::CAIntegrator, ::Any)
CellularAutomataCore.SciMLBase.check_error(::CAIntegrator)
terminate!(::CAIntegrator)
reinit!(::CAIntegrator)
CellularAutomataCore.SciMLBase.add_tstop!(::CAIntegrator, ::Any)
CASolution
CellularAutomataCore.SciMLBase.successful_retcode(::CASolution)
final_state
trajectory
```

## Callbacks

```@docs
PeriodicCallback
PresetTimeCallback
SavingCallback
SavedValues
```

## Ensembles

```@docs
EnsembleProblem(::CAProblem)
seed_trajectory
```

## Update schemes

```@docs
AbstractScheme
Synchronous
Stochastic
Checkerboard
RandomSequential
Margolus
Gillespie
```
