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

```@docs
CAProblem
remake(::CAProblem)
init(::CAProblem)
solve(::CAProblem)
CAIntegrator
u_modified!(::CAIntegrator, ::Bool)
terminate!(::CAIntegrator)
reinit!(::CAIntegrator)
CellularAutomataCore.SciMLBase.add_tstop!(::CAIntegrator, ::Any)
CASolution
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
