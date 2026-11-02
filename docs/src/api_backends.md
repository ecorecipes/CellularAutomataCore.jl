# Backends

The engines that advance a problem, and the interface for writing one.

## Backends

```@docs
AbstractBackend
ArrayBackend
SparseBackend
GPUBackend
ReferenceBackend
supports
stored_tiles
GraphNeighbourStates
```

## Agent backends

Hybrid models: the state lives in an Agents.jl model or an Ark.jl world shared
with mobile agents.

```@docs
AgentsBackend
ArkBackend
CACell
agent_model
read_cell
write_cell!
```

## Writing a backend

A backend is a subtype of [`AbstractBackend`](@ref) with methods of
[`supports`](@ref), [`read_cell`](@ref), [`write_cell!`](@ref) and the
functions below, which are public but not exported. A backend that runs its own
engine can evaluate the rule at a cell with
[`CellularAutomataCore.update_cell`](@ref).

```@docs
CellularAutomataCore.check_problem
CellularAutomataCore.allocate_state
CellularAutomataCore.advance!
CellularAutomataCore.current_state
CellularAutomataCore.set_state!
CellularAutomataCore.update_cell
```
