# Spaces, rules and states

Lattices and their boundaries, neighbourhoods, local rules, counter-based randomness, environmental fields, states and the model layer.

## Spaces and boundaries

```@docs
AbstractSpace
Lattice
Torus
Grid
HexGrid
GraphSpace
cells
boundaries
geometry
AbstractGeometry
Square
Hex
AbstractBoundary
Periodic
Reflecting
Constant
```

## Neighbourhoods

```@docs
AbstractNeighbourhood
Moore
VonNeumann
Hexagonal
Offsets
GraphNeighbours
default_neighbourhood
neighbourhood_offsets
```

## Rules

```@docs
AbstractRule
local_rule
neighbourhood
validate_state
LocalRule
Wolfram
Totalistic
Life
Generations
LIFE_PRESETS
GENERATIONS_PRESETS
rulestring
BlockRule
EventRule
Lenia
ORBIUM
```

## Randomness

```@docs
CellContext
uniform
bernoulli
CellularAutomataCore.philox4x32
```

## Covariates, kernels and fields

```@docs
covariate
field
fields
KernelField
kernel_weights
DispersalKernel
GaussianKernel
ExponentialKernel
PowerLawKernel
UniformKernel
LeniaKernel
AnisotropicKernel
EmpiricalKernel
```

## Landscapes from rasters

With Rasters.jl loaded (and ArchGDAL or NCDatasets to read GeoTIFF or NetCDF
files), a raster gives a space, a mask of the cells that hold data, and
covariates; a state can be written back on the raster's grid.

```@docs
raster_space
raster_mask
raster_covariate
state_raster
```

## States and patterns

```@docs
single_seed
place!
parse_rle
read_rle
rle_string
write_rle
```

## Models

A ModelingToolkit-style layer: declare a model's states, parameters, rule,
fields and covariates once, then build problems by name with
`CAProblem(model, u0map, tspan, pmap; space)`.

```@docs
CAModel
@camodel
StateVariable
Parameter
states
parameters
defaults
CAProblem(::CAModel, ::Any, ::Any)
```
