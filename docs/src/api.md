# Overview and exceptions

```@docs
CellularAutomataCore
```

The reference is split into [spaces, rules and states](api_model.md), [solving](api_solving.md), [backends](api_backends.md) and [analysis](api_analysis.md).

## Exceptions

Every exception the package defines subtypes the abstract root
[`CellularAutomataCoreError`](@ref). Invalid arguments and keywords raise
Base's `ArgumentError`, which is outside it.

```@docs
CellularAutomataCoreError
RuleParseError
RLEParseError
StateShapeError
InvalidStateError
IncompatibleSpaceError
UnsupportedCombinationError
```
