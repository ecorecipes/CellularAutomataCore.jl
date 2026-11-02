# Contributing to CellularAutomataCore.jl

## Local setup

The ecosystem packages depend on each other through `[sources]` entries that point at
sibling directories (`../OtherPackage.jl`). Clone every package you need side by side:

```sh
mkdir -p ca && cd ca
git clone https://github.com/ecorecipes/CellularAutomataCore.jl
cd CellularAutomataCore.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`Pkg.test()` is the only run that counts as "the suite passes".

## Style

- Julia ≥ 1.12. Every dependency, including stdlibs and test-only extras, has a `[compat]` entry.
- Format with JuliaFormatter using the repository `.JuliaFormatter.toml` (`style = "yas"`).
- `snake_case` for functions and variables, `CamelCase` for types, a leading underscore for internal helpers.
- Public functions have docstrings with a signature line, a one-sentence summary, and an example where practical.
- Exceptions are typed, carry the offending names, live in `src/errors.jl` and subtype
  `CellularAutomataCoreError`; library code never calls a bare `error("...")`. Invalid arguments and keywords raise
  `ArgumentError`.
- Every optimised code path is tested against a slower reference implementation on small models.

## Adding a vignette

1. Create `vignettes/NN_short_name/NN_short_name.qmd` with front matter including `engine: julia` and
   `pdf: default` under `format:` (copy an existing vignette's front matter).
2. Add any new dependencies to `vignettes/Project.toml`.
3. `cd vignettes && quarto render && quarto render --to gfm` renders HTML, GitHub-flavoured Markdown and PDF
   (the second command restores `NN_short_name_files/`, which a multi-format render deletes) (the PDF needs
   `lualatex` and the JuliaMono font in `../fonts/JuliaMono/`; see `vignettes/README.md`). Commit the
   `.qmd`, the rendered `.md`, `.html` and `.pdf`, and `NN_short_name_files/`.
4. `julia scripts/sync_vignettes.jl` and commit `docs/src/tutorials/`.

The `Vignettes` workflow re-renders everything weekly and on demand; it is not a required check because
rendering is slow and depends on a Quarto installation. The `vignette-sync` job in CI is required.

## Pull requests

Branch from `main`, keep PRs focused, and make sure CI is green (tests on Julia 1.12 and latest,
vignette sync check, docs build).
