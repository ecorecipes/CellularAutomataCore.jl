ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Documenter
using DocumenterCitations
using CellularAutomataCore

DocMeta.setdocmeta!(CellularAutomataCore, :DocTestSetup, :(using CellularAutomataCore);
                    recursive=true)

# Tutorials are rendered quarto vignettes copied into docs/src/tutorials by
# scripts/sync_vignettes.jl. The page list is built from the files on disk so
# it never has to be maintained by hand.
function tutorial_pages()
    dir = joinpath(@__DIR__, "src", "tutorials")
    isdir(dir) || return Pair{String,String}[]
    files = sort(filter(f -> endswith(f, ".md"), readdir(dir)))
    map(files) do f
        title = f
        for line in eachline(joinpath(dir, f))
            m = match(r"^#\s+(.*)", line)
            if m !== nothing
                title = String(strip(m.captures[1]))
                break
            end
        end
        return title => "tutorials/" * f
    end
end

pages = Any["Home" => "index.md",
            "API Reference" => ["Overview and exceptions" => "api.md",
                                "Spaces, rules and states" => "api_model.md",
                                "Solving" => "api_solving.md",
                                "Backends" => "api_backends.md",
                                "Analysis" => "api_analysis.md"],
            "References" => "references.md"]
tutorials = tutorial_pages()
isempty(tutorials) || push!(pages, "Tutorials" => tutorials)

# Source links need a commit to point at. A checkout without commits (a fresh
# scaffold, a tarball) builds without them instead of failing.
has_commit = success(pipeline(`git -C $(@__DIR__) rev-parse HEAD`; stderr=devnull))
remote_kw = has_commit ? (; repo=Remotes.GitHub("ecorecipes", "CellularAutomataCore.jl")) :
            (; remotes=nothing)

# Bibliography for `[Key](@cite)` citations in docstrings and pages. The .bib is
# synced from the workspace-level docs/references.bib by scripts/sync_references.jl.
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib");
                           style=:authoryear)

makedocs(;
         remote_kw...,
         modules=[CellularAutomataCore],
         sitename="CellularAutomataCore.jl",
         authors="Simon Frost",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://ecorecipes.github.io/CellularAutomataCore.jl",
                                repolink="https://github.com/ecorecipes/CellularAutomataCore.jl",
                                edit_link="main"),
         pages=pages,
         plugins=[bib])

"--no-deploy" in ARGS || deploydocs(;
                                    repo="github.com/ecorecipes/CellularAutomataCore.jl.git",
                                    devbranch="main")
