# Render the quarto vignettes to HTML, GitHub-flavoured Markdown and PDF, running each
# vignette's code once.
#
#   julia scripts/render_vignettes.jl                 # every vignette that is out of date
#   julia scripts/render_vignettes.jl 04_foo 05_bar   # only these
#   julia scripts/render_vignettes.jl --force         # run the code again regardless
#   julia scripts/render_vignettes.jl --dry-run       # say what would run
#
# Quarto's `freeze` keeps one result per format, so `quarto render` runs the code once for
# each, and it looks only at the .qmd, so a change to a package leaves the old results in
# place. Here the code runs once, and the markdown it leaves, straight after execution and before
# any of quarto's filters (`keep-md`; citations and cross-references still unresolved), is kept in
# `vignettes/_exec_cache`, not committed. Every format is rendered from that, so all of them show
# the same run and the same figures. A cached result is reused only if its key matches: the
# files in the vignette's directory and top-level helpers, the vignette environment's
# Manifest, the Julia version, and the source of each package the vignette names, with the
# packages those depend on through `[sources]`. So a change to a package runs its vignettes
# again.
using SHA
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const VIG = joinpath(ROOT, "vignettes")
const CACHE = joinpath(VIG, "_exec_cache")
const FIGURES = "figure-commonmark"  # where the committed Markdown has always kept its figures
const RETRIES = 3
const PIPELINE = "keep-md 1"  # part of the key: change it when the cached form changes

function is_output(v, f)
    return f in (v * ".md", v * ".html", v * ".pdf", v * ".gfm.md", v * "_exec.qmd",
                 v * "_exec.md")
end

function vignette_names()
    return sort(filter(readdir(VIG)) do d
                    return occursin(r"^\d+_", d) && isfile(joinpath(VIG, d, d * ".qmd"))
                end)
end

function hash_file!(ctx, label, path)
    SHA.update!(ctx, Vector{UInt8}(label * "\0"))
    SHA.update!(ctx, read(path))
    return nothing
end

function hash_tree!(ctx, label, dir)
    isdir(dir) || return nothing
    files = String[]
    for (root, _, fs) in walkdir(dir), f in fs
        push!(files, joinpath(root, f))
    end
    for p in sort(files)
        hash_file!(ctx, joinpath(label, relpath(p, dir)), p)
    end
    return nothing
end

# The packages in the vignette environment's `[sources]`, as name => directory.
function source_packages()
    project = TOML.parsefile(joinpath(VIG, "Project.toml"))
    return Dict(name => normpath(joinpath(VIG, spec["path"]))
                for (name, spec) in get(project, "sources", Dict())
                if spec isa AbstractDict && haskey(spec, "path"))
end

# Those a vignette needs: the ones its text names (all of them if it names none, to be safe),
# and what they depend on through `[sources]`.
function packages_for(text, packages)
    wanted = Set(name for name in keys(packages) if occursin(name, text))
    isempty(wanted) && (wanted = Set(keys(packages)))
    grew = true
    while grew
        grew = false
        for name in collect(wanted)
            file = joinpath(packages[name], "Project.toml")
            isfile(file) || continue
            for dep in keys(get(TOML.parsefile(file), "deps", Dict()))
                if haskey(packages, dep) && !(dep in wanted)
                    push!(wanted, dep)
                    grew = true
                end
            end
        end
    end
    return sort(collect(wanted))
end

function execution_key(v, packages)
    dir = joinpath(VIG, v)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, Vector{UInt8}(PIPELINE * string(VERSION)))
    manifest = joinpath(VIG, "Manifest.toml")
    isfile(manifest) && hash_file!(ctx, "Manifest.toml", manifest)
    for f in sort(readdir(dir))
        p = joinpath(dir, f)
        (isfile(p) && !is_output(v, f)) && hash_file!(ctx, f, p)
    end
    for f in sort(readdir(VIG))
        endswith(f, ".jl") && hash_file!(ctx, f, joinpath(VIG, f))
    end
    for name in packages_for(read(joinpath(dir, v * ".qmd"), String), packages)
        pkg = packages[name]
        hash_file!(ctx, joinpath(name, "Project.toml"), joinpath(pkg, "Project.toml"))
        for sub in ("src", "ext", "models")
            hash_tree!(ctx, joinpath(name, sub), joinpath(pkg, sub))
        end
    end
    return bytes2hex(SHA.digest!(ctx))
end

# The quarto call, retried: the Julia server can time out when the machine is busy.
function quarto(args...)
    log = tempname()
    for attempt in 1:RETRIES
        ok = success(pipeline(`quarto render $(collect(args))`; stdout=log, stderr=log))
        ok && return true
        attempt < RETRIES && println("    attempt $attempt failed, retrying")
    end
    println(read(log, String))
    return false
end

# The executed markdown as a document with nothing left to run: a .qmd, so that it keeps the
# project's format options, which a plain .md outside the project's render list would lose.
function as_document(markdown, v)
    m = match(r"\A---\n(.*?)\n---\n"s, markdown)
    m === nothing && error("$v: the executed markdown has no front matter")
    front = replace(m.captures[1] * "\n", r"^engine:.*\n"m => "")
    body = markdown[(m.offset + ncodeunits(m.match)):end]
    return "---\nengine: markdown\n" * front * "---\n" * body
end

function run_code(v, key)
    println("  running the code")
    dir = joinpath(VIG, v)
    # from the vignette's directory: quarto writes `--output` relative to where it is run
    ok = cd(dir) do
        return quarto("$v.qmd", "--to", "gfm", "--output", "$(v)_exec.md", "-M",
                      "keep-md:true")
    end
    ok || return false
    entry = joinpath(CACHE, v, key)
    mkpath(entry)
    kept = joinpath(dir, "$v.gfm.md")
    write(joinpath(entry, "document.qmd"), as_document(read(kept, String), v))
    rm(kept)
    rm(joinpath(dir, "$(v)_exec.md"); force=true)
    figures = joinpath(dir, "$(v)_files", FIGURES)
    isdir(figures) && mv(figures, joinpath(entry, "figures"); force=true)
    return true
end

function render_key(entry)
    ctx = SHA.SHA256_CTX()
    hash_file!(ctx, "document.qmd", joinpath(entry, "document.qmd"))
    hash_file!(ctx, "_quarto.yml", joinpath(VIG, "_quarto.yml"))
    bib = joinpath(VIG, "references.bib")
    isfile(bib) && hash_file!(ctx, "references.bib", bib)
    return bytes2hex(SHA.digest!(ctx))
end

function render_formats(v, entry)
    dir = joinpath(VIG, v)
    files = joinpath(dir, "$(v)_files")
    figures = joinpath(files, FIGURES)
    rm(figures; force=true, recursive=true)
    if isdir(joinpath(entry, "figures"))
        mkpath(files)
        cp(joinpath(entry, "figures"), figures)
    end
    input = joinpath(dir, "$(v)_exec.qmd")
    cp(joinpath(entry, "document.qmd"), input; force=true)
    try
        for (format, ext) in (("html", "html"), ("pdf", "pdf"), ("gfm", "md"))
            println("  ", format)
            ok = cd(() -> quarto("$(v)_exec.qmd", "--to", format, "--output", "$v.$ext"),
                    dir)
            ok || return false
        end
    finally
        rm(input; force=true)
        rm(joinpath(dir, "$(v)_exec_files"); force=true, recursive=true)
    end
    # quarto leaves converted figures beside the originals when it makes the PDF
    isdir(figures) && foreach(readdir(figures)) do f
        return endswith(f, ".pdf") && rm(joinpath(figures, f))
    end
    isdir(files) && isempty(readdir(files)) && rm(files)
    return true
end

function main(args)
    force = "--force" in args
    dry = "--dry-run" in args
    unknown = filter(a -> startswith(a, "--") && !(a in ("--force", "--dry-run")), args)
    isempty(unknown) || error("unknown options: ", join(unknown, ", "))
    asked = filter(a -> !startswith(a, "--"), args)
    names = vignette_names()
    for a in asked
        a in names || error("no vignette $a; vignettes are ", join(names, ", "))
    end
    isempty(asked) || (names = asked)
    packages = source_packages()
    failed = String[]
    for v in names
        key = execution_key(v, packages)
        entry = joinpath(CACHE, v, key)
        cached = !force && isfile(joinpath(entry, "document.qmd"))
        stamp = joinpath(CACHE, v, "rendered")
        outputs = all(ext -> isfile(joinpath(VIG, v, "$v.$ext")), ("md", "html", "pdf"))
        current = cached && outputs && isfile(stamp) &&
                  read(stamp, String) == render_key(entry)
        if current
            println(v, ": up to date")
            continue
        end
        println(v, ": ", cached ? "rendering from the cached run" : "out of date")
        dry && continue
        if !cached && !run_code(v, key)
            push!(failed, v)
            continue
        end
        if !render_formats(v, entry)
            push!(failed, v)
            continue
        end
        write(stamp, render_key(entry))
        # only the current run is worth keeping
        for old in readdir(joinpath(CACHE, v))
            old in (key, "rendered") || rm(joinpath(CACHE, v, old); recursive=true)
        end
    end
    if !isempty(failed)
        println("failed: ", join(failed, ", "))
        exit(1)
    end
    return nothing
end

main(ARGS)
