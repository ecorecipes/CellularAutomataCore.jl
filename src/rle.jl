# Run-length-encoded (RLE) patterns, the format of Golly and the LifeWiki pattern
# collections. A pattern is a matrix indexed [row, column] = [y, x], first row at the top.

"""
    parse_rle(text) -> (pattern, rule)

Parse run-length-encoded pattern text. Returns the pattern as a `Matrix{Int8}`
indexed `[row, column]` (row 1 is the top line of the pattern) and the rule
string from the header, or `nothing` if the header names none. Lines starting
with `#` are comments. Two-state patterns use `b` (dead) and `o` (alive);
multi-state patterns use `.` for state 0 and `A` to `X` for states 1 to 24. A
run count may precede any tag, `\$` ends a row and `!` ends the pattern. Raises
[`RLEParseError`](@ref) on malformed text.

```jldoctest
julia> pattern, rule = parse_rle("x = 3, y = 3, rule = B3/S23\\nbo\\\$2bo\\\$3o!");

julia> pattern
3×3 Matrix{Int8}:
 0  1  0
 0  0  1
 1  1  1

julia> rule
"B3/S23"
```
"""
function parse_rle(text::AbstractString)
    lines = split(text, '\n')
    header = 0
    for (i, line) in enumerate(lines)
        l = strip(line)
        (isempty(l) || startswith(l, '#')) && continue
        header = i
        break
    end
    header == 0 && throw(RLEParseError(0, "no header line \"x = ..., y = ...\""))
    m = match(r"^\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*(?:,\s*rule\s*=\s*([^\s,]+))?",
              lines[header])
    m === nothing &&
        throw(RLEParseError(header, "expected a header \"x = <width>, y = <height>\""))
    width, height = parse(Int, m.captures[1]), parse(Int, m.captures[2])
    rule = m.captures[3] === nothing ? nothing : String(m.captures[3])
    pattern = zeros(Int8, height, width)
    row, col, count = 1, 1, 0
    for i in (header + 1):length(lines)
        line = strip(lines[i])
        startswith(line, '#') && continue
        for ch in line
            if isdigit(ch)
                count = 10count + (ch - '0')
                continue
            end
            n = max(count, 1)
            count = 0
            if ch == '!'
                return pattern, rule
            elseif ch == '$'
                row += n
                col = 1
            elseif isspace(ch)
                continue
            else
                state = ch == 'b' || ch == '.' ? 0 :
                        ch == 'o' ? 1 :
                        'A' <= ch <= 'X' ? Int(ch - 'A') + 1 :
                        throw(RLEParseError(i, "unknown cell tag '$ch'"))
                if state != 0
                    (row <= height && col + n - 1 <= width) ||
                        throw(RLEParseError(i,
                                            "the pattern extends beyond the declared size $width × $height"))
                    pattern[row, col:(col + n - 1)] .= state
                end
                col += n
            end
        end
    end
    return throw(RLEParseError(0, "the pattern is not terminated by '!'"))
end

"""
    read_rle(io_or_path) -> (pattern, rule)

Read a run-length-encoded pattern from an `IO` or a file; see
[`parse_rle`](@ref).
"""
read_rle(io::IO) = parse_rle(read(io, String))
read_rle(path::AbstractString) = open(read_rle, path)

"""
    rle_string(pattern; rule=nothing) -> String

Run-length-encode a pattern (a matrix of states indexed `[row, column]`), with
`rule` (a string or a [`Life`](@ref)/[`Generations`](@ref) rule) in the header
if given. Patterns with states `0` and `1` only use the two-state tags `b`/`o`;
others use `.` and `A`–`X`. Lines are wrapped at 70 characters.
[`parse_rle`](@ref) reads the result back.
"""
function rle_string(pattern::AbstractMatrix{<:Integer}; rule=nothing)
    maximum(pattern; init=0) <= 24 && minimum(pattern; init=0) >= 0 ||
        throw(ArgumentError("RLE supports states 0 to 24"))
    twostate = all(x -> x == 0 || x == 1, pattern)
    tag(s) = twostate ? (s == 0 ? 'b' : 'o') : (s == 0 ? '.' : Char('A' + s - 1))
    h, w = size(pattern)
    items = String[]
    run(n, c) = push!(items, (n == 1 ? "" : string(n)) * c)
    cursor = 1  # the row the encoder is on; `$` moves it down
    for r in 1:h
        last = findlast(!=(0), view(pattern, r, :))
        last === nothing && continue
        r > cursor && run(r - cursor, '$')
        cursor = r
        c = 1
        while c <= last
            s = pattern[r, c]
            n = 1
            while c + n <= last && pattern[r, c + n] == s
                n += 1
            end
            run(n, tag(s))
            c += n
        end
    end
    push!(items, "!")
    rulestr = rule === nothing ? "" :
              ", rule = " * (rule isa AbstractString ? rule : rulestring(rule))
    io = IOBuffer()
    println(io, "x = ", w, ", y = ", h, rulestr)
    len = 0
    for it in items
        if len + length(it) > 70
            println(io)
            len = 0
        end
        print(io, it)
        len += length(it)
    end
    println(io)
    return String(take!(io))
end

"""
    write_rle(io_or_path, pattern; rule=nothing)

Write a pattern in run-length-encoded form; see [`rle_string`](@ref).
"""
write_rle(io::IO, pattern; rule=nothing) = print(io, rle_string(pattern; rule))
function write_rle(path::AbstractString, pattern; rule=nothing)
    return open(io -> write_rle(io, pattern; rule), path, "w")
end
