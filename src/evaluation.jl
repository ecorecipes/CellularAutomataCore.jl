# Comparing simulated and observed maps, as used to validate and calibrate spatial models:
# agreement (Cohen's kappa), discrimination (ROC AUC) and agreement on change (figure of merit).

function _same_size(a, b, what)
    size(a) == size(b) ||
        throw(ArgumentError("$what: arrays have sizes $(size(a)) and $(size(b))"))
    return nothing
end

"""
    confusion_counts(predicted, observed) -> NamedTuple

The counts `(tp, fp, fn, tn)` of true and false positives and negatives between
two `Bool` maps of the same size (or maps converted with a predicate first, as
in `confusion_counts(sim .== :I, obs .== :I)`).
"""
function confusion_counts(predicted::AbstractArray{Bool}, observed::AbstractArray{Bool})
    _same_size(predicted, observed, "confusion_counts")
    tp = count(predicted .& observed)
    fp = count(predicted .& .!observed)
    fn = count(.!predicted .& observed)
    return (tp=tp, fp=fp, fn=fn, tn=length(observed) - tp - fp - fn)
end

"""
    cohens_kappa(predicted, observed) -> Float64

Cohen's kappa [Cohen1960](@cite) between two categorical maps of the same
size: the agreement `p_o` corrected for the agreement `p_e` expected by chance
from the category frequencies, `(p_o - p_e) / (1 - p_e)`. 1 is perfect
agreement, 0 is no better than chance. When both maps hold a single, identical
category, it is 1.

```jldoctest
julia> cohens_kappa([1 1; 0 0], [1 1; 0 0])
1.0

julia> cohens_kappa([1 0; 1 0], [1 1; 0 0])
0.0
```
"""
function cohens_kappa(predicted::AbstractArray, observed::AbstractArray)
    _same_size(predicted, observed, "cohens_kappa")
    n = length(observed)
    n > 0 || throw(ArgumentError("cohens_kappa: empty maps"))
    po = count(predicted .== observed) / n
    cp, co = state_counts(predicted), state_counts(observed)
    pe = sum(get(cp, k, 0) * v for (k, v) in co; init=0) / n^2
    pe == 1 && return po == 1 ? 1.0 : 0.0
    return (po - pe) / (1 - pe)
end

"""
    auc(scores, observed) -> Float64

The area under the ROC curve of real-valued `scores` (for example the fraction
of replicates in which each cell was infested) against a `Bool` map of observed
presence: the probability that a randomly chosen present cell scores higher than
a randomly chosen absent one, ties counting one half (the Mann–Whitney
statistic).

```jldoctest
julia> auc([0.9, 0.8, 0.3, 0.1], [true, true, false, false])
1.0

julia> auc([0.5, 0.5, 0.5, 0.5], [true, false, true, false])
0.5
```
"""
function auc(scores::AbstractArray{<:Real}, observed::AbstractArray{Bool})
    _same_size(scores, observed, "auc")
    npos = count(observed)
    nneg = length(observed) - npos
    (npos > 0 && nneg > 0) ||
        throw(ArgumentError("auc needs both present and absent cells in the observed map"))
    s = vec(scores)
    order = sortperm(s)
    ranks = similar(s, Float64)
    i = 1
    while i <= length(order)
        j = i
        while j < length(order) && s[order[j + 1]] == s[order[i]]
            j += 1
        end
        r = (i + j) / 2  # average rank of a run of ties
        for k in i:j
            ranks[order[k]] = r
        end
        i = j + 1
    end
    rpos = sum(ranks[vec(observed)])
    return (rpos - npos * (npos + 1) / 2) / (npos * nneg)
end

"""
    figure_of_merit(initial, predicted, observed) -> Float64

Agreement on change between an initial map and predicted and observed later
maps: the number of cells that changed in both, to the same state, divided by
the number that changed in either (the figure of merit of land-change models
[Pontius2008](@cite)). 1 means every change was predicted and no other; 0 means
no overlap.
"""
function figure_of_merit(initial::AbstractArray, predicted::AbstractArray,
                         observed::AbstractArray)
    _same_size(initial, predicted, "figure_of_merit")
    _same_size(initial, observed, "figure_of_merit")
    cp = predicted .!= initial
    co = observed .!= initial
    either = count(cp .| co)
    either == 0 && return 1.0
    both = count(cp .& co .& (predicted .== observed))
    return both / either
end
