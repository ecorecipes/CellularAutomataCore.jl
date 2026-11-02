# Kernel fields by FFT on periodic lattices: the field at I is the cross-correlation
# sum_o w_o src[I + o], computed as irfft(conj(rfft(k)) .* rfft(src)) with the kernel wrapped
# onto the lattice.
module CellularAutomataCoreFFTWExt

using CellularAutomataCore
using CellularAutomataCore: _ResolvedField
using FFTW: irfft, plan_rfft

const CAC = CellularAutomataCore

struct _FFTCache{P,K}
    plan::P
    kernel_fft::K
end

function CAC._fft_cache(rf::_ResolvedField, dims::Tuple)
    kpad = zeros(eltype(rf.weights), dims)
    for (o, w) in zip(rf.offsets, rf.weights)
        kpad[CartesianIndex(mod1.(Tuple(o) .+ 1, dims))] += w
    end
    plan = plan_rfft(kpad)
    return _FFTCache(plan, conj.(plan * kpad))
end

function CAC._compute_field_fft!(out::AbstractArray{<:AbstractFloat}, rf::_ResolvedField,
                                 src::Array{<:AbstractFloat}, cache::_FFTCache)
    out .= irfft((cache.plan * src) .* cache.kernel_fft, size(src, 1))
    return out
end

end # module
