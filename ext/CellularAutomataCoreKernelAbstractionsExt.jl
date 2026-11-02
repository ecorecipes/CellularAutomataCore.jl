# GPUBackend: the halo-padded synchronous engine of ArrayBackend, as KernelAbstractions
# kernels, so one implementation runs on CUDA, ROCm, Metal, oneAPI, JLArrays and the CPU.
module CellularAutomataCoreKernelAbstractionsExt

using Adapt: Adapt
using CellularAutomataCore
using CellularAutomataCore: _Env, _GatedScheme, _HaloState, _gates
using KernelAbstractions
using KernelAbstractions: @index, @kernel

const CAC = CellularAutomataCore

# Move host arrays (inside rules, fields, covariates and states) to the device.
struct _ToDevice{D}
    device::D
end
function Adapt.adapt_storage(to::_ToDevice, a::Array)
    d = KernelAbstractions.allocate(to.device, eltype(a), size(a))
    copyto!(d, a)
    return d
end
Adapt.adapt_storage(to::_ToDevice, a::BitArray) = Adapt.adapt_storage(to, Array(a))
_to_device(device, x) = Adapt.adapt(_ToDevice(device), x)

function CAC.GPUBackend(a::AbstractArray; workgroupsize::Integer=256)
    return GPUBackend(KernelAbstractions.get_backend(a), Int(workgroupsize))
end

mutable struct _DeviceState{U,N,F,R,C,M,Fl}
    cur::U
    next::U
    halo::Int
    dims::NTuple{N,Int}
    shift::CartesianIndex{N}
    fieldvals::F
    rule::R
    covariates::C
    mask::M
    fields::Fl
end

function CAC.check_problem(b::GPUBackend, prob, scheme)
    prob.space isa Lattice ||
        throw(IncompatibleSpaceError("GPUBackend", CAC._describe(prob.space),
                                     "needs a lattice"))
    any(f -> f.fft, values(prob.fields)) &&
        throw(ArgumentError("GPUBackend computes kernel fields directly; use method=:direct"))
    return nothing
end

function CAC.allocate_state(b::GPUBackend{<:KernelAbstractions.Backend}, prob, scheme)
    host = CAC.allocate_state(ArrayBackend(false), prob, scheme)::_HaloState
    dev = b.device
    mask = prob.mask === nothing ? nothing : _to_device(dev, Array{Bool}(prob.mask))
    return _DeviceState(_to_device(dev, host.cur), _to_device(dev, host.next), host.halo,
                        host.dims, host.shift, _to_device(dev, host.fieldvals),
                        _to_device(dev, prob.rule), _to_device(dev, prob.covariates), mask,
                        _to_device(dev, prob.fields))
end

@inline function _outside(K::CartesianIndex{N}, dims) where {N}
    return !all(ntuple(d -> 1 <= K[d] <= dims[d], Val(N)))
end

# KernelAbstractions forbids `return` in a kernel, and the formatter's yas style adds one.
#! format: off
@kernel function _halo_kernel!(cur, space, dims, shift)
    P = @index(Global, Cartesian)
    K = P - shift
    if _outside(K, dims)
        CAC._write!(cur, P, CAC._boundary_read_padded(space, cur, K, shift))
    end
end

@kernel function _field_kernel!(out, cur, rf, shift)
    I = @index(Global, Cartesian)
    @inbounds out[I] = CAC._field_at(cur, rf, I + shift)
end

@kernel function _sweep_kernel!(next, cur, rule, offs, shift, lin, env, t, gate)
    I = @index(Global, Cartesian)
    CAC._update_cell!(next, cur, rule, offs, shift, lin, env, t, gate, I)
end
#! format: on

function _fill_halo!(b::GPUBackend, st::_DeviceState, space)
    kernel! = _halo_kernel!(b.device, b.workgroupsize)
    kernel!(st.cur, space, st.dims, st.shift; ndrange=st.dims .+ 2 * st.halo)
    return nothing
end

function CAC.advance!(b::GPUBackend, st::_DeviceState, prob, scheme::_GatedScheme, p, t)
    _fill_halo!(b, st, prob.space)
    field! = _field_kernel!(b.device, b.workgroupsize)
    foreach(values(st.fields), values(st.fieldvals)) do rf, out
        return field!(out, st.cur, rf, st.shift; ndrange=st.dims)
    end
    env = _Env(_to_device(b.device, p), st.covariates, st.fieldvals, st.mask, prob.seed)
    sweep! = _sweep_kernel!(b.device, b.workgroupsize)
    lin = LinearIndices(st.dims)
    for (k, gate) in enumerate(_gates(scheme))
        k > 1 && _fill_halo!(b, st, prob.space)
        sweep!(st.next, st.cur, st.rule, prob.neighbours.offsets, st.shift, lin, env, t,
               gate;
               ndrange=st.dims)
        st.cur, st.next = st.next, st.cur
    end
    _synchronize(b.device)
    return nothing
end

# Not every KernelAbstractions backend defines `synchronize` (JLArrays' does not).
function _synchronize(device)
    hasmethod(KernelAbstractions.synchronize, Tuple{typeof(device)}) &&
        KernelAbstractions.synchronize(device)
    return nothing
end

_host(a::AbstractArray) = Array(a)
_host(u::NamedTuple) = map(Array, u)

function _interior(st::_DeviceState{U,N}) where {U,N}
    return ntuple(i -> (st.halo + 1):(st.halo + st.dims[i]), N)
end

function CAC.current_state(::GPUBackend, st::_DeviceState)
    h = _host(st.cur)
    inner = _interior(st)
    return h isa AbstractArray ? h[inner...] : map(a -> a[inner...], h)
end

function CAC.set_state!(b::GPUBackend, st::_DeviceState, u)
    h = _host(st.cur)
    for I in CartesianIndices(st.dims)
        CAC._write!(h, I + st.shift, CAC._read(u, I))
    end
    st.cur isa AbstractArray ? copyto!(st.cur, h) :
    foreach(copyto!, values(st.cur), values(h))
    return nothing
end

CAC.read_cell(b::GPUBackend, st::_DeviceState, I) = CAC._read(CAC.current_state(b, st), I)
function CAC.write_cell!(b::GPUBackend, st::_DeviceState, I, v)
    u = CAC.current_state(b, st)
    CAC._write!(u, I, CAC._convert_cell(u, v))
    CAC.set_state!(b, st, u)
    return nothing
end

end # module
