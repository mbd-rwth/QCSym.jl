"""
    AbstractStateBackend

A simulator that [`evolve!`](@ref) runs the summands of a [`LoweredState`](@ref) on. A backend
implements `_run_gates!(ψ, backend, n, gates, matrices)`: apply `gates` (a vector of
[`GateRef`](@ref)s, in application order) to the `n`-qubit state `ψ` in place, each with
the matrix `matrices[gate.slot]`.

A backend that keeps state between parameter sets, e.g. compiled programs, returns it from
`_backend_buffers(backend, lowered)` and implements `_run_summand!(ψ, backend, lowered, j, ws)`
instead, which runs summand `j` with the buffers `ws.backend`. Every task of `evolve!` makes
its own buffers and reuses them for all its parameter sets.
"""
abstract type AbstractStateBackend end

"""
    ReferenceBackend()

Applies the gates one by one with QCSym's own kernel (`QCSym._apply_gate!`): no
dependencies, no fusion, no specialised kernels. Meant for tests and small registers, and as
the reference the other backends are checked against.
"""
struct ReferenceBackend <: AbstractStateBackend end

"""
    LENSBackend(; fuse=true, atol=0.0, threads=1)

Simulates with LENS.jl. Every task of [`evolve!`](@ref) compiles a `LENS.MutableProgram` per
summand for its first parameter set (passing `fuse` and `atol` on) and recompiles it in place
for the next ones, so that these allocate nothing; it `simulate!`s with `threads` kernel
threads. It is implemented in the package extension `QCSymLENSExt`, which `using LENS` loads.
"""
Base.@kwdef struct LENSBackend <: AbstractStateBackend
    fuse::Bool = true
    atol::Float64 = 0.0
    threads::Int = 1
end

function _run_gates!(ψ, backend::AbstractStateBackend, n, gates, matrices)
    backend isa LENSBackend &&
        error("LENSBackend needs LENS.jl: run `using LENS`, which loads QCSym's LENS extension")
    error("$(typeof(backend)) does not implement QCSym.Simulation._run_gates!")
end

function _run_gates!(ψ, ::ReferenceBackend, n, gates, matrices)
    for g in gates
        QCSym._apply_gate!(ψ, matrices[g.slot], g.qubits, n)
    end
    return ψ
end

# Runs summand `j` of `ls` on ψ: by default `_run_gates!`, see AbstractStateBackend.
_run_summand!(ψ, backend::AbstractStateBackend, ls::LoweredState, j::Int, ws) =
    _run_gates!(ψ, backend, ls.n, ls.summands[j].gates, ws.matrices)

_backend_buffers(::AbstractStateBackend, ::LoweredState) = nothing

# Buffers of one task: the slot matrices (constant ones shared with the LoweredState,
# parametric ones overwritten per parameter set), the coefficients, for trees with more
# than one summand the initial state, a work state and an accumulator, the entries of the
# parametric slots as the in-place `matrix_values` writes them, and the backend's own.
struct _Workspace{B}
    matrices::Vector{Matrix{ComplexF64}}
    coeffs::Vector{ComplexF64}
    input::Vector{ComplexF64}
    work::Vector{ComplexF64}
    acc::Vector{ComplexF64}
    values::Vector{ComplexF64}
    backend::B
end

_runs_in_place(ls::LoweredState) = length(ls.summands) == 1 && ls.summands[1].coeff == 0

function _Workspace(ls::LoweredState, backend::AbstractStateBackend=ReferenceBackend())
    matrices = [A === nothing ? zeros(ComplexF64, d, d) : A
                for (A, d) in zip(ls.const_matrices, ls.slot_dims)]
    N = _runs_in_place(ls) ? 0 : 1 << ls.n
    input = _reads_input(ls) ? zeros(ComplexF64, N) : ComplexF64[]
    return _Workspace(matrices, zeros(ComplexF64, ls.ncoeffs), input, zeros(ComplexF64, N),
                      zeros(ComplexF64, N), zeros(ComplexF64, max(ls.nvalues, 0)),
                      _backend_buffers(backend, ls))
end

function _fill_values!(ws::_Workspace, ls::LoweredState, p)
    if ls.matrix_values !== nothing && ls.nvalues >= 0
        ls.matrix_values(ws.values, p)
        offset = 0
        for slot in ls.param_slots
            A = ws.matrices[slot]
            copyto!(A, 1, ws.values, offset + 1, length(A))
            offset += length(A)
        end
    elseif ls.matrix_values !== nothing
        for (A, slot) in zip(ls.matrix_values(p), ls.param_slots)
            size(A) == size(ws.matrices[slot]) || throw(DimensionMismatch(
                "a gate matrix evaluated to size $(size(A)), its gate needs $(size(ws.matrices[slot]))"))
            copyto!(ws.matrices[slot], A)
        end
    end
    ls.coeff_values === nothing || ls.coeff_values(ws.coeffs, p)
    return ws
end

function _set_basis!(ψ, k::QCSym.BasisKet)
    fill!(ψ, 0)
    ψ[k.index + 1] = 1
    return ψ
end

function _evolve_one!(ψ, ls::LoweredState, p, ws::_Workspace, backend)
    _fill_values!(ws, ls, p)
    if _runs_in_place(ls)
        s = only(ls.summands)
        s.start == 0 || _set_basis!(ψ, ls.kets[s.start])
        return _run_summand!(ψ, backend, ls, 1, ws)
    end
    # every summand starts from the same state, so keep the input before ψ is overwritten
    _reads_input(ls) && copyto!(ws.input, ψ)
    fill!(ws.acc, 0)
    for (j, s) in enumerate(ls.summands)
        s.start == 0 ? copyto!(ws.work, ws.input) : _set_basis!(ws.work, ls.kets[s.start])
        _run_summand!(ws.work, backend, ls, j, ws)
        LinearAlgebra.axpy!(s.coeff == 0 ? one(ComplexF64) : ws.coeffs[s.coeff], ws.work, ws.acc)
    end
    return copyto!(ψ, ws.acc)
end

"""
    evolve!(ψ, lowered::LoweredState, p; backend=ReferenceBackend()) -> ψ
    evolve!(Ψ, lowered::LoweredState, P; backend=ReferenceBackend(), ntasks=Threads.nthreads()) -> Ψ

Evolve the state `ψ` in place under `lowered` for the parameter vector `p` (ordered like the
`params` given to [`lower_state`](@ref)): `ψ` holds the initial state on entry and the final
state on return. A tree that already contains its initial state, e.g. one built with
`Circuits.gcol2tree_chain_ket0`, does not read `ψ` and only writes the result into it.

For a matrix `Ψ`, column `k` is evolved with the parameter set `P[:, k]`, so every column may
start from its own state. The columns are split over `ntasks` tasks, each simulating its
columns one after another with its own buffers.

A tree with several summands, e.g. a derivative, runs one simulation per summand from the
same initial state and sums the results. `backend` chooses the simulator: `ReferenceBackend()`
or, after `using LENS`, `LENSBackend()`.
"""
function evolve!(ψ::AbstractVector{ComplexF64}, ls::LoweredState, p::AbstractVector;
                 backend::AbstractStateBackend=ReferenceBackend())
    length(ψ) == 1 << ls.n || throw(DimensionMismatch(
        "a state on $(ls.n) qubits has length $(1 << ls.n), got $(length(ψ))"))
    return _evolve_one!(ψ, ls, p, _Workspace(ls, backend), backend)
end

function evolve!(Ψ::AbstractMatrix{ComplexF64}, ls::LoweredState, P::AbstractMatrix;
                 backend::AbstractStateBackend=ReferenceBackend(),
                 ntasks::Integer=Threads.nthreads())
    size(Ψ, 1) == 1 << ls.n || throw(DimensionMismatch(
        "states on $(ls.n) qubits have length $(1 << ls.n), got $(size(Ψ, 1))"))
    size(Ψ, 2) == size(P, 2) || throw(DimensionMismatch(
        "$(size(Ψ, 2)) states but $(size(P, 2)) parameter sets"))
    ncols = size(Ψ, 2)
    ncols == 0 && return Ψ
    chunks = Iterators.partition(1:ncols, cld(ncols, clamp(ntasks, 1, ncols)))
    tasks = map(chunks) do cols
        Threads.@spawn begin
            ws = _Workspace(ls, backend)
            for k in cols
                _evolve_one!(view(Ψ, :, k), ls, view(P, :, k), ws, backend)
            end
        end
    end
    foreach(fetch, tasks)
    return Ψ
end

"""
    evolve(ψ0, lowered::LoweredState, p_or_P; kwargs...)
    evolve(lowered::LoweredState, p_or_P; kwargs...)

Like [`evolve!`](@ref), on a copy: the state(s) evolved from `ψ0` for the parameter vector
`p`, or one column per parameter set of the matrix `P`, all starting from `ψ0`. Without
`ψ0`, for trees that contain their initial state (`Circuits.gcol2tree_chain_ket0`).
"""
evolve(ψ0::AbstractVector, ls::LoweredState, p::AbstractVector; kwargs...) =
    evolve!(Vector{ComplexF64}(ψ0), ls, p; kwargs...)

evolve(ψ0::AbstractVector, ls::LoweredState, P::AbstractMatrix; kwargs...) =
    evolve!(repeat(Vector{ComplexF64}(ψ0), 1, size(P, 2)), ls, P; kwargs...)

function evolve(ls::LoweredState, p_or_P::AbstractVecOrMat; kwargs...)
    _reads_input(ls) && throw(ArgumentError(
        "this tree is applied to an initial state; pass one: evolve(ψ0, lowered, p)"))
    N = 1 << ls.n
    out = p_or_P isa AbstractMatrix ? zeros(ComplexF64, N, size(p_or_P, 2)) : zeros(ComplexF64, N)
    return evolve!(out, ls, p_or_P; kwargs...)
end
