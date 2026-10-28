# Gate-level terms: one factor per gate.
#
# gcol2tree2 spells every step as a ⊗ over all qubit lines and a two-qubit gate as the
# two-term sum of its 2x2 decomposition, so the tree carries n factors per step and needs a
# decomposition for every multi-qubit gate. A GateOn term instead keeps "matrix M acting on
# these qubits" as one node: the tree grows with the number of gates, gates of any size work,
# and it is exactly what a state-vector simulator consumes (see QCSym.Simulation).

"""
    GateOn(n, qubits)

The operator on an `n`-qubit register that applies a `2^k x 2^k` matrix to `qubits`
(`k = length(qubits)`) and acts as the identity on every other qubit. Calling it builds the
term, `GateOn(n, qubits)(M)`, the same way `Differential(x)(f)` does.

`qubits` are global qubit indices in the kron order of `M`: `qubits[1]` is the most
significant bit of `M`'s row and column index, and qubit 1 is the most significant bit of an
amplitude index. That is the placement of `Circuits.assemble_unitary` and of LENS.jl.

The placement lives in the operation and `M` is the only argument, so `substitute`,
`get_variables` and [`kron_derivative`](@ref) only ever see `M`: the product rule rebuilds
`GateOn(n, qubits)(dM)` without knowing about qubits. Applied to a numeric `M` it returns a
lazy [`GateStep`](@ref) instead of a term.
"""
struct GateOn{K}
    n::Int
    qubits::NTuple{K, Int}
    function GateOn(n::Integer, qubits)
        qs = Tuple(Int(q) for q in qubits)
        isempty(qs) && throw(ArgumentError("a gate needs at least one qubit"))
        all(q -> 1 <= q <= n, qs) ||
            throw(ArgumentError("qubits $(qs) are not all in 1:$(n)"))
        allunique(qs) || throw(ArgumentError("qubits must be distinct, got $(qs)"))
        return new{length(qs)}(Int(n), qs)
    end
end

Base.show(io::IO, g::GateOn) = print(io, "GateOn[", join(g.qubits, ","), "]")

# Shape is nominal, as for ⊙: QCSym's matrix-valued terms do not track their true size.
(g::GateOn)(M) = SymbolicUtils.term(g, Symbolics.unwrap(M); type=SymbolicUtils.SymReal,
                                    shape=(1:2, 1:2))

# `Num <: Real`, so `AbstractMatrix{<:Number}` would also catch a matrix of symbolic entries
# and evaluate it eagerly; only plain numbers make a gate numeric.
const _NumericEntry = Union{AbstractFloat, Integer, Complex{<:Union{AbstractFloat, Integer}}}

(g::GateOn)(M::AbstractMatrix{<:_NumericEntry}) = GateStep(g, M)

"""
    GateStep(g::GateOn, M)

A numeric [`GateOn`](@ref): the matrix `M` on `g.qubits` of a `g.n`-qubit register, kept
unexpanded like the other [`LazyStep`](@ref)s. Applying it to a state costs
`O(2^n * 2^k)` and never builds the `2^n x 2^n` embedding; indexing it computes single
entries of that embedding.
"""
struct GateStep{T, MT<:AbstractMatrix} <: LazyStep{T}
    n::Int
    qubits::Vector{Int}
    matrix::MT
end

function GateStep(g::GateOn, M::AbstractMatrix)
    D = 1 << length(g.qubits)
    size(M) == (D, D) || throw(DimensionMismatch(
        "a gate on $(length(g.qubits)) qubit(s) needs a $(D)x$(D) matrix, got $(join(size(M), 'x'))"))
    return GateStep{promote_type(eltype(M), ComplexF64), typeof(M)}(g.n, collect(g.qubits), M)
end

Base.size(s::GateStep) = (1 << s.n, 1 << s.n)

function Base.getindex(s::GateStep{T}, i::Int, j::Int) where {T}
    r, c = i - 1, j - 1
    mask = _gate_bits(s.n, s.qubits)
    # the embedding is the identity on all other qubits
    (r & ~mask) == (c & ~mask) || return zero(T)
    return T(s.matrix[_gate_index(r, s.n, s.qubits) + 1, _gate_index(c, s.n, s.qubits) + 1])
end

# Bit of the amplitude index that holds qubit q (qubit 1 is the most significant).
_qubit_bit(n, q) = 1 << (n - q)
_gate_bits(n, qubits) = mapreduce(q -> _qubit_bit(n, q), |, qubits)

# Row/column of the gate matrix that the amplitude index `idx` falls on.
function _gate_index(idx, n, qubits)
    k = length(qubits)
    s = 0
    for (j, q) in enumerate(qubits)
        s |= ((idx >> (n - q)) & 1) << (k - j)
    end
    return s
end

_apply(a::GateStep, v::AbstractVector) =
    _apply_gate!(Vector{promote_type(eltype(a), eltype(v))}(v), a.matrix, a.qubits, a.n)

"""
    _apply_gate!(ψ, M, qubits, n) -> ψ

Apply the `2^k x 2^k` matrix `M` to `qubits` of the `n`-qubit state `ψ` in place, with the
placement convention of [`GateOn`](@ref). A plain reference kernel: every group of `2^k`
amplitudes that differ only in the gate's qubits is gathered, multiplied by `M` and written
back.
"""
function _apply_gate!(ψ::AbstractVector, M::AbstractMatrix, qubits, n::Integer)
    k = length(qubits)
    D = 1 << k
    length(ψ) == 1 << n || throw(DimensionMismatch(
        "a state on $(n) qubits has length $(1 << n), got $(length(ψ))"))
    size(M) == (D, D) || throw(DimensionMismatch(
        "a gate on $(k) qubit(s) needs a $(D)x$(D) matrix, got $(join(size(M), 'x'))"))
    # where the gate's basis state s sits relative to the first amplitude of its group
    offsets = Vector{Int}(undef, D)
    for s in 0:D-1
        o = 0
        for (j, q) in enumerate(qubits)
            ((s >> (k - j)) & 1) == 1 && (o |= _qubit_bit(n, q))
        end
        offsets[s+1] = o
    end
    mask = offsets[D]
    tmp = Vector{eltype(ψ)}(undef, D)
    @inbounds for base in 0:length(ψ)-1
        (base & mask) == 0 || continue
        for s in 1:D
            tmp[s] = ψ[base + offsets[s] + 1]
        end
        for r in 1:D
            acc = zero(eltype(ψ))
            for s in 1:D
                acc += M[r, s] * tmp[s]
            end
            ψ[base + offsets[r] + 1] = acc
        end
    end
    return ψ
end

# Operator products without a state in between: an operator-valued chain such as
# gcol2tree_chain's unitary. Only small registers ever evaluate these densely; applied to a
# state, a chain goes through ⊙(::LazyStep, ::AbstractVector) instead.
_dense(x) = x isa LazyStep ? collect(x) : x
⊙(a::GateStep, b::GateStep) = _dense(a) * _dense(b)
⊙(a::GateStep, b::AbstractMatrix{<:Number}) = _dense(a) * _dense(b)
⊙(a::AbstractMatrix{<:Number}, b::GateStep) = _dense(a) * _dense(b)

"""
    BasisKet(n, index=0)

The computational basis state with 0-based amplitude `index` on `n` qubits (qubit 1 most
significant). [`ket0`](@ref) wraps one into a tree leaf. It is deliberately not an
`AbstractVector`: hash-consing the leaf compares two integers instead of `2^n` amplitudes.
"""
struct BasisKet
    n::Int
    index::Int
    function BasisKet(n::Integer, index::Integer=0)
        1 <= n <= 62 || throw(ArgumentError("number of qubits must be in 1:62, got $(n)"))
        0 <= index < (1 << n) ||
            throw(ArgumentError("basis index $(index) out of range for $(n) qubits"))
        return new(Int(n), Int(index))
    end
end

Base.show(io::IO, k::BasisKet) = print(io, "|", string(k.index; base=2, pad=k.n), "⟩")

function Base.collect(k::BasisKet)
    ψ = zeros(ComplexF64, 1 << k.n)
    ψ[k.index + 1] = 1
    return ψ
end

"""
    ket0(n)

`|0…0⟩` on `n` qubits as a tree leaf: a constant holding a [`BasisKet`](@ref). It is what
[`Circuits.gcol2tree_chain_ket0`](@ref) applies the circuit to.
"""
ket0(n::Integer) = SymbolicUtils.Const{SymbolicUtils.SymReal}(BasisKet(n, 0))

# Evaluating `G ⊙ |k⟩`: materialise the ket, then the vector methods take over (a GateStep
# applies itself in place, a dense matrix multiplies).
⊙(a::AbstractMatrix{<:Number}, k::BasisKet) = ⊙(a, collect(k))
