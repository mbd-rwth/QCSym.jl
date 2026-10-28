import LinearAlgebra
import StaticArrays
import Kronecker
import Base

function _promote_shape_nD(a::Symbolics.Arr{T}, b::Symbolics.Arr{T2}) where {T, T2}
    sizea = size(a)
    sizeb = size(b)
    dimsa = length(sizea)
    dimsb = length(sizeb)
    dimsout, dimsother, sizesmore = dimsa>=dimsb ? (dimsa, dimsb, sizea) : (dimsb, dimsa, sizeb)

    sout = zeros(Int, dimsout)
    for i in 1:dimsother
        sout[i] = sizea[i] * sizeb[i]
    end
    for i in dimsout-dimsother+1:dimsout
        sout[i] = sizesmore[i]
    end

    return Tuple(sout)
end


⊗(x1, x2) = begin
    return SymbolicUtils.term(⊗, x1, x2; type=SymbolicUtils.SymReal)
end

⊗(x1::Complex{Symbolics.Num}, x2) = begin
    return SymbolicUtils.term(⊗, x1, x2; type=SymbolicUtils.SymReal)
end
⊗(x1, x2::Complex{Symbolics.Num}) = begin
    return SymbolicUtils.term(⊗, x1, x2; type=SymbolicUtils.SymReal)
end
⊗(xs::Vararg{Any}) = begin
    return SymbolicUtils.term(⊗, xs...; type=SymbolicUtils.SymReal)
end

⊗(x1::Complex{Symbolics.Num}, x2::Symbolics.Num) = begin
    return SymbolicUtils.term(⊗, x1, x2; type=SymbolicUtils.SymReal)
end

# LinearAlgebra.kron on StaticArrays.StaticArray operands returns another
# StaticArray, fully unrolled at compile time. That is a big win while the
# operands are small (single-qubit 2x2 factors), but the *output* dimension of
# ⊗ grows with qubit count (2^n), so staying static past a handful of qubits
# would blow up compile time and code size instead of avoiding allocations.
# Widen any static operands back to a regular Array before kron once the
# combined output size would exceed this bound.
const _MAX_STATIC_KRON_DIM = 16

_widen_if_static(x::StaticArrays.StaticArray) = Array(x)
_widen_if_static(x) = x

"""
    LazyStep{T} <: AbstractMatrix{T}

A `2^n x 2^n` step operator kept in factored form. Under `gcol2tree_col_1` with
`distribute=false` the tree is `step_d ⊙ (... ⊙ (step_1 applied to a column))`, so every
step is only ever applied to a `2^n` state vector. Building the step's matrix first costs
`4^n` for nothing: `kron(A_1,...,A_n) * v` is the vec trick, `O(n * 2^n)` time and `O(2^n)`
memory. `Kronecker.jl` implements that trick; what it has no notion of is a lazy *sum*,
which is what a two-qubit gate's `|0><0| ⊗ I + |1><1| ⊗ U` decomposition produces, hence
[`SumStep`](@ref).

These are `AbstractMatrix`, so anything that genuinely wants the dense step still gets it
via `collect`/indexing -- the point is only that nothing builds it by accident.
"""
abstract type LazyStep{T} <: AbstractMatrix{T} end

"""
    KronStep(kp)

One `A_1 ⊗ ... ⊗ A_n` held as a lazy `Kronecker.jl` product. See [`LazyStep`](@ref).
"""
struct KronStep{T, K} <: LazyStep{T}
    kp::K
end
KronStep(kp) = KronStep{eltype(kp), typeof(kp)}(kp)

"""
    SumStep(terms)

A sum of step operators, kept unevaluated so that `(A + B) * v` is `A*v + B*v` -- two vec
tricks and a vector add, rather than two `2^n x 2^n` matrices and a matrix add.

`terms` is a **tuple**, not a vector: it has to be concretely typed or `_apply` infers as
`Any`, every application becomes a dynamic dispatch, and the generated sampling function
boxes each intermediate. A two-qubit gate contributes two summands, so these tuples stay
short; a step holding many two-qubit gates would instead cost compile time.
"""
struct SumStep{T, TT<:Tuple} <: LazyStep{T}
    terms::TT
end
SumStep(terms::Tuple) = SumStep{promote_type(map(eltype, terms)...), typeof(terms)}(terms)
SumStep(terms) = SumStep(Tuple(terms))

Base.size(s::KronStep) = size(s.kp)
Base.size(s::SumStep)  = size(first(s.terms))
Base.getindex(s::KronStep, i::Int, j::Int) = s.kp[i, j]
Base.getindex(s::SumStep, i::Int, j::Int)  = sum(t -> t[i, j], s.terms)
# Materialise through the factors rather than element by element.
Base.collect(s::KronStep) = collect(s.kp)
Base.collect(s::SumStep)  = sum(collect, s.terms)

# Applying a step to a state vector: the whole reason these types exist.
_apply(a::KronStep, v::AbstractVector) = a.kp * v
_apply(a::AbstractMatrix, v::AbstractVector) = a * v
# Recursive over the tuple rather than `sum(generator)`, so the result type is inferred.
_apply(a::SumStep, v::AbstractVector) = _apply_sum(v, a.terms...)
_apply_sum(v, t) = _apply(t, v)
_apply_sum(v, t, rest...) = _apply(t, v) + _apply_sum(v, rest...)

_factor_sizes(k) = map(size, Kronecker.getallfactors(k))

"""
    _LAZY_STEP_MIN_DIM

Output dimension above which `⊗` of square factors returns a [`LazyStep`](@ref) instead of
a dense matrix. Staying factored is asymptotically right -- `O(n*2^n)` per step instead of
`O(4^n)` -- but it carries a constant: the vec trick makes `n` strided passes over the
state, while a dense step is one BLAS call, and BLAS wins while the matrix still fits in
cache. Raise this to force dense evaluation, set it to 0 to always stay factored.
"""
const _LAZY_STEP_MIN_DIM = Ref(16)

_kron_size_capped(xs::AbstractArray{<:Number}...) = begin
    outdim = prod(size(x, 1) for x in xs)
    if outdim > _MAX_STATIC_KRON_DIM
        #return LinearAlgebra.kron(map(_widen_if_static, xs)...)
        # Several matrix factors: keep them factored, so a step applied to a state vector
        # never materialises 2^n x 2^n. Vector factors are the col_1 leaves, whose kron is
        # the 2^n state vector itself and has to be dense.
        if outdim > _LAZY_STEP_MIN_DIM[] && length(xs) > 1 && all(x -> ndims(x) == 2, xs)
            return KronStep(Kronecker.kronecker(xs...))
        end
        out = LinearAlgebra.kron(xs...)
        # Array, not Matrix: under col_1 the factors are 2-element columns and the kron is
        # a 2^n vector, which Matrix{ComplexF64} cannot construct.
        return Array{ComplexF64}(out)
    end
    return LinearAlgebra.kron(xs...)
end

⊗(xs::Vararg{AbstractArray{<:Number}}) = begin
    return _kron_size_capped(xs...)
end

⊗(x1::Number, x2::Number) = begin
    return x1 * x2
end

⊗(x1::AbstractArray{<:Number}, x2::AbstractArray{<:Number}) = begin
    return _kron_size_capped(x1, x2)
end


# I_Gate_Filler places this reserved symbol on every untouched qubit line at every
# step (see Gates.I_Gate_Filler). ⊙ with it is always a structural no-op, so fold it
# away at construction time instead of emitting a dead matrix multiply into the tree.
_is_identity_filler(x) = SymbolicUtils.issym(x) && Symbolics.getname(x) === :I

⊙(x1::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, x2::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}) = begin
    _is_identity_filler(x1) && return x2
    _is_identity_filler(x2) && return x1
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
end

⊙(x1::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, x2::Union{Symbolics.Num, Complex{Symbolics.Num}}) = begin
    _is_identity_filler(x1) && return x2
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
end

⊙(x1::Union{Symbolics.Num, Complex{Symbolics.Num}}, x2::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}) = begin
    _is_identity_filler(x2) && return x1
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
end

⊙(x1::Union{Symbolics.Num, Complex{Symbolics.Num}}, x2::Union{Symbolics.Num, Complex{Symbolics.Num}}) = begin
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
end

# ⊙(xs::Vararg{<:Union{Symbolics.Num, Complex{Symbolics.Num}}}) = begin
#     println("⊙ vararg2 called with ", length(xs), " args")
#     return SymbolicUtils.term(⊙, xs; type=SymbolicUtils.SymReal)
# end

# ⊙(xs::Vararg{SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}) = begin
#     println("⊙ vararg called with ", length(xs), " args")
#     return SymbolicUtils.term(⊙, (xxs for xxs in xs if !_is_identity_filler(xxs))...; type=SymbolicUtils.SymReal)
# end

# ⊙(xs::AbstractMatrix...) = begin
#     return *(xs...)
# end

# First column of a factor. Column 1 of a Kronecker product is the Kronecker
# product of the factors' first columns, and column 1 of a matrix chain only
# needs the first column of its rightmost factor, so wrapping the rightmost
# (earliest step) factor of every per-qubit chain in col_1 turns a tree for the
# whole unitary into a tree for its first column, without ever forming a
# 2^n x 2^n intermediate. See Circuits.gcol2tree_col_1.
col_1(x) = begin
    return SymbolicUtils.term(col_1, x; type=SymbolicUtils.SymReal, shape=(1:2, 1:1))
end

col_1(x::AbstractMatrix{<:Number}) = begin
    return x[:, 1]
end

col_1(x::AbstractVector{<:Number}) = x

col_1(x::Number) = x

⊙(x1::Number, x2::Number) = begin
    return x1 * x2
end

#⊙(x1::AbstractMatrix{<:Number}, x2::AbstractMatrix{<:Number}) = begin
#    return x1 * x2
#end

⊙(x1::Number, x2::AbstractMatrix{<:Number}) = begin
    return x1 * x2
end

# A col_1 factor only ever sits at the right end of a chain, so the remaining
# factors keep multiplying onto a column instead of widening back to a matrix.
⊙(x1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number}) = begin
    return x1 * x2
end

⊙(x1::Number, x2::AbstractVector{<:Number}) = begin
    return x1 * x2
end

⊙(x1::AbstractMatrix{<:Number}, x2::Number) = begin
    return x1 * x2
end

# ⊙(x1::AbstractMatrix{<:Number}, x2::StaticArrays.SMatrix{2, 2, ComplexF64}) = begin
#     return StaticArrays.SMatrix{2, 2, ComplexF64}(x1 * x2)
# end

# A distributed tree (gcol2tree2 with distribute=true) only ever multiplies the 2x2
# factors of one qubit line, so the StaticArray keeps those chains allocation-free. An
# undistributed tree instead multiplies whole step unitaries, which are 2^n x 2^n, so the
# size has to be checked rather than assumed -- forcing MMatrix{2,2} here is what used to
# restrict the runtime to the distributed form.
⊙(x1::AbstractMatrix{<:Number}, x2::AbstractMatrix{<:Number}) = begin
    if size(x1) == (2, 2) && size(x2) == (2, 2)
        return StaticArrays.MMatrix{2, 2, ComplexF64}(x1 * x2)
    end
    return x1 * x2
end

# Chains of three or more factors: _first_column builds n-ary ⊙ nodes, and an
# undistributed step chain is n-ary too. Folding right matches the nesting the binary
# methods produce, so both spellings of the same chain evaluate identically.
⊙(xs::Vararg{AbstractMatrix{<:Number}}) = foldr(⊙, xs)

# --- lazy steps -------------------------------------------------------------------
# A chain under col_1 ends in a state vector, so this is the method that actually runs
# for every step of the circuit; it is what keeps the cost at O(depth * 2^n).
⊙(a::LazyStep, v::AbstractVector{<:Number}) = _apply(a, v)

# Two steps composed without a vector between them: a step holding both single- and
# two-qubit gates builds ⊙(sq_step, mq_step). Every factor here is 2x2, so the mixed
# product rule kron(A,B)*kron(C,D) = kron(A*C, B*D) applies and the result stays factored.
# The size check is what Kronecker.jl's own `*` would otherwise throw on.
function ⊙(a::KronStep, b::KronStep)
    if _factor_sizes(a.kp) == _factor_sizes(b.kp)
        return KronStep(a.kp * b.kp)
    end
    return collect(a) * collect(b)
end
⊙(a::KronStep, b::SumStep) = SumStep(map(t -> ⊙(a, t), b.terms))
⊙(a::SumStep, b::KronStep) = SumStep(map(t -> ⊙(t, b), a.terms))
⊙(a::SumStep, b::SumStep)  = SumStep(_cross(a.terms, b.terms))

_flatten_tuple(t::Tuple) = reduce((x, y) -> (x..., y...), t; init=())
_cross(as::Tuple, bs::Tuple) = _flatten_tuple(map(x -> map(y -> ⊙(x, y), bs), as))

# A two-qubit gate's two ⊗ branches meet here. Base's `+(::AbstractArray...)` broadcasts,
# which would materialise both operands, so LazyStep needs its own -- including the
# vararg form, or a three-way sum would fall back to Base and densify.
Base.:+(a::LazyStep, b::LazyStep) = SumStep((a, b))
Base.:+(a::LazyStep, b::LazyStep, cs::LazyStep...) = SumStep((a, b, cs...))
Base.:+(a::LazyStep, b::AbstractMatrix) = SumStep((a, b))
Base.:+(a::AbstractMatrix, b::LazyStep) = SumStep((a, b))

#const one_2x2_mmatrix_complex = StaticArrays.MMatrix{2, 2, ComplexF64}([1 0; 0 1])

#⊙(xs::Vararg{AbstractMatrix{<:Number}}) = begin
# ⊙(xs::Tuple{<:Union{AbstractMatrix{<:Number}, Symbolics.Num, Complex{Symbolics.Num}, SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}}) = begin
#     #return *(xs...)
#     if size(xs[1]) == (2, 2)
#         out = one_2x2_mmatrix_complex
#     else
#         out = one(Matrix{ComplexF64}, size(xs[1], 1), size(xs[1], 2))
#     end
#     for i in eachindex(xs)
#         out = LinearAlgebra.mul!(out, out, xs[i])
#     end
    
#     return out
# end

# ⊙(xs::Vararg{<:Union{AbstractMatrix{<:Number}, Symbolics.Num, Complex{Symbolics.Num}, SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}}) = begin
#     return SymbolicUtils.term(⊙, xs...; type=SymbolicUtils.SymReal)
# end

#Symbolics.@register_symbolic ⊙(x1, x2) false

# ⊙(xs::Vararg{Any}) = begin
#     return SymbolicUtils.term(⊙, xs...; type=SymbolicUtils.SymReal)
# end

# Base.:+(xs::Vararg{SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}) = begin
#     return SymbolicUtils.term(+, xs...; type=SymbolicUtils.SymReal)
# end

# function Base.:*(a::StaticArrays.SArray{Tuple{2, 2}, Complex{Float64}, 2, 4}, b::Matrix{ComplexF64})
#     if size(b) == (2, 2)
#         return StaticArrays.SMatrix{2, 2, ComplexF64}(a * StaticArrays.MMatrix{2, 2, ComplexF64}(b))
#     else
#         return Matrix{ComplexF64}(a) * b        
#     end
# end
# function Base.:*(a::Matrix{ComplexF64}, b::StaticArrays.SArray{Tuple{2, 2}, Complex{Float64}, 2, 4})
#     if size(a) == (2, 2)
#         return StaticArrays.SMatrix{2, 2, ComplexF64}(StaticArrays.MMatrix{2, 2, ComplexF64}(a) * b)
#     else
#         return a * Matrix{ComplexF64}(b)
#     end
# end

SymbolicUtils.islike(a::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, ::Type{Number}) = true

# col_1 is linear, so kron_derivative's multilinear product rule differentiates
# it correctly as the one-argument case: d(col_1(X)) = col_1(dX). So is a GateOn term,
# d(GateOn(n, qubits)(M)) = GateOn(n, qubits)(dM). Every placement is a different operator
# value, hence the type test.
is_matop(f) = f === (⊗) || f === (⊙) || f === (col_1) || f isa GateOn

depends_on(ex, v) = begin
    if any(isequal(Symbolics.value(v)), Symbolics.get_variables(ex))
        return true
    elseif SymbolicUtils.isterm(ex)
        return isequal(ex, v) || any([depends_on(exx, v) for exx in ex.args])
    elseif SymbolicUtils.isconst(ex)
        return isequal(ex, v) || isequal(Symbolics.value(ex), v)
    elseif SymbolicUtils.issym(ex)
        return isequal(ex, v)
    else
        return false#isequal(ex, v)
    end
end

# Memoised depends_on over the DAG. Terms are hash-consed, so a shared subtree is the same
# object; and on a chain (gcol2tree_chain) the unmemoised check walks the whole remaining
# chain at every level, O(m^2) for m gates. A call depends on v if one of its arguments
# does, which is what get_variables finds; leaves, and calls whose operation could carry v
# itself (a Differential(v)), keep the original test.
function _depends_on(ex, v, memo::IdDict)
    exv = Symbolics.value(ex)
    get!(memo, exv) do
        if SymbolicUtils.iscall(exv) && SymbolicUtils.operation(exv) isa Union{Function, GateOn}
            isequal(exv, v) || any(a -> _depends_on(a, v, memo), SymbolicUtils.arguments(exv))
        else
            depends_on(exv, v)
        end
    end::Bool
end

has_matop(ex) =
    SymbolicUtils.iscall(ex) && (is_matop(SymbolicUtils.operation(ex)) || any(has_matop ∘ Symbolics.value, SymbolicUtils.arguments(ex)))

function const_integer(n)
    nv = SymbolicUtils.unwrap_const(Symbolics.value(n))
    nv isa Integer && return Int(nv)
    nv isa Union{AbstractFloat, Rational} && isinteger(nv) && return Int(nv)
    return nothing
end

# The value `SymbolicUtils.term`'s default `shape` keyword computes, folded pairwise
# over a vector. The default is `promote_shape(+, shape.(args)...)`, which broadcasts
# over the splatted argument *tuple* and then splats the result again; differentiating
# a sum of thousands of summands makes that, rather than the differentiation, the whole
# cost. `shape` is called on exactly the values the default would have called it on, so
# the result is the same -- and `term` is hash-consed, so the term is the same object.
function _sum_shape(terms)
    sh = SymbolicUtils.shape(terms[1])
    for i in 2:length(terms)
        sh = SymbolicUtils.promote_shape(+, sh, SymbolicUtils.shape(terms[i]))
    end
    return sh
end

_sum_term(terms) = SymbolicUtils.term(+, terms...;
    type=QCSym.SymbolicUtils.SymReal, shape=_sum_shape(terms))

kron_derivative(ex, v) = _kron_derivative(ex, v, IdDict(), IdDict())

# `dmemo` holds the derivative of every node already differentiated, so a subtree shared in
# the DAG is differentiated once; `depmemo` holds the answers of _depends_on.
_kron_derivative(ex, v, dmemo::IdDict, depmemo::IdDict) =
    get!(() -> _kron_derivative_node(ex, v, dmemo, depmemo), dmemo, Symbolics.value(ex))

function _kron_derivative_node(ex, v, dmemo::IdDict, depmemo::IdDict)
    _depends_on(ex, v, depmemo) || return 0
    exv = Symbolics.value(ex)
    has_matop(exv) || return Symbolics.derivative(ex, v)
    op = SymbolicUtils.operation(exv)
    args = map(Symbolics.wrap, SymbolicUtils.arguments(exv))
    deriv(x) = _kron_derivative(x, v, dmemo, depmemo)
    if op === (+)
        #return sum(kron_derivative(arg, v) for arg in args)
        return _sum_term(Any[deriv(arg) for arg in args])
    elseif op === (*) || is_matop(op)
        # product rule for multilinear ops, preserving argument order
        terms = Any[]
        for i in eachindex(args)
            if !_depends_on(args[i], v, depmemo)
                continue
            end
            di = deriv(args[i])
            isequal(di, 0) && continue
            push!(terms, op(args[1:(i - 1)]..., di, args[(i + 1):end]...))
        end
        #return isempty(terms) ? 0 : sum(terms)
        if isempty(terms)
            return 0
        elseif length(terms) == 1
            return terms[1]
        else
            return _sum_term(terms)
        end
    elseif op === (^)
        X, n = args
        nval = const_integer(n)
        if nval !== nothing && nval >= 1 && has_matop(Symbolics.value(X))
            dX = deriv(X)
            isequal(dX, 0) && return 0
            nval == 1 && return dX
            # d(X^n) = Σ_{k=0}^{n-1} X^k ⊙ dX ⊙ X^(n-1-k)
            terms = Any[]
            for k in 0:(nval - 1)
                t = k == 0        ? dX ⊙ X^(nval - 1) :
                    k == nval - 1 ? X^(nval - 1) ⊙ dX :
                                    X^k ⊙ dX ⊙ X^(nval - 1 - k)
                push!(terms, t)
            end
            return sum(terms)
        end
        return Symbolics.Differential(v)(ex)
    elseif op === (/)
        N, den = args
        if !has_matop(Symbolics.value(den))
            dN, dden = deriv(N), Symbolics.derivative(den, v)
            terms = Any[]
            isequal(dN, 0) || push!(terms, dN / den)
            isequal(dden, 0) || push!(terms, -(dden / den^2) * N)
            return isempty(terms) ? 0 : sum(terms)
        end
        return Symbolics.Differential(v)(ex)
    else
        return Symbolics.Differential(v)(ex)
    end
end





# ⊗(a::QCSym.Gates.AbstractQuantumGate, b::QCSym.Gates.AbstractQuantumGate) = begin
#     #out_shape = (a.shape[1]*b.shape[1], a.shape[2]*b.shape[2])
#     _1 = length(a.shape[1])*length(b.shape[1])
#     _2 = length(a.shape[2])*length(b.shape[2])
#     out_shape = SymbolicUtils.ShapeVecT([1:_1,1:_2])
#     #SymbolicUtils.term(⊗, a, b; shape=out_shape, type=SymbolicUtils.BasicSymbolicImpl.Term{SymbolicUtils.SymReal})
#     #SymbolicUtils.term(⊗, a, b; shape=out_shape, type=QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit})
#     SymbolicUtils.term(⊗, a, b; shape = SymbolicUtils.SmallVec{UnitRange{Int64}, Vector{UnitRange{Int64}}}([1:1]), type=QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit})
# end

# function ⊗(xs::AbstractVector{T}) where {T<:QCSym.Gates.AbstractGate}
#     isempty(xs) && throw(ArgumentError("⊗ requires at least one matrix/gate"))
#     # returned value matches r = SymbolicUtils.@rule  QCSym.:⊗(~~ys) => 1.0
#     # Kronecker product of a single factor is the factor itself
#     length(xs) == 1 && return only(xs)

#     _1 = prod(length(x.shape[1]) for x in xs)
#     _2 = prod(length(x.shape[2]) for x in xs)

#     out_shape = SymbolicUtils.ShapeVecT([1:_1, 1:_2])

#     return SymbolicUtils.term(
#         ⊗,
#         xs...;
#         #shape = out_shape,
#         shape = SymbolicUtils.SmallVec{UnitRange{Int64}, Vector{UnitRange{Int64}}}([1:1]), 
#         type = QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit},
#     )
# end

# ⊗(a::SymbolicUtils.BasicSymbolic, b::QCSym.Gates.AbstractQuantumGate) = begin
#     println("Ended up here")
#     println(a.shape)
#     println(b.shape)
#     println(QCSym.SymbolicUtils.shape(a))

#     _1 = length(a.shape[1])*length(b.shape[1])
#     _2 = length(a.shape[2])*length(b.shape[2])
#     out_shape = SymbolicUtils.ShapeVecT([1:_1,1:_2])
#     #SymbolicUtils.term(⊗, a, b; shape=out_shape, type=SymbolicUtils.BasicSymbolicImpl.Term{SymbolicUtils.SymReal})
#     #SymbolicUtils.term(⊗, a, b; shape=out_shape, type=QCSym.Gates._CMQGate{QCSym.BitsRegs.Bit})
#     SymbolicUtils.term(⊗, a, b; shape=SymbolicUtils.SmallVec{UnitRange{Int64}, Vector{UnitRange{Int64}}}([1:1]), type=QCSym.Gates._CMQGate{QCSym.BitsRegs.Bit})
# end

# mul(a::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, b::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}) = begin
    
#     @assert SymbolicUtils.symtype(a) == QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit} || (SymbolicUtils.symtype(a) == QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit}) "Gate types must be either _CMQGate or _CMSMQGate for basic multiplication, but got types a $(SymbolicUtils.symtype(a))"
    
#     @assert SymbolicUtils.symtype(b) == QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit} || (SymbolicUtils.symtype(b) == QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit}) "Gate types must be either _CMQGate or _CMSMQGate for basic multiplication, but got types b $(SymbolicUtils.symtype(b))"
        
    
#     @assert a.shape == b.shape "Gate shapes must match for basic multiplication, but got shapes $(a.shape) and $(b.shape)"
#     out_shape = SymbolicUtils.promote_shape(*, a.shape, b.shape)
    
#     #SymbolicUtils.term(mul, a, b; shape=out_shape, type=QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit})
#     SymbolicUtils.term(mul, a, b;shape=SymbolicUtils.SmallVec{UnitRange{Int64}, Vector{UnitRange{Int64}}}([1:1]), type=QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit})
# end

# mul(a::AbstractVector{SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}) = begin
    
#     for i in eachindex(a)
#         @assert SymbolicUtils.symtype(a[i]) == QCSym.Gates._CMQGate{QCSym.BitsRegs.QBit} "Gate types must be either _CMQGate for basic multiplication, but got types a[$i] $(SymbolicUtils.symtype(a[i]))"
#     end
#     # for i in eachindex(a)
#     #     @assert a[1].shape == a[i].shape "Gate shapes must match for basic multiplication, but got shapes $(a[1].shape) and $(a[i].shape)"
#     # end    
    
#     out_shape = a[1].shape
    
#     #SymbolicUtils.term(mul, a...; shape=out_shape, type=QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit})
#     SymbolicUtils.term(mul, a...; shape=SymbolicUtils.SmallVec{UnitRange{Int64}, Vector{UnitRange{Int64}}}([1:1]), type=QCSym.Gates._CMSMQGate{QCSym.BitsRegs.QBit})
# end

