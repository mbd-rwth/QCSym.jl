import LinearAlgebra
import StaticArrays

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

_kron_size_capped(xs::AbstractArray{<:Number}...) = begin
    outdim = prod(size(x, 1) for x in xs)
    if outdim > _MAX_STATIC_KRON_DIM
        return LinearAlgebra.kron(map(_widen_if_static, xs)...)
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
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal)
end

⊙(x1::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, x2::Union{Symbolics.Num, Complex{Symbolics.Num}}) = begin
    _is_identity_filler(x1) && return x2
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal)
end

⊙(x1::Union{Symbolics.Num, Complex{Symbolics.Num}}, x2::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}) = begin
    _is_identity_filler(x2) && return x1
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal)
end

⊙(x1::Union{Symbolics.Num, Complex{Symbolics.Num}}, x2::Union{Symbolics.Num, Complex{Symbolics.Num}}) = begin
    return SymbolicUtils.term(⊙, x1, x2; type=SymbolicUtils.SymReal)
end

⊙(xs::Vararg{SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}) = begin
    return SymbolicUtils.term(⊙, xs...; type=SymbolicUtils.SymReal)
end
⊙(xs::AbstractMatrix...) = begin
    return *(xs...)
end

⊙(x1::Number, x2::Number) = begin
    return x1 * x2
end

⊙(x1::AbstractMatrix{<:Number}, x2::AbstractMatrix{<:Number}) = begin
    return x1 * x2
end

⊙(x1::Number, x2::AbstractMatrix{<:Number}) = begin
    return x1 * x2
end

⊙(x1::AbstractMatrix{<:Number}, x2::Number) = begin
    return x1 * x2
end

⊙(xs::Vararg{AbstractMatrix{<:Number}}) = begin
    return *(xs...)
end

# ⊙(xs::Vararg{Any}) = begin
#     return SymbolicUtils.term(⊙, xs...; type=SymbolicUtils.SymReal)
# end

# Base.:+(xs::Vararg{SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}) = begin
#     return SymbolicUtils.term(+, xs...; type=SymbolicUtils.SymReal)
# end


SymbolicUtils.islike(a::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}, ::Type{Number}) = true

is_matop(f) = f === (⊗) || f === (⊙)

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

has_matop(ex) =
    SymbolicUtils.iscall(ex) && (is_matop(SymbolicUtils.operation(ex)) || any(has_matop ∘ Symbolics.value, SymbolicUtils.arguments(ex)))

function const_integer(n)
    nv = SymbolicUtils.unwrap_const(Symbolics.value(n))
    nv isa Integer && return Int(nv)
    nv isa Union{AbstractFloat, Rational} && isinteger(nv) && return Int(nv)
    return nothing
end

function kron_derivative(ex, v)
    depends_on(ex, v) || return 0
    exv = Symbolics.value(ex)
    has_matop(exv) || return Symbolics.derivative(ex, v)
    op = SymbolicUtils.operation(exv)
    args = map(Symbolics.wrap, SymbolicUtils.arguments(exv))
    if op === (+)
        #return sum(kron_derivative(arg, v) for arg in args)
        return SymbolicUtils.term(+, (kron_derivative(arg, v) for arg in args)...; type=QCSym.SymbolicUtils.SymReal)
    elseif op === (*) || is_matop(op)
        # product rule for multilinear ops, preserving argument order
        terms = Any[]
        for i in eachindex(args)
            if !depends_on(args[i], v)
                continue
            end
            di = kron_derivative(args[i], v)
            isequal(di, 0) && continue
            push!(terms, op(args[1:(i - 1)]..., di, args[(i + 1):end]...))
        end
        #return isempty(terms) ? 0 : sum(terms)
        if isempty(terms)
            return 0
        elseif length(terms) == 1
            return terms[1]
        else
            return SymbolicUtils.term(+, terms...; type=QCSym.SymbolicUtils.SymReal)
        end
    elseif op === (^)
        X, n = args
        nval = const_integer(n)
        if nval !== nothing && nval >= 1 && has_matop(Symbolics.value(X))
            dX = kron_derivative(X, v)
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
            dN, dden = kron_derivative(N, v), Symbolics.derivative(den, v)
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

