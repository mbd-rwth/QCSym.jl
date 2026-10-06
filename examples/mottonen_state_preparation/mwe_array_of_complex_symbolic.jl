import Symbolics
import SymbolicUtils

SymbolicUtils.SymbolicIndexingInterface.symbolic_type(::Type{Complex{Symbolics.Num}}) = SymbolicUtils.ScalarSymbolic()

θ, λ = Symbolics.@variables θ::Real λ::Real

real_f = Symbolics.variable(:real_f; T=Symbolics.FnType{Tuple{Vararg{Number}}, Number, Nothing})(θ)
complex_f = Symbolics.variable(:complex_f; T=Symbolics.FnType{Tuple{Vararg{Number}}, Number, Nothing})(λ)

real_mat =    Matrix([exp(    θ) 0.0; 0.0 0.0])
complex_mat = Matrix([exp(1im*λ) 0.0; 0.0 0.0])

expr = real_f * complex_f

real_subs = Symbolics.substitute(expr, Dict(real_f => real_mat))
complex_subs = Symbolics.substitute(expr, Dict(complex_f => complex_mat))

real_vars = Symbolics.get_variables(real_subs)
complex_vars = Symbolics.get_variables(complex_subs)

SymbolicUtils.SymbolicIndexingInterface.symbolic_type(::Type{Complex{Symbolics.Num}}) = SymbolicUtils.ScalarSymbolic()

function my_is_array_of_symbolics(O)
    # O is an array, not a symbolic array, and either has a non-symbolic eltype or contains elements that are
    # symbolic or arrays of symbolics
    return O isa AbstractArray && SymbolicUtils.symbolic_type(O) == SymbolicUtils.NotSymbolic() &&
        (SymbolicUtils.symbolic_type(eltype(O)) != SymbolicUtils.NotSymbolic() && !(eltype(O) <: Union{Symbol, Expr}) ||
        false)
end


SymbolicUtils.symbolic_type(eltype(eltype(complex_mat)))
SymbolicUtils.symtype(real_mat)
SymbolicUtils.symtype(complex_mat)

SymbolicUtils.isconst(real_mat)
SymbolicUtils.isconst(complex_mat)

SymbolicUtils.symbolic_type(eltype(real_mat))
SymbolicUtils.symbolic_type(eltype(complex_mat))


SymbolicUtils.symbolic_type(eltype(real_mat))

methods(SymbolicUtils.symbolic_type)
stacktrace()