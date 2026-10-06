include("src.jl")
import QCSym
import Symbolics
import SymbolicUtils
import StaticArrays
target_state = Array{ComplexF64}(collect(1:4) .+ 1im)

sp2 = MottonenStatePreparation.StatePreparator(target_state, true)

num_qubits = ceil(Int, log2(length(target_state)))
qc = QCSym.Circuits.QuantumCircuit(name="MottonenStatePrep")
qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)

MottonenStatePreparation.implement_state_prep!(qc, sp2, qreg)

expr = QCSym.Circuits.gcol2tree(qc.gatecollection)


param_gates_XX = Dict()
param_angles_XX = []
dict_to_subs_XX = Dict()
for n in [QCSym.Gates.RY_Gate, QCSym.Gates.RZ_Gate, QCSym.Gates.GP_Gate]
    param_gates_XX[n] = qc.gatecollection.collections[n]
    param_angles_XX = vcat(param_angles_XX, [pg.parameters[first(keys(pg.parameters))]["sym"] for pg in param_gates_XX[n]])
    dict_to_subs_XX = merge(dict_to_subs_XX, Dict(pg.symbol => pg.matrix_alt for pg in param_gates_XX[n]))
end

expr2 = Symbolics.substitute(expr, dict_to_subs_XX)
vars = collect(v for v in Symbolics.get_variables(expr2))

function get_val_for_sym(s)
    _a = QCSym.Circuits.s2g(s, qc.gatecollection)
    if hasfield(typeof(_a), :symbol) && isequal(s, _a.symbol)
        _g = _a.matrix_numeric
        return _g
    end
    if !isnothing(_a)
        for (_, p) in _a.parameters
            if isequal(p["sym"], s)
                return nothing
            end
        end
        for p in Symbolics.get_variables(_a.symbol)
            if isequal(p, s)
                _g = _a.matrix_alt
                return nothing
            end
        end
    end
    if isequal(s.name, :I)
        return StaticArrays.SMatrix{2, 2}([1 0; 0 1])
    end
    return nothing
end


vars_vals = Dict(v => get_val_for_sym(v) for v in vars)
vars_vals = Dict(k => v for (k,v) in vars_vals if !isnothing(v))

expr3 = Symbolics.substitute(expr2, vars_vals)

params = collect(v for v in Symbolics.get_variables(expr3))
function _recurse(e)
    println(e)
    if ! SymbolicUtils.iscall(e)
        println(e)
        println(SymbolicUtils.isexpr(e))
        println(SymbolicUtils.isconst(e))
        println(SymbolicUtils.is_function_symbolic(e))
        println(SymbolicUtils.issym(e))
        println(SymbolicUtils.isterm(e))
        println(SymbolicUtils.default_is_atomic(e))
        println(typeof(e))
        println(typeof(SymbolicUtils.unwrap(e)))
        return false
    else
        return true
    end
end
params = collect(v for v in SymbolicUtils.search_variables(expr3; recurse=_recurse))


ary = first(param_gates_XX[QCSym.Gates.RY_Gate]).matrix_alt
θ, λ = Symbolics.@variables θ::Real λ::Real
bob = Symbolics.variable(:bob; T=Symbolics.FnType{Tuple{Vararg{Number}}, Number, Nothing})
bob2 = SymbolicUtils.unwrap(bob(θ))
a = Matrix([exp(-1im*θ/2) 0.0; 0.0 exp(1im*θ/2)])

Symbolics.get_variables(a)


b = λ*bob2
Symbolics.substitute(b, Dict(bob2 => a))
Symbolics.get_variables(b)

bb = QCSym.:⊗(λ, bob2)
Symbolics.substitute(bb, Dict(bob2 => a))
Symbolics.get_variables(b)

SymbolicUtils.symbolic_type(Symbolics.Num)
SymbolicUtils.symbolic_type(Complex{Symbolics.Num})


function my_is_array_of_symbolics(O)
    # O is an array, not a symbolic array, and either has a non-symbolic eltype or contains elements that are
    # symbolic or arrays of symbolics
    return O isa AbstractArray && SymbolicUtils.symbolic_type(O) == SymbolicUtils.NotSymbolic() &&
        (SymbolicUtils.symbolic_type(eltype(O)) != SymbolicUtils.NotSymbolic() && !(eltype(O) <: Union{Symbol, Expr}) ||
        false)
end

my_is_array_of_symbolics(ary)
SymbolicUtils.symbolic_type(ary) == SymbolicUtils.NotSymbolic()
(SymbolicUtils.symbolic_type(eltype(ary)) != SymbolicUtils.NotSymbolic() && !(eltype(ary) <: Union{Symbol, Expr}))

my_is_array_of_symbolics(a)
SymbolicUtils.symbolic_type(a) == SymbolicUtils.NotSymbolic()
(SymbolicUtils.symbolic_type(eltype(a)) != SymbolicUtils.NotSymbolic() && !(eltype(a) <: Union{Symbol, Expr}))
SymbolicUtils.symbolic_type(eltype(a)) != SymbolicUtils.NotSymbolic()
!(eltype(a) <: Union{Symbol, Expr})

vec(a)

a_al = SymbolicUtils.array_literal((2,2), vec(a)...)
my_is_array_of_symbolics(a_al)

abcd = QCSym.:⊗(first(param_gates_XX[QCSym.Gates.RY_Gate]).symbol, first(param_gates_XX[QCSym.Gates.RZ_Gate]).symbol)
abcd = first(param_gates_XX[QCSym.Gates.RY_Gate]).symbol * first(param_gates_XX[QCSym.Gates.RZ_Gate]).symbol
Symbolics.substitute(abcd, Dict(first(param_gates_XX[QCSym.Gates.RY_Gate]).symbol => first(param_gates_XX[QCSym.Gates.RY_Gate]).matrix_alt))

Symbolics.substitute(abcd, Dict(first(param_gates_XX[QCSym.Gates.RZ_Gate]).symbol => first(param_gates_XX[QCSym.Gates.RZ_Gate]).matrix_alt))