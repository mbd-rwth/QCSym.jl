import Symbolics
import StaticArrays
import ..BitsRegs.AbstractBit
#import ..BitsRegs.Bit
import ..BitsRegs.MapBitID
import ..BitsRegs.QBit
import ..BitsRegs.BitRegister

# kronecker product of single qubit gates, including 2x2 decompositions of multi qubit gates
struct _CMQGate{T<:AbstractBit} <: AbstractQuantumGate{T}
end

# matrix-matrix product of kronecker products ...
struct _CMSMQGate{T<:AbstractBit} <: AbstractQuantumGate{T}
end


struct _00_Gate{T<:AbstractBit} <: AbstractInternalSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(_00_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function _00_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="_00", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 0])
    return _00_Gate(base_gate)
end

struct _11_Gate{T<:AbstractBit} <: AbstractInternalSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(_11_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function _11_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="_11", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([0 0; 0 1])
    return _11_Gate(base_gate)
end

struct I_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(I_Gate, mutable_BaseQuantumGate_for_construction)
end

function I_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="I", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 1])
    return I_Gate(base_gate)
end

struct I_Gate_Filler{T<:Int} <: AbstractSingleQubitQuantumGate{T}
    qbit_glob_id::T
    shape::SymbolicUtils.ShapeT
    symbol::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}
    matrix_numeric::StaticArrays.SMatrix
    #@insert_fields_AbstractQuantumGate()
    #@constructor_from_mutable_base(I_Gate, mutable_BaseQuantumGate_for_construction)
    #I_Gate_Filler(qbit_glob_id::T) where {T<:Int} = new{T}(qbit_glob_id, SymbolicUtils.ShapeVecT([1:2,1:2]), SymbolicUtils.one_of_vartype(SymbolicUtils.SymReal), [1 0; 0 1])
    I_Gate_Filler(qbit_glob_id::T) where {T<:Int} = new{T}(qbit_glob_id, SymbolicUtils.ShapeVecT([1:2,1:2]), eval(:(Symbolics.@variables($(Symbol("I"))::Complex{Real})[1])), StaticArrays.SMatrix{2,2}([1 0; 0 1]))
end

Base.show(io::IO, gate::I_Gate_Filler) = begin
    println(io, "Quantum Gate: I_Gate_Filler")
    for name in fieldnames(typeof(gate))
        println(io, "  ", name, ": ", getfield(gate, name))
    end
end


struct X_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(X_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function X_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="X", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([0 1; 1 0])
    return X_Gate(base_gate)
end

struct Y_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(Y_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function Y_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="Y", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([0.0 0.0-1im; 0.0+1im 0.0])
    return Y_Gate(base_gate)
end

struct Z_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(Z_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function Z_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="Z", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 -1])
    return Z_Gate(base_gate)
end

struct H_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(H_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function H_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="H", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = 1/sqrt(2)*StaticArrays.SMatrix{2,2}([1.0 1.0; 1.0 -1.0])
    return H_Gate(base_gate)
end

struct S_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(S_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function S_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="S", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 1im])
    return S_Gate(base_gate)
end

struct Sdg_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(Sdg_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function Sdg_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="Sdg", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 -1im])
    return Sdg_Gate(base_gate)
end

struct T_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(T_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function T_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="T", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 (1+1im)/sqrt(2)])
    return T_Gate(base_gate)
end

struct Tdg_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(Tdg_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function Tdg_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="Tdg", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = StaticArrays.SMatrix{2,2}([1 0; 0 (1-1im)/sqrt(2)])
    return Tdg_Gate(base_gate)
end

struct SX_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(SX_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function SX_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, _...)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="SX", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    base_gate.matrix_numeric = 0.5*StaticArrays.SMatrix{2,2}([1+1im 1-1im; 1-1im 1+1im])
    return SX_Gate(base_gate)
end

struct U_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(U_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function U_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false, _...)
    """NOTE: There is a difference in the definition between OpenQasm2 and OpenQasm3, OQ3 = e^i(ϕ+λ)/2 OQ2"""
    params = ["θ", "ϕ", "λ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="U", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    ϕ = params["ϕ"]["sym"]
    λ = params["λ"]["sym"]
    base_gate.matrix_alt = Matrix([cos(θ/2) -exp(1im*λ)*sin(θ/2);
                                   exp(1im*ϕ)*sin(θ/2) exp(1im*(ϕ+λ))*cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return U_Gate(base_gate)
end

struct GP_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(GP_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function GP_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool=false, is_treat_alt_only::Bool=false, param_values::Dict{String, <:Real}, _...)
    params = ["γ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="GP", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    γ = base_gate.parameters["γ"]["sym"]
    if !isempty(param_values)
        base_gate.parameters["γ"]["val"] = param_values["γ"]
    end
    base_gate.matrix_alt = Matrix([exp(1im*γ) 0.0;
                                   0.0 exp(1im*(γ))])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return GP_Gate(base_gate)
end

struct RX_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(RX_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function RX_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool=false, is_treat_alt_only::Bool=false, param_values::Dict{String, <:Real}, _...)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="RX", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    θ = base_gate.parameters["θ"]["sym"]
    if !isempty(param_values)
        base_gate.parameters["θ"]["val"] = param_values["θ"]
    end

    base_gate.matrix_alt = Matrix([cos(θ/2) -1im*sin(θ/2);
                                   -1im*sin(θ/2) cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return RX_Gate(base_gate)
end

struct RY_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(RY_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function RY_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool=false, is_treat_alt_only::Bool=false, param_values::Dict{String, <:Real}, _...)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="RY", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    θ = base_gate.parameters["θ"]["sym"]
    if !isempty(param_values)
        base_gate.parameters["θ"]["val"] = param_values["θ"]
    end
    base_gate.matrix_alt = Matrix([cos(θ/2) -sin(θ/2);
                                   sin(θ/2) cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return RY_Gate(base_gate)
end

struct RZ_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(RZ_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function RZ_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool=false, is_treat_alt_only::Bool=false, param_values::Dict{String, <:Real}, _...)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="RZ", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    θ = base_gate.parameters["θ"]["sym"]
    if !isempty(param_values)
        base_gate.parameters["θ"]["val"] = param_values["θ"]
    end
    base_gate.matrix_alt = Matrix([exp(-1im*θ/2) 0.0;
                                   0.0 exp(1im*θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return RZ_Gate(base_gate)
end

struct RZ_OQ3_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(RZ_OQ3_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function RZ_OQ3_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false, _...)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="RZ", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    θ = base_gate.parameters["θ"]["sym"]

    base_gate.matrix_alt = Matrix([cos(θ/2) sin(θ/2);
                                   sin(θ/2) -cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    throw(MethodError("There probably is a typo in the docs of OpenQasm3. Until this is clarified you probably want to use the usual RZ gate instead. See https://github.com/openqasm/openqasm/issues/660"))
    return RZ_OQ3_Gate(base_gate)
end

struct P_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(P_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function P_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false, _...)
    params = ["λ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="P", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1, parameters = params)
    λ = base_gate.parameters["λ"]["sym"]
    
    # base_gate.matrix_alt = Matrix([1.0 0.0;
    #                                0.0 exp(1im*λ)])
    # base_gate.matrix_alt = [1.0 0.0;
    #                                0.0 cos(λ)+1im*sin(λ)]
    base_gate.matrix_alt = [1.0 0.0;
                                   0.0 λ] # cannot directly be exp(1im*λ) is not correctly identified by build_function if λ is substituted by another symbolic expression. This has to do with the handling of complex numbers in general and especially in combination with matrices.
    base_gate.is_treat_alt_only = is_treat_alt_only
    return P_Gate(base_gate)
end

struct CX_Gate{T<:AbstractBit} <: AbstractMultiQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CX_Gate, mutable_BaseQuantumGate_for_construction)
end

function CX_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool=false)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="CX", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2,
        decomposition_t=GateDecomposition2x2Types(1=>[I_Gate, X_Gate]),
        decomposition_c=GateDecomposition2x2Types(1=>[_00_Gate, _11_Gate]))
    base_gate.matrix_numeric = StaticArrays.SMatrix{4,4}([1.0 0.0 0.0 0.0;
                                0.0 1.0 0.0 0.0;
                                0.0 0.0 0.0 1.0;
                                0.0 0.0 1.0 0.0])
    
    return CX_Gate(base_gate)
end

struct CY_Gate{T<:AbstractBit} <: AbstractMultiQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CY_Gate, mutable_BaseQuantumGate_for_construction)
end

function CY_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="CY", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2)
    base_gate.matrix_numeric = StaticArrays.SMatrix{4,4}([1.0 0.0 0.0 0.0;
                                0.0 1.0 0.0 0.0;
                                0.0 0.0 0.0 -1im;
                                0.0 0.0 1im 0.0])
    
    return CY_Gate(base_gate)
end

struct CZ_Gate{T<:AbstractBit} <: AbstractMultiQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CZ_Gate, mutable_BaseQuantumGate_for_construction)
end

function CZ_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only, 
        name_prefix=name_prefix, name_short="CZ", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2)
    base_gate.matrix_numeric = StaticArrays.SMatrix{4,4}([1.0 0.0 0.0 0.0;
                                0.0 1.0 0.0 0.0;
                                0.0 0.0 1.0 0.0;
                                0.0 0.0 0.0 -1.0])
    
    return CZ_Gate(base_gate)
end

struct CP_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CP_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CP_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["λ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CP", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    λ = base_gate.parameters["λ"]["sym"]
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 1.0 0.0;
                                   0.0 0.0 0.0 exp(1im*λ)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CP_Gate(base_gate)
end

struct CRX_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CRX_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CRX_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CRX", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 cos(θ/2) -sin(θ/2)*1im;
                                   0.0 0.0 -sin(θ/2)*1im cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CRX_Gate(base_gate)
end

struct CRY_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CRY_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CRY_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CRY", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 cos(θ/2) -sin(θ/2);
                                   0.0 0.0 sin(θ/2) cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CRY_Gate(base_gate)
end

struct CRY_OQ3_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CRY_OQ3_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CRY_OQ3_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CRY_OQ3", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 cos(θ/2) -sin(θ/2);
                                   0.0 0.0 sin(θ/2) cos(θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CRY_OQ3_Gate(base_gate)
end

struct CRZ_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CRZ_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CRZ_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["θ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CRZ", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 exp(-1im*θ/2) 0.0;
                                   0.0 0.0 0.0 exp(1im*θ/2)])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CRZ_Gate(base_gate)
end


struct CH_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CH_Gate, mutable_BaseQuantumGate_for_construction)
    
end
function CH_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CH", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2)
    
    base_gate.matrix_numeric = StaticArrays.SMatrix{4,4}([1.0 0.0 0.0 0.0;
                                       0.0 1.0 0.0 0.0;
                                       0.0 0.0 1.0 1.0;
                                       0.0 0.0 1.0 -1.0]) / sqrt(2)
    return CH_Gate(base_gate)
end

struct CU_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(CU_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function CU_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, qubits_c::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    params = ["θ", "ϕ", "λ", "γ"]
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="CU", qubits_t=qubits_t, qubits_c=qubits_c,
        step=step, num_summands_decomposed=2, parameters = params)
    params = base_gate.parameters
    θ = params["θ"]["sym"]
    ϕ = params["ϕ"]["sym"]
    λ = params["λ"]["sym"]
    γ = params["γ"]["sym"]
    umat = [exp(im*γ)*cos(θ/2)    -exp(im*(γ+λ))*sin(θ/2);
            exp(im*(γ+ϕ))*sin(θ/2) exp(im*(γ+ϕ+λ))*cos(θ/2)]
    #umat = exp(1im*γ)*Matrix([cos(θ/2) -exp(1im*λ)*sin(θ/2);
    #                          exp(1im*ϕ)*sin(θ/2) exp(1im*(ϕ+λ))*cos(θ/2)])
    base_gate.matrix_alt = Matrix([1.0 0.0 0.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 umat[1,1] umat[1,2];
                                   0.0 0.0 umat[2,1] umat[2,2]])
    base_gate.is_treat_alt_only = is_treat_alt_only
    return CU_Gate(base_gate)
end

struct SWAP_Gate{T<:AbstractBit} <: AbstractSingleQubitQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    @constructor_from_mutable_base(SWAP_Gate, mutable_BaseQuantumGate_for_construction)
    
end

function SWAP_Gate_for_Circuit(;name_prefix::String="", qubits_t::AbstractVector{QBit}, step::Int, is_treat_numeric_only::Bool, is_treat_alt_only::Bool=false)
    base_gate = mutable_BaseQuantumGate_for_construction(is_treat_numeric_only=is_treat_numeric_only,
        name_prefix=name_prefix, name_short="SWAP", qubits_t=qubits_t, qubits_c=nothing,
        step=step, num_summands_decomposed=1)
    
    base_gate.matrix_numeric = StaticArrays.SMatrix{4,4}([1.0 0.0 0.0 0.0;
                                   0.0 0.0 1.0 0.0;
                                   0.0 1.0 0.0 0.0;
                                   0.0 0.0 0.0 1.0])
    return SWAP_Gate(base_gate)
end