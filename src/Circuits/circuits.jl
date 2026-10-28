import Symbolics
#import ..BitsRegs
#import ..Gates
#import ..Gates
import ..QCSym
#import QCSym.Gates
#import QCSym.Gates

include("./gatecollection.jl")
include("./symbolic_tree.jl")

@kwdef struct QuantumCircuit
    name::String = "MyCircuit"
    context::QCSym.BitsRegs.MapBitID = QCSym.BitsRegs.MapBitID()
    qregs::Vector{QCSym.BitsRegs.BitRegister{QCSym.BitsRegs.QBit}} = QCSym.BitsRegs.BitRegister{QCSym.BitsRegs.QBit}[]
    cregs::Vector{QCSym.BitsRegs.BitRegister{QCSym.BitsRegs.CBit}} = QCSym.BitsRegs.BitRegister{QCSym.BitsRegs.CBit}[]
    gatecollection::GateCollection = GateCollection(Dict())
    barriers::Vector{Int} = Int[]
end

function get_num_qubits(qc::QuantumCircuit)
    n = 0
     for qreg in qc.qregs
        n += qreg.size
     end
     return n
end
function get_num_cbits(qc::QuantumCircuit)
    n = 0
     for creg in qc.cregs
        n += creg.size
     end
     return n
end

include("./gate_chain.jl")

Base.show(io::IO, qc::QuantumCircuit) = begin
    println(io, "Quantum Circuit: ", qc.name)
    println(io, "  Number of qubits: ", get_num_qubits(qc))
    println(io, "  Number of classical bits: ", get_num_cbits(qc))
    println(io, "  Quantum registers:")
    for qreg in qc.qregs
        println(io, "    ", qreg.name, " (size: ", qreg.size, ")")
    end
    println(io, "  Classical registers:")
    for creg in qc.cregs
        println(io, "    ", creg.name, " (size: ", creg.size, ")")
    end
    println(io, "  Gates:")
    for (gate_type, gates) in qc.gatecollection.collections
        println(io, "    ", gate_type)
        for gate in gates
            println(io, "      ", gate.name)
        end
    end
    println(io, "  Barriers:")
    for barrier in qc.barriers
        println(io, "    ", barrier)
    end
end

function add_qreg(qc::QuantumCircuit, name::String="", size::Int=-1)
    qreg = QCSym.BitsRegs.add_qreg(qc.context, name, size)
    #QCSym.BitsRegs.add_qreg(qc.context,qreg)
    push!(qc.qregs, qreg)
    return qreg
end

function add_creg(circuit::QuantumCircuit, name::String="", size::Int=-1)
    @assert !has_creg(circuit, name) "Classical register with name $name already exists"
    creg = QCSym.BitsRegs.add_creg(circuit.context, name, size)
    push!(circuit.cregs, creg)
    return creg
end

function has_creg(qc::QuantumCircuit, name::String)
    return QCSym.BitsRegs.has_creg(qc.context, name)
end

function has_qreg(qc::QuantumCircuit, name::String)
    return QCSym.BitsRegs.has_qreg(qc.context, name)
end

"""kwargs are exclusively forwarded to the added gate. See their respective documentation."""
function add_gate(circuit::QuantumCircuit, gate::Type{T}; kwargs...) where {T <: QCSym.Gates.AbstractGate}
    if !haskey(circuit.gatecollection.collections, gate)
        circuit.gatecollection.collections[gate] = T[]
    end
    #println(nameof(T))
    #println("$(gate)_for_Circuit")
    #println(Meta.parse("$(gate)_for_Circuit"))
    gate_constructor = eval(Meta.parse("$(gate)_for_Circuit"))
    push!(circuit.gatecollection.collections[gate], gate_constructor(; kwargs...))
end

function assemble_symbolic_unitary(qc::QuantumCircuit, replace_symbolic_zeros::Bool=false, replace_symbolic_ones::Bool=false)
    return assemble_unitary(qc, replace_symbolic_zeros, replace_symbolic_ones)
end

function _replace_symbolic_zeros_and_ones(U::Matrix{T}, gates::AbstractVector{<:QCSym.Gates.AbstractGate}, replace_symbolic_zeros::Bool, replace_symbolic_ones::Bool) where {T<:Union{Symbolics.Num, Complex{Symbolics.Num}}}
    
    dict_to_replace = Dict{Union{Symbolics.Num, Complex{Symbolics.Num}}, Complex}()
    for g in gates
        if g.matrix_numeric === nothing
            continue
        end
        if replace_symbolic_zeros
            for id_zero in findall(x -> x == 0, g.matrix_numeric)
                #dict_to_replace[g.matrix[id_zero]] = 0
                dict_to_replace[real(g.matrix[id_zero])] = 0
                dict_to_replace[imag(g.matrix[id_zero])] = 0
            end
        end
        if replace_symbolic_ones
            for id_one in findall(x -> x == 1, g.matrix_numeric)
                #dict_to_replace[g.matrix[id_one]] = 1
                dict_to_replace[real(g.matrix[id_one])] = 1
                dict_to_replace[imag(g.matrix[id_one])] = 0
            end
        end
    end
    return Symbolics.substitute.(U, (dict_to_replace,), fold=Val(true))
end

function substitute_numerics_from_gates(expr::AbstractArray{T}, gates::AbstractVector{<:QCSym.Gates.AbstractGate}; fold=Val(true)) where {T<:Union{Symbolics.Num, Complex{Symbolics.Num}}}
    dict_to_replace = Dict{Union{Symbolics.Num, Complex{Symbolics.Num}}, Float64}()
    for g in gates
        if g.matrix_numeric === nothing
            continue
        end
        for id in eachindex(g.matrix)
            #println(typeof(g.matrix[id]), " ", eltype(g.matrix[id]), " ", g.matrix[id], " => ", typeof(g.matrix_numeric[id]))
            dict_to_replace[real(g.matrix[id])] = real(g.matrix_numeric[id])
            dict_to_replace[imag(g.matrix[id])] = imag(g.matrix_numeric[id])
        end
    end
    #println("to replace: ", dict_to_replace)
    return Symbolics.substitute.(expr, (dict_to_replace,))
end