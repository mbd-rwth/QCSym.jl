import Symbolics
import SymbolicUtils
import StaticArrays
import ..BitsRegs.AbstractBit
import ..BitsRegs.MapBitID
import ..BitsRegs.Bit
import InteractiveUtils

abstract type AbstractGate end
abstract type AbstractQuantumGate{T} <: AbstractGate end
abstract type AbstractSingleQubitQuantumGate{T} <: AbstractQuantumGate{T} end
abstract type AbstractMultiQubitQuantumGate{T} <: AbstractQuantumGate{T} end

abstract type AbstractSingleQubitQuantumGateParametric{T} <: AbstractSingleQubitQuantumGate{T} end
abstract type AbstractSingleQubitQuantumGateNonParametric{T} <: AbstractSingleQubitQuantumGate{T} end

abstract type AbstractMultiQubitQuantumGateParametric{T} <: AbstractMultiQubitQuantumGate{T} end
abstract type AbstractMultiQubitQuantumGateNonParametric{T} <: AbstractMultiQubitQuantumGate{T} end

abstract type FillerGate end

const GateDecomposition2x2Types = Dict{UInt, Vector{Type{<:AbstractSingleQubitQuantumGate}}}
const GateDecomposition2x2Gates = Dict{UInt, Vector{<:AbstractSingleQubitQuantumGate}}

macro insert_fields_AbstractQuantumGate()
    quote
        $(esc(:(num_qubits::Int)))
        $(esc(:(num_qubits_t::Int)))
        $(esc(:(num_qubits_c::Union{Nothing, Int})))
        $(esc(:(is_parametric::Bool)))
        $(esc(:(is_treat_numeric_only::Bool)))
        $(esc(:(is_treat_alt_only::Bool)))
        $(esc(:(name::String)))
        $(esc(:(symbol::SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal})))
        $(esc(:(name_short::String)))
        #$(esc(:(shape::Tuple{Int, Int})))
        $(esc(:(shape::SymbolicUtils.ShapeT)))
        $(esc(:(qubits::Array{T,1})))
        $(esc(:(qubits_t::Array{T,1})))
        $(esc(:(qubits_c::Union{Nothing, Array{T,1}})))
        $(esc(:(step::Int)))
        $(esc(:(num_summands_decomposed::Int)))
        $(esc(:(parameters::Union{Nothing, Dict{String, Dict{String, Union{Symbolics.Num, <:Real}}}})))
        $(esc(:(atomics::Vector{<:Complex{Symbolics.Num}})))
        $(esc(:(atomics_alt::Union{Nothing, Vector{<:Symbolics.Num}})))
        $(esc(:(matrix::Symbolics.Arr{Complex{Symbolics.Num},2})))
        #$(esc(:(matrix_alt::Union{Nothing, Matrix{Symbolics.Num}, Matrix{Complex{Symbolics.Num}}, Symbolics.Arr{Symbolics.Num,2}, SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}})))
        $(esc(:(matrix_alt::Union{Nothing, StaticArrays.MMatrix{2,2,Symbolics.Num}, StaticArrays.MMatrix{2,2,Complex{Symbolics.Num}}, StaticArrays.MMatrix{4,4,Symbolics.Num}, StaticArrays.MMatrix{4,4,Complex{Symbolics.Num}}, Symbolics.Arr{Symbolics.Num,2}})))
        $(esc(:(ids_matrix_zeros::Union{Nothing, Array{Int, 2}})))
        $(esc(:(matrix_numeric::Union{Nothing, StaticArrays.SMatrix})))
        $(esc(:(matrix22_t::Union{Nothing, Dict{Int, Vector{Symbolics.Arr{Complex{Symbolics.Num},2}}}})))
        $(esc(:(matrix22_t_alt::Union{Nothing, Dict{Int, Vector{Union{Nothing, Matrix{Symbolics.Num}, Matrix{Complex{Symbolics.Num}}, Symbolics.Arr{Symbolics.Num,2}, SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}}}})))
        $(esc(:(matrix22_c::Union{Nothing, Dict{Int, Vector{Symbolics.Arr{Complex{Symbolics.Num},2}}}})))
        $(esc(:(matrix22_t_numeric::Union{Nothing, Dict{Int, Vector{Array{Complex,2}}}})))
        $(esc(:(matrix22_c_numeric::Union{Nothing, Dict{Int, Vector{Array{Complex,2}}}})))
        $(esc(:(gates22_t::Union{Nothing, GateDecomposition2x2Gates})))
        $(esc(:(gates22_c::Union{Nothing, GateDecomposition2x2Gates})))
    end
end

mutable struct mutable_BaseQuantumGate_for_construction{T<:AbstractBit} <: AbstractQuantumGate{T}
    @insert_fields_AbstractQuantumGate()
    
    # function mutable_BaseQuantumGate_for_construction(;name="", name_short="", shape=(0,0),
    # qubits_t=[Bit(context=MapBitID(), index_local=-1, is_quantum=true, name_reg="default")], qubits_c=nothing,
    # step=0, num_summands_decomposed=0, parameters=nothing, is_treat_numeric_only=false,
    # ids_matrix_zeros=nothing, matrix_numeric=nothing, matrix22_t=nothing, matrix22_t_alt=nothing,
    # matrix22_c=nothing, matrix22_t_numeric=nothing, matrix22_c_numeric=nothing)
    function mutable_BaseQuantumGate_for_construction(; 
        is_treat_numeric_only::Bool,
        name_prefix::String, name_short::String, qubits_t::Vector{T}, qubits_c::Union{Nothing, Vector{T}}=nothing,
        step::Int, num_summands_decomposed::Int, parameters::Union{Nothing, AbstractVector{String}, Dict{String, Dict{Symbolics.Num, <:Real}}}=nothing,
        decomposition_t::Union{Nothing, GateDecomposition2x2Types}=nothing,
        decomposition_c::Union{Nothing, GateDecomposition2x2Types}=nothing) where {T<:AbstractBit}
        
        num_qubits, num_qubits_t, num_qubits_c = _determine_num_qubits(qubits_t, qubits_c)
        is_parametric = parameters !== nothing
        is_treat_numeric_only = is_treat_numeric_only
        is_treat_alt_only = false
        name=_generate_name_str(name_prefix*name_short, step, qubits_t, qubits_c)
        name_short=name_short
        #shape=(2^num_qubits, 2^num_qubits)
        _N = 2^num_qubits
        shape=SymbolicUtils.ShapeVecT([1:_N, 1:_N])
        qubits_t=qubits_t
        qubits_c=qubits_c
        qubits = isnothing(qubits_c) ? qubits_t : vcat(qubits_t, qubits_c)
        step=step
        num_summands_decomposed=num_summands_decomposed
        if is_parametric
            if parameters isa Vector{String}
                _ps = collect((Symbol(_generate_name_str(p*"_"*name_prefix*name_short, step, qubits_t, qubits_c))) for p in parameters)
                _ps = collect(Symbolics.@variables($(_ps[i]))[1] for i in eachindex(_ps))
                parameters = Dict(parameters[i] => Dict("sym" => _ps[i], "val" => 0.0) for i in eachindex(parameters))
            else
                error("parameters must be either a Vector{Symbolics.Num} or a Dict{Symbolics.Num, <:Complex} or a Dict{Symbolics.Num, <:Real}")
            end
        end        
        #parameter_symbols = parameters === nothing ? Vector{Union{Complex{Symbolics.Num}}}() : collect(Complex{Symbolics.Num}, v["sym"] for (k,v) in parameters)
        parameter_symbols = parameters === nothing ? Vector{Symbolics.Num}() : collect(Symbolics.Num, v["sym"] for (k,v) in parameters)
        
        symbol = if isempty(parameter_symbols)
            eval(:(Symbolics.@variables($(Symbol(name))::Complex{Real})[1]))
        else
            out_sym = Symbolics.variable(Symbol(name); T=Symbolics.FnType{Tuple{Vararg{Number}}, Number, Nothing})
            SymbolicUtils.unwrap(out_sym(parameter_symbols...))
        end
        atomics = nothing # will be set after matrix
        atomics_alt = parameters === nothing ? nothing : collect(v["sym"] for (k,v) in parameters)
        matrix = eval(Meta.parse("Symbolics.@variables(($(name)::Complex)[$(shape[1]),$(shape[2])])"))
        matrix = matrix[1]
        matrix_alt = nothing
        atomics = Symbolics.scalarize(matrix)[:]
        
        ids_matrix_zeros=nothing
        matrix_numeric=nothing
        matrix22_t=nothing
        matrix22_t_alt=nothing
        matrix22_c=nothing
        matrix22_t_numeric=nothing
        matrix22_c_numeric=nothing

        gates22_t=nothing
        gates22_c=nothing

        if decomposition_t !==nothing && decomposition_c !== nothing
            @assert allequal([num_summands_decomposed, length(decomposition_t[1]), length(decomposition_c[1])]) "decomposition_t and decomposition_c must have the same length"
            gates22_t = GateDecomposition2x2Gates()
            for (idloc, qt) in enumerate(qubits_t)
                gates22_t[qt.index_global] = [eval(Meta.parse((String(nameof(decomposition_t[idloc][i]))*"_for_Circuit")))(;
                name_prefix=name, qubits_t=[qubits_t[1]], step=step,is_treat_numeric_only=is_treat_numeric_only,
                is_treat_alt_only=is_treat_alt_only) for i in 1:num_summands_decomposed]
            end
            gates22_c = GateDecomposition2x2Gates()
            for (idloc, qc) in enumerate(qubits_c)
                gates22_c[qc.index_global] = [eval(Meta.parse((String(nameof(decomposition_c[idloc][i]))*"_for_Circuit")))(;
                name_prefix=name, qubits_t=[qubits_t[1]], step=step,is_treat_numeric_only=is_treat_numeric_only,
                is_treat_alt_only=is_treat_alt_only) for i in 1:num_summands_decomposed]
            end
           
           
        #    Dict(i => eval(Meta.parse((String(nameof(decomposition_t[i]))*"_for_Circuit")))(;
        #    name_prefix=name, qubits_t=[qubits_t[1]], step=step,is_treat_numeric_only=is_treat_numeric_only,
        #    is_treat_alt_only=is_treat_alt_only) for i in 1:num_summands_decomposed)
               
        # elseif decomposition_t === nothing && decomposition_c === nothing
        #     continue
        elseif decomposition_t !== nothing && decomposition_c === nothing
            @error("decomposition_t is provided but decomposition_c is not. Either both or none must be provided.")
        elseif decomposition_t === nothing && decomposition_c !== nothing
            @error("decomposition_c is provided but decomposition_t is not. Either both or none must be provided.")
        end

        return new{T}(num_qubits, num_qubits_t, num_qubits_c,
            is_parametric, is_treat_numeric_only, is_treat_alt_only,
            name, symbol, name_short, shape, qubits, qubits_t, qubits_c,
            step, num_summands_decomposed, parameters, atomics, atomics_alt,
            matrix, matrix_alt, ids_matrix_zeros, matrix_numeric,
            matrix22_t, matrix22_t_alt, matrix22_c,
            matrix22_t_numeric, matrix22_c_numeric, gates22_t, gates22_c
        )
    end
end


"""
    @constructor_from_mutable_base(struct_name, base_varname)

Generates an inner constructor for an immutable struct named `struct_name`.
The generated constructor accepts a single `mutable_BaseQuantumGate_for_construction`
instance (bound to parameter `base_varname`) and copies all its fields positionally
into `new(...)`.

Intended to be placed inside a struct body, where a separate user-facing constructor
builds and mutates a `mutable_BaseQuantumGate_for_construction`, then delegates to
this generated constructor.

# Usage

    struct MyGate{T<:AbstractBit} <: AbstractQuantumGate{T}
        @insert_fields_AbstractQuantumGate()
        @constructor_from_mutable_base(MyGate, base_gate)

        function MyGate(; qubits_t, qubits_c=nothing, step)
            base_gate = mutable_BaseQuantumGate_for_construction(...)
            # ... mutate base_gate fields as needed ...
            return MyGate(base_gate)   # dispatches to the generated constructor
        end
    end
"""
macro constructor_from_mutable_base(struct_name, base_varname)
    quote
        function $(esc(struct_name))($(esc(base_varname))::mutable_BaseQuantumGate_for_construction{_COMB_T}) where {_COMB_T <: AbstractBit}
            $(esc(:new)){_COMB_T}(Tuple(getfield($(esc(base_varname)), f) for f in fieldnames(typeof($(esc(base_varname)))))...)
        end
    end
end


function _generate_name_str(name::AbstractString, step::Int, qubits_t::Array{T,1}, qubits_c::Union{Nothing, Array{T,1}}) where {T<:AbstractBit}
    return name * "_s" * string(step) * "qt" * join((q.index_global for q in qubits_t), "") * "qc" * (isnothing(qubits_c) ? "" : join((q.index_global for q in qubits_c), ""))
end

function _determine_num_qubits(qubits_t::Array{T,1}, qubits_c::Union{Nothing, Array{T,1}}) where {T<:AbstractBit}
    num_qubits_t = size(qubits_t,1)
    num_qubits_c = isnothing(qubits_c) ? 0 : size(qubits_c,1)
    return num_qubits_t + num_qubits_c, num_qubits_t, num_qubits_c
end



#@kwdef struct BaseQuantumGate{T<:AbstractBit} <: AbstractQuantumGate{T}
#    @insert_fields_AbstractQuantumGate()
#end

#println(fieldnames(BaseQuantumGate))

# @kwdef struct BaseQuantumGate{T<:AbstractBit} <: AbstractQuantumGate
#     #is_should_be_listed_in_gate_collection::Bool
#     #name_gate_collection::Union{Nothing, String}
#     num_qubits::Int
#     num_qubits_t::Int
#     num_qubits_c::Union{Nothing, Int}
#     is_parametric::Bool
#     is_treat_numeric_only::Bool
    
#     name::String
#     name_short::String
#     shape::Tuple{Int, Int}
#     qubits::                 Array{T,1}
#     qubits_t::               Array{T,1}
#     qubits_c::Union{Nothing, Array{T,1}}
#     step::Int
#     num_summands_decomposed::Int
#     parameters::Union{Nothing, Vector{Dict{String, Complex}}}
#     atomics::                   Vector{<:Symbolics.Num}
#     atomics_alt::Union{Nothing, Vector{<:Symbolics.Num}}
#     matrix::                   Symbolics.Arr{Symbolics.Num,2}
#     matrix_alt::Union{Nothing, Matrix{Symbolics.Num}, Symbolics.Arr{Symbolics.Num,2}, SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{SymbolicUtils.SymReal}}
#     ids_matrix_zeros::Union{Nothing, Array{Int, 2}}
#     matrix_numeric::Union{Nothing, Array{Complex,2}}
#     matrix22_t::    Union{Nothing, Dict{Int, Vector{Symbolics.Arr{Symbolics.Num,2}}}}
#     matrix22_t_alt::Union{Nothing, Dict{Int, Vector{Symbolics.Arr{Symbolics.Num,2}}}}
#     matrix22_c::    Union{Nothing, Dict{Int, Vector{Symbolics.Arr{Symbolics.Num,2}}}}
#     matrix22_t_numeric::Union{Nothing, Dict{Int, Vector{Array{Complex,2}}}}
#     matrix22_c_numeric::Union{Nothing, Dict{Int, Vector{Array{Complex,2}}}}
# end


Base.show(io::IO, gate::T) where {T <: AbstractQuantumGate} = begin
    println(io, "Quantum Gate: ", gate.name)
    for name in fieldnames(typeof(gate))
        println(io, "  ", name, ": ", getfield(gate, name))
    end
end

function get_all_concrete_gates()
    return _subtypes(AbstractQuantumGate)
end


function _subtypes(type::Type)
    out = Any[]
    _subtypes!(out, type)
end

function _subtypes!(out, type::Type)
    if !isabstracttype(type)
        push!(out, type)
    else
        foreach(T->_subtypes!(out, T), InteractiveUtils.subtypes(type))
    end
    out
end
