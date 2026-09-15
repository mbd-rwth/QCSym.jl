import Symbolics
import SymbolicUtils
import BenchmarkTools
import StaticArrays
import LoopVectorization
import DataFrames
import CSV
import LinearAlgebra
using Revise
import QCSym

include("src/src.jl")

Threads.threadpoolsize(:default)
Threads.threadpoolsize(:interactive)

function sample!(o, f, samples)
    Threads.@threads :static for i in axes(o,2)
       #println(f(@view(samples[:,i]))[1][:,1])
        #o[:,i] = f(@view(samples[:,i]))[1][:,1]
        f(@view(o[i]), @view(samples[:,i]))


    end
end

function make_outer_params(num_qubits)
    data = [Symbolics.variable(Symbol("x$i")) for i in 1:2*num_qubits]
    weights = [Symbolics.variable(Symbol("η$i")) for i in 1:2*num_qubits]
    var = Symbolics.variable(:σ)
    return data, weights, var
end


function main(;num_qubits::Int, num_layers::Int, num_samples::Int, cse::Bool, parallel)

    data_params, weight_params, var = make_outer_params(num_qubits)

    qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
    qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)

    #################
    # Build circuit
    #################
    num_p_gates_forward, num_p_gates_backward = add_circuit_zzfeaturemap!(qc, qreg, num_layers, 1; for_back_config=(true, true))
    #################
    # Build circuit
    #################

    steps_p, steps_zz = get_steps_of_p_and_zz_gates(qc)

    #################
    # Build expression tree
    #################
    t_build_exprtree = @timed exprtree = QCSym.Circuits.gcol2tree2(qc.gatecollection)
    exprtree_init = copy(exprtree)

    #################
    # Modify expression tree for derivative
    #################
    param_gates_P = qc.gatecollection.collections[QCSym.Gates.P_Gate]
    dict_to_subsP = Dict(pg.symbol => pg.matrix_alt for pg in param_gates_P)
    
    dict_to_subs_outer_pars = get_dict_to_subs_outer_pars(param_gates_P, steps_p, steps_zz, data_params, weight_params, num_qubits)
    
    dict_to_subsP = Dict(pg.symbol => Symbolics.substitute(pg.matrix_alt, dict_to_subs_outer_pars) for pg in param_gates_P)
    dict_to_subsP_syms = Dict(k => Symbolics.substitute(k, dict_to_subs_outer_pars) for (k, v) in dict_to_subsP)
    dict_to_subsP_mats = Dict(k => Symbolics.substitute(v, dict_to_subs_outer_pars) for (k, v) in dict_to_subsP)
    dict_to_subsP_remaining = Dict(v => dict_to_subsP_mats[k] for (k, v) in dict_to_subsP_syms)
    exprtree = SymbolicUtils.substitute(exprtree, dict_to_subs_outer_pars)
    t_derivative = @timed exprtree = QCSym.kron_derivative(exprtree, weight_params[1])

    t_convert_exprtree = @timed begin
    dict_to_subs_P_derv = Dict(Symbolics.Differential(v.args[1])(v) => StaticArrays.SMatrix{2,2}(Symbolics.derivative(dict_to_subsP_mats[k], v.args[1])) for (k,v) in dict_to_subsP_syms)

    exprtree = SymbolicUtils.substitute(exprtree, dict_to_subs_P_derv)
    exprtree_deriv = copy(exprtree)
    exprtree = SymbolicUtils.substitute(exprtree, dict_to_subsP_remaining)


    #################
    # Modify expression tree substitute non-differentiated matrices
    #################
    vars = collect(v for v in Symbolics.get_variables(exprtree))

    vars_vals = Dict(v => get_val_for_sym(v, qc) for v in vars)
    vars_vals = Dict(k => v for (k,v) in vars_vals if !isnothing(v))

    exprtree_unsubs = copy(exprtree)
    exprtree = SymbolicUtils.substitute(exprtree, vars_vals)

    params = collect(v for v in Symbolics.get_variables(exprtree))
    end

    #################
    # Prepare sampling
    #################
    num_samples = num_samples
    inputs = rand(length(params), num_samples)
    #outputs = StaticArrays.MMatrix{2^num_qubits, num_samples, ComplexF64}(undef)
    #outputs = StaticArrays.SizedMatrix{2^num_qubits, num_samples, ComplexF64}(zeros(ComplexF64, 2^num_qubits, num_samples))
    #outputs = zeros(ComplexF64, 2^num_qubits, num_samples)
    #outputs = Vector{StaticArrays.MMatrix{2^num_qubits, 2^num_qubits, ComplexF64}}(undef, num_samples)
    outputs = Vector{Matrix{ComplexF64}}(undef, num_samples)
    for i in 1:num_samples
        outputs[i] = zeros(ComplexF64, 2^num_qubits, 2^num_qubits)
    end


    #################
    # Build functions
    #################
    ef = Symbolics.build_function([exprtree,], params; cse=cse, parallel=parallel, expression=Val{true}, checkbounds=false, nanmath=false, force_SA=true, convert_oop=false)

    t_code_generation = @timed bf = Symbolics.build_function([exprtree,], params; cse=cse, parallel=parallel, expression=Val{false}, checkbounds=false, nanmath=false, force_SA=true, convert_oop=false)

    #################
    # Sample
    #################
    t_sampling = @timed sample!(outputs, bf[2], inputs)
    return Dict("build_exprtree" => t_build_exprtree, "derivative" => t_derivative, "convert_exprtree" => t_convert_exprtree, "code_generation" => t_code_generation, "sampling" => t_sampling,
    "bf" => bf, "ef" => ef, "exprtree" => exprtree, "params" => params, "exprtree_init" => exprtree_init, "exprtree_unsubs" => exprtree_unsubs, "exprtree_deriv" => exprtree_deriv, "dict_to_subs_P_derv" => dict_to_subs_P_derv)
    #return ("build_exprtree" => t_build_exprtree, "code_generation" => t_code_generation, "sampling" => t_sampling)
end

out = main(num_qubits=7, num_layers=4, num_samples=100000, cse=true, parallel=Symbolics.SerialForm()); println(out["sampling"])
out = main(num_qubits=4, num_layers=1, num_samples=100000, cse=true, parallel=Symbolics.SerialForm()); println(out["sampling"])
out = main(num_qubits=4, num_layers=1, num_samples=100000, cse=true, parallel=Symbolics.SerialForm()); println(out["sampling"])
exit()
#MultithreadedForm
#out["convert_exprtree"]
out["sampling"]
# out["code_generation"]
# out["ef"]
# out["bf"]
out["exprtree"]
out["exprtree_init"]
# out["exprtree_deriv"]
# out["build_exprtree"]
pwd()
cd(pwd() * "/../QCSym.jl/examples/zz_feature_map")
open(pwd() * "/output.txt", "w") do io
    redirect_stdout(io) do
        println(@code_warntype(Symbolics.RuntimeGeneratedFunctions.generated_callfunc(out["bf"], rand(length(out["params"])))))
        #println(methods(*))
    end
end

@code_warntype(Symbolics.RuntimeGeneratedFunctions.generated_callfunc(out["bf"], rand(length(out["params"]))))