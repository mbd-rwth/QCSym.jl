# derivative_zz_feature_map.jl on the gate-level path: the circuit as a chain of GateOn
# factors (gcol2tree_chain_ket0), the same outer-parameter substitution, kron_derivative and
# two substitution loops, then lower_state and evolve! on LENS instead of build_function.
#
# Needs an environment with LENS.jl, e.g. the one of the LENS extension tests:
#   julia --project=test/LENSExt -t 8 examples/zz_feature_map/derivative_zz_feature_map_lens.jl
import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym
import LENS

include("src/src.jl")

function make_outer_params(num_qubits)
    data = [Symbolics.variable(Symbol("x$i")) for i in 1:2*num_qubits]
    weights = [Symbolics.variable(Symbol("η$i")) for i in 1:2*num_qubits]
    return data, weights
end

function main(; num_qubits::Int, num_layers::Int, num_samples::Int,
              backend=QCSym.Simulation.LENSBackend())
    data_params, weight_params = make_outer_params(num_qubits)

    qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
    qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)

    #################
    # Build circuit
    #################
    add_circuit_zzfeaturemap!(qc, qreg, num_layers, 1; for_back_config=(true, true))
    steps_p, steps_zz = get_steps_of_p_and_zz_gates(qc)

    #################
    # Build expression tree: one GateOn factor per gate, applied to |0…0⟩
    #################
    t_build_exprtree = @timed exprtree = QCSym.Circuits.gcol2tree_chain_ket0(qc)

    #################
    # Modify expression tree for derivative
    #################
    param_gates_P = qc.gatecollection.collections[QCSym.Gates.P_Gate]
    dict_to_subs_outer_pars = get_dict_to_subs_outer_pars(param_gates_P, steps_p, steps_zz, data_params, weight_params, num_qubits)

    dict_to_subsP_syms = Dict(pg.symbol => Symbolics.substitute(pg.symbol, dict_to_subs_outer_pars) for pg in param_gates_P)
    # Substituted once. derivative_zz_feature_map.jl substitutes the outer parameters into
    # matrix_alt a second time, after which Symbolics.derivative below fails on the zz-step
    # P gates, whose angle contains π ("Differentiation with array expressions is not yet
    # supported").
    dict_to_subsP_mats = Dict(pg.symbol => Symbolics.substitute(pg.matrix_alt, dict_to_subs_outer_pars) for pg in param_gates_P)
    dict_to_subsP_remaining = Dict(v => dict_to_subsP_mats[k] for (k, v) in dict_to_subsP_syms)
    exprtree = SymbolicUtils.substitute(exprtree, dict_to_subs_outer_pars)
    t_derivative = @timed dexprtree = QCSym.kron_derivative(exprtree, weight_params[1])

    t_convert_exprtree = @timed begin
        dict_to_subs_P_derv = Dict(Symbolics.Differential(v.args[1])(v) => StaticArrays.SMatrix{2,2}(Symbolics.derivative(dict_to_subsP_mats[k], v.args[1])) for (k,v) in dict_to_subsP_syms)
        dexprtree = SymbolicUtils.substitute(dexprtree, dict_to_subs_P_derv)
        dexprtree = SymbolicUtils.substitute(dexprtree, dict_to_subsP_remaining)
        exprtree = SymbolicUtils.substitute(exprtree, dict_to_subsP_remaining)
    end

    #################
    # Lower for simulation. The gate leaves still in the trees -- the H and CX gates, and the
    # P gates the second loop above misses -- are filled in from the circuit.
    #################
    params = vcat(data_params, weight_params)
    t_lowering = @timed begin
        lowered = QCSym.Simulation.lower_state(exprtree, params; gcol=qc)
        dlowered = QCSym.Simulation.lower_state(dexprtree, params; gcol=qc)
    end

    #################
    # Sample: ψ and dψ/dη1 for every column of `inputs`
    #################
    inputs = rand(length(params), num_samples)
    outputs = zeros(ComplexF64, 2^num_qubits, num_samples)
    doutputs = zeros(ComplexF64, 2^num_qubits, num_samples)
    # the first call compiles the generated functions, so time it on its own
    t_first_call = @timed QCSym.Simulation.evolve(dlowered, inputs[:, 1]; backend=backend)
    t_sampling = @timed begin
        QCSym.Simulation.evolve!(outputs, lowered, inputs; backend=backend)
        QCSym.Simulation.evolve!(doutputs, dlowered, inputs; backend=backend)
    end

    return Dict("build_exprtree" => t_build_exprtree, "derivative" => t_derivative,
                "convert_exprtree" => t_convert_exprtree, "lowering" => t_lowering,
                "first_call" => t_first_call, "sampling" => t_sampling,
                "lowered" => lowered, "dlowered" => dlowered, "params" => params,
                "inputs" => inputs, "outputs" => outputs, "doutputs" => doutputs)
end

out = main(num_qubits=4, num_layers=1, num_samples=1000)
println(out["dlowered"])
for k in ["build_exprtree", "derivative", "convert_exprtree", "lowering", "first_call", "sampling"]
    println(rpad(k, 17), round(out[k].time; digits=4), " s")
end

# the derivative of sample 1 against central finite differences in η1 through the reference
# backend
let l = out["lowered"], dl = out["dlowered"], p = out["inputs"][:, 1], h = 1e-6
    e = zeros(length(p)); e[length(p) ÷ 2 + 1] = 1
    fd = (QCSym.Simulation.evolve(l, p .+ h .* e) - QCSym.Simulation.evolve(l, p .- h .* e)) / (2h)
    println("|dψ - finite differences| = ", LinearAlgebra.norm(out["doutputs"][:, 1] - fd))
end
