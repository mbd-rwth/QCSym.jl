using Test

import Symbolics
import SymbolicUtils
import StaticArrays
import QCSym

# A circuit whose tree exercises every operation the loop code generator has to
# handle: parameterised 2x2 gates (`array_literal` over `sin`/`cos`/`complex`),
# per-qubit chains of `⊙`, `⊗` across qubits and `+` from differentiation.
function _codegen_test_circuit(num_qubits, num_layers)
    qc = QCSym.Circuits.QuantumCircuit(name="CodegenTestCircuit")
    qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)
    step = 1
    for _ in 1:num_layers
        for i in 1:num_qubits
            QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]],
                step=step, is_treat_numeric_only=false)
        end
        step += 1
        for i in 1:num_qubits
            QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i]],
                step=step, is_treat_numeric_only=false, param_values=Dict("λ" => 0.0))
        end
        step += 1
        for i in 1:num_qubits-1
            QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]],
                qubits_c=[qreg[i]], step=step, is_treat_numeric_only=false)
            step += 1
        end
    end
    return qc
end

# Substitute a data-dependent phase into every P gate, differentiate with respect to
# the weight, then resolve the remaining gate symbols to their numeric matrices --
# the same sequence the zz_feature_map example goes through.
function _codegen_test_tree(num_qubits, num_layers)
    qc = _codegen_test_circuit(num_qubits, num_layers)
    data = [Symbolics.variable(Symbol("x$i")) for i in 1:num_qubits]
    w = Symbolics.variable(:w)

    exprtree = QCSym.Circuits.gcol2tree2(qc.gatecollection)
    pgates = qc.gatecollection.collections[QCSym.Gates.P_Gate]

    subs_pars = Dict()
    for pg in pgates
        qid = pg.qubits_t[1].index_global
        subs_pars[pg.parameters["λ"]["sym"]] = exp(1im * w * data[qid])
    end

    syms = Dict(pg.symbol => Symbolics.substitute(pg.symbol, subs_pars) for pg in pgates)
    mats = Dict(pg.symbol => Symbolics.substitute(
        Symbolics.substitute(pg.matrix_alt, subs_pars), subs_pars) for pg in pgates)

    exprtree = SymbolicUtils.substitute(exprtree, subs_pars)
    exprtree = QCSym.kron_derivative(exprtree, w)

    derv = Dict(Symbolics.Differential(v.args[1])(v) =>
        StaticArrays.SMatrix{2,2}(Symbolics.derivative(mats[k], v.args[1]))
        for (k, v) in syms)
    exprtree = SymbolicUtils.substitute(exprtree, derv)
    exprtree = SymbolicUtils.substitute(exprtree, Dict(v => mats[k] for (k, v) in syms))

    vars = collect(Symbolics.get_variables(exprtree))
    vals = Dict{Any,Any}()
    for v in vars
        g = QCSym.Circuits.s2g(v, qc.gatecollection)
        if !isnothing(g) && hasfield(typeof(g), :symbol) && isequal(v, g.symbol)
            vals[v] = g.matrix_numeric
        elseif isequal(Symbolics.getname(v), :I)
            vals[v] = StaticArrays.SMatrix{2,2}([1 0; 0 1])
        end
    end
    exprtree = SymbolicUtils.substitute(exprtree, vals)

    return exprtree, collect(Symbolics.get_variables(exprtree))
end

@testset verbose = true "Loop Codegen Tests" begin
    println("Running loop codegen tests...")

    # Single-qubit circuits are left out: their tree contains a one-argument `⊗`,
    # which `Symbolic_Tensor_Algebra` does not define, so neither code generator can
    # evaluate them.
    @testset "matches build_function on $(nq) qubits, $(nl) layers" for (nq, nl) in
                                                                        ((2, 1), (2, 2), (3, 1), (3, 2))
        exprtree, params = _codegen_test_tree(nq, nl)
        @test !isempty(params)

        looped = QCSym.Codegen.build_function_loops(exprtree, params)
        reference = Symbolics.build_function(exprtree, params; cse=true,
            expression=Val{false}, checkbounds=false, nanmath=false, force_SA=true,
            convert_oop=false)

        for _ in 1:5
            p = rand(length(params))
            got = Matrix{ComplexF64}(looped(p))
            want = Matrix{ComplexF64}(reference(p))
            @test size(got) == (2^nq, 2^nq)
            @test got ≈ want
        end
    end

    @testset "emits loops rather than straight-line code" begin
        exprtree, params = _codegen_test_tree(3, 2)
        looped = QCSym.Codegen.build_function_loops(exprtree, params)
        # the whole point: far fewer emitted loops than DAG nodes
        @test looped.nloops < looped.nnodes / 4
    end

    @testset "loop count grows much slower than the DAG" begin
        small = QCSym.Codegen.build_function_loops(_codegen_test_tree(2, 2)...)
        large = QCSym.Codegen.build_function_loops(_codegen_test_tree(4, 2)...)
        @test large.nnodes > small.nnodes
        # the amount of code Julia has to compile must not track circuit size
        @test large.nloops / small.nloops < large.nnodes / small.nnodes
    end

    @testset "rejects symbols that are not parameters" begin
        exprtree, params = _codegen_test_tree(2, 1)
        @test_throws ArgumentError QCSym.Codegen.build_function_loops(exprtree, params[2:end])
    end
end
