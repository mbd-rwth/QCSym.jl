using Test

import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym

isdefined(Main, :ZZHelpers) || include(joinpath(@__DIR__, "zz_helpers.jl"))
import .ZZHelpers

@testset verbose=true "Simulation Tests" begin
    println("Running simulation tests...")
    C = QCSym.Circuits
    G = QCSym.Gates
    S = QCSym.Simulation

    # Independent dense reference: the product of the gates' dense embeddings (kron and a
    # qubit permutation, no QCSym kernel), in the gcol2tree_chain placement.
    function dense_unitary(qc)
        n = C.get_num_qubits(qc)
        U = Matrix{ComplexF64}(LinearAlgebra.I, 2^n, 2^n)
        stepwise = C._extract_gates_stepwise(qc.gatecollection)
        for s in sort(collect(keys(stepwise))), g in stepwise[s]
            U = ZZHelpers.dense_embed(g.matrix_numeric, C._chain_qubits(g), n) * U
        end
        return U
    end

    # H, CX in both directions, CZ, SWAP and CCX on non-adjacent qubits; qubit 3 idle
    function mixed_circuit()
        qc = C.QuantumCircuit(name="mixed")
        qr = C.add_qreg(qc, "q", 5)
        for q in [1, 2, 4, 5]
            C.add_gate(qc, G.H_Gate, qubits_t=[qr[q]], step=1, is_treat_numeric_only=false)
        end
        C.add_gate(qc, G.CX_Gate, qubits_t=[qr[5]], qubits_c=[qr[1]], step=2)
        C.add_gate(qc, G.CX_Gate, qubits_t=[qr[2]], qubits_c=[qr[4]], step=2)
        C.add_gate(qc, G.CZ_Gate, qubits_t=[qr[1]], qubits_c=[qr[5]], step=3, is_treat_numeric_only=false)
        C.add_gate(qc, G.SWAP_Gate, qubits_t=[qr[2], qr[5]], step=4, is_treat_numeric_only=false)
        C.add_gate(qc, G.CCX_Gate, qubits_t=[qr[1]], qubits_c=[qr[5], qr[2]], step=5)
        C.add_gate(qc, G.H_Gate, qubits_t=[qr[4]], step=5, is_treat_numeric_only=false)
        return qc
    end

    @testset "a constant circuit from |0…0⟩" begin
        qc = mixed_circuit()
        l = S.lower_state(C.gcol2tree_chain_ket0(qc), []; gcol=qc)
        @test length(l.summands) == 1
        @test l.summands[1].start != 0
        @test !S._reads_input(l)
        @test isempty(l.param_slots)
        @test S.evolve(l, Float64[]) ≈ dense_unitary(qc)[:, 1]
        # a matrix of parameter sets: here every column is the same state
        @test S.evolve(l, zeros(0, 3)) ≈ repeat(dense_unitary(qc)[:, 1], 1, 3)
    end

    @testset "an operator applied to given states" begin
        qc = mixed_circuit()
        U = dense_unitary(qc)
        l = S.lower_state(C.gcol2tree_chain(qc), []; gcol=qc)
        @test S._reads_input(l)
        ψ0 = LinearAlgebra.normalize!(rand(ComplexF64, 32))
        ψ = copy(ψ0)
        @test S.evolve!(ψ, l, Float64[]) === ψ
        @test ψ ≈ U * ψ0
        @test S.evolve(ψ0, l, Float64[]) ≈ U * ψ0
        # every column starts from its own state
        Ψ0 = rand(ComplexF64, 32, 7)
        @test S.evolve!(copy(Ψ0), l, zeros(0, 7); ntasks=3) ≈ U * Ψ0
        @test S.evolve(ψ0, l, zeros(0, 4)) ≈ repeat(U * ψ0, 1, 4)
        # without an initial state there is nothing to evolve
        @test_throws ArgumentError S.evolve(l, Float64[])
        @test_throws DimensionMismatch S.evolve!(zeros(ComplexF64, 16), l, Float64[])
    end

    @testset "derivative of a zz map, $(nq) qubits, $(nl) layers" for (nq, nl) in [(2, 1), (3, 1), (3, 2)]
        qc, x, η, outer = ZZHelpers.zz_circuit(nq, nl)
        params = vcat(x, η)
        ψ, dψ = ZZHelpers.derivative_two_loops(C.gcol2tree_chain_ket0(qc), qc, outer, η[1])
        lψ = S.lower_state(ψ, params; gcol=qc)
        ldψ = S.lower_state(dψ, params; gcol=qc)
        # one summand per gate that depends on η1: the P layer, nq gates per layer
        @test length(ldψ.summands) == nq * nl
        @test length(lψ.summands) == 1

        p = rand(length(params))
        @test LinearAlgebra.norm(S.evolve(lψ, p)) ≈ 1
        # against finite differences of the state
        h = 1e-6
        e = zeros(length(params)); e[nq + 1] = 1
        fd = (S.evolve(lψ, p .+ h .* e) - S.evolve(lψ, p .- h .* e)) / (2h)
        @test isapprox(S.evolve(ldψ, p), fd; atol=1e-8)

        # the leftover P leaves of the second loop are filled in from the gate collection and
        # give what a single substitution pass gives, evaluated as a whole tree
        ψ1, dψ1 = ZZHelpers.derivative_one_pass(C.gcol2tree_chain_ket0(qc), qc, outer, η[1])
        vals = Dict(Symbolics.unwrap.(params) .=> p)
        @test S.evolve(ldψ, p) ≈ QCSym.eval_numeric(SymbolicUtils.substitute(dψ1, vals))
        @test S.evolve(ldψ, p) ≈ S.evolve(S.lower_state(dψ1, params), p)

        # parameter sets as columns, split over tasks
        P = rand(length(params), 6)
        out = S.evolve(ldψ, P; ntasks=4)
        @test out ≈ S.evolve(ldψ, P; ntasks=1)
        @test out[:, 5] ≈ S.evolve(ldψ, P[:, 5])
    end

    @testset "leftover gate leaves" begin
        qc, x, η, outer = ZZHelpers.zz_circuit(2, 1)
        params = vcat(x, η)
        _, dψ = ZZHelpers.derivative_two_loops(C.gcol2tree_chain_ket0(qc), qc, outer, η[1])
        dψ = SymbolicUtils.substitute(dψ, ZZHelpers.numeric_gate_values(dψ, qc))
        # the second loop missed the P gate that does not depend on η1 ...
        err = try S.lower_state(dψ, params); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("P_s", sprint(showerror, err))
        # ... which only the gate collection can fill in; a missing parameter stays an error
        @test S.lower_state(dψ, params; gcol=qc) isa S.LoweredState
        @test_throws ArgumentError S.lower_state(dψ, x; gcol=qc)
        # an unsubstituted derivative cannot be filled in
        dψraw = QCSym.kron_derivative(SymbolicUtils.substitute(C.gcol2tree_chain_ket0(qc), outer), η[1])
        @test_throws ArgumentError S.lower_state(dψraw, params; gcol=qc)
    end

    @testset "coefficients" begin
        # SymbolicUtils' `*` cannot multiply a state term, so coefficients come in as explicit
        # terms, e.g. from rewriting
        qc = mixed_circuit()
        ψ = C.gcol2tree_chain_ket0(qc)
        α = Symbolics.variable(:α)
        mul(c, t) = SymbolicUtils.term(*, c, t; type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
        t = SymbolicUtils.term(+, mul(Symbolics.unwrap(α), ψ), mul(2, ψ),
                               SymbolicUtils.term(/, ψ, Symbolics.unwrap(α); type=SymbolicUtils.SymReal,
                                                  shape=(1:2, 1:2));
                               type=SymbolicUtils.SymReal, shape=(1:2, 1:2))
        l = S.lower_state(t, [α]; gcol=qc)
        @test length(l.summands) == 3
        @test S.evolve(l, [0.5]) ≈ (0.5 + 2 + 2) * dense_unitary(qc)[:, 1]
    end

    @testset "what cannot be lowered" begin
        qc = mixed_circuit()
        # a gcol2tree2 tree has no GateOn factors
        qc2, _, _, _ = ZZHelpers.zz_circuit(2, 1)
        @test_throws ArgumentError S.lower_state(C.gcol2tree2(qc2.gatecollection; distribute=false), [])
        # a symbol that is not a parameter
        θ = Symbolics.variable(:θ)
        t = QCSym.:⊙(QCSym.GateOn(2, (1,))(θ), QCSym.ket0(2))
        @test_throws ArgumentError S.lower_state(t, [])
        @test S.lower_state(t, [θ]) isa S.LoweredState
        # a state as the left factor of ⊙
        ψ = C.gcol2tree_chain_ket0(qc)
        @test_throws ArgumentError S.lower_state(QCSym.:⊙(ψ, ψ), []; gcol=qc)
        # gates of different register sizes
        mixed = QCSym.:⊙(QCSym.GateOn(3, (1,))(θ), QCSym.ket0(2))
        @test_throws ArgumentError S.lower_state(mixed, [θ])
    end

    @testset "fold_constants keeps the gates" begin
        qc = mixed_circuit()
        ψ = C.gcol2tree_chain_ket0(qc)
        ψn = SymbolicUtils.substitute(ψ, ZZHelpers.numeric_gate_values(ψ, qc))
        folded = QCSym.fold_constants(ψn)
        @test QCSym.count_nodes(folded) == QCSym.count_nodes(ψn)
        @test S.evolve(S.lower_state(folded, []), Float64[]) ≈ dense_unitary(qc)[:, 1]
        @test_throws ArgumentError QCSym.Codegen.build_function_loops(ψn, [])
    end

    @testset "LENSBackend without LENS" begin
        if Base.get_extension(QCSym, :QCSymLENSExt) === nothing
            qc = mixed_circuit()
            l = S.lower_state(C.gcol2tree_chain_ket0(qc), []; gcol=qc)
            err = try S.evolve(l, Float64[]; backend=S.LENSBackend()); nothing catch e; e end
            @test occursin("using LENS", sprint(showerror, err))
        else
            @test_skip "LENS is loaded"
        end
    end
end
