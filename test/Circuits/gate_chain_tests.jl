using Test

import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym

isdefined(Main, :ZZHelpers) || include(joinpath(@__DIR__, "..", "Simulation", "zz_helpers.jl"))
import .ZZHelpers

@testset verbose=true "Gate chain Tests" begin
    println("Running gate chain tests...")
    C = QCSym.Circuits
    G = QCSym.Gates

    # numeric gate matrices for every gate leaf, then the tree evaluated
    evaltree(t, qc) = QCSym.eval_numeric(SymbolicUtils.substitute(t, ZZHelpers.numeric_gate_values(t, qc)))
    basis(n, bits) = (ψ = zeros(ComplexF64, 2^n); ψ[foldl((a, b) -> 2a + b, bits; init=0) + 1] = 1; ψ)

    @testset "GateOn term" begin
        θ = Symbolics.variable(:θ)
        A = Symbolics.variable(:A)
        g = QCSym.GateOn(3, (3, 1))
        @test g.qubits == (3, 1)
        @test_throws ArgumentError QCSym.GateOn(3, (1, 4))
        @test_throws ArgumentError QCSym.GateOn(3, (2, 2))
        @test_throws ArgumentError QCSym.GateOn(3, ())

        t = g(A)
        @test SymbolicUtils.operation(t) === g
        @test string(t) == "GateOn[3,1](A)"
        # hash-consed: the same placement and matrix give the same object
        @test QCSym.GateOn(3, (3, 1))(A) === t
        @test isequal(collect(Symbolics.get_variables(t)), [Symbolics.unwrap(A)])

        # substitute and kron_derivative only touch the matrix; the placement is kept
        M = SymbolicUtils.Const{SymbolicUtils.SymReal}(StaticArrays.SMatrix{2,2}([θ 0; 0 1]))
        s = SymbolicUtils.substitute(t, Dict(A => M))
        @test SymbolicUtils.operation(s) === g
        dθ = QCSym.kron_derivative(QCSym.GateOn(3, (2,))(θ * Symbolics.unwrap(A)), θ)
        @test SymbolicUtils.operation(Symbolics.unwrap(dθ)) === QCSym.GateOn(3, (2,))
        @test isequal(only(SymbolicUtils.arguments(Symbolics.unwrap(dθ))), Symbolics.unwrap(A))

        # a numeric matrix gives a lazy step instead of a term; a symbolic one stays a term
        @test g(rand(ComplexF64, 4, 4)) isa QCSym.GateStep
        @test SymbolicUtils.iscall(g(StaticArrays.SMatrix{4,4}(fill(θ, 4, 4))))
        @test_throws DimensionMismatch g(rand(ComplexF64, 2, 2))
    end

    @testset "gate kernel on $(n) qubits, qubits $(qs)" for (n, qs) in
            [(4, (1,)), (4, (4,)), (4, (2, 3)), (4, (3, 2)), (4, (1, 4)), (4, (4, 1)),
             (4, (2, 4, 1)), (4, (3, 1, 2)), (3, (1, 2, 3))]
        M = rand(ComplexF64, 2^length(qs), 2^length(qs))
        ψ = rand(ComplexF64, 2^n)
        D = ZZHelpers.dense_embed(M, qs, n)
        S = QCSym.GateOn(n, qs)(M)
        @test QCSym._apply_gate!(copy(ψ), M, qs, n) ≈ D * ψ
        @test QCSym.:⊙(S, ψ) ≈ D * ψ
        @test collect(S) ≈ D
    end

    @testset "dense reference by kron" begin
        # adjacent qubits in order are plain kron products
        M = rand(ComplexF64, 4, 4)
        I2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        @test ZZHelpers.dense_embed(M, (2, 3), 4) ≈ kron(I2, M, I2)
        @test collect(QCSym.GateOn(4, (2, 3))(M)) ≈ kron(I2, M, I2)
    end

    @testset "CCX gate" begin
        qc = C.QuantumCircuit(name="ccx")
        qr = C.add_qreg(qc, "q", 3)
        C.add_gate(qc, G.CCX_Gate, qubits_t=[qr[1]], qubits_c=[qr[2], qr[3]], step=1)
        g = only(qc.gatecollection.collections[G.CCX_Gate])
        @test g.num_qubits == 3
        @test_throws ArgumentError C.add_gate(qc, G.CCX_Gate, qubits_t=[qr[1]], qubits_c=[qr[2]], step=2)
        # truth table: the target flips exactly when both controls are 1
        U = evaltree(C.gcol2tree_chain(qc), qc)
        for b in 0:7
            bits = [(b >> 2) & 1, (b >> 1) & 1, b & 1]
            out = copy(bits)
            bits[2] == 1 && bits[3] == 1 && (out[1] = 1 - out[1])
            @test U * basis(3, bits) ≈ basis(3, out)
        end
        # gcol2tree2 cannot represent it, and says so instead of dropping it
        @test_throws ErrorException C.gcol2tree2(qc.gatecollection; distribute=false)
    end

    @testset "gcol2tree_chain on known circuits" begin
        # X on 1 and 3, CCX(1,3 -> 5), CX(5 -> 2) with the control below the target, CZ(2, 5),
        # SWAP(3, 6); qubit 4 carries no gate
        qc = C.QuantumCircuit(name="known")
        qr = C.add_qreg(qc, "q", 6)
        C.add_gate(qc, G.X_Gate, qubits_t=[qr[1]], step=1, is_treat_numeric_only=false)
        C.add_gate(qc, G.X_Gate, qubits_t=[qr[3]], step=1, is_treat_numeric_only=false)
        C.add_gate(qc, G.CCX_Gate, qubits_t=[qr[5]], qubits_c=[qr[1], qr[3]], step=2)
        C.add_gate(qc, G.CX_Gate, qubits_t=[qr[2]], qubits_c=[qr[5]], step=3)
        C.add_gate(qc, G.CZ_Gate, qubits_t=[qr[5]], qubits_c=[qr[2]], step=4, is_treat_numeric_only=false)
        C.add_gate(qc, G.SWAP_Gate, qubits_t=[qr[3], qr[6]], step=5, is_treat_numeric_only=false)

        ψ = C.gcol2tree_chain_ket0(qc)
        v = evaltree(ψ, qc)
        # |000000> -> X1 X3 -> |101000> -> CCX -> |101010> -> CX(5->2) -> |111010>
        # -> CZ(2,5) gives -1 -> SWAP(3,6) -> -|110011>
        @test v isa AbstractVector
        @test v ≈ -basis(6, [1, 1, 0, 0, 1, 1])
        # one factor per gate, register size from the circuit despite the idle qubit
        U = C.gcol2tree_chain(qc)
        @test QCSym.count_nodes(U) == 2 * 6 + 5
        @test SymbolicUtils.operation(first(SymbolicUtils.arguments(U))) == QCSym.GateOn(6, (6, 3))
        @test evaltree(U, qc)[:, 1] ≈ v
        # counting the qubits the gates touch would make the register too small
        @test_throws ArgumentError C.gcol2tree_chain(qc.gatecollection)
    end

    @testset "gates of one step must not overlap" begin
        qc = C.QuantumCircuit(name="overlap")
        qr = C.add_qreg(qc, "q", 2)
        C.add_gate(qc, G.H_Gate, qubits_t=[qr[1]], step=1, is_treat_numeric_only=false)
        C.add_gate(qc, G.CX_Gate, qubits_t=[qr[2]], qubits_c=[qr[1]], step=1)
        @test_throws ErrorException C.gcol2tree_chain(qc)
    end

    @testset "matches gcol2tree2, $(nq) qubits, $(nl) layers" for (nq, nl) in [(2, 1), (3, 2), (4, 1)]
        qc, x, η, outer = ZZHelpers.zz_circuit(nq, nl)
        params = vcat(x, η)
        vals = Dict(Symbolics.unwrap.(params) .=> rand(length(params)))
        numeric(t) = QCSym.eval_numeric(SymbolicUtils.substitute(t, vals))
        # U and dU/dη1 through both builders, with the one-pass substitution
        tc, dtc = ZZHelpers.derivative_one_pass(C.gcol2tree_chain(qc), qc, outer, η[1])
        tk, dtk = ZZHelpers.derivative_one_pass(C.gcol2tree2(qc.gatecollection; distribute=false),
                                                qc, outer, η[1])
        # gcol2tree2's untouched qubit lines carry I_Gate_Filler's reserved `I` symbol
        ident = StaticArrays.SMatrix{2,2,ComplexF64}([1 0; 0 1])
        fill_I(t) = SymbolicUtils.substitute(t, Dict(v => ident for v in Symbolics.get_variables(t)
                                                     if Symbolics.getname(v) === :I))
        @test numeric(tc) ≈ collect(numeric(fill_I(tk)))
        @test numeric(dtc) ≈ collect(numeric(fill_I(dtk)))
    end

    @testset "deep chain, $(m) gates" for m in [300, 2000]
        # kron_derivative used to re-walk the rest of a chain at every level, O(m^2) for m
        # gates (7.6 s for 2000); now it is linear, about 0.1 s.
        qc = C.QuantumCircuit(name="deep")
        qr = C.add_qreg(qc, "q", 4)
        for s in 1:m
            C.add_gate(qc, G.P_Gate, qubits_t=[qr[mod1(s, 4)]], step=s, is_treat_numeric_only=false)
        end
        η = Symbolics.variable(:η)
        outer = Dict(g.parameters["λ"]["sym"] => exp(1im * η * k)
                     for (k, g) in enumerate(qc.gatecollection.collections[G.P_Gate]))
        t = SymbolicUtils.substitute(C.gcol2tree_chain_ket0(qc), outer)
        QCSym.kron_derivative(t, η)   # compile
        elapsed = @elapsed dt = QCSym.kron_derivative(t, η)
        @test elapsed < 5
        if m <= 300
            # one summand per gate, as every gate depends on η; each summand is a full chain,
            # which is why the long chain is not lowered here
            derv, remaining = ZZHelpers.p_leaf_dicts(qc, outer)
            dt = SymbolicUtils.substitute(dt, merge(derv, remaining))
            @test length(QCSym.Simulation.lower_state(dt, [η]).summands) == m
        end
    end
end
