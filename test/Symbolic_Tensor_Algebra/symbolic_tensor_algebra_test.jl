using Test

import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym


@testset verbose=true "Symbolic Tensor Algebra Tests" begin
    println("Running symbolic tensor algebra tests...")
    
    @testset verbose=true "Kronecker defined" begin
        @testset "Symbolic arrays" begin
            a, b = Symbolics.@variables a[1:2], b[1:2,1:2]
            @test_nowarn QCSym.:⊗(a, b)
        end

        @testset "QCSym gates" begin
            gate1 = QCSym.Gates.H_Gate
            gate2 = QCSym.Gates.X_Gate
            @test_nowarn QCSym.:⊗(gate1, gate2)
        end

        @testset "Kron of gates' symbols" begin
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            gate1 = QCSym.Gates.H_Gate
            gate2 = QCSym.Gates.X_Gate
            QCSym.Circuits.add_gate(qc, gate1, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            QCSym.Circuits.add_gate(qc, gate2, qubits_t=[qreg[2]], step=1, is_treat_numeric_only=false)
            symg1 = qc.gatecollection.collections[gate1][1].symbol
            symg2 = qc.gatecollection.collections[gate2][1].symbol
            @test_nowarn QCSym.:⊗(symg1, symg2)
        end
    end

    @testset verbose=true "⊙ runtime beyond 2x2" begin
        # The distributed tree only ever multiplies 2x2 factors, so ⊙ used to force its
        # result to MMatrix{2,2}. An undistributed tree multiplies whole step unitaries.
        a2 = ComplexF64[0 1; 1 0]
        a4 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        a8 = Matrix{ComplexF64}(LinearAlgebra.I, 8, 8); a8[1,8] = 2.0

        @test QCSym.:⊙(a2, a2) ≈ a2 * a2
        @test size(QCSym.:⊙(a4, a4)) == (4, 4)
        @test QCSym.:⊙(a4, a4) ≈ a4 * a4
        @test QCSym.:⊙(a8, a8) ≈ a8 * a8
        # n-ary chains (produced by _first_column and by undistributed step chains)
        @test QCSym.:⊙(a4, a4, a4) ≈ a4 * a4 * a4
        @test QCSym.:⊙(a2, a2, a2, a2) ≈ a2 * a2 * a2 * a2
    end

    @testset verbose=true "gcol2tree2 distribute=false" begin
        H  = ComplexF64[1 1; 1 -1] / sqrt(2)
        X  = ComplexF64[0 1; 1 0]
        I2 = ComplexF64[1 0; 0 1]
        P0 = ComplexF64[1 0; 0 0]
        P1 = ComplexF64[0 0; 0 1]

        # Independent dense reference: plain LinearAlgebra.kron, no QCSym involved.
        # Comparing the two builders against each other would hide a shared bug.
        function ref_unitary(nq, nl)
            U = Matrix{ComplexF64}(LinearAlgebra.I, 2^nq, 2^nq)
            for _ in 1:nl
                U = kron([H for _ in 1:nq]...) * U
                for c in 1:nq-1
                    cx = kron([i == c ? P0 : I2 for i in 1:nq]...) +
                         kron([i == c ? P1 : (i == c+1 ? X : I2) for i in 1:nq]...)
                    U = cx * U
                end
                U = kron([X for _ in 1:nq]...) * U
            end
            return U
        end

        function build(nq, nl)
            qc = QCSym.Circuits.QuantumCircuit(name="T")
            qreg = QCSym.Circuits.add_qreg(qc, "q", nq)
            step = 1
            for _ in 1:nl
                for i in 1:nq
                    QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]],
                                            step=step, is_treat_numeric_only=false)
                end
                step += 1
                for i in 1:nq-1
                    QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]],
                                            qubits_c=[qreg[i]], step=step,
                                            is_treat_numeric_only=false)
                    step += 1
                end
                for i in 1:nq
                    QCSym.Circuits.add_gate(qc, QCSym.Gates.X_Gate, qubits_t=[qreg[i]],
                                            step=step, is_treat_numeric_only=false)
                end
                step += 1
            end
            return qc
        end

        ident = StaticArrays.SMatrix{2,2,ComplexF64}([1 0; 0 1])
        function symvals(expr, qc)
            d = Dict()
            for v in Symbolics.get_variables(expr)
                g = QCSym.Circuits.s2g(v, qc.gatecollection)
                if g !== nothing && hasfield(typeof(g), :symbol) && isequal(v, g.symbol)
                    d[v] = g.matrix_numeric
                elseif isequal(v.name, :I)
                    # I_Gate_Filler's symbol is not in the gate collection
                    d[v] = ident
                end
            end
            return d
        end
        # substitute() only rewrites leaves; evaluating the ⊗/⊙ nodes is what
        # build_function's generated code does.
        function evalnode(e)
            if !SymbolicUtils.iscall(e)
                return SymbolicUtils.isconst(e) ? SymbolicUtils.unwrap_const(e) : e
            end
            op = SymbolicUtils.operation(e)
            return op([evalnode(a) for a in SymbolicUtils.arguments(e)]...)
        end
        evaltree(t, qc) = evalnode(SymbolicUtils.substitute(t, symvals(t, qc)))

        nnodes(ex) = SymbolicUtils.iscall(ex) ?
            1 + sum(nnodes, SymbolicUtils.arguments(ex); init=0) : 1

        @testset "matches dense reference, $(nq) qubits, $(nl) layers" for (nq, nl) in
                [(2,1), (2,2), (3,1), (3,2), (4,1)]
            qc = build(nq, nl)
            Uref = ref_unitary(nq, nl)

            td = QCSym.Circuits.gcol2tree2(qc.gatecollection; distribute=true)
            tu = QCSym.Circuits.gcol2tree2(qc.gatecollection; distribute=false)
            @test ComplexF64.(evaltree(td, qc)) ≈ Uref
            @test ComplexF64.(evaltree(tu, qc)) ≈ Uref

            cu = QCSym.Circuits.gcol2tree_col_1(qc.gatecollection; distribute=false)
            @test ComplexF64.(vec(collect(evaltree(cu, qc)))) ≈ Uref[:, 1]
        end

        @testset "lazy steps keep evaluation at 2^n" begin
            # Every case above sits at or below _MAX_STATIC_KRON_DIM, where ⊗ still
            # returns a dense kron. Past it a step must stay factored, or applying it to a
            # state vector costs 4^n.
            n = 6
            v = rand(ComplexF64, 2^n)
            fs1 = [rand(ComplexF64, 2, 2) for _ in 1:n]
            fs2 = [rand(ComplexF64, 2, 2) for _ in 1:n]
            K1 = QCSym.:⊗(fs1...)
            K2 = QCSym.:⊗(fs2...)
            d1 = kron(fs1...)
            d2 = kron(fs2...)

            @test K1 isa QCSym.LazyStep
            @test size(K1) == (2^n, 2^n)
            @test collect(K1) ≈ d1

            # the operation the whole design exists for
            @test QCSym.:⊙(K1, v) ≈ d1 * v
            @test QCSym.:⊙(K1, v) isa AbstractVector

            # a two-qubit gate's two branches: the sum must stay lazy, and applying it
            # must distribute over the terms rather than add two 2^n x 2^n matrices
            S = K1 + K2
            @test S isa QCSym.LazyStep
            @test collect(S) ≈ d1 + d2
            @test QCSym.:⊙(S, v) ≈ (d1 + d2) * v

            # composing two steps stays factored (all factors are 2x2, so the mixed
            # product rule applies exactly)
            @test QCSym.:⊙(K1, K2) isa QCSym.LazyStep
            @test collect(QCSym.:⊙(K1, K2)) ≈ d1 * d2
            @test QCSym.:⊙(QCSym.:⊙(K1, K2), v) ≈ d1 * d2 * v
            @test QCSym.:⊙(S, K2) isa QCSym.LazyStep
            @test QCSym.:⊙(S, K2) ≈ (d1 + d2) * d2
        end

        @testset "col_1 past the dense cap, $(nq) qubits" for nq in [5, 6]
            qc = build(nq, 1)
            Uref = ref_unitary(nq, 1)
            tu = QCSym.Circuits.gcol2tree2(qc.gatecollection; distribute=false)
            @test ComplexF64.(collect(evaltree(tu, qc))) ≈ Uref
            cu = QCSym.Circuits.gcol2tree_col_1(qc.gatecollection; distribute=false)
            col = evaltree(cu, qc)
            # the state vector itself, never a 2^n x 2^n intermediate
            @test col isa AbstractVector
            @test ComplexF64.(vec(collect(col))) ≈ Uref[:, 1]
        end

        @testset "fold_constants" begin
            qc = build(3, 2)
            Uref = ref_unitary(3, 2)
            tree = QCSym.Circuits.gcol2tree2(qc.gatecollection; distribute=false)
            sub = SymbolicUtils.substitute(tree, symvals(tree, qc))

            folded = QCSym.fold_constants(sub)
            # this circuit has no free parameters at all, so the whole tree folds to one
            # value -- and it has to be the right one
            @test QCSym.count_nodes(folded) < QCSym.count_nodes(sub)
            @test ComplexF64.(collect(QCSym.eval_numeric(folded))) ≈ Uref
            @test ComplexF64.(collect(QCSym.eval_numeric(sub))) ≈ Uref

            # ... and must not fold a leaf whose *entries* are symbolic. Folding one of
            # those would bake a parameter value into the generated function, which is
            # silent and wrong rather than an error.
            θ = Symbolics.variable(:θ)
            M = StaticArrays.SMatrix{2,2}([θ 0; 0 1])
            N = StaticArrays.SMatrix{2,2,ComplexF64}([0 1; 1 0])
            param_tree = QCSym.:⊗(SymbolicUtils.Const{SymbolicUtils.SymReal}(M),
                                  SymbolicUtils.Const{SymbolicUtils.SymReal}(N))
            @test QCSym._leaf_has_vars(SymbolicUtils.Const{SymbolicUtils.SymReal}(M))
            @test !QCSym._leaf_has_vars(SymbolicUtils.Const{SymbolicUtils.SymReal}(N))
            @test SymbolicUtils.iscall(QCSym.fold_constants(param_tree))
        end

        @testset "tree size is linear in depth" begin
            # The point of distribute=false: node count must grow with the circuit, not
            # with 2^(number of two-qubit gates). Doubling the layers must not do more
            # than roughly double the tree.
            n1 = nnodes(QCSym.Circuits.gcol2tree2(build(3, 1).gatecollection; distribute=false))
            n2 = nnodes(QCSym.Circuits.gcol2tree2(build(3, 2).gatecollection; distribute=false))
            n4 = nnodes(QCSym.Circuits.gcol2tree2(build(3, 4).gatecollection; distribute=false))
            @test n2 <= 2.2 * n1
            @test n4 <= 2.2 * n2

            # ... whereas the distributed builder is exponential in the same sweep, which
            # is what makes it unusable at depth.
            d1 = nnodes(QCSym.Circuits.gcol2tree2(build(3, 1).gatecollection; distribute=true))
            d2 = nnodes(QCSym.Circuits.gcol2tree2(build(3, 2).gatecollection; distribute=true))
            @test d2 > 4 * d1
        end
    end
end
