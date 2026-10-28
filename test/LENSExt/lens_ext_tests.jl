using Test

import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import Random
import QCSym
import LENS

isdefined(Main, :ZZHelpers) || include(joinpath(@__DIR__, "..", "Simulation", "zz_helpers.jl"))
import .ZZHelpers

# Run in the environment next to this file, which provides LENS:
#   julia --project=test/LENSExt test/testrunner.jl lens_ext
@testset verbose=true "LENS extension Tests" begin
    println("Running LENS extension tests...")
    C = QCSym.Circuits
    G = QCSym.Gates
    S = QCSym.Simulation
    Random.seed!(1)

    @test Base.get_extension(QCSym, :QCSymLENSExt) !== nothing

    backends = [S.LENSBackend(), S.LENSBackend(fuse=false), S.LENSBackend(atol=1e-12)]

    @testset "random circuits from given states, $(n) qubits" for n in [3, 5, 7]
        # random 1-, 2- and 3-qubit gates on random (also non-adjacent, reversed) qubits; the
        # non-unitary ones are what a derivative or an integral hands to the simulator
        terms = Any[]
        for _ in 1:4n
            k = rand(1:min(3, n))
            qs = Random.randperm(n)[1:k]
            M = SymbolicUtils.Const{SymbolicUtils.SymReal}(rand(ComplexF64, 2^k, 2^k))
            push!(terms, QCSym.GateOn(n, qs)(M))
        end
        U = foldl((acc, g) -> QCSym.:⊙(g, acc), terms[2:end]; init=terms[1])
        l = S.lower_state(U, [])
        Ψ0 = rand(ComplexF64, 2^n, 5)
        ref = S.evolve!(copy(Ψ0), l, zeros(0, 5))
        @test ref ≈ QCSym.eval_numeric(U) * Ψ0
        for b in backends
            @test S.evolve!(copy(Ψ0), l, zeros(0, 5); backend=b, ntasks=2) ≈ ref
        end
    end

    @testset "CCX and friends from |0…0⟩" begin
        qc = C.QuantumCircuit(name="lens")
        qr = C.add_qreg(qc, "q", 4)
        for q in 1:4
            C.add_gate(qc, G.H_Gate, qubits_t=[qr[q]], step=1, is_treat_numeric_only=false)
        end
        C.add_gate(qc, G.CCX_Gate, qubits_t=[qr[2]], qubits_c=[qr[4], qr[1]], step=2)
        C.add_gate(qc, G.CZ_Gate, qubits_t=[qr[3]], qubits_c=[qr[2]], step=3, is_treat_numeric_only=false)
        C.add_gate(qc, G.SWAP_Gate, qubits_t=[qr[1], qr[4]], step=4, is_treat_numeric_only=false)
        C.add_gate(qc, G.CX_Gate, qubits_t=[qr[1]], qubits_c=[qr[3]], step=5)
        l = S.lower_state(C.gcol2tree_chain_ket0(qc), []; gcol=qc)
        ref = S.evolve(l, Float64[])
        for b in backends
            @test S.evolve(l, Float64[]; backend=b) ≈ ref
        end
    end

    @testset "zz map and its derivative, $(nq) qubits, $(nl) layers" for (nq, nl) in [(3, 1), (4, 2)]
        qc, x, η, outer = ZZHelpers.zz_circuit(nq, nl)
        params = vcat(x, η)
        # the operator, so that every parameter set starts from its own state
        U, dU = ZZHelpers.derivative_two_loops(C.gcol2tree_chain(qc), qc, outer, η[2])
        lU = S.lower_state(U, params; gcol=qc)
        ldU = S.lower_state(dU, params; gcol=qc)
        @test length(ldU.summands) == nl

        P = rand(length(params), 8)
        Ψ0 = rand(ComplexF64, 2^nq, 8)
        for b in backends
            @test S.evolve!(copy(Ψ0), lU, P; backend=b, ntasks=3) ≈ S.evolve!(copy(Ψ0), lU, P)
            @test S.evolve!(copy(Ψ0), ldU, P; backend=b, ntasks=3) ≈ S.evolve!(copy(Ψ0), ldU, P)
        end

        # finite differences, through LENS
        h = 1e-6
        e = zeros(length(params)); e[nq + 2] = 1
        p, ψ0 = P[:, 1], Ψ0[:, 1]
        lens = S.LENSBackend()
        fd = (S.evolve(ψ0, lU, p .+ h .* e; backend=lens) - S.evolve(ψ0, lU, p .- h .* e; backend=lens)) / (2h)
        @test isapprox(S.evolve(ψ0, ldU, p; backend=lens), fd; atol=1e-7)
    end

    @testset "LENS kernel threads" begin
        qc, x, η, outer = ZZHelpers.zz_circuit(10, 1)
        params = vcat(x, η)
        ψ, dψ = ZZHelpers.derivative_two_loops(C.gcol2tree_chain_ket0(qc), qc, outer, η[1])
        l = S.lower_state(dψ, params; gcol=qc)
        p = rand(length(params))
        @test S.evolve(l, p; backend=S.LENSBackend(threads=2)) ≈ S.evolve(l, p)
    end
end
