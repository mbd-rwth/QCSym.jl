# Shared by the gate-chain, Simulation and LENS extension tests: a zz-feature-map-like
# circuit and the derivative workflow of examples/zz_feature_map, written out here so the
# tests do not depend on the examples.
module ZZHelpers

import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym

const C = QCSym.Circuits
const G = QCSym.Gates

"""
    zz_circuit(nq, layers) -> (qc, x, η, outer)

Per layer: H on every qubit, P(exp(i x_q η_1)) on every qubit, then CX-P-CX on every pair of
neighbours with P(exp(i η_{q+1} (π - x_q)(π - x_{q+1}))). `outer` maps the P gates' `λ` to
these expressions in the data `x` and weights `η`.
"""
function zz_circuit(nq, layers)
    qc = C.QuantumCircuit(name="zz")
    qr = C.add_qreg(qc, "q", nq)
    x = [Symbolics.variable(Symbol("x", i)) for i in 1:nq]
    η = [Symbolics.variable(Symbol("η", i)) for i in 1:nq]
    outer = Dict()
    add_p!(q, λ, step) = begin
        C.add_gate(qc, G.P_Gate, qubits_t=[qr[q]], step=step, is_treat_numeric_only=false)
        outer[last(qc.gatecollection.collections[G.P_Gate]).parameters["λ"]["sym"]] = λ
    end
    step = 1
    for _ in 1:layers
        for q in 1:nq
            C.add_gate(qc, G.H_Gate, qubits_t=[qr[q]], step=step, is_treat_numeric_only=false)
        end
        step += 1
        for q in 1:nq
            add_p!(q, exp(1im * x[q] * η[1]), step)
        end
        step += 1
        for q in 1:nq-1
            C.add_gate(qc, G.CX_Gate, qubits_t=[qr[q+1]], qubits_c=[qr[q]], step=step)
            add_p!(q + 1, exp(1im * η[q+1] * (π - x[q]) * (π - x[q+1])), step + 1)
            C.add_gate(qc, G.CX_Gate, qubits_t=[qr[q+1]], qubits_c=[qr[q]], step=step + 2)
            step += 3
        end
    end
    return qc, x, η, outer
end

# The substitution dictionaries of the derivative examples.
function p_leaf_dicts(qc, outer)
    pg = qc.gatecollection.collections[G.P_Gate]
    syms = Dict(g.symbol => Symbolics.substitute(g.symbol, outer) for g in pg)
    mats = Dict(g.symbol => Symbolics.substitute(g.matrix_alt, outer) for g in pg)
    remaining = Dict(s => mats[k] for (k, s) in syms)
    derv = Dict(Symbolics.Differential(only(SymbolicUtils.arguments(s)))(s) =>
                StaticArrays.SMatrix{2,2}(Symbolics.derivative(mats[k], only(SymbolicUtils.arguments(s))))
                for (k, s) in syms)
    return derv, remaining
end

"""
    derivative_two_loops(tree, qc, outer, v) -> (tree, dtree)

The workflow of examples/zz_feature_map: outer parameters into `tree`, `kron_derivative`
with respect to `v`, then the differentiated P leaves and, in a second loop, the remaining
ones. The second loop leaves the P leaves that do not depend on `v` in place.
"""
function derivative_two_loops(tree, qc, outer, v)
    derv, remaining = p_leaf_dicts(qc, outer)
    tree = SymbolicUtils.substitute(tree, outer)
    dtree = QCSym.kron_derivative(tree, v)
    dtree = SymbolicUtils.substitute(SymbolicUtils.substitute(dtree, derv), remaining)
    return SymbolicUtils.substitute(tree, remaining), dtree
end

"""
    derivative_one_pass(tree, qc, outer, v) -> (tree, dtree)

The same, with both dictionaries and the numeric gate matrices substituted in one pass,
which leaves nothing in place.
"""
function derivative_one_pass(tree, qc, outer, v)
    derv, remaining = p_leaf_dicts(qc, outer)
    tree = SymbolicUtils.substitute(tree, outer)
    dtree = QCSym.kron_derivative(tree, v)
    leaves = merge(derv, remaining, numeric_gate_values(tree, qc))
    return SymbolicUtils.substitute(tree, leaves), SymbolicUtils.substitute(dtree, leaves)
end

# matrix_numeric of every non-parametric gate leaf in `tree`
function numeric_gate_values(tree, qc)
    vals = Dict()
    for v in Symbolics.get_variables(tree)
        g = C.s2g(v, qc.gatecollection)
        if g !== nothing && hasfield(typeof(g), :symbol) && isequal(v, g.symbol)
            vals[v] = g.matrix_numeric
        end
    end
    return vals
end

"""
    dense_embed(M, qubits, n)

`M` on `qubits` of an `n`-qubit register as a dense matrix, built as `kron(M, I)` in the qubit
order `(qubits..., the other qubits...)` and permuted back. An independent reference for the
gate kernel, which never builds the embedding.
"""
function dense_embed(M, qubits, n)
    k = length(qubits)
    order = vcat(collect(qubits), [q for q in 1:n if !(q in qubits)])
    K = kron(Matrix{ComplexF64}(M), Matrix{ComplexF64}(LinearAlgebra.I, 2^(n - k), 2^(n - k)))
    perm = map(0:2^n-1) do b
        foldl((acc, q) -> (acc << 1) | ((b >> (n - q)) & 1), order; init=0) + 1
    end
    return K[perm, perm]
end

end
