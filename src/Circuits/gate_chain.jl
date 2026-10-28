"""
    gcol2tree_chain(gcol::GateCollection; num_qubits=_get_num_qubits(gcol))
    gcol2tree_chain(qc::QuantumCircuit)

The circuit unitary as a chain with one factor per gate,

    U = G_m ⊙ (… ⊙ (G_2 ⊙ G_1)),    G = QCSym.GateOn(n, qubits)(gate.symbol)

[`gcol2tree2`](@ref) spells every step as a `⊗` over all qubit lines and a two-qubit gate
as the sum of its 2x2 decomposition. Here the tree grows only with the number of gates, so
multi-qubit gates add no summands, and gates of any size work without a decomposition.

Each factor's argument is the gate's own symbol, the same leaf `gcol2tree2` uses. The usual
substitutions (outer parameters, `matrix_alt`, `matrix_numeric`, differentiated leaves) and
[`QCSym.kron_derivative`](@ref) therefore work unchanged.

Gates are ordered by step. Gates of one step must act on disjoint qubits (checked), so they
commute; they are ordered by their first qubit to keep the tree deterministic. A gate's
qubits are placed as in `assemble_unitary` (simple_dense.jl), `reverse(gate.qubits)`, which
puts the controls first.

Pass `num_qubits`, or the circuit itself, if some qubit of the register carries no gate:
counting the qubits the gates touch (`_get_num_qubits`) would size the register too small.

See [`gcol2tree_chain_ket0`](@ref) for the state the circuit prepares from `|0…0⟩`, and
`QCSym.Simulation.lower_state` for running either on a simulator.
"""
function gcol2tree_chain(gcol::GateCollection; num_qubits::Int=_get_num_qubits(gcol))
    stepwise_gates = _extract_gates_stepwise(gcol)
    U = nothing
    for s in sort(collect(keys(stepwise_gates)))
        used = Set{Int}()
        for g in sort(stepwise_gates[s]; by=g -> minimum(q.index_global for q in g.qubits))
            qubits = _chain_qubits(g)
            isempty(intersect(used, qubits)) || error(
                "gates of step $(s) share a qubit ($(g.name)); the gates of one step must " *
                "act on disjoint qubits")
            union!(used, qubits)
            G = QCSym.GateOn(num_qubits, qubits)(g.symbol)
            U = U === nothing ? G : QCSym.:⊙(G, U)
        end
    end
    U === nothing && throw(ArgumentError("the gate collection holds no gates"))
    return U
end

gcol2tree_chain(qc::QuantumCircuit) =
    gcol2tree_chain(qc.gatecollection; num_qubits=get_num_qubits(qc))

# Kron order of a gate's matrix: the placement assemble_unitary uses, controls first.
_chain_qubits(g) = Tuple(reverse([q.index_global for q in g.qubits]))

"""
    gcol2tree_chain_ket0(gcol::GateCollection; num_qubits=_get_num_qubits(gcol))
    gcol2tree_chain_ket0(qc::QuantumCircuit)

[`gcol2tree_chain`](@ref) applied to `|0…0⟩`, the chain counterpart of
[`gcol2tree_col_1`](@ref):

    ψ = G_m ⊙ (… ⊙ (G_1 ⊙ |0…0⟩))

The state leaf is [`QCSym.ket0`](@ref) and sits under the first gate, so a numeric
evaluation applies one gate after the other to a state vector.
"""
gcol2tree_chain_ket0(gcol::GateCollection; num_qubits::Int=_get_num_qubits(gcol)) =
    _first_column(gcol2tree_chain(gcol; num_qubits=num_qubits))

gcol2tree_chain_ket0(qc::QuantumCircuit) = _first_column(gcol2tree_chain(qc))
