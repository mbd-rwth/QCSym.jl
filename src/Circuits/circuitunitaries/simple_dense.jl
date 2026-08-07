#import Symbolics
#import ..BitsRegs
#import ..Gates
import LinearAlgebra

"""
    assemble_unitary(qc::QuantumCircuit) -> Matrix{Symbolics.Num}

Assembles the full symbolic unitary matrix of the quantum circuit.

The circuit's qubits are ordered by register insertion order (first qubit of the
first register = most-significant bit, etc.). Gates are applied in ascending `step`
order; multiple gates at the same step must act on disjoint qubits.

Returns a `2^n × 2^n` symbolic matrix, where `n = get_num_qubits(qc)`.
"""
function assemble_unitary(qc::QuantumCircuit, replace_symbolic_zeros::Bool=false, replace_symbolic_ones::Bool=false)
    n   = get_num_qubits(qc)
    dim = 2^n

    # ── 1. Global qubit ordering: (name_reg, index_local) => index_global ──────
    qubit_pos = Dict{Tuple{String,Int}, Int}()
    for qreg in qc.qregs
        for bit in qreg.bits
            qubit_pos[(bit.name_reg, bit.index_local)] = bit.index_global
        end
    end

    # ── 2. Collect every gate from the gate collection ───────────────────────────
    all_gates = QCSym.Gates.AbstractGate[]
    for (_, gates) in qc.gatecollection.collections
        append!(all_gates, gates)
    end

    isempty(all_gates) && return Matrix{Complex{Symbolics.Num}}(LinearAlgebra.I, dim, dim)

    # ── 3. Per-step assembly, steps applied in ascending order ───────────────────
    sorted_steps = sort(unique(g.step for g in all_gates))

    # Start with the identity; steps are pre-multiplied on the left: U = U_last * … * U_1
    I2        = Matrix{Int}([1 0; 0 1])
    U_total   = Matrix{Complex{Symbolics.Num}}(LinearAlgebra.I, dim, dim)

    for step in sorted_steps
        gates_at_step = [g for g in all_gates if g.step == step]

        # Separate single-qubit from multi-qubit gates
        single_gates = [g for g in gates_at_step if g.num_qubits == 1]
        multi_gates  = [g for g in gates_at_step if g.num_qubits  > 1]

        # Build per-qubit 2×2 matrix map for single-qubit gates
        qubit_mat = Dict{Int, AbstractMatrix}()
        for gate in single_gates
            #p = qubit_pos[(gate.qubits_t[1].name_reg, Int(gate.qubits_t[1].index_local))]
            p = gate.qubits_t[1].index_global
            #qubit_mat[p] = Matrix(Symbolics.scalarize(gate.matrix))
            qubit_mat[p] = begin
                if gate.is_treat_numeric_only
                    gate.matrix_numeric
                elseif gate.is_treat_alt_only
                    gate.matrix_alt
                else
                    gate.matrix
                end
            end
        end

        # Assemble step unitary for single-qubit layer via Kronecker product
        mats   = [haskey(qubit_mat, p) ? qubit_mat[p] : I2 for p in 1:n]
        step_U = _kron_sequence(mats)

        # Embed and multiply in multi-qubit gates
        for gate in multi_gates
            #positions  = sort([qubit_pos[(q.name_reg, Int(q.index_local))] for q in gate.qubits])
            #positions  = sort([q.index_global for q in gate.qubits])
            positions  = collect(reverse([q.index_global for q in gate.qubits]))
            gate_mat   = begin
                if gate.is_treat_numeric_only
                    gate.matrix_numeric
                elseif gate.is_treat_alt_only
                    gate.matrix_alt
                else
                    gate.matrix
                end
            end
            embedded   = _embed_gate(gate_mat, positions, n)
            step_U     = embedded * step_U
        end

        step_U = _replace_symbolic_zeros_and_ones(step_U, gates_at_step, replace_symbolic_zeros, replace_symbolic_ones)
        U_total = step_U * U_total
        #display(step_U)
        #display(U_total)
    end

    return U_total
end

# ── helper: Kronecker product of a sequence of matrices ─────────────────────────
function _kron_sequence(mats::Vector)
    result = mats[1]
    for i in 2:length(mats)
        result = LinearAlgebra.kron(result, mats[i])
    end
    return result
end

# ── helper: embed a k-qubit gate matrix into the full 2^n space ─────────────────
"""
    _embed_gate(gate_matrix, qubits_global_indices, n_total) -> Matrix{Symbolics.Num}

Embeds a `2^k × 2^k` gate matrix acting on qubits `qubits_global_indices` (sorted, 1-indexed,
MSB = position 1) into the full `2^n_total × 2^n_total` Hilbert space by
distributing gate matrix elements over the correct basis indices while treating
all non-gate qubits as spectators.
"""
function _embed_gate(gate_matrix::Union{AbstractMatrix, Symbolics.Arr{Complex{Symbolics.Num}, 2}}, qubits_global_indices::Vector{Int}, n_total::Int)
    k = length(qubits_global_indices)
    @assert size(gate_matrix) == (2^k, 2^k) "Gate matrix size $(size(gate_matrix)) inconsistent with $(k) qubits"

    dim_full        = 2^n_total
    result          = zeros(Complex{Symbolics.Num}, dim_full, dim_full)
    other_qubits_global_indices = sort(setdiff(1:n_total, qubits_global_indices))
    n_other         = length(other_qubits_global_indices)

    for col_gate in 1:2^k
        for row_gate in 1:2^k
            elem = gate_matrix[row_gate, col_gate]
            # k-bit patterns within the gate subspace (MSB first)
            col_bits = [(col_gate - 1) >> (k - i) & 1 for i in 1:k]
            row_bits = [(row_gate - 1) >> (k - i) & 1 for i in 1:k]

            # Iterate over all 2^(n-k) spectator-qubit configurations
            for other_idx in 0:(2^n_other - 1)
                other_bits    = [other_idx >> (n_other - j) & 1 for j in 1:n_other]
                col_full_bits = zeros(Int, n_total)
                row_full_bits = zeros(Int, n_total)
                for (i, p) in enumerate(qubits_global_indices)
                    col_full_bits[p] = col_bits[i]
                    row_full_bits[p] = row_bits[i]
                end
                for (j, p) in enumerate(other_qubits_global_indices)
                    col_full_bits[p] = other_bits[j]
                    row_full_bits[p] = other_bits[j]
                end
                # Convert bit arrays to 1-based indices
                col_full = sum(col_full_bits[i] << (n_total - i) for i in 1:n_total) + 1
                row_full = sum(row_full_bits[i] << (n_total - i) for i in 1:n_total) + 1
                result[row_full, col_full] += elem
            end
        end
    end

    return result
end

