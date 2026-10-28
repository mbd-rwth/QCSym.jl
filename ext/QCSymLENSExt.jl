module QCSymLENSExt
# LENS.jl as a simulation backend: QCSym.Simulation.LENSBackend.

import QCSym
import LENS

# One summand, rebuilt and recompiled for every parameter set: the gate matrices change
# from one set to the next, and a compiled LENS program is fixed. LENS numbers qubits like
# QCSym (qubit 1 is the most significant bit, a gate's first qubit the most significant bit
# of its matrix), so the qubits pass through unchanged.
function QCSym.Simulation._run_gates!(ψ::AbstractVector{ComplexF64},
                                      backend::QCSym.Simulation.LENSBackend, n::Integer,
                                      gates, matrices)
    c = LENS.Circuit(n)
    for g in gates
        push!(c, matrices[g.slot], g.qubits)
    end
    prog = LENS.compile(c; eltype=ComplexF64, fuse=backend.fuse, atol=backend.atol)
    LENS.simulate!(ψ, prog; threads=backend.threads)
    return ψ
end

end
