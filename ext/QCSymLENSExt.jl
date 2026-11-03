module QCSymLENSExt
# LENS.jl as a simulation backend: QCSym.Simulation.LENSBackend.

import QCSym
import LENS

# LENS numbers qubits like QCSym (qubit 1 is the most significant bit, a gate's first qubit the
# most significant bit of its matrix), so the qubits pass through unchanged.

# The buffers of one task of evolve!: a MutableProgram per summand, compiled for the first
# parameter set the task runs and recompiled in place for the others, and per summand the gates
# whose matrices depend on the parameters, the only ones to update.
struct LENSPrograms
    programs::Vector{Union{Nothing, LENS.MutableProgram{ComplexF64}}}
    parametric::Vector{Vector{Int}}
end

function QCSym.Simulation._backend_buffers(::QCSym.Simulation.LENSBackend,
                                           ls::QCSym.Simulation.LoweredState)
    parametric = [[i for (i, g) in enumerate(s.gates) if ls.const_matrices[g.slot] === nothing]
                  for s in ls.summands]
    return LENSPrograms(Union{Nothing, LENS.MutableProgram{ComplexF64}}[nothing for _ in ls.summands],
                        parametric)
end

function QCSym.Simulation._run_summand!(ψ::AbstractVector{ComplexF64},
                                        backend::QCSym.Simulation.LENSBackend,
                                        ls::QCSym.Simulation.LoweredState, j::Int, ws)
    gates = ls.summands[j].gates
    buffers = ws.backend
    mp = buffers.programs[j]
    if mp === nothing
        c = LENS.Circuit(ls.n)
        for g in gates
            push!(c, ws.matrices[g.slot], g.qubits)
        end
        mp = buffers.programs[j] = LENS.MutableProgram(c; fuse=backend.fuse, atol=backend.atol)
    else
        for i in buffers.parametric[j]
            LENS.setmatrix!(mp, i, ws.matrices[gates[i].slot])
        end
    end
    LENS.simulate!(ψ, mp; threads=backend.threads)
    return ψ
end

end
