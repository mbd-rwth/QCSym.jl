# Cost per parameter set of the LENS backend, which rebuilds and recompiles every summand for
# every parameter set (no parametric LENS programs yet). Split into gate-matrix evaluation,
# LENS circuit build + compile, and simulation, for ψ and dψ/dη1 of a 2-layer zz map. The
# simulation column is what a compiled program reused across parameter sets would cost.
# The batch columns run evolve! on enough parameter sets for about half a second on one task,
# and compare 1 task with all threads in the same process.
#
# Pin it to one NUMA node; this host interleaves the nodes' CPUs:
#   numactl --cpunodebind=0 --membind=0 julia --project=test/LENSExt -t 8 test/LENSExt/lens_ext_benchmark.jl
import Symbolics
import SymbolicUtils
import LinearAlgebra
import Printf
import QCSym
import LENS

isdefined(Main, :ZZHelpers) || include(joinpath(@__DIR__, "..", "Simulation", "zz_helpers.jl"))
import .ZZHelpers

const C = QCSym.Circuits
const S = QCSym.Simulation

# best of `reps` after one warm-up call
best(f; reps=5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function summand_programs(l, ws)
    map(l.summands) do s
        c = LENS.Circuit(l.n)
        for g in s.gates
            push!(c, ws.matrices[g.slot], g.qubits)
        end
        LENS.compile(c)
    end
end

function row(name, l)
    p = rand(length(l.params))
    ψ0 = zeros(ComplexF64, 1 << l.n)
    ψ0[1] = 1
    ws = S._Workspace(l)
    first_call = @elapsed S.evolve(ψ0, l, p; backend=S.LENSBackend())
    t_values = best(() -> S._fill_values!(ws, l, p))
    t_compile = best(() -> summand_programs(l, ws))
    progs = summand_programs(l, ws)
    ψ = similar(ψ0)
    t_sim = best() do
        for prog in progs
            copyto!(ψ, ψ0)
            LENS.simulate!(ψ, prog)
        end
    end
    t_set = best(() -> S.evolve(ψ0, l, p; backend=S.LENSBackend()))
    nsets = clamp(round(Int, 0.5 / t_set), 16, 4096)
    P = rand(length(l.params), nsets)
    Ψ = repeat(ψ0, 1, nsets)
    t_serial = best(() -> S.evolve!(Ψ, l, P; backend=S.LENSBackend(), ntasks=1); reps=3) / nsets
    t_threads = best(() -> S.evolve!(Ψ, l, P; backend=S.LENSBackend()); reps=3) / nsets
    us(t) = Printf.@sprintf("%9.1f", 1e6 * t)
    println(rpad(name, 12), lpad(length(l.summands), 4), " | ", us(t_values), us(t_compile), us(t_sim),
            " | ", us(t_set), us(t_serial), us(t_threads), Printf.@sprintf("%7.1fx", t_serial / t_threads),
            " | ", Printf.@sprintf("%6.2f", first_call))
end

println("Julia threads: ", Threads.nthreads(), ". Times in µs per parameter set unless noted.")
println(rpad("", 12), lpad("sums", 4), " | ", lpad("values", 9), lpad("compile", 9), lpad("simulate", 9),
        " | ", lpad("evolve", 9), lpad("batch 1t", 9), lpad("batch nt", 9), lpad("speedup", 8),
        " | first call [s]")
for nq in [4, 8, 12, 16]
    qc, x, η, outer = ZZHelpers.zz_circuit(nq, 2)
    params = vcat(x, η)
    t_sym = @elapsed begin
        ψ, dψ = ZZHelpers.derivative_two_loops(C.gcol2tree_chain_ket0(qc), qc, outer, η[1])
        lψ = S.lower_state(ψ, params; gcol=qc)
        ldψ = S.lower_state(dψ, params; gcol=qc)
    end
    row("n=$(nq) ψ", lψ)
    row("n=$(nq) dψ", ldψ)
    println(rpad("", 12), "     symbolic work (tree, derivative, substitution, lowering): ",
            Printf.@sprintf("%.2f s", t_sym))
end
