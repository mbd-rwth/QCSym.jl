using Test


#tests = ["assem", "disassem", "simulator"]
tests = ["gates", "symbolic_tensor_algebra", "symjit", "symbolics", "codegen"]
tests = ["symbolic_tensor_algebra", "symjit", "symbolics", "codegen", "gate_chain", "simulation", "perf_matmul"]
# tests = ["perf_matmul"]
# "lens_ext" needs LENS.jl, so it runs in its own environment:
#   julia --project=test/LENSExt test/testrunner.jl lens_ext

if !isempty(ARGS)
	tests = ARGS  # Set list to same as command line args
end


dict_testfiles = Dict(
    "gates" => "Gates/gates_tests.jl",
    "symbolic_tensor_algebra" => "Symbolic_Tensor_Algebra/symbolic_tensor_algebra_test.jl",
    "symjit" => "SymJit/symjit_tests.jl",
    "symbolics" => "Symbolics/symbolics_test.jl",
    "codegen" => "Codegen/loop_codegen_tests.jl",
    "perf_matmul" => "Perf_MatMul/perf_matmul_test.jl",
    "gate_chain" => "Circuits/gate_chain_tests.jl",
    "simulation" => "Simulation/simulation_tests.jl",
    "lens_ext" => "LENSExt/lens_ext_tests.jl"
)



for t in tests
    println("Running tests: $t")
    include(dict_testfiles[t])
end
