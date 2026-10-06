using Test


#tests = ["assem", "disassem", "simulator"]
tests = ["gates", "symbolic_tensor_algebra", "symjit", "symbolics", "codegen"]
#tests = ["symbolic_tensor_algebra"]
if !isempty(ARGS)
	tests = ARGS  # Set list to same as command line args
end


dict_testfiles = Dict(
    "gates" => "Gates/gates_tests.jl",
    "symbolic_tensor_algebra" => "Symbolic_Tensor_Algebra/symbolic_tensor_algebra_test.jl",
    "symjit" => "SymJit/symjit_tests.jl",
    "symbolics" => "Symbolics/symbolics_test.jl",
    "codegen" => "Codegen/loop_codegen_tests.jl"
)



for t in tests
    println("Running tests: $t")
    include(dict_testfiles[t])
end
