module QCSym
import Symbolics
import SymbolicUtils
import PrecompileTools
Symbolics.@register_derivative complex(a,b) 1 Symbolics.SConst(1.0)
Symbolics.@register_derivative complex(a,b) 2 Symbolics.SConst(im)
const PI = SymbolicUtils.Const{SymbolicUtils.SymReal}(π)
include("./macros.jl")
include("./Rewriting.jl")
include("./Bits_Regs.jl")
include("./Gates.jl")
include("./Circuits.jl")
include("./States.jl")
include("./Symbolic_Tensor_Algebra.jl")

include("./Precompile.jl")
end