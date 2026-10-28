module Simulation
# Statevector simulation of gate-level trees: lower_state turns a tree of GateOn factors
# into gate chains, evolve! runs them on a backend (ReferenceBackend here, LENSBackend in
# the package extension that `using LENS` loads).
import LinearAlgebra
import Symbolics
import SymbolicUtils
import ..QCSym

include("Simulation/lower_state.jl")
include("Simulation/evolve.jl")
end
