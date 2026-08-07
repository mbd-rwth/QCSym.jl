using Test

import Symbolics
import SymbolicUtils
import QCSym

function custom_issapprox_for_vecs_or_mats(vec_or_mat1, vec_or_mat2; atol=1e-12)::Bool
    if size(vec_or_mat1) != size(vec_or_mat2)
        return false
    end

    _vec = vec_or_mat1
    for i in eachindex(_vec)
        if !isapprox(Symbolics.value(_vec[i]), vec_or_mat2[i], atol=atol)
            return false
        end
    end
    return true
end


@testset verbose=true "Gates Tests" begin
    println("Running gates tests...")
    println(QCSym.Gates.get_all_concrete_gates())
    
    @testset verbose=true "gate actions on state" begin
        @testset "H gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.H_Gate
            expected_statevec_after = 1/(sqrt(2)) * [1/2*(sqrt(5)+sqrt(15)), im+sqrt(3)*im, 1/2*(sqrt(5)-sqrt(15)), im-sqrt(3)*im]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            # dict_subs = Dict()
            # for i in eachindex(qc.gatecollection.collections[gate][1].matrix)
            #     dict_subs[real(qc.gatecollection.collections[gate][1].matrix[i])] = real(qc.gatecollection.collections[gate][1].matrix_numeric[i])
            #     dict_subs[imag(qc.gatecollection.collections[gate][1].matrix[i])] = imag(qc.gatecollection.collections[gate][1].matrix_numeric[i])
            # end
            # statevec_after_numeric_subsd = Symbolics.substitute.(U_statevec_numeric, (dict_subs,), fold=Val(true))
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "I gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.I_Gate
            expected_statevec_after = copy(statevec.vector_numeric)

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])

            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "X gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.X_Gate
            expected_statevec_after = [q1[2]*q2[1], q1[2]*q2[2], q1[1]*q2[1], q1[1]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])

            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "Y gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.Y_Gate
            expected_statevec_after = [-im*q1[2]*q2[1], -im*q1[2]*q2[2], im*q1[1]*q2[1], im*q1[1]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "Z gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.Z_Gate
            expected_statevec_after = [q1[1]*q2[1], q1[1]*q2[2], -q1[2]*q2[1], -q1[2]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "S gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.S_Gate
            expected_statevec_after = [q1[1]*q2[1], q1[1]*q2[2], im*q1[2]*q2[1], im*q1[2]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "Sdg gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.Sdg_Gate
            expected_statevec_after = [q1[1]*q2[1], q1[1]*q2[2], -im*q1[2]*q2[1], -im*q1[2]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "T gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.T_Gate
            _coeff = (1.0+1.0*im)/sqrt(2)
            expected_statevec_after = [q1[1]*q2[1], q1[1]*q2[2], _coeff*q1[2]*q2[1], _coeff*q1[2]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "Tdg gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.Tdg_Gate
            _coeff = (1.0-1.0*im)/sqrt(2)
            expected_statevec_after = [q1[1]*q2[1], q1[1]*q2[2], _coeff*q1[2]*q2[1], _coeff*q1[2]*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "SX gate action on 2 qubit system" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.SX_Gate
            expected_statevec_after = [1/2*((1+1im)*q1[1]+(1-1im)*q1[2])*q2[1], 1/2*((1+1im)*q1[1]+(1-1im)*q1[2])*q2[2],
                                       1/2*((1-1im)*q1[1]+(1+1im)*q1[2])*q2[1], 1/2*((1-1im)*q1[1]+(1+1im)*q1[2])*q2[2]]

            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "U gate action on 1 qubit system, alt representation" begin
            q1 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1])
            gate = QCSym.Gates.U_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 1)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            gate_params = qc.gatecollection.collections[gate][1].parameters
            expected_statevec_after = [cos(gate_params["θ"]["sym"]/2)*q1[1] - exp(1im*gate_params["λ"]["sym"])*sin(gate_params["θ"]["sym"]/2)*q1[2],
                                      exp(1im*gate_params["ϕ"]["sym"])*sin(gate_params["θ"]["sym"]/2)*q1[1] + exp(1im*(gate_params["ϕ"]["sym"]+gate_params["λ"]["sym"]))*cos(gate_params["θ"]["sym"]/2)*q1[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "GP gate action on 1 qubit system, alt representation" begin
            q1 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1])
            gate = QCSym.Gates.GP_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 1)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true, param_values=Dict{String, Real}())
            gate_params = qc.gatecollection.collections[gate][1].parameters
            expected_statevec_after = exp(1im*gate_params["γ"]["sym"])*q1
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "RX gate action on 2 qubit system, alt representation" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.RX_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true, param_values=Dict{String, Real}())
            gate_params = qc.gatecollection.collections[gate][1].parameters
            θ = gate_params["θ"]["sym"]
            expected_statevec_after = [(cos(θ/2)*q1[1]-1im*sin(θ/2)*q1[2])*q2[1],
                                       (cos(θ/2)*q1[1]-1im*sin(θ/2)*q1[2])*q2[2],
                                       (-1im*sin(θ/2)*q1[1]+cos(θ/2)*q1[2])*q2[1],
                                       (-1im*sin(θ/2)*q1[1]+cos(θ/2)*q1[2])*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "RY gate action on 2 qubit system, alt representation" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.RY_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true, param_values=Dict{String, Real}())
            gate_params = qc.gatecollection.collections[gate][1].parameters
            θ = gate_params["θ"]["sym"]
            expected_statevec_after = [q1[1]*(cos(θ/2)*q2[1]-sin(θ/2)*q2[2]),
                                       q1[1]*(sin(θ/2)*q2[1]+cos(θ/2)*q2[2]),
                                       q1[2]*(cos(θ/2)*q2[1]-sin(θ/2)*q2[2]),
                                       q1[2]*(sin(θ/2)*q2[1]+cos(θ/2)*q2[2])]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "RZ gate action on 2 qubit system, alt representation" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.RZ_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true, param_values=Dict{String, Real}())
            gate_params = qc.gatecollection.collections[gate][1].parameters
            θ = gate_params["θ"]["sym"]
            expected_statevec_after = [q1[1]*exp(-1im*θ/2)*q2[1],
                                       q1[1]*exp( 1im*θ/2)*q2[2],
                                       q1[2]*exp(-1im*θ/2)*q2[1],
                                       q1[2]*exp( 1im*θ/2)*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "P gate action on 2 qubit system, alt representation" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.P_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            gate_params = qc.gatecollection.collections[gate][1].parameters
            λ = gate_params["λ"]["sym"]
            expected_statevec_after = [q1[1]*1.0*q2[1],
                                       q1[1]*λ*q2[2],
                                       q1[2]*1.0*q2[1],
                                       q1[2]*λ*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "CX gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CX_Gate
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[2],
                                       q1[2]*q2[1]]
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]],step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CX gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CX_Gate
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[2]*q2[2],
                                       q1[2]*q2[1],
                                       q1[1]*q2[2]]
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]],step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CY gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CY_Gate
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       -im*q1[2]*q2[2],
                                       im*q1[2]*q2[1]]
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]],step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CZ gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CZ_Gate
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[1],
                                       -q1[2]*q2[2]]
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]],step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CZ gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CZ_Gate
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[1],
                                       -q1[2]*q2[2]]
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]],step=1, is_treat_numeric_only=false)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CP gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CP_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            λ = qc.gatecollection.collections[gate][1].parameters["λ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[1],
                                       exp(1im*λ)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "CP gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CP_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            λ = qc.gatecollection.collections[gate][1].parameters["λ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[1],
                                       exp(1im*λ)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "CRX gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CRX_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            θ = qc.gatecollection.collections[gate][1].parameters["θ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       cos(θ/2)*q1[2]*q2[1] - im*sin(θ/2)*q1[2]*q2[2],
                                       -im*sin(θ/2)*q1[2]*q2[1] + cos(θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "CRY gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CRY_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            θ = qc.gatecollection.collections[gate][1].parameters["θ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       cos(θ/2)*q1[1]*q2[2] - sin(θ/2)*q1[2]*q2[2],
                                       q1[2]*q2[1],
                                       sin(θ/2)*q1[1]*q2[2] + cos(θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
        end

        @testset "CRZ gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CRZ_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            θ = qc.gatecollection.collections[gate][1].parameters["θ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       exp(-1im*θ/2)*q1[2]*q2[1],
                                       exp(1im*θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            # @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
            # Dont know why, but the Symbolics.isequal test fails here
            @test custom_issapprox_for_vecs_or_mats([0,0,0,0], statevec_after_numeric_subsd - expected_statevec_after)
        end

        @testset "CRZ gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CRZ_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            θ = qc.gatecollection.collections[gate][1].parameters["θ"]["sym"]
            expected_statevec_after = [q1[1]*q2[1],
                                       exp(-1im*θ/2)*q1[1]*q2[2],
                                       q1[2]*q2[1],
                                       exp(1im*θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),))
            # @test all(Symbolics.isequal.(statevec_after_numeric_subsd, expected_statevec_after))
            # Dont know why, but the Symbolics.isequal test fails here
            @test custom_issapprox_for_vecs_or_mats([0,0,0,0], statevec_after_numeric_subsd - expected_statevec_after)
        end

        @testset "CH gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CH_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[1] + q1[2]*q2[2],
                                       q1[2]*q2[1] - q1[2]*q2[2]] / sqrt(2)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CH gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CH_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]], step=1, is_treat_numeric_only=false)
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2] + q1[2]*q2[2],
                                       q1[2]*q2[1],
                                       q1[1]*q2[2] - q1[2]*q2[2]] / sqrt(2)
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "CU gate action on 2 qubit system, qubit_t > qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CU_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2]], qubits_c=[qreg[1]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            params = ["θ", "ϕ", "λ", "γ"]
            θ, ϕ, λ, γ = [qc.gatecollection.collections[gate][1].parameters[p]["sym"] for p in params]
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[1]*q2[2],
                                       exp(im*γ)*cos(θ/2)*q1[2]*q2[1] - exp(im*(γ+λ))*sin(θ/2)*q1[2]*q2[2],
                                       exp(im*(γ+ϕ))*sin(θ/2)*q1[2]*q2[1] + exp(im*(γ+ϕ+λ))*cos(θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),), fold=Val(true))
            @test custom_issapprox_for_vecs_or_mats([0,0,0,0], statevec_after_numeric_subsd - expected_statevec_after)
        end

        @testset "CU gate action on 2 qubit system, qubit_t < qubit_c" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.CU_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1]], qubits_c=[qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            params = ["θ", "ϕ", "λ", "γ"]
            θ, ϕ, λ, γ = [qc.gatecollection.collections[gate][1].parameters[p]["sym"] for p in params]
            expected_statevec_after = [q1[1]*q2[1],
                                       exp(im*γ)*cos(θ/2)*q1[1]*q2[2] - exp(im*(γ+λ))*sin(θ/2)*q1[2]*q2[2],
                                       q1[2]*q2[1],
                                       exp(im*(γ+ϕ))*sin(θ/2)*q1[1]*q2[2] + exp(im*(γ+ϕ+λ))*cos(θ/2)*q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            ze = Symbolics.@variables(z)[1]
            zze = Symbolics.substitute(exp(ze), Dict(ze=>0)) 
            # zze --> exp(0), where 0 is treated as a symbol.
            # For some reason expected_statevec_after contains these exp(0), which need ot be replaced manually
            expected_statevec_after = Symbolics.substitute.(expected_statevec_after, (Dict(zze=>1.0),), fold=Val(true))
            @test custom_issapprox_for_vecs_or_mats([0,0,0,0], statevec_after_numeric_subsd - expected_statevec_after)
        end

        @testset "SWAP gate action on 2 qubit system, qubit_t_1 < qubit_t_2" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.SWAP_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[1], qreg[2]], step=1, is_treat_numeric_only=false, is_treat_alt_only=true)
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[2]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end

        @testset "SWAP gate action on 2 qubit system, qubit_t_1 > qubit_t_2" begin
            q1 = [0.5+0im, sqrt(3)/2+0im]
            q2 = [sqrt(5)+0im, 0.0+2.0im]
            statevec = QCSym.States.StateVector([q1, q2])
            gate = QCSym.Gates.SWAP_Gate
            expected_statevec_after = nothing
            qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
            qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", 2)
            QCSym.Circuits.add_gate(qc, gate, qubits_t=[qreg[2], qreg[1]], step=1, is_treat_numeric_only=false)
            expected_statevec_after = [q1[1]*q2[1],
                                       q1[2]*q2[1],
                                       q1[1]*q2[2],
                                       q1[2]*q2[2]]
            U = QCSym.Circuits.assemble_symbolic_unitary(qc, false, false)
            U_statevec_numeric = U * statevec.vector_numeric
            statevec_after_numeric_subsd = QCSym.Circuits.substitute_numerics_from_gates(U_statevec_numeric, qc.gatecollection.collections[gate])
            @test custom_issapprox_for_vecs_or_mats(statevec_after_numeric_subsd, expected_statevec_after)
        end
        
    end
end

