function phi_single(x1, w)
    return w*x1
end
function phi_double(x1, x2, w)
    #return w*(pi-x1)*(pi-x2)
    return w*(QCSym.PI-x1)*(QCSym.PI-x2)
end

function add_circuit_zzfeaturemap!(qc, qreg, num_layers, initial_step_id; for_back_config=(true, true))
    num_qubits = QCSym.BitsRegs.num_qbits(qc.context)
    num_p_gates_forward = 0
    num_p_gates_backward = 0
    if for_back_config[1]
        #################
        # Build data encoding i
        #################
        step_id = initial_step_id
        for _ in 1:num_layers
            for i in 1:num_qubits
                QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false)
            end
            step_id += 1
            for i in 1:num_qubits
                QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("λ"=> 0.0))
                num_p_gates_forward += 1
            end
            step_id += 1
            for i in 1:num_qubits-1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
                step_id += 1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i+1]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
                num_p_gates_forward += 1
                step_id += 1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
                step_id += 1    
            end
        end
    end

    if for_back_config[2]
        #################
        # Build data encoding j (adjoint)
        #################
        for _ in 1:num_layers
            for i in num_qubits-1:-1:1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
                step_id += 1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i+1]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
                num_p_gates_backward += 1
                step_id += 1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
                step_id += 1    
            end
            for i in num_qubits:-1:1
                QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("λ"=> 0.0))
                num_p_gates_backward += 1
            end
            step_id += 1
            for i in 1:num_qubits
                QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false)
            end
        end
    end
    #################
    # End data encoding
    #################
    return num_p_gates_forward, num_p_gates_backward

end


function get_steps_of_p_and_zz_gates(qc::QCSym.Circuits.QuantumCircuit)
    p_gates = qc.gatecollection.collections[QCSym.Gates.P_Gate]
    _steps_p = sort([pg.step for pg in p_gates])
    u=unique(_steps_p)
    d=Dict([(i,count(x->x==i,_steps_p)) for i in u])
    steps_p = Vector{Int}([k for (k,v) in d if v>1])
    steps_zz = Vector{Int}([k for (k,v) in d if v==1])
    return steps_p, steps_zz
end

function get_dict_to_subs_outer_pars(param_gates_P, steps_p, steps_zz, data_params, weight_params, num_qubits)
    dict_to_subs_outer_pars = Dict()
    adj_sign = 1
    for s in steps_p
        for pg in param_gates_P
            if pg.step == s
                qid = pg.qubits_t[1].index_global
                dict_to_subs_outer_pars[pg.parameters["λ"]["sym"]] = exp(adj_sign * 1im * phi_single(data_params[qid], weight_params[1]))
                adj_sign *= -1
            end
        end
    end
    adj_sign = 1
    adj_count = 1
    for s in steps_zz
        for pg in param_gates_P
            if pg.step == s
                qid = pg.qubits_t[1].index_global
                dict_to_subs_outer_pars[pg.parameters["λ"]["sym"]] = exp(adj_sign * 1im * phi_double(data_params[qid-1], data_params[qid], weight_params[1+qid]))
                adj_count += 1
                if mod(qid, num_qubits) == 0; (adj_sign *= -1); (adj_count = 1) end
            end
        end
    end
    return dict_to_subs_outer_pars
end

function get_val_for_sym(s, qc)
    _a = QCSym.Circuits.s2g(s, qc.gatecollection)
    if hasfield(typeof(_a), :symbol) && isequal(s, _a.symbol)
        _g = _a.matrix_numeric
        return _g
    end
    if !isnothing(_a)
        for (_, p) in _a.parameters
            if isequal(p["sym"], s)
                return nothing
            end
        end
        for p in Symbolics.get_variables(_a.symbol)
            if isequal(p, s)
                _g = _a.matrix_alt
                return nothing
            end
        end
    end
    if isequal(s.name, :I)
        return StaticArrays.SMatrix{2, 2}([1 0; 0 1])
    end
    return nothing
end

function get_data(num_samples, num_dims)
    data = rand(Float64, num_dims, num_samples)
    return data
end