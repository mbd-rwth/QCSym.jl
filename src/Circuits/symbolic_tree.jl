function gcol2tree(gcol::GateCollection)
    stepwise_gates = _extract_gates_stepwise(gcol)
    steps = sort(collect(keys(stepwise_gates)))
    
    num_qubits = _get_num_qubits(gcol)
    
    symbolic_steps = Dict{Int, Vector{<:QCSym.Gates.AbstractGate}}()
    # println("Converting gate collection to symbolic tree representation...")
    # println("   Number of unique steps: $num_steps")
    # println("   Number of qubits: $num_qubits")

    U = nothing
    U_intermediate = QCSym.SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{QCSym.SymbolicUtils.SymReal}[]
    U_step_mq = Vector{Union{Complex{Symbolics.Num}, Symbolics.Num, QCSym.SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{QCSym.SymbolicUtils.SymReal}}}()
    for s in steps
        U_step = nothing
        gates_at_step = stepwise_gates[s]
        symbolic_steps[s] = gates_at_step

        sq_gates = [g for g in gates_at_step if g.num_qubits == 1]
        mq_gates = [g for g in gates_at_step if g.num_qubits == 2]

        sq_gates_sorted = _sort_by_glob_qbit_id(sq_gates)
        vec_filled_sq_gates = Vector{QCSym.Gates.AbstractGate}()
        for i in 1:num_qubits
            push!(vec_filled_sq_gates, QCSym.Gates.I_Gate_Filler(i))
            #push!(vec_filled_sq_gates, QCSym.Gates.I_Gate_for_Circuit(qubits_t=stepwise_gates[1][1], step::Int, is_treat_numeric_only::Bool, _...))
        end
        for sqg in sq_gates_sorted
            qid = sqg.qubits_t[1].index_global
            vec_filled_sq_gates[qid] = sqg
        end

        sqsyms_from_gates = [g.symbol for g in vec_filled_sq_gates]

        U_step_sq = QCSym.:⊗(sqsyms_from_gates...)


        for mqg in mq_gates
           vec_filled_mq_sq_gates_0 = Vector{QCSym.Gates.AbstractGate}()
           vec_filled_mq_sq_gates_1 = Vector{QCSym.Gates.AbstractGate}()
            for i in 1:num_qubits
                push!(vec_filled_mq_sq_gates_0, QCSym.Gates.I_Gate_Filler(i))
                push!(vec_filled_mq_sq_gates_1, QCSym.Gates.I_Gate_Filler(i))
            end
            for (qid, gates) in mqg.gates22_t
                vec_filled_mq_sq_gates_0[qid] = gates[1]
                vec_filled_mq_sq_gates_1[qid] = gates[2]
            end
            for (qid, gates) in mqg.gates22_c
                vec_filled_mq_sq_gates_0[qid] = gates[1]
                vec_filled_mq_sq_gates_1[qid] = gates[2]
            end

            mqsyms_from_gates_0 = [g.symbol for g in vec_filled_mq_sq_gates_0]
            mqsyms_from_gates_1 = [g.symbol for g in vec_filled_mq_sq_gates_1]
            
            U_step_mq = push!(U_step_mq, QCSym.SymbolicUtils.term(+, QCSym.:⊗(mqsyms_from_gates_0...), QCSym.:⊗(mqsyms_from_gates_1...); type=QCSym.SymbolicUtils.SymReal))
        end




        #U_step = U_step_sq
        if !isempty(sq_gates)
            U_intermediate = push!(U_intermediate, U_step_sq)
        end
        if !isempty(mq_gates)
            U_intermediate = push!(U_intermediate, U_step_mq...)
        end
    end
    U = QCSym.:⊙(U_intermediate...)
    
    return U
end

function _unroll_prod(x)
    if SymbolicUtils.isterm(x) && x.f == QCSym.:⊙
        return (xx for xx in x.args)
    else
        return (x,)
    end
end

# function _mul_kron(x, y)
#     println("x.f: ", x.f, " y.f: ", y.f)
#     println("x.args: ", x.args, " y.args: ", y.args)
    
#     res = QCSym.:⊗((QCSym.:⊙((_unroll_prod(_x)..., _unroll_prod(_y)...)) for (_x, _y) in zip(x.args, y.args))...)
#     return res
# end

function _mul_kron(x, y)
    res = QCSym.:⊗((QCSym.:⊙(_x, _y) for (_x, _y) in zip(x.args, y.args))...)
    return res
end

function _mul_kron_sum_x(x, y)
    res = QCSym.SymbolicUtils.term(+, (_mul_kron(_x, y) for _x in x.args)...; type=QCSym.SymbolicUtils.SymReal)
    return res
end

function _mul_kron_sum_y(x, y)
    res = QCSym.SymbolicUtils.term(+, (_mul_kron(x, _y) for _y in y.args)...; type=QCSym.SymbolicUtils.SymReal)
    return res
end

function _mul_kron_sum_sum(x, y)
    res = QCSym.SymbolicUtils.term(+, (_mul_kron(_x, _y) for (_x, _y) in zip(x.args, y.args))...; type=QCSym.SymbolicUtils.SymReal)
    return res
end

function _mul_sum_sum(x, y)

    res = QCSym.SymbolicUtils.term(+, (_mul_unknown(_x, _y) for (_x, _y) in zip(x.args, y.args))...; type=QCSym.SymbolicUtils.SymReal)
    return res
end

function _mul_unknown(x, y)
    if x.f==QCSym.:+ && y.f==QCSym.:+
        return _mul_sum_sum(x, y)
    elseif x.f==QCSym.:+ && y.f==QCSym.:⊗
        return _mul_kron_sum_x(x, y)
    elseif x.f==QCSym.:⊗ && y.f==QCSym.:+
        return _mul_kron_sum_y(x, y)
    elseif x.f==QCSym.:⊗ && y.f==QCSym.:⊗
        return _mul_kron(x, y)
    else
        error("Unsupported operation for _mul_unknown with x.f=$(x.f) and y.f=$(y.f)")
    end
end

function gcol2tree2(gcol::GateCollection)
    stepwise_gates = _extract_gates_stepwise(gcol)
    steps = sort(collect(keys(stepwise_gates)))
    
    num_qubits = _get_num_qubits(gcol)
    
    symbolic_steps = Dict{Int, Vector{<:QCSym.Gates.AbstractGate}}()
    # println("Converting gate collection to symbolic tree representation...")
    # println("   Number of unique steps: $num_steps")
    # println("   Number of qubits: $num_qubits")

    U = nothing
    U_intermediate = QCSym.SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{QCSym.SymbolicUtils.SymReal}[]
    U_step_mq = Vector{Union{Complex{Symbolics.Num}, Symbolics.Num, QCSym.SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{QCSym.SymbolicUtils.SymReal}}}()
    for s in steps
        U_step = nothing
        gates_at_step = stepwise_gates[s]
        symbolic_steps[s] = gates_at_step

        sq_gates = [g for g in gates_at_step if g.num_qubits == 1]
        mq_gates = [g for g in gates_at_step if g.num_qubits == 2]

        sq_gates_sorted = _sort_by_glob_qbit_id(sq_gates)
        vec_filled_sq_gates = Vector{QCSym.Gates.AbstractGate}()
        for i in 1:num_qubits
            push!(vec_filled_sq_gates, QCSym.Gates.I_Gate_Filler(i))
            #push!(vec_filled_sq_gates, QCSym.Gates.I_Gate_for_Circuit(qubits_t=stepwise_gates[1][1], step::Int, is_treat_numeric_only::Bool, _...))
        end
        for sqg in sq_gates_sorted
            qid = sqg.qubits_t[1].index_global
            vec_filled_sq_gates[qid] = sqg
        end

        sqsyms_from_gates = [g.symbol for g in vec_filled_sq_gates]

        U_step_sq = QCSym.:⊗(sqsyms_from_gates...)


        for mqg in mq_gates
           vec_filled_mq_sq_gates_0 = Vector{QCSym.Gates.AbstractGate}()
           vec_filled_mq_sq_gates_1 = Vector{QCSym.Gates.AbstractGate}()
            for i in 1:num_qubits
                push!(vec_filled_mq_sq_gates_0, QCSym.Gates.I_Gate_Filler(i))
                push!(vec_filled_mq_sq_gates_1, QCSym.Gates.I_Gate_Filler(i))
            end
            for (qid, gates) in mqg.gates22_t
                vec_filled_mq_sq_gates_0[qid] = gates[1]
                vec_filled_mq_sq_gates_1[qid] = gates[2]
            end
            for (qid, gates) in mqg.gates22_c
                vec_filled_mq_sq_gates_0[qid] = gates[1]
                vec_filled_mq_sq_gates_1[qid] = gates[2]
            end

            mqsyms_from_gates_0 = [g.symbol for g in vec_filled_mq_sq_gates_0]
            mqsyms_from_gates_1 = [g.symbol for g in vec_filled_mq_sq_gates_1]
            
            #U_step_mq = push!(U_step_mq, QCSym.SymbolicUtils.term(+, QCSym.:⊗(mqsyms_from_gates_0...), QCSym.:⊗(mqsyms_from_gates_1...); type=QCSym.SymbolicUtils.SymReal))
            U_step_mq = QCSym.SymbolicUtils.term(+, QCSym.:⊗(mqsyms_from_gates_0...), QCSym.:⊗(mqsyms_from_gates_1...); type=QCSym.SymbolicUtils.SymReal)
            #U_step_mq = +(QCSym.:⊗(mqsyms_from_gates_0...), QCSym.:⊗(mqsyms_from_gates_1...))
        end

        
        
        #U_step = U_step_sq
        if !isempty(sq_gates) && isempty(mq_gates)
            #U_intermediate = push!(U_intermediate, U_step_sq)
            U_step = U_step_sq
        end
        if !isempty(mq_gates) && isempty(sq_gates)
            #U_intermediate = push!(U_intermediate, U_step_mq...)
            U_step = U_step_mq
        end
        if !isempty(sq_gates) && !isempty(mq_gates)
            U_step = _mul_unknown(U_step_sq, U_step_mq)
        end
        #println("U_step: ", U_step)
        U = U===nothing ? U_step : _mul_unknown(U_step, U)
        #println("U: ")
        #println(U)
    end
    #U = QCSym.:⊙(U_intermediate...)
    
    return U
end