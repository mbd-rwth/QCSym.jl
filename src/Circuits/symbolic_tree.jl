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

function extract_col_0(expr)
    if SymbolicUtils.iscall(expr) && SymbolicUtils.operation(expr)==QCSym.:⊗
        return 0
    end
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

const _SymT = QCSym.SymbolicUtils.BasicSymbolicImpl.var"typeof(BasicSymbolicImpl)"{QCSym.SymbolicUtils.SymReal}

"""
    _ProjectorSymbols

The symbols of the `_00_Gate` / `_11_Gate` factors that controlled gates decompose into
on their control lines. Multiplying one directly into the other gives zero
(`|0><0| * |1><1| = 0`), so a summand containing such a chain can be dropped.

Direct adjacency is what makes the test sound: `_00 ⊙ X ⊙ _11 = |0><1|` is *not* zero, so
only neighbouring factors may be compared.
"""
struct _ProjectorSymbols
    proj0::Set{_SymT}
    proj1::Set{_SymT}
end

function _projector_symbols(stepwise_gates)
    projs = _ProjectorSymbols(Set{_SymT}(), Set{_SymT}())
    for (_, gates) in stepwise_gates, g in gates
        hasfield(typeof(g), :gates22_c) || continue
        isnothing(g.gates22_c) && continue
        for (_, gates22) in g.gates22_c, g22 in gates22
            g22 isa QCSym.Gates._00_Gate && push!(projs.proj0, g22.symbol)
            g22 isa QCSym.Gates._11_Gate && push!(projs.proj1, g22.symbol)
        end
    end
    return projs
end

_is_chain(x) = SymbolicUtils.iscall(x) && SymbolicUtils.operation(x) === QCSym.:⊙

# Leftmost and rightmost factor of a ⊙ chain. Chains are usually nested to the right,
# but a step holding both single- and two-qubit gates builds left-nested ones, so both
# ends have to be followed recursively.
_chain_head(x) = _is_chain(x) ? _chain_head(SymbolicUtils.arguments(x)[1]) : x
_chain_tail(x) = _is_chain(x) ? _chain_tail(SymbolicUtils.arguments(x)[end]) : x

_new_pair_is_zero(_x, _y, ::Nothing) = false

# Multiplying _x onto _y creates exactly one new neighbouring pair, between the rightmost
# factor of _x and the leftmost factor of _y; every other pair in the result was tested
# when it was created, so a surviving chain never holds an adjacent orthogonal pair.
# Folding (⊙ with an identity filler collapses) cannot produce a false positive here: the
# filler is neither projector, so a folded-away factor simply fails the test.
function _new_pair_is_zero(_x, _y, projs::_ProjectorSymbols)
    a, b = _chain_tail(_x), _chain_head(_y)
    return (a in projs.proj0 && b in projs.proj1) || (a in projs.proj1 && b in projs.proj0)
end

_skip_zeros(itr) = Any[t for t in itr if t !== nothing]

function _sum_of_terms(terms)
    isempty(terms) && error("every summand was pruned as zero, which cannot happen for a "*
                            "circuit unitary: the zero test dropped a non-zero term")
    length(terms) == 1 && return terms[1]
    return QCSym.SymbolicUtils.term(+, terms...; type=QCSym.SymbolicUtils.SymReal)
end

# Returns nothing when the summand is structurally zero, so the caller can drop it.
function _mul_kron(x, y, projs=nothing)
    slots = Any[]
    for (_x, _y) in zip(x.args, y.args)
        _new_pair_is_zero(_x, _y, projs) && return nothing
        push!(slots, QCSym.:⊙(_x, _y))
    end
    return QCSym.:⊗(slots...)
end

function _mul_kron_sum_x(x, y, projs=nothing)
    return _sum_of_terms(_skip_zeros(_mul_kron(_x, y, projs) for _x in x.args))
end

function _mul_kron_sum_y(x, y, projs=nothing)
    return _sum_of_terms(_skip_zeros(_mul_kron(x, _y, projs) for _y in y.args))
end

function _mul_kron_sum_sum(x, y, projs=nothing)
    return _sum_of_terms(_skip_zeros(_mul_kron(_x, _y, projs) for _x in x.args for _y in y.args))
end

# Every summand of x multiplies every summand of y. Zipping the two argument lists
# instead would silently drop the mixed terms, which only happens to be harmless when
# the dropped products vanish anyway (e.g. two-qubit gates sharing a control qubit,
# where the mixed terms contain _00 ⊙ _11 = 0).
function _mul_sum_sum(x, y, projs=nothing)
    return _sum_of_terms(_skip_zeros(_mul_unknown(_x, _y, projs) for _x in x.args for _y in y.args))
end

function _mul_unknown(x, y, projs=nothing)
    if x.f==QCSym.:+ && y.f==QCSym.:+
        return _mul_sum_sum(x, y, projs)
    elseif x.f==QCSym.:+ && y.f==QCSym.:⊗
        return _mul_kron_sum_x(x, y, projs)
    elseif x.f==QCSym.:⊗ && y.f==QCSym.:+
        return _mul_kron_sum_y(x, y, projs)
    elseif x.f==QCSym.:⊗ && y.f==QCSym.:⊗
        return _mul_kron(x, y, projs)
    else
        error("Unsupported operation for _mul_unknown with x.f=$(x.f) and y.f=$(y.f)")
    end
end

"""
    _first_column(expr)

Rewrite an expression tree for a matrix into one for its first column, pushing the
column extraction down to the leaves so that no full-size intermediate is ever
formed. The rewrite is exact, it just applies column 1 of each operation:

  * `(A + B)[:,1] == A[:,1] + B[:,1]`
  * `(A ⊗ B)[:,1] == A[:,1] ⊗ B[:,1]`  (column 1 of a Kronecker product only uses
    column 1 of every factor)
  * `(A ⊙ B)[:,1] == A ⊙ B[:,1]`       (only the rightmost factor of a chain is reduced)

Leaves are wrapped in `QCSym.col_1`, which numerically takes the first column of a
gate's 2x2 matrix.
"""
function _first_column(expr)
    if !SymbolicUtils.iscall(expr)
        return QCSym.col_1(expr)
    end

    op = SymbolicUtils.operation(expr)
    args = SymbolicUtils.arguments(expr)

    if op === (+)
        return QCSym.SymbolicUtils.term(+, (_first_column(a) for a in args)...; type=QCSym.SymbolicUtils.SymReal)
    elseif op === QCSym.:⊗
        return QCSym.:⊗((_first_column(a) for a in args)...)
    elseif op === QCSym.:⊙
        # Left factors keep acting on the column produced further right.
        col_args = (args[1:end-1]..., _first_column(args[end]))
        return length(col_args) == 2 ? QCSym.:⊙(col_args...) :
               QCSym.SymbolicUtils.term(QCSym.:⊙, col_args...; type=QCSym.SymbolicUtils.SymReal)
    else
        # A parametric gate symbol such as P(λ) is a call, but still a leaf matrix.
        return QCSym.col_1(expr)
    end
end

"""
    gcol2tree_col_1(gcol::GateCollection)

Like [`gcol2tree2`](@ref), but the returned expression represents only the first
column of the circuit unitary instead of the whole unitary.

The tree has the same shape as the one from `gcol2tree2`; the difference is that the
rightmost (earliest step) factor of every per-qubit chain is wrapped in `QCSym.col_1`.
Evaluating it therefore krons 2-element columns rather than 2x2 matrices, so the
intermediates grow as `2^n` instead of `2^n x 2^n`.

The result is the unitary applied to |0...0>, i.e. the state of a circuit started in
the all-zero basis state, as a length-`2^n` vector.
"""
function gcol2tree_col_1(gcol::GateCollection; prune_zeros::Bool=true)
    return _first_column(gcol2tree2(gcol; prune_zeros=prune_zeros))
end

"""
    gcol2tree2(gcol::GateCollection; prune_zeros=true)

Build the symbolic expression tree of the whole circuit unitary.

Each two-qubit gate turns the running expression into a sum, and the next one multiplies
every summand with every summand, so the term count doubles per two-qubit gate. With
`prune_zeros` the summands that are structurally zero are dropped as they are formed,
which is what keeps the growth tractable: they are the large majority, and pruning them
while accumulating means the intermediate expression never holds them. See
[`_ProjectorSymbols`](@ref) for the test used. Pass `prune_zeros=false` to get the
unpruned, mathematically identical expression for cross-checking.
"""
function gcol2tree2(gcol::GateCollection; prune_zeros::Bool=true)
    stepwise_gates = _extract_gates_stepwise(gcol)
    steps = sort(collect(keys(stepwise_gates)))
    projs = prune_zeros ? _projector_symbols(stepwise_gates) : nothing

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
            U_step = _mul_unknown(U_step_sq, U_step_mq, projs)
        end
        #println("U_step: ", U_step)
        if U === nothing
            U = U_step
        else
            U = _mul_unknown(U_step, U, projs)
            U === nothing && error("the whole step product was pruned as zero, which cannot "*
                                   "happen for a circuit unitary: the zero test dropped a "*
                                   "non-zero term")
        end
        #println("U: ")
        #println(U)
    end
    #U = QCSym.:⊙(U_intermediate...)
    
    return U
end