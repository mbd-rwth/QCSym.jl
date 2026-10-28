"""
    GateRef

One gate of a lowered summand: the qubits it acts on, in the kron order of its matrix, and
the slot that holds the matrix.
"""
struct GateRef
    qubits::Vector{Int}
    slot::Int
end

"""
    Summand

One term `c * G_m ⋯ G_1 |start⟩` of a [`LoweredState`](@ref), simulated on its own.
`gates` are in application order. `coeff == 0` stands for the coefficient 1, any other value
indexes the coefficients. `start == 0` stands for the state handed to [`evolve!`](@ref), any
other value indexes `LoweredState.kets`.
"""
struct Summand
    coeff::Int
    gates::Vector{GateRef}
    start::Int
end

"""
    LoweredState

A statevector-like tree prepared by [`lower_state`](@ref): a flat sum of gate chains, the
gate matrices deduplicated into slots, and generated functions that evaluate the
parameter-dependent slots and the coefficients for one parameter vector. It does not depend
on the simulator; [`evolve!`](@ref) runs it on a backend.
"""
struct LoweredState{FM, FC}
    n::Int
    params::Vector{Any}
    summands::Vector{Summand}
    kets::Vector{QCSym.BasisKet}
    # matrix size of every slot, and the value of the constant ones (nothing if parametric)
    slot_dims::Vector{Int}
    const_matrices::Vector{Union{Nothing, Matrix{ComplexF64}}}
    # the slots `matrix_values(p)` returns, in its output order
    param_slots::Vector{Int}
    matrix_values::FM
    coeff_values::FC
    ncoeffs::Int
end

_reads_input(ls::LoweredState) = any(s -> s.start == 0, ls.summands)

function Base.show(io::IO, ls::LoweredState)
    print(io, "LoweredState(", ls.n, " qubits, ", length(ls.summands), " summand(s), ",
          length(ls.slot_dims), " gate matrices of which ", length(ls.param_slots),
          " parametric")
    _reads_input(ls) && print(io, ", applied to an input state")
    print(io, ")")
end

"""
    lower_state(tree, params; gcol=nothing) -> LoweredState

Prepare a statevector-like `tree` for [`evolve!`](@ref). `tree` is either

* an operator built from `QCSym.GateOn` factors, e.g. `Circuits.gcol2tree_chain` or a
  derivative of it: `evolve!` applies it to the state it is given; or
* a state, such an operator applied to basis-state leaves (`QCSym.ket0`), e.g.
  `Circuits.gcol2tree_chain_ket0`: `evolve!` then starts from that basis state.

Products `⊙`, sums `+` and scalar factors (`*`, `/`) of these are expanded into a flat sum of
gate chains, and every chain is simulated on its own. A `kron_derivative` produces exactly
such a sum, with one chain per gate that depends on the variable. Anything else, e.g. a `⊗`
tree from `gcol2tree2`, is rejected.

`params` are the symbols that vary between parameter sets, in the order of the parameter
vectors passed to `evolve!`. Gate matrices and coefficients may depend on nothing else.

Gate leaves left in a gate matrix by the substitutions before are filled in from `gcol`, the
gate collection (or circuit) the tree was built from. These are a parametric gate's call such
as `P_s7qt3qc(...)`, which becomes the gate's `matrix_alt` with the call's arguments for its
parameters, and a gate symbol, which becomes its `matrix_numeric`; both are matched by the
gate's name. Without `gcol`, or for anything else unresolved (a `Differential`, a symbol that
is not a parameter), lowering stops with an error that lists the leaves.
"""
function lower_state(tree, params; gcol=nothing)
    root = Symbolics.unwrap(tree)
    terms = _expand(root, IdDict(), IdDict())
    isempty(terms) && throw(ArgumentError("the tree has no summands"))
    allequal(t.start === nothing for t in terms) ||
        _cannot_lower(root, "a sum of operators and states")
    n = _register_size(terms)

    paramset = Set{Any}(Symbolics.unwrap(p) for p in params)
    gates_by_name = gcol === nothing ? nothing : _gates_by_name(gcol)

    slots = IdDict{Any, Int}()
    slot_dims = Int[]
    slot_exprs = Any[]
    coeff_exprs = Any[]
    kets = QCSym.BasisKet[]
    summands = Summand[]
    # every summand of a derivative repeats the circuit's gates, so share their qubit lists
    qubit_lists = Dict{QCSym.GateOn, Vector{Int}}()
    for t in terms
        gates = GateRef[]
        for (op, M) in _gate_vector(t.gates)
            dim = 1 << length(op.qubits)
            slot = get!(slots, M) do
                push!(slot_exprs, _resolve_leaves(M, paramset, gates_by_name))
                push!(slot_dims, dim)
                length(slot_exprs)
            end
            slot_dims[slot] == dim || throw(DimensionMismatch(
                "the same gate matrix is used on $(slot_dims[slot]) and $(dim)-dimensional gates"))
            push!(gates, GateRef(get!(() -> collect(op.qubits), qubit_lists, op), slot))
        end
        coeff = 0
        if !_isone(t.coeff)
            _check_free(t.coeff, paramset, "a coefficient")
            coeff = something(findfirst(c -> isequal(c, t.coeff), coeff_exprs), 0)
            coeff == 0 && (push!(coeff_exprs, t.coeff); coeff = length(coeff_exprs))
        end
        start = 0
        if t.start !== nothing
            start = something(findfirst(==(t.start), kets), 0)
            start == 0 && (push!(kets, t.start); start = length(kets))
        end
        push!(summands, Summand(coeff, gates, start))
    end

    const_matrices = Union{Nothing, Matrix{ComplexF64}}[]
    param_slots = Int[]
    for (slot, M) in enumerate(slot_exprs)
        if _has_free_symbols(M)
            push!(const_matrices, nothing)
            push!(param_slots, slot)
        else
            A = _numeric_matrix(QCSym.eval_numeric(M))
            size(A) == (slot_dims[slot], slot_dims[slot]) || throw(DimensionMismatch(
                "a gate on $(trailing_zeros(slot_dims[slot])) qubit(s) needs a " *
                "$(slot_dims[slot])x$(slot_dims[slot]) matrix, got $(join(size(A), 'x'))"))
            push!(const_matrices, A)
        end
    end
    matrix_values = isempty(param_slots) ? nothing :
                    _build_values(Any[slot_exprs[s] for s in param_slots], params)
    coeff_values = isempty(coeff_exprs) ? nothing : _build_values(coeff_exprs, params)

    return LoweredState(n, collect(Any, params), summands, kets, slot_dims, const_matrices,
                        param_slots, matrix_values, coeff_values, length(coeff_exprs))
end

# The out-of-place function of `build_function`: called with a parameter vector, it returns
# the values of `exprs` in order.
function _build_values(exprs, params)
    f = Symbolics.build_function(exprs, params; expression=Val{false})
    return f isa Tuple ? first(f) : f
end

# --- expansion into gate chains -----------------------------------------------------

# The gates of one summand while expanding: a linked list, last applied gate first. Putting
# an operator in front of a chain shared in the DAG then costs O(1) instead of a copy.
struct _GateList
    op::QCSym.GateOn
    matrix::Any
    prev::Union{_GateList, Nothing}
end

# One summand while expanding. `start === nothing` makes it an operator, otherwise it starts
# from that basis state. `coeff` is 1 or a scalar expression.
struct _Term
    coeff::Any
    gates::Union{_GateList, Nothing}
    start::Union{QCSym.BasisKet, Nothing}
end

function _gate_vector(gates::Union{_GateList, Nothing})
    out = Tuple{QCSym.GateOn, Any}[]
    while gates !== nothing
        push!(out, (gates.op, gates.matrix))
        gates = gates.prev
    end
    return reverse!(out)
end

# `a`'s gates applied after `b`'s.
function _append(a::Union{_GateList, Nothing}, b::Union{_GateList, Nothing})
    nodes = _GateList[]
    while a !== nothing
        push!(nodes, a)
        a = a.prev
    end
    for node in Iterators.reverse(nodes)
        b = _GateList(node.op, node.matrix, b)
    end
    return b
end

_isone(c) = c isa Number && isone(c)
_mulcoeff(a, b) = _isone(a) ? b : _isone(b) ? a : a * b

_basis_ket(x) = x isa SymbolicUtils.BasicSymbolic && SymbolicUtils.isconst(x) &&
                SymbolicUtils.unwrap_const(x) isa QCSym.BasisKet ?
                SymbolicUtils.unwrap_const(x) : nothing

# Does x contain a gate or a basis state, i.e. is it not a plain scalar?
function _has_structure(x, memo::IdDict)
    get!(memo, x) do
        if SymbolicUtils.iscall(x)
            SymbolicUtils.operation(x) isa QCSym.GateOn ||
                any(a -> _has_structure(a, memo), SymbolicUtils.arguments(x))
        else
            _basis_ket(x) !== nothing
        end
    end::Bool
end

_cannot_lower(x, what) = throw(ArgumentError(
    "lower_state handles products, sums and scalar multiples of GateOn factors and basis " *
    "states (e.g. Circuits.gcol2tree_chain); got $(what) in $(x)"))

_expand(x, memo::IdDict, smemo::IdDict) = get!(() -> _expand_node(x, memo, smemo), memo, x)

function _expand_node(x, memo, smemo)
    k = _basis_ket(x)
    k !== nothing && return [_Term(1, nothing, k)]
    SymbolicUtils.iscall(x) || _cannot_lower(x, "a leaf where a gate or a state belongs")
    op = SymbolicUtils.operation(x)
    args = SymbolicUtils.arguments(x)
    if op isa QCSym.GateOn
        return [_Term(1, _GateList(op, only(args), nothing), nothing)]
    elseif op === QCSym.:⊙
        # the rightmost factor acts first
        terms = _expand(args[end], memo, smemo)
        for a in Iterators.reverse(args[1:end-1])
            terms = _compose(_expand(a, memo, smemo), terms, x)
        end
        return terms
    elseif op === (+)
        all(a -> _has_structure(a, smemo), args) ||
            _cannot_lower(x, "a sum of a gate or state with a scalar")
        terms = reduce(vcat, (_expand(a, memo, smemo) for a in args))
        allequal(t.start === nothing for t in terms) ||
            _cannot_lower(x, "a sum of operators and states")
        return terms
    elseif op === (*) || op === (/)
        structural = [a for a in args if _has_structure(a, smemo)]
        (length(structural) == 1 && (op === (*) || first(args) === only(structural))) ||
            _cannot_lower(x, "a product of several gates or states, or a division by one")
        scalars = [a for a in args if !_has_structure(a, smemo)]
        c = op === (*) ? prod(scalars) : 1 / only(scalars)
        return [_Term(_mulcoeff(c, t.coeff), t.gates, t.start)
                for t in _expand(only(structural), memo, smemo)]
    elseif op === QCSym.col_1
        # the first column of an operator is the operator applied to |0…0⟩
        ops = _expand(only(args), memo, smemo)
        return _compose(ops, [_Term(1, nothing, QCSym.BasisKet(_register_size(ops), 0))], x)
    end
    _cannot_lower(x, "the operation $(op)")
end

function _compose(A, B, x)
    out = _Term[]
    for a in A
        a.start === nothing || _cannot_lower(x, "a state as the left factor of ⊙")
        for b in B
            push!(out, _Term(_mulcoeff(a.coeff, b.coeff), _append(a.gates, b.gates), b.start))
        end
    end
    return out
end

function _register_size(terms)
    ns = Set{Int}()
    for t in terms
        t.start === nothing || push!(ns, t.start.n)
        g = t.gates
        while g !== nothing
            push!(ns, g.op.n)
            g = g.prev
        end
    end
    length(ns) == 1 || throw(ArgumentError(isempty(ns) ?
        "the tree holds neither gates nor states" :
        "gates and states of different register sizes $(sort!(collect(ns)))"))
    return only(ns)
end

# --- gate matrices ------------------------------------------------------------------

# Every free symbol of x, looking inside array-valued constants: a parametric gate
# substitutes in as a matrix of symbolic entries (cf. QCSym._leaf_has_vars).
function _free_symbols!(out, x, seen::IdDict)
    haskey(seen, x) && return out
    seen[x] = true
    if x isa Union{Symbolics.Num, Complex{Symbolics.Num}}
        foreach(v -> push!(out, v), Symbolics.get_variables(x))
    elseif x isa AbstractArray
        foreach(e -> _free_symbols!(out, e, seen), x)
    elseif x isa SymbolicUtils.BasicSymbolic
        if SymbolicUtils.iscall(x)
            foreach(a -> _free_symbols!(out, a, seen), SymbolicUtils.arguments(x))
        elseif SymbolicUtils.isconst(x)
            _free_symbols!(out, SymbolicUtils.unwrap_const(x), seen)
        else
            push!(out, x)
        end
    end
    return out
end

_has_free_symbols(x) = !isempty(_free_symbols!(Set{Any}(), x, IdDict()))

# Leaves of a gate matrix that keep it from being a function of the parameters alone: a
# call of a symbolic function (a parametric gate's leaf), a Differential, and any symbol
# that is not a parameter (a gate's leaf, or a variable missing from `params`).
function _unresolved(M, paramset)
    out = Any[]
    seen = IdDict()
    function walk(x)
        haskey(seen, x) && return
        seen[x] = true
        if x isa SymbolicUtils.BasicSymbolic && SymbolicUtils.iscall(x)
            op = SymbolicUtils.operation(x)
            if op isa SymbolicUtils.BasicSymbolic || op isa Symbolics.Differential
                push!(out, x)
            else
                foreach(walk, SymbolicUtils.arguments(x))
            end
        else
            for v in _free_symbols!(Set{Any}(), x, IdDict())
                v in paramset || push!(out, v)
            end
        end
    end
    walk(Symbolics.unwrap(M))
    return unique(out)
end

function _resolve_leaves(M, paramset, gates_by_name)
    left = _unresolved(M, paramset)
    if !isempty(left) && gates_by_name !== nothing
        subs = Dict{Any, Any}()
        for x in left
            A = _gate_matrix(x, gates_by_name)
            A === nothing || (subs[x] = A)
        end
        if !isempty(subs)
            M = SymbolicUtils.substitute(M, subs)
            left = _unresolved(M, paramset)
        end
    end
    isempty(left) || throw(ArgumentError(
        "a gate matrix still contains $(join(string.(left), ", ")). Substitute them, pass " *
        "them as parameters, or pass the gate collection as `gcol` for gate leaves; gate " *
        "matrix: $(M)"))
    return M
end

function _check_free(x, paramset, what)
    extra = [v for v in _free_symbols!(Set{Any}(), x, IdDict()) if !(v in paramset)]
    isempty(extra) || throw(ArgumentError(
        "$(what) depends on $(join(string.(extra), ", ")), which are not parameters: $(x)"))
    return nothing
end

function _gates_by_name(gcol)
    gcol isa QCSym.Circuits.QuantumCircuit && (gcol = gcol.gatecollection)
    byname = Dict{Symbol, Any}()
    for (_, gates) in gcol.collections, g in gates
        s = Symbolics.unwrap(g.symbol)
        byname[Symbolics.getname(SymbolicUtils.iscall(s) ? SymbolicUtils.operation(s) : s)] = g
    end
    return byname
end

# The matrix a leftover gate leaf stands for, or nothing if it is not a gate's leaf.
function _gate_matrix(x, gates_by_name)
    if SymbolicUtils.iscall(x) && SymbolicUtils.operation(x) isa SymbolicUtils.BasicSymbolic
        g = get(gates_by_name, Symbolics.getname(SymbolicUtils.operation(x)), nothing)
        (g === nothing || g.matrix_alt === nothing) && return nothing
        # the gate's own call holds its parameter symbols, in the order of x's arguments
        formal = SymbolicUtils.arguments(Symbolics.unwrap(g.symbol))
        return Symbolics.substitute(g.matrix_alt, Dict(zip(formal, SymbolicUtils.arguments(x))))
    elseif SymbolicUtils.issym(x)
        g = get(gates_by_name, Symbolics.getname(x), nothing)
        (g === nothing || g.matrix_numeric === nothing) && return nothing
        return g.matrix_numeric
    end
    return nothing
end

_to_number(x::Number) = x
_to_number(x::Symbolics.Num) = _to_number(Symbolics.value(x))
_to_number(x::Complex{Symbolics.Num}) = complex(_to_number(real(x)), _to_number(imag(x)))
function _to_number(x::SymbolicUtils.BasicSymbolic)
    SymbolicUtils.isconst(x) && return _to_number(SymbolicUtils.unwrap_const(x))
    throw(ArgumentError("expected a number, got $(x)"))
end

_numeric_matrix(A::AbstractMatrix) = Matrix{ComplexF64}(map(_to_number, A))
_numeric_matrix(A) = throw(ArgumentError("a gate matrix evaluated to $(typeof(A)), not a matrix"))
