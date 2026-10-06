import SymbolicUtils
import StaticArrays
import RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

# `Symbolics.build_function` emits one Julia statement per node of the (CSE'd)
# expression DAG. For a circuit unitary that is thousands to tens of thousands of
# statements in a single method body, and since type inference and the LLVM passes are
# superlinear in body size, the first call disappears into the JIT for a long time.
#
# The DAG is extremely repetitive though: almost every statement is the *same*
# operation (a 2x2 matrix product, a Kronecker product, an add) applied to different
# slots. So instead of emitting `nnodes` statements we emit a handful of
# `SymbolicUtils.Code.ForLoop`s, each running a single statement shape over a
# precomputed table of operand/result indices. Julia then compiles a number of loop
# bodies that depends only on how many *kinds* of statement the tree contains, not on
# how many there are; the DAG itself moves out of the code and into data (the tables).
#
# Values live in "banks": flat vectors, one per (shape, realness) class, so every loop
# body stays type stable. See `_bank_eltype`.

# ------------------------------------------------------------------------------
# shape / realness lattice
# ------------------------------------------------------------------------------

# A node's shape is `SCALAR` for numbers, or `(nrows, ncols)` for arrays. Column
# vectors (as produced by `col_1`) are normalised to `(n, 1)`.
const SCALAR = (0, 0)

_is_scalar(s) = s === SCALAR

_normalize_shape(sz::Tuple{Int}) = (sz[1], 1)
_normalize_shape(sz::Tuple{Int,Int}) = (sz[1], sz[2])
_normalize_shape(sz::Tuple) = throw(ArgumentError(
    "only 1- and 2-dimensional shapes are supported, got $sz"))

# Result shape of an operation. Anything not listed here is assumed to return a
# number, so keep this in sync with the array-valued operations
# `Symbolic_Tensor_Algebra` can put in a tree.
function _infer_shape(f, argshapes)
    if f === QCSym.:⊙
        ms = filter(!_is_scalar, argshapes)
        isempty(ms) && return SCALAR
        return (first(ms)[1], last(ms)[2])
    elseif f === QCSym.:⊗
        r = 1
        c = 1
        for s in argshapes
            _is_scalar(s) && continue
            r *= s[1]
            c *= s[2]
        end
        return (r == 1 && c == 1) ? SCALAR : (r, c)
    elseif f === QCSym.col_1
        s = argshapes[1]
        return _is_scalar(s) ? SCALAR : (s[1], 1)
    elseif f === (+) || f === (-)
        for s in argshapes
            _is_scalar(s) || return s
        end
        return SCALAR
    elseif f === (*)
        ms = filter(!_is_scalar, argshapes)
        isempty(ms) && return SCALAR
        return (first(ms)[1], last(ms)[2])
    end
    return SCALAR
end

# Operations that map reals to reals. `complex` is deliberately absent: it is the one
# way a real-valued subtree turns complex in these trees.
const _REAL_PRESERVING = Set{Any}([
    +, -, *, /, ^, inv, muladd, ifelse, max, min, conj,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    exp, log, log2, log10, sqrt, cbrt,
])

# Real whatever the argument is.
const _REAL_VALUED = Set{Any}([abs, abs2, real, imag, angle])

function _infer_real(f, argreals)
    f in _REAL_VALUED && return true
    f in _REAL_PRESERVING && return all(argreals)
    return false
end

# ------------------------------------------------------------------------------
# DAG linearisation
# ------------------------------------------------------------------------------

mutable struct DagNode
    kind::Symbol            # :const, :param or :call
    f::Any                  # operation, for :call
    args::Vector{Int}       # indices of argument nodes, for :call
    val::Any                # literal value, for :const
    pidx::Int               # position in `params`, for :param
    shape::Tuple{Int,Int}
    isreal::Bool
    bank::Int               # index into the bank list, filled by `assign_banks!`
    slot::Int               # index within that bank
end

_callnode(f, args, shape, isreal) = DagNode(:call, f, args, nothing, 0, shape, isreal, 0, 0)
_constnode(v, shape, isreal) = DagNode(:const, nothing, Int[], v, 0, shape, isreal, 0, 0)
_paramnode(i) = DagNode(:param, nothing, Int[], nothing, i, SCALAR, true, 0, 0)

struct Dag
    nodes::Vector{DagNode}
    cache::Dict{Any,Int}                    # symbolic subexpression => node
    opcache::Dict{Tuple{Any,Vector{Int}},Int}  # (operation, operands) => node
    params::Dict{Any,Int}                   # param symbol => position in the argument vector
end

Dag(params) = Dag(DagNode[], Dict{Any,Int}(), Dict{Tuple{Any,Vector{Int}},Int}(),
    Dict{Any,Int}(SymbolicUtils.unwrap(p) => i for (i, p) in enumerate(params)))

function _push!(dag::Dag, node::DagNode)
    push!(dag.nodes, node)
    return length(dag.nodes)
end

# Associative operations are rebuilt as binary trees so the set of distinct statement
# shapes stays tiny: an n-ary `+` would otherwise need its own loop body for every
# arity that happens to occur in the tree.
_is_associative(f) = f === (+) || f === (*) || f === QCSym.:⊙ || f === QCSym.:⊗

"""
    linearize(expr, params) -> (dag, root)

Flatten the symbolic DAG rooted at `expr` into a topologically ordered node list,
hash-consing structurally equal subexpressions (this is the CSE pass) and inferring a
shape and a realness flag for every node.
"""
function linearize(expr, params)
    dag = Dag(params)
    root = _visit!(dag, SymbolicUtils.unwrap(expr))
    return dag, root
end

function _visit!(dag::Dag, x)
    cached = get(dag.cache, x, 0)
    cached != 0 && return cached
    idx = _build_node!(dag, x)
    dag.cache[x] = idx
    return idx
end

function _build_node!(dag::Dag, x)
    xu = SymbolicUtils.unwrap(x)

    if xu isa SymbolicUtils.BasicSymbolic
        if SymbolicUtils.iscall(xu)
            f = SymbolicUtils.operation(xu)
            rawargs = collect(SymbolicUtils.arguments(xu))
            # `array_literal`'s first argument is a size tuple, not a value: it is
            # baked into the node's shape instead of becoming a node of its own.
            if f === SymbolicUtils.array_literal
                sz = _normalize_shape(SymbolicUtils.unwrap_const(rawargs[1]))
                argidx = Int[_visit!(dag, a) for a in rawargs[2:end]]
                return _emit_node!(dag, f, argidx, sz)
            end
            argidx = Int[_visit!(dag, a) for a in rawargs]
            return _emit_call!(dag, f, argidx)
        end
        if SymbolicUtils.issym(xu)
            pidx = get(dag.params, xu, 0)
            pidx == 0 && throw(ArgumentError(
                "free symbol $(xu) is not among the supplied parameters"))
            return _push!(dag, _paramnode(pidx))
        end
        xu = SymbolicUtils.unwrap_const(xu)
    end

    return _push!(dag, _const_from(xu))
end

function _const_from(v)
    if v isa Number
        return _constnode(v, SCALAR, v isa Real)
    elseif v isa AbstractArray
        return _constnode(v, _normalize_shape(size(v)), false)
    end
    throw(ArgumentError("unsupported literal of type $(typeof(v)) in expression tree"))
end

function _emit_call!(dag::Dag, f, argidx::Vector{Int})
    if _is_associative(f) && length(argidx) > 2
        acc = argidx[1]
        for k in 2:length(argidx)
            acc = _emit_node!(dag, f, Int[acc, argidx[k]])
        end
        return acc
    end
    return _emit_node!(dag, f, argidx)
end

# Keyed on operands rather than on the symbolic term, so that the partial sums and
# products introduced by binarisation are shared too. `forced_shape` is for
# `array_literal`, whose result shape comes from the term rather than its operands.
function _emit_node!(dag::Dag, f, argidx::Vector{Int}, forced_shape = nothing)
    key = ((f, forced_shape), argidx)
    cached = get(dag.opcache, key, 0)
    cached != 0 && return cached

    isr = false
    shape = forced_shape
    if shape === nothing
        shape = _infer_shape(f, Tuple{Int,Int}[dag.nodes[i].shape for i in argidx])
        isr = _is_scalar(shape) && _infer_real(f, Bool[dag.nodes[i].isreal for i in argidx])
    end
    idx = _push!(dag, _callnode(f, argidx, shape, isr))
    dag.opcache[key] = idx
    return idx
end

# ------------------------------------------------------------------------------
# banks
# ------------------------------------------------------------------------------

struct Bank
    shape::Tuple{Int,Int}
    isreal::Bool
    eltype::Type
    nodes::Vector{Int}      # node stored in each slot
end

# 2x2 (and 2x1) values dominate these trees and are what `⊙` is specialised for, so
# keep them as `StaticArrays`: the bank is then an inline, pointer-free buffer.
# Larger blocks grow as 2^nqubits and stay plain `Matrix`, mirroring the cutoff
# `Symbolic_Tensor_Algebra` applies to `⊗` itself.
const _MAX_STATIC_BANK_LEN = 4

function _bank_eltype(shape, isreal)
    if _is_scalar(shape)
        return isreal ? Float64 : ComplexF64
    end
    r, c = shape
    if r * c <= _MAX_STATIC_BANK_LEN
        return StaticArrays.SMatrix{r,c,ComplexF64,r * c}
    end
    return Matrix{ComplexF64}
end

"""
    assign_banks!(dag) -> (banks, nconst)

Give every node a (bank, slot) home. Constants are placed in the *leading* slots of
their bank so the whole constant pool can be installed with one `copyto!` from a
prebuilt vector, rather than one assignment per constant.
"""
function assign_banks!(dag::Dag)
    banks = Bank[]
    bykey = Dict{Tuple{Tuple{Int,Int},Bool},Int}()

    function place!(node, i)
        # only scalars get split by realness; arrays are complex throughout
        isr = _is_scalar(node.shape) ? node.isreal : false
        key = (node.shape, isr)
        b = get(bykey, key, 0)
        if b == 0
            push!(banks, Bank(node.shape, isr, _bank_eltype(node.shape, isr), Int[]))
            b = length(banks)
            bykey[key] = b
        end
        push!(banks[b].nodes, i)
        node.bank = b
        node.slot = length(banks[b].nodes)
        return nothing
    end

    for (i, node) in enumerate(dag.nodes)
        node.kind === :const && place!(node, i)
    end
    nconst = Int[length(b.nodes) for b in banks]

    for (i, node) in enumerate(dag.nodes)
        node.kind === :const || place!(node, i)
    end
    # banks created after the constant pass have no constants
    append!(nconst, zeros(Int, length(banks) - length(nconst)))

    return banks, nconst
end

"""
    bank_templates(dag, banks, nconst) -> Tuple

The constant pool of each bank, as one vector per bank. The generated function
allocates its scratch banks `undef` and `copyto!`s these over their leading slots;
every other slot is written by exactly one loop iteration before anything reads it.
"""
function bank_templates(dag::Dag, banks::Vector{Bank}, nconst::Vector{Int})
    tmpls = Any[]
    for (bi, bank) in enumerate(banks)
        v = Vector{bank.eltype}(undef, nconst[bi])
        for s in 1:nconst[bi]
            v[s] = _to_bank(bank.eltype, dag.nodes[bank.nodes[s]].val)
        end
        push!(tmpls, v)
    end
    return Tuple(tmpls)
end

_to_bank(::Type{T}, v) where {T<:Number} = convert(T, v)
_to_bank(::Type{T}, v) where {T<:StaticArrays.SMatrix} = T(v)
_to_bank(::Type{Matrix{ComplexF64}}, v) = Matrix{ComplexF64}(v)
# Column vectors are normalised to shape (n, 1), so a constant one lands in a matrix bank.
# `Matrix{T}(::Vector)` has no method, hence the reshape. Constant array-valued nodes only
# arise once `fold_constants` has run -- before that every constant is a 2x2 gate matrix.
_to_bank(::Type{Matrix{ComplexF64}}, v::AbstractVector) =
    Matrix{ComplexF64}(reshape(v, length(v), 1))

# ------------------------------------------------------------------------------
# scheduling
# ------------------------------------------------------------------------------

# A group is one emitted `ForLoop`: nodes sharing an operation, an arity, a result
# bank and the same argument banks, and mutually independent.
struct Group
    f::Any
    nargs::Int
    outbank::Int
    argbanks::Vector{Int}
    nodes::Vector{Int}
end

_signature(dag::Dag, i::Int) = (dag.nodes[i].f, length(dag.nodes[i].args),
    dag.nodes[i].bank, Tuple(dag.nodes[a].bank for a in dag.nodes[i].args))

"""
    schedule(dag) -> Vector{Group}

Greedy list schedule. Each round picks the statement signature with the most ready
nodes, then grows that batch by *closure*: any further node of the same signature that
becomes ready as the batch is filled joins it too.

The closure step is what makes this worth doing. These trees are deep and narrow — a
per-qubit gate chain is a strictly serial run of `⊙` — so a schedule that only batched
mutually independent nodes would emit a loop per chain link and barely beat
straight-line code. A `ForLoop` runs its iterations in order, so a whole serial chain
is legal in one loop as long as the index tables list it in dependency order, which is
exactly the order the closure produces.
"""
function schedule(dag::Dag)
    n = length(dag.nodes)
    ready = falses(n)
    done = falses(n)
    ndeps = zeros(Int, n)
    users = [Int[] for _ in 1:n]

    for (i, node) in enumerate(dag.nodes)
        if node.kind === :call
            ndeps[i] = length(node.args)
            for a in node.args
                push!(users[a], i)
            end
        else
            # constants and parameters are seeded before any loop runs
            done[i] = true
        end
    end
    for i in 1:n
        !done[i] && ndeps[i] == 0 && (ready[i] = true)
    end
    for i in 1:n
        done[i] || continue
        for u in users[i]
            ndeps[u] -= 1
            ndeps[u] == 0 && (ready[u] = true)
        end
    end

    # signatures are compared once per ready node per round, so intern them to plain
    # integers up front instead of rebuilding tuples in the inner loops
    sigids = zeros(Int, n)
    sigmap = Dict{Any,Int}()
    for i in 1:n
        dag.nodes[i].kind === :call || continue
        sigids[i] = get!(sigmap, _signature(dag, i), length(sigmap) + 1)
    end

    groups = Group[]
    counts = zeros(Int, length(sigmap))
    remaining = count(!, done)

    while remaining > 0
        fill!(counts, 0)
        for i in 1:n
            ready[i] && (counts[sigids[i]] += 1)
        end
        sig, best = 0, 0
        for (s, c) in enumerate(counts)
            c > best && ((sig, best) = (s, c))
        end
        best == 0 && throw(ErrorException(
            "expression DAG is cyclic or malformed: $remaining nodes never became ready"))

        batch = Int[]
        for i in 1:n
            ready[i] && sigids[i] == sig || continue
            push!(batch, i)
            ready[i] = false
            done[i] = true
        end

        k = 1
        while k <= length(batch)
            i = batch[k]
            k += 1
            for u in users[i]
                ndeps[u] -= 1
                ndeps[u] == 0 || continue
                if sigids[u] == sig
                    push!(batch, u)
                    done[u] = true
                else
                    ready[u] = true
                end
            end
        end
        remaining -= length(batch)

        head = dag.nodes[batch[1]]
        push!(groups, Group(head.f, length(head.args), head.bank,
            Int[dag.nodes[a].bank for a in head.args], batch))
    end

    return groups
end

# ------------------------------------------------------------------------------
# code emission
# ------------------------------------------------------------------------------

const _ARG = :ˍ₋arg1
const _TABS = :ˍ₋tabs
const _TMPL = :ˍ₋tmpl
const _IVAR = :ˍ₋i

_banksym(b::Int) = Symbol("ˍ₋b", b)
_tabsym(t::Int) = Symbol("ˍ₋t", t)

"""
    emit(dag, banks, nconst, groups, root) -> (Func, tables)

Turn the schedule into a `SymbolicUtils.Code.Func` whose body is a fixed number of
`SymbolicUtils.Code.ForLoop`s, plus the index tables those loops read. The tables are
passed in as an argument rather than interpolated as array literals: splicing them
into the AST would only move the size problem from statements into constants.
"""
function emit(dag::Dag, banks::Vector{Bank}, nconst::Vector{Int},
              groups::Vector{Group}, root::Int)
    tables = Vector{Int32}[]
    setup = SymbolicUtils.Code.Assignment[]
    body = Any[]

    function table!(v::Vector{Int32})
        push!(tables, v)
        ti = length(tables)
        push!(setup, SymbolicUtils.Code.Assignment(_tabsym(ti), Expr(:ref, _TABS, ti)))
        return ti
    end

    for (bi, bank) in enumerate(banks)
        push!(setup, SymbolicUtils.Code.Assignment(_banksym(bi),
            Expr(:call, Vector{bank.eltype}, UndefInitializer(), length(bank.nodes))))
        nconst[bi] == 0 && continue
        push!(body, Expr(:call, copyto!, _banksym(bi), 1,
            Expr(:ref, _TMPL, bi), 1, nconst[bi]))
    end

    # parameters: ˍ₋arg1[k] -> the slot holding parameter k. Parameters the tree does
    # not actually mention have no slot, so both ends are tabulated rather than
    # assuming the argument vector lines up with the bank.
    paramnodes = [nd for nd in dag.nodes if nd.kind === :param]
    if !isempty(paramnodes)
        sti = table!(Int32[nd.slot for nd in paramnodes])
        ati = table!(Int32[nd.pidx for nd in paramnodes])
        push!(body, SymbolicUtils.Code.ForLoop(_IVAR, :(1:$(length(paramnodes))),
            Expr(:(=),
                Expr(:ref, _banksym(paramnodes[1].bank), Expr(:ref, _tabsym(sti), _IVAR)),
                Expr(:ref, _ARG, Expr(:ref, _tabsym(ati), _IVAR)))))
    end

    for g in groups
        oti = table!(Int32[dag.nodes[i].slot for i in g.nodes])
        atis = Int[table!(Int32[dag.nodes[dag.nodes[i].args[k]].slot for i in g.nodes])
                   for k in 1:g.nargs]

        call = Expr(:call, g.f)
        if g.f === SymbolicUtils.array_literal
            # the size tuple is not a value node; it is recovered from the result
            # bank, which by construction has exactly one shape
            push!(call.args, banks[g.outbank].shape)
        end
        for k in 1:g.nargs
            push!(call.args,
                Expr(:ref, _banksym(g.argbanks[k]), Expr(:ref, _tabsym(atis[k]), _IVAR)))
        end

        push!(body, SymbolicUtils.Code.ForLoop(_IVAR, :(1:$(length(g.nodes))),
            Expr(:(=),
                Expr(:ref, _banksym(g.outbank), Expr(:ref, _tabsym(oti), _IVAR)),
                call)))
    end

    rootnode = dag.nodes[root]
    push!(body, Expr(:ref, _banksym(rootnode.bank), rootnode.slot))

    func = SymbolicUtils.Code.Func(
        Any[_ARG, _TABS, _TMPL],
        SymbolicUtils.Code.Assignment[],
        SymbolicUtils.Code.Let(setup,
            SymbolicUtils.Code.LiteralExpr(Expr(:block, body...)), false))

    return func, tables
end

# `@inbounds` is applied to the whole body in one go rather than per loop: every index
# comes from a table this module built, so they are correct by construction, and the
# bounds checks would otherwise sit in the innermost loops.
function _wrap_inbounds!(ex::Expr)
    ex.args[2] = Expr(:block,
        Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), ex.args[2]))
    return ex
end

# ------------------------------------------------------------------------------
# public entry point
# ------------------------------------------------------------------------------

"""
    LoopFunction

Callable wrapper returned by [`build_function_loops`](@ref). Holds the generated
function together with the index tables and bank templates it reads, so it can be
called as `f(parameter_vector)` just like a `Symbolics.build_function` result.
"""
struct LoopFunction{F,T,M}
    f::F
    tables::T
    templates::M
    expr::Expr          # the generated source, kept for inspection
    nloops::Int
    nnodes::Int
end

(lf::LoopFunction)(p) = lf.f(p, lf.tables, lf.templates)

Base.show(io::IO, lf::LoopFunction) =
    print(io, "LoopFunction(", lf.nnodes, " nodes in ", lf.nloops, " loops)")

"""
    build_function_loops(expr, params; expression = Val{false})

Loop-based alternative to `Symbolics.build_function` for `QCSym` expression trees.

Where `build_function` emits one statement per DAG node — a body that grows with the
circuit and dominates first-call JIT time — this emits a fixed handful of
`SymbolicUtils.Code.ForLoop`s driven by index tables, so the amount of code Julia has
to compile stops growing with the circuit.

With `expression = Val{true}` the generated `Expr` is returned together with the
tables and templates it expects; with `Val{false}` (the default) a callable
[`LoopFunction`](@ref) is returned.

Supported operations are the ones `Symbolic_Tensor_Algebra` puts in a tree (`⊙`, `⊗`,
`col_1`, `array_literal`, `+`, `*`) plus scalar maths; see `_infer_shape`.
"""
function build_function_loops(expr, params; expression = Val{false})
    dag, root = linearize(expr, params)
    banks, nconst = assign_banks!(dag)
    groups = schedule(dag)
    func, tables = emit(dag, banks, nconst, groups, root)
    templates = bank_templates(dag, banks, nconst)

    ex = _wrap_inbounds!(SymbolicUtils.Code.toexpr(func))

    expression === Val{true} && return ex, tables, templates

    ex.args[1] = Expr(:call, gensym(:qcsym_loops), ex.args[1].args...)
    rgf = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ex)
    return LoopFunction(rgf, tables, templates, ex, length(groups), length(dag.nodes))
end
