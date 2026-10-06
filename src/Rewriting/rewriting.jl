# r2 = SymbolicUtils.@rule (~a ⊗ ~b)*(~c ⊗ ~d) => (~a*~c) ⊗ (~b*~d)
# r3 = SymbolicUtils.@rule ((~a+~b) ⊗ ~c)*(~d ⊗ ~e) => ((~a+~b)*~d) ⊗ (~c*~e)

#r = SymbolicUtils.@rule (~a::isa(a,QCSym.Gates.H_Gate) ⊗ ~b::isa(b,QCSym.Gates.H_Gate)) => 1.0

"""
    eval_numeric(ex)

Evaluate a fully substituted expression tree by applying each node's operation to its
evaluated arguments.

`SymbolicUtils.substitute` only rewrites *leaves*: after substituting the gate matrices,
the `⊗`/`⊙`/`+` nodes are still symbolic terms holding numeric arguments. Evaluating them
is what `build_function`'s generated code does, and this is the same walk done eagerly.
"""
function eval_numeric(ex)
    if !SymbolicUtils.iscall(ex)
        return SymbolicUtils.isconst(ex) ? SymbolicUtils.unwrap_const(ex) : ex
    end
    op = SymbolicUtils.operation(ex)
    return op([eval_numeric(a) for a in SymbolicUtils.arguments(ex)]...)
end

# Memoised "does this subtree mention any free variable?". Terms are hash-consed, so a
# shared subtree is literally the same object and an IdDict turns this from O(size^2)
# over the tree into O(size) over the DAG.
function _has_vars(ex, memo::IdDict)
    get!(memo, ex) do
        SymbolicUtils.iscall(ex) ?
            any(a -> _has_vars(a, memo), SymbolicUtils.arguments(ex)) :
            _leaf_has_vars(ex)
    end::Bool
end

# A parametric gate substitutes in as a *matrix of symbolic entries*, which reaches here as
# one leaf. Asking the leaf itself for variables can miss those, and folding such a leaf
# away would silently bake one sample's parameter values into the function, so look inside
# array-valued constants elementwise.
function _leaf_has_vars(ex)
    isempty(Symbolics.get_variables(ex)) || return true
    val = SymbolicUtils.isconst(ex) ? SymbolicUtils.unwrap_const(ex) : ex
    val isa AbstractArray && return any(x -> !isempty(Symbolics.get_variables(x)), val)
    return false
end

_is_foldable_op(f) = f === (⊗) || f === (⊙) || f === (col_1) || f === (+)

"""
    fold_constants(ex) -> folded

Evaluate every subtree of `ex` that has no free variables left, replacing it with its
value.

After the substitutions that precede code generation, only the parametric gates still
carry variables -- in a `zz` feature map that is the `P` gates, while every `H`, `X` and
`CX` factor is already a number. But the `⊗`/`⊙` nodes above them stay symbolic, so
`build_function` emits a call for each and the generated function rebuilds those constant
step operators *on every sample*. Folding them once here moves that work from sampling to
build time; what remains per sample is applying them, which is irreducible.

Run this last: after `kron_derivative` (which rewrites the tree and introduces the
differentiated gate leaves) and after the final substitution of numeric gate matrices.
Nothing is folded if variables are still present, so running it too early is a no-op
rather than an error.
"""
function fold_constants(ex)
    hasvars = IdDict()
    folded  = IdDict()

    function fold(e)
        SymbolicUtils.iscall(e) || return e
        get!(folded, e) do
            if !_has_vars(e, hasvars) && _is_foldable_op(SymbolicUtils.operation(e))
                return SymbolicUtils.Const{SymbolicUtils.SymReal}(eval_numeric(e))
            end
            args = SymbolicUtils.arguments(e)
            new = map(fold, args)
            # Rebuilding a term is what makes `shape` matter, so leave the node alone
            # unless folding actually changed one of its arguments.
            all(p -> p[1] === p[2], zip(new, args)) && return e
            return SymbolicUtils.maketerm(typeof(e), SymbolicUtils.operation(e), new,
                                          SymbolicUtils.metadata(e))
        end
    end

    return fold(ex)
end

"""
    count_nodes(ex)

Number of nodes in `ex` counted as a tree (shared subtrees counted once per occurrence).
Useful for seeing what [`fold_constants`](@ref) removed.
"""
count_nodes(ex) = SymbolicUtils.iscall(ex) ?
    1 + sum(count_nodes, SymbolicUtils.arguments(ex); init=0) : 1
