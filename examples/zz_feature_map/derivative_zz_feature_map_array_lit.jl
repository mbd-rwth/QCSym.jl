import Symbolics
import SymbolicUtils
import BenchmarkTools
import StaticArrays
import LoopVectorization
import DataFrames
import CSV
import LinearAlgebra
#using Revise
import QCSym


function sample!(o, f, samples)
    for i in axes(o,2)
       o[:,i] = f(@view samples[:,i])[:,1]
    end
end


a, b, c = Symbolics.@variables a::Real b::Real c::Real
function f_of_outer_parms(i)
    return a + b^2 + c^3 + i
end

function f_adj_outer_parms(i)
    return a^3 + b^2 + c + i
end



function main(;num_qubits::Int, num_layers::Int, num_samples::Int, cse::Bool, parallel)


    qc = QCSym.Circuits.QuantumCircuit(name="TestCircuit")
    qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)

    #################
    # Build data encoding i
    #################
    step_id = 1
    for _ in 1:num_layers
        for i in 1:num_qubits
            QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false)
            #QCSym.Circuits.add_gate(qc, QCSym.Gates.RY_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
        end
        step_id += 1
        for i in 1:num_qubits
            QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("λ"=> 0.0))
            #QCSym.Circuits.add_gate(qc, QCSym.Gates.RY_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
        end
        step_id += 1
        for i in 1:num_qubits-1
            QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
            step_id += 1
            QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i+1]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
            #QCSym.Circuits.add_gate(qc, QCSym.Gates.RY_Gate, qubits_t=[qreg[i+1]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
            step_id += 1
            QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
            step_id += 1    
        end
    end

    #################
    # Build data encoding j (adjoint)
    #################
    num_p_gates_so_far = length(qc.gatecollection.collections[QCSym.Gates.P_Gate])
    # for _ in 1:num_layers
    #     for i in 1:num_qubits-1
    #         QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
    #         step_id += 1
    #         QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i+1]], step=step_id, is_treat_numeric_only=false, param_values=Dict("θ"=> 0.0))
    #         step_id += 1
    #         QCSym.Circuits.add_gate(qc, QCSym.Gates.CX_Gate, qubits_t=[qreg[i+1]], qubits_c=[qreg[i]], step=step_id, is_treat_numeric_only=false)
    #         step_id += 1    
    #     end
    #     for i in 1:num_qubits
    #         QCSym.Circuits.add_gate(qc, QCSym.Gates.P_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false, param_values=Dict("λ"=> 0.0))
    #     end
    #     step_id += 1
    #     for i in 1:num_qubits
    #         QCSym.Circuits.add_gate(qc, QCSym.Gates.H_Gate, qubits_t=[qreg[i]], step=step_id, is_treat_numeric_only=false)
    #     end
    # end

    #################
    # End data encoding
    #################

    #abcd = QCSym.Circuits.gcol2tree(qc.gatecollection)
    t_build_exprtree = @timed abcd2 = QCSym.Circuits.gcol2tree2(qc.gatecollection)
    abcd2_init = copy(abcd2)
    p1 = qc.gatecollection.collections[QCSym.Gates.P_Gate][1] 
    param_gates_P = qc.gatecollection.collections[QCSym.Gates.P_Gate]
    param_angles_P = [pg.parameters["λ"]["sym"] for pg in param_gates_P]
    dict_to_subsP = Dict(pg.symbol => pg.matrix_alt for pg in param_gates_P)
    dict_to_subs_outer_pars = Dict(pa => (i>num_p_gates_so_far ? exp(-1im*f_of_outer_parms(i)) : exp(1im*f_of_outer_parms(i))) for (i, pa) in enumerate(param_angles_P))
    dict_to_subsP = Dict(pg.symbol => Symbolics.substitute(pg.matrix_alt, dict_to_subs_outer_pars) for pg in param_gates_P)
    dict_to_subsP_syms = Dict(k => Symbolics.substitute(k, dict_to_subs_outer_pars) for (k, v) in dict_to_subsP)
    dict_to_subsP_mats = Dict(k => Symbolics.substitute(v, dict_to_subs_outer_pars) for (k, v) in dict_to_subsP)
    dict_to_subsP_remaining = Dict(v => dict_to_subsP_mats[k] for (k, v) in dict_to_subsP_syms)
    abcd2 = SymbolicUtils.substitute(abcd2, dict_to_subs_outer_pars)
    t_derivative = @timed abcd2 = QCSym.kron_derivative(abcd2, b)

    t_convert_exprtree = @timed begin
    dict_to_subs_P_derv = Dict(Symbolics.Differential(v.args[1])(v) => StaticArrays.SMatrix{2,2}(Symbolics.derivative(dict_to_subsP_mats[k], v.args[1])) for (k,v) in dict_to_subsP_syms)

    abcd2 = SymbolicUtils.substitute(abcd2, dict_to_subs_P_derv)
    abcd2_deriv = copy(abcd2)
    abcd2 = SymbolicUtils.substitute(abcd2, dict_to_subsP_remaining)

    vars = collect(v for v in Symbolics.get_variables(abcd2))

    function get_val_for_sym(s)
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


    vars_vals = Dict(v => get_val_for_sym(v) for v in vars)
    vars_vals = Dict(k => v for (k,v) in vars_vals if !isnothing(v))

    abcd2_unsubs = copy(abcd2)
    abcd2 = SymbolicUtils.substitute(abcd2, vars_vals)

    params = collect(v for v in Symbolics.get_variables(abcd2))
    end

    num_samples = num_samples
    inputs = rand(length(params), num_samples)
    outputs = StaticArrays.MMatrix{2^num_qubits, num_samples, ComplexF64}(undef)

    ef = Symbolics.build_function(abcd2, params; cse=cse, parallel=parallel, expression=Val{true}, checkbounds=false, nanmath=false, force_SA=true, convert_oop=false)

    t_code_generation = @timed bf = Symbolics.build_function(abcd2, params; cse=cse, parallel=parallel, expression=Val{false}, checkbounds=false, nanmath=false, force_SA=true, convert_oop=false, iip_config=(true, true), postprocess_fbody=ex->println("fbody = ", ex))

    t_sampling = @timed sample!(outputs, bf, inputs)
    return Dict("build_exprtree" => t_build_exprtree, "derivative" => t_derivative, "convert_exprtree" => t_convert_exprtree, "code_generation" => t_code_generation, "sampling" => t_sampling,
    "bf" => bf, "ef" => ef, "abcd2" => abcd2, "params" => params, "abcd2_init" => abcd2_init, "abcd2_unsubs" => abcd2_unsubs, "abcd2_deriv" => abcd2_deriv, "dict_to_subs_P_derv" => dict_to_subs_P_derv)
    #return ("build_exprtree" => t_build_exprtree, "code_generation" => t_code_generation, "sampling" => t_sampling)
end

out = main(num_qubits=2, num_layers=1, num_samples=100000, cse=true, parallel=Symbolics.MultithreadedForm())

out["convert_exprtree"]
out["sampling"]
out["code_generation"]
out["ef"]
out["bf"]
out["abcd2"]
out["abcd2_init"]
out["abcd2_deriv"]
out["build_exprtree"]

@code_warntype out["bf"](rand(length(out["params"])))

@code_warntype(Symbolics.RuntimeGeneratedFunctions.generated_callfunc(out["bf"], rand(length(out["params"]))))

BenchmarkTools.@btime out["bf"](rand(length(out["params"])))

cd(pwd() * "/..")

open(pwd() * "/QCSym.jl/output.txt", "w") do io
    redirect_stdout(io) do
        println(@code_warntype(Symbolics.RuntimeGeneratedFunctions.generated_callfunc(out["bf"], rand(length(out["params"])))))
        #println(methods(*))
    end
end



Symbolics.build_function([out["abcd2"]], out["params"]; cse=true, parallel=Symbolics.SerialForm(), expression=Val{false}, checkbounds=false, nanmath=false, force_SA=true, convert_oop=false, iip_config=(true, true))[1](rand(length(out["params"])))




typeof(Symbolics.terms(out["abcd2"])[1].args[2].args[2].args[2].args[2].args[1])

typeof(StaticArrays.MMatrix{2,2, Int64})

SymbolicUtils.Code.function_to_expr_no_nanmath(@nospecialize(op), O::SymbolicUtils.BasicSymbolic{T}, st) where {T} = begin
    expr = Expr(:call)
    if op isa SymbolicUtils.BasicSymbolic{T}
        push!(expr.args, SymbolicUtils.manual_dispatch_toexpr(op, st))
    else
        push!(expr.args, op)
    end
    args = parent(SymbolicUtils.arguments(O))
    SymbolicUtils.@union_split_smallvec args for arg in args
        push!(expr.args, Symbolics.toexpr(arg, st))
    end
    return expr
end

SymbolicUtils.operation(Symbolics.terms(out["abcd2"])[1].args[2].args[2].args[2].args[2].args[1])

SymbolicUtils.StaticArrays = StaticArrays
mytoexpr(op, st) = begin
    println("op = ", op)
    try 
        if SymbolicUtils.operation(op) == SymbolicUtils.array_literal
            println("op is array literal")
            args = SymbolicUtils.arguments(op)
            expr = Base.Expr(:call)
            push!(expr.args, :(Symbolics.StaticArrays.MMatrix{2,2, ComplexF64}))
            for arg in args[2:end]
                push!(expr.args, Symbolics.toexpr(arg))
            end
            return expr
        else
            return Symbolics.toexpr(op, st)
        end
    catch
        return Symbolics.toexpr(op, st)
    end

end

SymbolicUtils.Code.manual_dispatch_toexpr(@nospecialize(x), st) = begin
    if x isa SymbolicUtils.BasicSymbolic{SymbolicUtils.SymReal}
        mytoexpr(x, st)
    elseif x isa Vector{SymbolicUtils.BasicSymbolic{SymbolicUtils.SymReal}}
        Symbolics.toexpr(x, st)
    elseif x isa Matrix{SymbolicUtils.BasicSymbolic{SymbolicUtils.SymReal}}
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Symbol
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Expr
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.Let
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.Assignment
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.DestructuredArgs
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.MakeArray
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.SetArray
        Symbolics.toexpr(x, st)
    elseif x isa SymbolicUtils.Code.SetArray
        Symbolics.toexpr(x, st)
    else
        Symbolics.toexpr(x, st)
    end
end

Symbolics.toexpr(Symbolics.terms(out["abcd2"])[1].args[2].args[2].args[2].args[2].args[1])
