# derivative_zz_feature_map.jl on the gate-level path: the circuit as a chain of GateOn
# factors (gcol2tree_chain_ket0), the same outer-parameter substitution, kron_derivative and
# two substitution loops, then lower_state and evolve! on LENS instead of build_function.
#
# Needs an environment with LENS.jl, e.g. the one of the LENS extension tests:
#   julia --project=test/LENSExt -t 8 examples/zz_feature_map/derivative_zz_feature_map_lens.jl
import Symbolics
import SymbolicUtils
import StaticArrays
import LinearAlgebra
import QCSym
#import LENS
import DataFrames
import CSV
import Optim
import Random
# LENS as a package if the environment has it (e.g. brute_force_sampling/Project.toml), which
# also loads QCSym's LENS extension, i.e. QCSym.Simulation.LENSBackend; otherwise its source
if Base.find_package("LENS") === nothing
    include("/Users/adam-1aeqn8vhvpjnv4u/git_repos/LENS.jl/src/LENS.jl")
    import .LENS
else
    import LENS
end

LinearAlgebra.BLAS.set_num_threads(1)


include("../zz_feature_map/src/src.jl")

df = DataFrames.DataFrame(CSV.File("/Users/adam-1aeqn8vhvpjnv4u/git_repos/zz_tmp/ParCFD/coffe_vs_nap/power_nap_vs_coffee_effectiveness_dataset.csv"))
DataFrames.names(df)
unique(df[!,"occupation"])
const PERIODICITY_P = 4*pi
function normalize_to_const(constant, data)
    mi, ma = extrema(data)
    return collect((x-mi)*constant/ma for x in data)
end

data_in = DataFrames.DataFrame("age" => normalize_to_const(PERIODICITY_P, df[!,"age"]), 
                                "occupation" => collect(ifelse(x=="Student", 0, ifelse(x=="Working Professional", 1, 2))/3*PERIODICITY_P for x in df[!,"occupation"]),
                                "sleep_hours_previous_night" => normalize_to_const(PERIODICITY_P, df[!,"sleep_hours_previous_night"]),
                                "intervention_duration_minutes" => normalize_to_const(PERIODICITY_P, df[!,"intervention_duration_minutes"]),
                                "alertness_score_before" => normalize_to_const(PERIODICITY_P, df[!,"alertness_score_before"]),
                                "alertness_score_after" => normalize_to_const(PERIODICITY_P, df[!,"alertness_score_after"]),
                                "productivity_rating" => normalize_to_const(PERIODICITY_P, df[!,"productivity_rating"]),
                                "mood_rating" => normalize_to_const(PERIODICITY_P, df[!,"mood_rating"]),
                                "side_effects" => collect(ifelse(x=="None", 0, ifelse(x=="Grogginess",1,ifelse(x=="Anxiety",2,3)))/4*PERIODICITY_P for x in df[!,"side_effects"]))
data_out = DataFrames.DataFrame("intervention_type" => collect(ifelse(x=="Coffee", 0, 1) for x in df[!,"intervention_type"]))


#################
# GP emulator, notation of
# https://mbd.pages.rwth-aachen.de/psimpy/emulator/robustgasp.html#theory-recap-of-gp-emulation
# The {0,1} labels are emulated by GP regression, and a point is classified as 1 where the
# predictive mean exceeds 0.5.
#
# Inputs are d×n matrices with one column per sample. The kernel is supplied by the user as
#   kernel(X1, X2; params...) -> R with R[i, j] = c(X1[:, i], X2[:, j]),
# a correlation, i.e. c(x, x) = 1. Its keyword arguments are fitted by maximum likelihood,
# with β and σ² profiled out in closed form.
#################

# Separable Matérn 5/2 correlation with one range parameter per input dimension, the
# RobustGaSP default. An example of a user kernel.
function matern52_correlation(X1, X2; ψ=ones(size(X1, 1)))
    R = ones(size(X1, 2), size(X2, 2))
    for l in axes(X1, 1), j in axes(X2, 2), i in axes(X1, 2)
        s = sqrt(5) * abs(X1[l, i] - X2[l, j]) / ψ[l]
        R[i, j] *= (1 + s + s^2/3) * exp(-s)
    end
    return R
end

# correlation_gradient(kernel, X; params...) -> (R, ∂R): R = kernel(X, X; params...) and its
# derivatives, a NamedTuple with the keys of params holding ∂R/∂θ for every entry θ of the
# parameter, as one matrix for a number and as an array of matrices shaped like the
# parameter for an array. Gradient-based training needs it for the kernel.
function correlation_gradient end

# A kernel may provide containers for the correlations of the inputs X with each other,
# created once by kernel_workspace(kernel, X; gradient) (gradient=false: none for ∂R) and
# overwritten by every correlation!(workspace, kernel, X; params...) -> R and
# correlation_gradient!(workspace, kernel, X; params...) -> (R, ∂R). Kernels without them get
# nothing and are called as kernel(X, X; params...) and correlation_gradient(kernel, X; params...).
kernel_workspace(kernel, X; gradient=true) = nothing
correlation!(::Nothing, kernel, X; params...) = kernel(X, X; params...)
correlation_gradient!(::Nothing, kernel, X; params...) = correlation_gradient(kernel, X; params...)

function correlation_gradient(::typeof(matern52_correlation), X; ψ)
    d, n = size(X)
    C = [ones(n, n) for _ in 1:d]    # correlation in dimension l
    dC = [zeros(n, n) for _ in 1:d]  # and its derivative with respect to ψ[l]
    for l in 1:d, j in 1:n, i in 1:n
        s = sqrt(5) * abs(X[l, i] - X[l, j]) / ψ[l]
        C[l][i, j] = (1 + s + s^2/3) * exp(-s)
        dC[l][i, j] = s^2 * (1 + s) * exp(-s) / (3ψ[l])
    end
    R = reduce((A, B) -> A .* B, C)
    dR = [reduce((A, B) -> A .* B, (m == l ? dC[m] : C[m] for m in 1:d)) for l in 1:d]
    return R, (ψ=dR,)
end

#################
# ZZ feature map kernel: the fidelity c(x, x') = |⟨ψ(x)|ψ(x')⟩|² of the states
# ψ(x) = U(x)|0…0⟩ of the forward ZZ feature map (add_circuit_zzfeaturemap!), one qubit per
# input dimension. Every P gate has its own weight: in layer l the P gate on qubit i gets the
# angle η_p[i, l] x_i, and the P gate on qubit i between the CX gates gets
# η_zz[i-1, l] (π - x_{i-1})(π - x_i). The weights η_p (num_qubits × num_layers) and η_zz
# ((num_qubits - 1) × num_layers) are the kernel parameters.
#################

struct ZZFeatureMapKernel{B<:QCSym.Simulation.AbstractStateBackend}
    num_qubits::Int
    num_layers::Int
    lowered::QCSym.Simulation.LoweredState                # ψ
    dlowered::Vector{QCSym.Simulation.LoweredState}       # ∂ψ/∂η for η in vcat(vec(η_p), vec(η_zz))
    backend::B
end

# Builds the circuit, its tree and the derivative trees with respect to every weight as in
# derivative_zz_feature_map_lens.jl, and lowers them for evolve!. Parameter vectors of the
# lowered states are vcat(x, vec(η_p), vec(η_zz)).
function ZZFeatureMapKernel(num_qubits::Int; num_layers::Int,
                            backend=QCSym.Simulation.ReferenceBackend())
    num_qubits >= 2 || throw(ArgumentError("the ZZ feature map needs at least 2 qubits"))
    data_params = [Symbolics.variable(Symbol("x$i")) for i in 1:num_qubits]
    η_p = [Symbolics.variable(Symbol("η_p$(i)_$(l)")) for i in 1:num_qubits, l in 1:num_layers]
    η_zz = [Symbolics.variable(Symbol("η_zz$(i)_$(l)")) for i in 1:num_qubits-1, l in 1:num_layers]
    weight_params = vcat(vec(η_p), vec(η_zz))

    qc = QCSym.Circuits.QuantumCircuit(name="ZZFeatureMap")
    qreg = QCSym.Circuits.add_qreg(qc, "q_reg_1", num_qubits)
    add_circuit_zzfeaturemap!(qc, qreg, num_layers, 1; for_back_config=(true, false))
    steps_p, _ = get_steps_of_p_and_zz_gates(qc)

    param_gates_P = qc.gatecollection.collections[QCSym.Gates.P_Gate]
    qubit(pg) = pg.qubits_t[1].index_global
    dict_to_subs_outer_pars = Dict()
    for pg in param_gates_P
        i = qubit(pg)
        single = pg.step in steps_p
        # the layer is the rank of the gate's step among the P gates of its kind on its qubit
        l = count(g -> qubit(g) == i && (g.step in steps_p) == single && g.step <= pg.step, param_gates_P)
        angle = single ? phi_single(data_params[i], η_p[i, l]) :
                         phi_double(data_params[i-1], data_params[i], η_zz[i-1, l])
        dict_to_subs_outer_pars[pg.parameters["λ"]["sym"]] = exp(1im * angle)
    end
    dict_to_subsP_syms = Dict(pg.symbol => Symbolics.substitute(pg.symbol, dict_to_subs_outer_pars) for pg in param_gates_P)
    dict_to_subsP_mats = Dict(pg.symbol => Symbolics.substitute(pg.matrix_alt, dict_to_subs_outer_pars) for pg in param_gates_P)
    dict_to_subsP_remaining = Dict(v => dict_to_subsP_mats[k] for (k, v) in dict_to_subsP_syms)
    dict_to_subs_P_derv = Dict(Symbolics.Differential(v.args[1])(v) => StaticArrays.SMatrix{2,2}(Symbolics.derivative(dict_to_subsP_mats[k], v.args[1])) for (k,v) in dict_to_subsP_syms)

    exprtree = SymbolicUtils.substitute(QCSym.Circuits.gcol2tree_chain_ket0(qc), dict_to_subs_outer_pars)
    params = vcat(data_params, weight_params)
    lowered = QCSym.Simulation.lower_state(SymbolicUtils.substitute(exprtree, dict_to_subsP_remaining), params; gcol=qc)
    dlowered = QCSym.Simulation.LoweredState[]
    for η in weight_params
        dexprtree = QCSym.kron_derivative(exprtree, η)
        dexprtree = SymbolicUtils.substitute(dexprtree, dict_to_subs_P_derv)
        dexprtree = SymbolicUtils.substitute(dexprtree, dict_to_subsP_remaining)
        push!(dlowered, QCSym.Simulation.lower_state(dexprtree, params; gcol=qc))
    end
    return ZZFeatureMapKernel(num_qubits, num_layers, lowered, dlowered, backend)
end

# writes the parameter sets vcat(x, η) of the lowered states, one per column x of X, into P
function zz_params!(P, k::ZZFeatureMapKernel, X, η)
    size(X, 1) == k.num_qubits || throw(DimensionMismatch(
        "the kernel has $(k.num_qubits) qubits, the inputs $(size(X, 1)) dimensions"))
    P[1:k.num_qubits, :] .= X
    P[k.num_qubits+1:end, :] .= η
    return P
end

# the weights in the order of the lowered states' parameter vectors
function zz_weights(k::ZZFeatureMapKernel, η_p, η_zz)
    size(η_p) == (k.num_qubits, k.num_layers) && size(η_zz) == (k.num_qubits - 1, k.num_layers) ||
        throw(DimensionMismatch("one weight per P gate: η_p must be $(k.num_qubits)×$(k.num_layers) " *
                                "and η_zz $(k.num_qubits - 1)×$(k.num_layers), got $(size(η_p)) and $(size(η_zz))"))
    return vcat(vec(η_p), vec(η_zz))
end

# The containers for the correlations of N inputs with each other and their derivatives,
# filled in place by evolve! and mul!.
struct ZZKernelWorkspace
    P::Matrix{Float64}            # parameter sets vcat(x, η), one column per input
    S::Matrix{ComplexF64}         # ψ, one column per input
    G::Matrix{ComplexF64}         # S'S, the overlaps ⟨ψ_i|ψ_j⟩
    R::Matrix{Float64}            # |G|²
    # for gradients only, empty otherwise
    dS::Matrix{ComplexF64}        # ∂ψ/∂η of one weight η at a time
    A::Matrix{ComplexF64}         # dS'S, the overlaps ⟨∂ψ_i|ψ_j⟩
    dR::Vector{Matrix{Float64}}   # ∂R/∂η for η in vcat(vec(η_p), vec(η_zz))
end

function kernel_workspace(k::ZZFeatureMapKernel, X; gradient=true)
    N, D, nη = size(X, 2), 1 << k.num_qubits, length(k.dlowered)
    m = gradient ? N : 0
    return ZZKernelWorkspace(zeros(k.num_qubits + nη, N), zeros(ComplexF64, D, N),
                             zeros(ComplexF64, N, N), zeros(N, N),
                             zeros(ComplexF64, gradient ? D : 0, m), zeros(ComplexF64, m, m),
                             [zeros(m, m) for _ in 1:(gradient ? nη : 0)])
end

# Every state is simulated once and stored; the correlations are the dot products of all
# pairs of them, S'S.
function correlation!(ws::ZZKernelWorkspace, k::ZZFeatureMapKernel, X; η_p, η_zz)
    zz_params!(ws.P, k, X, zz_weights(k, η_p, η_zz))
    QCSym.Simulation.evolve!(ws.S, k.lowered, ws.P; backend=k.backend)
    LinearAlgebra.mul!(ws.G, ws.S', ws.S)
    ws.R .= abs2.(ws.G)
    return ws.R
end

function correlation_gradient!(ws::ZZKernelWorkspace, k::ZZFeatureMapKernel, X; η_p, η_zz)
    isempty(ws.dR) && throw(ArgumentError("the workspace has no containers for ∂R; create it with gradient=true"))
    correlation!(ws, k, X; η_p=η_p, η_zz=η_zz)
    for (dl, D) in zip(k.dlowered, ws.dR)
        QCSym.Simulation.evolve!(ws.dS, dl, ws.P; backend=k.backend)
        LinearAlgebra.mul!(ws.A, ws.dS', ws.S)
        # ∂G = A + A' and ∂|G|² = 2 Re(conj(G) ∂G)
        D .= 2 .* real.(conj.(ws.G) .* (ws.A .+ ws.A'))
    end
    n_p = length(η_p)
    return ws.R, (η_p=reshape(view(ws.dR, 1:n_p), size(η_p)),
                  η_zz=reshape(view(ws.dR, n_p+1:length(ws.dR)), size(η_zz)))
end

# allocating versions, e.g. for predictions
correlation_gradient(k::ZZFeatureMapKernel, X; params...) =
    correlation_gradient!(kernel_workspace(k, X), k, X; params...)

function (k::ZZFeatureMapKernel)(X1, X2; η_p, η_zz)
    X2 === X1 && return correlation!(kernel_workspace(k, X1; gradient=false), k, X1; η_p=η_p, η_zz=η_zz)
    η = zz_weights(k, η_p, η_zz)
    states(X) = QCSym.Simulation.evolve!(zeros(ComplexF64, 1 << k.num_qubits, size(X, 2)), k.lowered,
                                         zz_params!(zeros(k.num_qubits + length(η), size(X, 2)), k, X, η);
                                         backend=k.backend)
    return abs2.(states(X1)' * states(X2))
end

struct GPEmulator{K, P<:NamedTuple}
    kernel::K
    params::P            # kernel parameters
    nugget::Float64
    X::Matrix{Float64}   # training inputs, one column per sample
    y::Vector{Float64}   # training outputs
    F::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}   # of R + nugget*I
    β::Vector{Float64}
    σ2::Float64
    α::Vector{Float64}   # R⁻¹(y - Hβ), shared by all predictions
    loglik::Float64      # profile log-likelihood
end

# h(x): the mean is constant, so H is a column of ones
gp_h(X) = ones(size(X, 2), 1)

# generalized least squares estimate of β, F is the Cholesky factorization of R
function gp_beta(H, F, y)
    RiH = F \ H
    return (H' * RiH) \ (RiH' * y)
end

# r(x*): correlations between the training inputs and each column of Xs, n×m
gp_r(gp::GPEmulator, Xs) = gp.kernel(gp.X, Xs; gp.params...)

# m'(x*): predictive mean
gp_m(gp::GPEmulator, Xs; r=gp_r(gp, Xs)) = gp_h(Xs) * gp.β + r' * gp.α

# c'(x*) = c(x*, x*) - r'R⁻¹r: predictive correlation, with c(x*, x*) = 1
gp_c(gp::GPEmulator, Xs; r=gp_r(gp, Xs)) = max.(1 .- vec(sum(r .* (gp.F \ r); dims=1)), 0)

# predictive mean m' and variance σ²c' at each column of Xs
function gp_predict(gp::GPEmulator, Xs)
    r = gp_r(gp, Xs)
    return gp_m(gp, Xs; r=r), gp.σ2 .* gp_c(gp, Xs; r=r)
end

gp_classify(gp::GPEmulator, Xs; threshold=0.5) = Int.(gp_m(gp, Xs) .> threshold)

# β, σ² and the profile log-likelihood for fixed kernel parameters; nothing if R + nugget*I
# is not positive definite. R + nugget*I is factorized in place, in a new matrix or in
# `buffer`, a matrix of the size of R, which the GPEmulator then refers to.
function gp_fit(kernel, X, y, params, nugget; R=kernel(X, X; params...), buffer=nothing)
    n = length(y)
    A = buffer === nothing ? similar(R, Float64) : buffer
    copyto!(A, R)
    for i in 1:n
        A[i, i] += nugget
    end
    F = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(A); check=false)
    LinearAlgebra.issuccess(F) || return nothing
    H = gp_h(X)
    β = gp_beta(H, F, y)
    res = y - H * β
    α = F \ res
    σ2 = LinearAlgebra.dot(res, α) / n
    σ2 > 0 || return nothing
    loglik = -(n * log(σ2) + LinearAlgebra.logdet(F) + n * (1 + log(2π))) / 2
    return GPEmulator(kernel, params, Float64(nugget), Matrix{Float64}(X), Vector{Float64}(y),
                      F, β, σ2, α, loglik)
end

# Gradient of the profile log-likelihood for ∂R/∂θ of every flattened kernel parameter θ. As
# β and σ² are stationary points of the likelihood, ∂ℓ/∂θ = (αᵀ ∂R α / σ² - tr(R⁻¹ ∂R)) / 2.
function gp_loglik_gradient(gp::GPEmulator, dR)
    Ri = inv(gp.F)
    return [(LinearAlgebra.dot(gp.α, D, gp.α) / gp.σ2 - LinearAlgebra.dot(Ri, D)) / 2 for D in dR]
end

# The optimizer works on one flat vector; the kernel parameters are a NamedTuple of numbers
# and arrays.
flatten_params(p::NamedTuple) = reduce(vcat, (vec(collect(Float64, v)) for v in values(p)); init=Float64[])

function unflatten_params(template::NamedTuple, v)
    i = 0
    vals = map(values(template)) do t
        x = t isa Number ? v[i+1] : reshape(v[i+1:i+length(t)], size(t))
        i += length(t)
        return x
    end
    return NamedTuple{keys(template)}(vals)
end

# the ∂R of correlation_gradient in the order of flatten_params(template)
flatten_gradient(template::NamedTuple, dR) =
    reduce(vcat, (t isa Number ? [dR[k]] : vec(dR[k]) for (k, t) in pairs(template)); init=Matrix{Float64}[])

# Bounds may be given for some of the parameters only, as numbers or as arrays shaped like
# the parameter.
bounds_vector(template, bounds, default) = flatten_params(NamedTuple{keys(template)}(
    map(k -> get(bounds, k, default) .+ zero.(template[k]), keys(template))))

# The bounds are enforced by optimizing an unconstrained z mapped into (lo, hi), log-scaled
# for one-sided bounds.
function from_unconstrained(z, lo, hi)
    isfinite(lo) && isfinite(hi) && return lo + (hi - lo) / (1 + exp(-z))
    isfinite(lo) && return lo + exp(z)
    isfinite(hi) && return hi - exp(z)
    return z
end

function to_unconstrained(x, lo, hi)
    lo < x < hi || throw(ArgumentError("initial parameter $x is not strictly inside ($lo, $hi)"))
    isfinite(lo) && isfinite(hi) && return log((x - lo) / (hi - x))
    isfinite(lo) && return log(x - lo)
    isfinite(hi) && return log(hi - x)
    return x
end

# dx/dz of from_unconstrained
function from_unconstrained_derivative(z, lo, hi)
    if isfinite(lo) && isfinite(hi)
        s = 1 / (1 + exp(-z))
        return (hi - lo) * s * (1 - s)
    end
    isfinite(lo) && return exp(z)
    isfinite(hi) && return -exp(z)
    return one(z)
end

# Maximum likelihood fit of the kernel parameters, starting at params0. method and options
# are passed to Optim.optimize; Nelder-Mead needs no derivatives of the kernel. Gradient-based
# methods (e.g. Optim.LBFGS()) get the gradient of the likelihood from
# correlation_gradient(kernel, X; params...); with gradient=false Optim uses finite
# differences instead. A Ref passed as `result` receives Optim's result.
function gp_train(kernel, X, y; params0::NamedTuple, lower::NamedTuple=(;), upper::NamedTuple=(;),
                  nugget=1e-8, method=Optim.NelderMead(), options=Optim.Options(iterations=10000, show_trace=true, show_every=1),
                  gradient=method isa Optim.FirstOrderOptimizer, result=nothing)
    isempty(params0) && return gp_fit(kernel, X, y, params0, nugget)
    lo = bounds_vector(params0, lower, -Inf)
    hi = bounds_vector(params0, upper, Inf)
    to_params(z) = unflatten_params(params0, from_unconstrained.(z, lo, hi))
    # the kernel's containers, reused by every evaluation
    ws = kernel_workspace(kernel, X; gradient=gradient)
    fit(params) = gp_fit(kernel, X, y, params, nugget; R=correlation!(ws, kernel, X; params...))
    function negloglik(z)
        gp = fit(to_params(z))
        return isnothing(gp) ? Inf : -gp.loglik
    end
    # Optim asks for the value, the gradient or both; the gradient's kernel evaluation also
    # gives the value
    function negloglik_fg!(F, G, z)
        G === nothing && return negloglik(z)
        params = to_params(z)
        R, dR = correlation_gradient!(ws, kernel, X; params...)
        gp = gp_fit(kernel, X, y, params, nugget; R=R)
        if isnothing(gp)
            fill!(G, 0)
            return Inf
        end
        G .= .-gp_loglik_gradient(gp, flatten_gradient(params0, dR)) .* from_unconstrained_derivative.(z, lo, hi)
        return -gp.loglik
    end
    objective = gradient ? Optim.NLSolversBase.only_fg!(negloglik_fg!) : negloglik
    res = Optim.optimize(objective, to_unconstrained.(flatten_params(params0), lo, hi), method, options)
    println(res)
    result === nothing || (result[] = res)
    Optim.converged(res) || @warn "maximum likelihood fit of the kernel parameters did not converge" res
    return fit(to_params(Optim.minimizer(res)))
end

# random split of 1:n into training and test indices
function train_test_split(n; train_fraction=0.8, rng=Random.default_rng())
    perm = Random.randperm(rng, n)
    n_train = round(Int, train_fraction * n)
    return perm[1:n_train], perm[n_train+1:end]
end


X = permutedims(Matrix{Float64}(data_in))
y = Float64.(data_out[!, "intervention_type"])
idx_train, idx_test = train_test_split(length(y); train_fraction=0.8, rng=Random.Xoshiro(1))

# The training below is skipped when another script includes this one for its definitions
# and data, after setting GP_DEFINITIONS_ONLY = true (benchmark_kernels.jl).
if !(@isdefined(GP_DEFINITIONS_ONLY) && GP_DEFINITIONS_ONLY)
    gp = gp_train(matern52_correlation, X[:, idx_train], y[idx_train];
                  params0=(ψ=fill(PERIODICITY_P / 2, size(X, 1)),), lower=(ψ=fill(0.0, size(X, 1)),), upper=(ψ=fill(10.0, size(X, 1)),), nugget=1e-8)
    y_pred = gp_classify(gp, X[:, idx_test])
    println("ψ = ", gp.params.ψ)
    println("β = ", gp.β, ", σ² = ", gp.σ2, ", log-likelihood = ", gp.loglik)
    println("test accuracy: ", sum(y_pred .== y[idx_test]) / length(idx_test))

    # ZZ feature map kernel with one qubit per input dimension, trained with the gradient of the
    # likelihood
    zz_kernel = ZZFeatureMapKernel(size(X, 1); num_layers=3)
    gp_zz = gp_train(zz_kernel, X[:, idx_train], y[idx_train];
                     params0=(η_p=ones(size(X, 1), zz_kernel.num_layers), η_zz=ones(size(X, 1) - 1, zz_kernel.num_layers)),
                     lower=(η_p=0.0, η_zz=0.0), upper=(η_p=2.0, η_zz=2.0), nugget=1e-8,
                     method = Optim.NelderMead())
                     #  method=Optim.Adam(alpha=0.1,
                    #                     beta_mean=0.6,
                    #                     beta_var=0.7,))
    y_pred_zz = gp_classify(gp_zz, X[:, idx_test])
    println("η_p = ", gp_zz.params.η_p)
    println("η_zz = ", gp_zz.params.η_zz)
    println("β = ", gp_zz.β, ", σ² = ", gp_zz.σ2, ", log-likelihood = ", gp_zz.loglik)
    println("test accuracy: ", sum(y_pred_zz .== y[idx_test]) / length(idx_test))


    exit()
end

