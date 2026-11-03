# Space-filling scan of the parameter space of the ZZ feature map kernel of GP_coffe_vs_nap.jl.
# For a number of layers, all weights of the circuit form one box, (2d - 1)·layers weights for
# d inputs, sampled by QuasiMonteCarlo.jl. A sample fixes the weights, so nothing is trained:
# the GP is fitted with them (β and σ² in closed form), and its profile log-likelihood, β, σ²,
# training and test accuracy are written to a CSV file, one row per sample, one file per
# number of layers. The columns eta_p_<i>_<l> and eta_zz_<i>_<l> are the weights η_p[i, l] (P
# gate on qubit i in layer l) and η_zz[i, l] (P gate between the CX gates of qubits i, i+1).
#
# The states run on LENS through QCSym's symbolic path: QCSym.Simulation.evolve! with
# LENSBackend(threads=1), one thread per state. Each of the `julia -t x` threads takes whole
# samples: it simulates their states one after another into its own containers, created once,
# and computes the likelihood with BLAS on one thread, so x threads keep x cores busy.
#
#   cd examples/GP_coffe_vs_nap/brute_force_sampling
#   julia --project=. -t 32 brute_force_sampling.jl
#   julia --project=. -t 32 brute_force_sampling.jl --layers=2 --samples=4096 --sampler=LatinHypercube
#   julia --project=. -t 32 brute_force_sampling.jl --layers=1 --upper=0.5     # all weights in [0, 0.5]
# Options:
#   --layers=1,2,3      numbers of layers, a scan and a CSV file each
#   --samples=n         samples per scan, by default N_SAMPLES
#   --sampler=name      one of SAMPLERS, by default Sobol
#   --lower=a,b,...     lower bounds of the weights: one number for all of them, or one per weight
#   --upper=a,b,...     in the order of the CSV columns (only with a single number of layers)
#   --seed=s            seeds the random samplers (LatinHypercube, Uniform)
#   --resume            skip the samples already in the CSV files; failed ones run again
#   --csv=prefix        write <prefix>_L<layers>.csv, by default brute_force_sampling_L<layers>.csv
#                       next to this script
# From the REPL, include this file and call e.g. run_scan(2; n_samples=1024).
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)   # every thread computes its own likelihoods
import LENS                             # as a package, which loads QCSym's LENS extension
import QuasiMonteCarlo
import Distributions
import Dates
import Printf

const GP_DEFINITIONS_ONLY = true
include(joinpath(@__DIR__, "..", "GP_coffe_vs_nap.jl"))

#################
# Settings
#################
const LAYERS = [1, 2, 3]
const N_SAMPLES = 2^14             # Sobol points are best balanced for powers of 2
const SAMPLER = "Sobol"
# Bounds of the weights: a number for all of them, a vector with one entry per weight (in the
# order of the CSV columns), or a Dict from the number of layers to either.
const LOWER = 0.0
const UPPER = 1.0
const SEED = 1
const NUGGET = 1e-8
const CSV_PREFIX = joinpath(@__DIR__, "brute_force_sampling")
const CHUNK = 100                  # rows per write to the CSV file
const PROGRESS_INTERVAL = 10.0     # seconds between runtime estimates

# The sampling methods of QuasiMonteCarlo.jl, and a distribution of Distributions.jl, which
# QuasiMonteCarlo samples from directly.
const SAMPLERS = Dict(
    "Sobol" => () -> QuasiMonteCarlo.SobolSample(),
    "Halton" => () -> QuasiMonteCarlo.HaltonSample(),
    "LatticeRule" => () -> QuasiMonteCarlo.LatticeRuleSample(),
    "Golden" => () -> QuasiMonteCarlo.GoldenSample(),
    "LatinHypercube" => () -> QuasiMonteCarlo.LatinHypercubeSample(),
    "Uniform" => () -> Distributions.Uniform(),
)

#################
# Parameter space
#################

# the CSV names of the weights, in the order of the lowered states' parameters vcat(vec(η_p), vec(η_zz))
weight_names(d, num_layers) = vcat(vec(["eta_p_$(i)_$(l)" for i in 1:d, l in 1:num_layers]),
                                   vec(["eta_zz_$(i)_$(l)" for i in 1:d-1, l in 1:num_layers]))

# The bounds of the n weights of `num_layers` layers from a setting like LOWER.
function bounds_of(spec, num_layers, n)
    spec isa AbstractDict && (spec = spec[num_layers])
    spec isa Number && return fill(Float64(spec), n)
    length(spec) == n || throw(ArgumentError(
        "$(length(spec)) bounds given for the $(n) weights of $(num_layers) layer(s)"))
    return Float64.(spec)
end

# the weights as the kernel parameters of GP_coffe_vs_nap.jl
function kernel_params(k::ZZFeatureMapKernel, η)
    n_p = k.num_qubits * k.num_layers
    return (η_p=reshape(η[1:n_p], k.num_qubits, k.num_layers),
            η_zz=reshape(η[n_p+1:end], k.num_qubits - 1, k.num_layers))
end

#################
# Evaluation of a sample
#################

# The containers of one thread, created once and overwritten by every sample it takes.
struct Worker
    Ptr::Matrix{Float64}          # parameter sets vcat(x, η) of the training inputs
    Pte::Matrix{Float64}          # and of the test inputs
    Ψtr::Matrix{ComplexF64}       # their states, one column per input, simulated by LENS
    Ψte::Matrix{ComplexF64}
    Gtr::Matrix{ComplexF64}       # overlaps ⟨ψ_i|ψ_j⟩ of the training states
    Gte::Matrix{ComplexF64}       # and of the training with the test states
    R::Matrix{Float64}            # |Gtr|², the correlations of the training inputs
    r::Matrix{Float64}            # |Gte|², those of the training with the test inputs
end

function Worker(k::ZZFeatureMapKernel, n_train, n_test)
    n_params, D = k.num_qubits + length(k.dlowered), 1 << k.num_qubits
    return Worker(zeros(n_params, n_train), zeros(n_params, n_test),
                  zeros(ComplexF64, D, n_train), zeros(ComplexF64, D, n_test),
                  zeros(ComplexF64, n_train, n_train), zeros(ComplexF64, n_train, n_test),
                  zeros(n_train, n_train), zeros(n_train, n_test))
end

# The GP with the weights η and its accuracies, on the calling thread only; nothing if
# R + nugget*I is not positive definite.
function evaluate!(w::Worker, k::ZZFeatureMapKernel, η, data)
    zz_params!(w.Ptr, k, data.Xtr, η)
    zz_params!(w.Pte, k, data.Xte, η)
    QCSym.Simulation.evolve!(w.Ψtr, k.lowered, w.Ptr; backend=k.backend, ntasks=1)
    QCSym.Simulation.evolve!(w.Ψte, k.lowered, w.Pte; backend=k.backend, ntasks=1)
    # Ψtr'Ψtr is Hermitian: herk computes its upper triangle only, half of the work of mul!
    LinearAlgebra.BLAS.herk!('U', 'C', 1.0, w.Ψtr, 0.0, w.Gtr)
    LinearAlgebra.mul!(w.Gte, w.Ψtr', w.Ψte)
    w.R .= abs2.(w.Gtr)
    LinearAlgebra.copytri!(w.R, 'U')
    w.r .= abs2.(w.Gte)
    gp = gp_fit(k, data.Xtr, data.ytr, kernel_params(k, η), NUGGET; R=w.R)
    gp === nothing && return nothing
    # the predictive mean of the inputs whose correlations with the training inputs are r
    accuracy(X, r, y) = sum((gp_m(gp, X; r=r) .> 0.5) .== (y .== 1)) / length(y)
    return (loglik=gp.loglik, beta=gp.β[1], sigma2=gp.σ2,
            train_accuracy=accuracy(data.Xtr, w.R, data.ytr),
            test_accuracy=accuracy(data.Xte, w.r, data.yte))
end

const NO_RESULT = (loglik=missing, beta=missing, sigma2=missing,
                   train_accuracy=missing, test_accuracy=missing)

#################
# CSV files
#################

csv_path(prefix, num_layers) = "$(prefix)_L$(num_layers).csv"

# Appends the rows, with the header if the file is new. A file with other columns is left alone.
function append_rows(path, rows)
    if isfile(path) && filesize(path) > 0
        header = join(string.(keys(first(rows))), ",")
        open(readline, path) == header || error("$(path) has other columns than this scan writes; pass another --csv")
        CSV.write(path, rows; append=true)
    else
        CSV.write(path, rows)
    end
end

# The samples of this scan already in the file, apart from failed ones. Checks that the file
# holds this scan: the same sampler, number of samples and seed, and the same weights.
function done_samples(path, scan, S, names)
    isfile(path) && filesize(path) > 0 || return Set{Int}()
    f = CSV.File(path; stringtype=String)
    for r in f
        (r.sampler, r.n_samples, r.seed) == (scan.sampler, scan.n_samples, scan.seed) || error(
            "$(path) holds another scan ($(r.sampler), $(r.n_samples) samples, seed $(r.seed)); pass another --csv")
    end
    for r in Iterators.take(f, 1)
        all(isapprox(r[Symbol(n)], S[j, r.sample]; rtol=1e-12) for (j, n) in enumerate(names)) || error(
            "the weights in $(path) differ from this scan's, other bounds?; pass another --csv")
    end
    return Set{Int}(r.sample for r in f if r.status != "error")
end

#################
# Scan
#################

function format_duration(s)
    s = round(Int, s)
    s < 60 && return "$(s) s"
    s < 3600 && return Printf.@sprintf("%d min %02d s", s ÷ 60, s % 60)
    return Printf.@sprintf("%d h %02d min", s ÷ 3600, (s % 3600) ÷ 60)
end

# Scans the weights of `num_layers` layers, the samples on all threads, and appends a row per
# sample to `csv`, CHUNK rows at a time, printing an estimate of the remaining time every
# PROGRESS_INTERVAL seconds from the rate measured so far.
function run_scan(num_layers; n_samples=N_SAMPLES, sampler=SAMPLER, lower=LOWER, upper=UPPER,
                  seed=SEED, csv=csv_path(CSV_PREFIX, num_layers), resume=false)
    haskey(SAMPLERS, sampler) || throw(ArgumentError(
        "unknown sampler $(sampler), use one of $(sort(collect(keys(SAMPLERS))))"))
    data = (Xtr=X[:, idx_train], ytr=y[idx_train], Xte=X[:, idx_test], yte=y[idx_test])
    d = size(X, 1)
    names = weight_names(d, num_layers)
    lb, ub = bounds_of(lower, num_layers, length(names)), bounds_of(upper, num_layers, length(names))
    all(lb .< ub) || throw(ArgumentError("every lower bound must be below its upper bound"))

    Random.seed!(seed)
    S = QuasiMonteCarlo.sample(n_samples, lb, ub, SAMPLERS[sampler]())
    scan = (num_layers=num_layers, sampler=sampler, n_samples=n_samples, seed=seed)
    done = resume ? done_samples(csv, scan, S, names) : Set{Int}()
    todo = filter(!in(done), 1:n_samples)
    println("$(num_layers) layer(s): $(length(names)) weights, $(n_samples) $(sampler) samples",
            isempty(done) ? "" : ", $(length(done)) of them already in $(csv)")
    isempty(todo) && return

    t_build = @elapsed k = ZZFeatureMapKernel(d; num_layers=num_layers,
                                              backend=QCSym.Simulation.LENSBackend(threads=1))
    workers = [Worker(k, length(data.ytr), length(data.yte)) for _ in 1:Threads.nthreads()]
    # the first evaluation compiles the kernel's generated functions, so that it does not
    # count towards the rate
    t_compile = @elapsed evaluate!(workers[1], k, S[:, todo[1]], data)
    println("    kernel built in ", format_duration(t_build), ", compiled in ", format_duration(t_compile),
            "; ", Threads.nthreads(), " threads, one sample each at a time, BLAS on ",
            LinearAlgebra.BLAS.get_num_threads(), " thread(s)")

    rows = Channel{Any}(Inf)
    next = Threads.Atomic{Int}(1)
    failed = Threads.Atomic{Bool}(false)
    tasks = map(workers) do w
        Threads.@spawn while (j = Threads.atomic_add!(next, 1)) <= length(todo)
            i = todo[j]
            status, result, t = "ok", NO_RESULT, NaN
            try
                t = @elapsed m = evaluate!(w, k, view(S, :, i), data)
                m === nothing ? (status = "not_positive_definite") : (result = m)
            catch e
                status = "error"
                # the first error in full, the others only in the CSV file
                Threads.atomic_xchg!(failed, true) ||
                    @error "sample $(i) failed" exception=(e, catch_backtrace())
            end
            put!(rows, (scan..., sample=i, status=status, result..., time=t,
                        NamedTuple{Tuple(Symbol.(names))}(Tuple(S[:, i]))...))
        end
    end

    buffer = NamedTuple[]
    t0 = time()
    last_report = t0
    for n_done in 1:length(todo)
        push!(buffer, take!(rows))
        if length(buffer) == CHUNK || n_done == length(todo)
            append_rows(csv, buffer)
            empty!(buffer)
        end
        now = time()
        if now - last_report >= PROGRESS_INTERVAL || n_done == length(todo)
            elapsed, left = now - t0, length(todo) - n_done
            rate = n_done / elapsed
            println("    ", n_done, "/", length(todo), " samples (",
                    round(100 * n_done / length(todo); digits=1), " %) in ", format_duration(elapsed),
                    ", ", round(rate; sigdigits=3), " samples/s",
                    left == 0 ? "" : string(", about ", format_duration(left / rate), " left, done around ",
                                           Dates.format(Dates.now() + Dates.Second(round(Int, left / rate)), "HH:MM")))
            last_report = now
        end
    end
    foreach(wait, tasks)
    println("    written to ", csv)
end

#################
# Command line
#################

if abspath(PROGRAM_FILE) == @__FILE__
    opts = Dict{String, String}()
    for a in ARGS
        startswith(a, "--") || error("unknown argument $(a), see the top of $(@__FILE__)")
        key, value = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "")
        key in ("layers", "samples", "sampler", "lower", "upper", "seed", "resume", "csv") ||
            error("unknown option --$(key), see the top of $(@__FILE__)")
        opts[key] = value
    end
    numbers(v) = (x = parse.(Float64, split(v, ',')); length(x) == 1 ? only(x) : x)
    layers = haskey(opts, "layers") ? parse.(Int, split(opts["layers"], ',')) : LAYERS
    lower = haskey(opts, "lower") ? numbers(opts["lower"]) : LOWER
    upper = haskey(opts, "upper") ? numbers(opts["upper"]) : UPPER
    length(layers) > 1 && (lower isa AbstractVector || upper isa AbstractVector) &&
        error("bounds per weight need a single number of layers, the number of weights depends on it")
    prefix = get(opts, "csv", CSV_PREFIX)
    for num_layers in layers
        run_scan(num_layers; n_samples=parse(Int, get(opts, "samples", string(N_SAMPLES))),
                 sampler=get(opts, "sampler", SAMPLER), lower=lower, upper=upper,
                 seed=parse(Int, get(opts, "seed", string(SEED))),
                 csv=csv_path(prefix, num_layers), resume=haskey(opts, "resume"))
    end
end
