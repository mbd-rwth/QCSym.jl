# Benchmark of the GP classifier of GP_coffe_vs_nap.jl: the ZZ feature map kernel against the
# Matérn 5/2 kernel, for every combination of kernel, number of layers (ZZ only), optimizer,
# iteration limit and seed, all on the same train/test split. Every run appends one row to
# the CSV file as soon as it ends, so the rows of earlier runs survive a crash or an
# interrupted script. Failed and non-converged runs get a row like any other: `status` is ok
# or failed, `converged` is true or false (false for a run that failed before the optimizer
# returned), `stopped_by` is Optim's termination code (e.g. Iterations, GradientNorm,
# NelderMeadCriterion, Time) or Error, and a failed run keeps the results it got to.
#
# Run all cases, or select some by their values or by the numbers --list prints:
#   julia --project=. -t 8 examples/GP_coffe_vs_nap/benchmark_kernels.jl
#   julia --project=. -t 8 examples/GP_coffe_vs_nap/benchmark_kernels.jl --kernels=zz --layers=1,2 --optimizers=LBFGS,NelderMead --iterations=50
#   julia --project=. -t 8 examples/GP_coffe_vs_nap/benchmark_kernels.jl --list
#   julia --project=. -t 8 examples/GP_coffe_vs_nap/benchmark_kernels.jl --ids=3,7
# Options:
#   --kernels, --layers, --optimizers, --iterations, --seeds
#                     comma-separated values to use instead of the defaults below
#   --ids=...         only these cases, numbered as --list prints them for the same other options
#   --list            print the selected cases and exit
#   --resume          skip cases that already have a successful row in the CSV file
#   --csv=path        the CSV file, by default benchmark_kernels.csv next to this script
#   --time-limit=s    stop each run after s seconds (Optim's time_limit)
# From the REPL, include this file and call e.g.
#   run_benchmarks(select_cases(kernels=["zz"], layers=[2], optimizers=["LBFGS"]))
import Dates
import Logging

const GP_DEFINITIONS_ONLY = true
include(joinpath(@__DIR__, "GP_coffe_vs_nap.jl"))

#################
# Settings: the values of every case dimension used unless others are selected
#################
const KERNELS = ["matern", "zz"]
const LAYERS = [1, 2, 3, 4, 5]                        # ZZ feature map only
const OPTIMIZERS = ["NelderMead", "ParticleSwarm", "SimulatedAnnealing",
                    "LBFGS", "BFGS", "ConjugateGradient", "GradientDescent",
                    "AcceleratedGradientDescent", "MomentumGradientDescent",
                    "Adam", "AdaMax", "NGMRES", "OACCEL"]
const ITERATIONS = [10, 50, 200, 500, 1000]
const SEEDS = [1]                               # seed the global RNG before every run
const NUGGET = 1e-8
const CSV_PATH = joinpath(@__DIR__, "benchmark_kernels.csv")

# The Optim methods selectable by name: all of Optim's unconstrained ones for several
# variables that need no Hessian, so not Newton and NewtonTrustRegion. The box-constrained
# ones (Fminbox, LBFGSB, SAMIN, IPNewton) take the bounds themselves, gp_train maps them away.
# Gradient-free: NelderMead, ParticleSwarm, SimulatedAnnealing; the others use the gradient.
# Adam and AdaMax share the settings of GP_coffe_vs_nap.jl (AdaMax's default step 0.002
# barely moves within a few hundred iterations); the others have Optim's defaults.
const OPTIMIZER_METHODS = Dict(
    "NelderMead" => () -> Optim.NelderMead(),
    "ParticleSwarm" => () -> Optim.ParticleSwarm(),
    "SimulatedAnnealing" => () -> Optim.SimulatedAnnealing(),
    "LBFGS" => () -> Optim.LBFGS(),
    "BFGS" => () -> Optim.BFGS(),
    "ConjugateGradient" => () -> Optim.ConjugateGradient(),
    "GradientDescent" => () -> Optim.GradientDescent(),
    "AcceleratedGradientDescent" => () -> Optim.AcceleratedGradientDescent(),
    "MomentumGradientDescent" => () -> Optim.MomentumGradientDescent(),
    "Adam" => () -> Optim.Adam(alpha=0.1, beta_mean=0.6, beta_var=0.7),
    "AdaMax" => () -> Optim.AdaMax(alpha=0.1, beta_mean=0.6, beta_var=0.7),
    "NGMRES" => () -> Optim.NGMRES(),
    "OACCEL" => () -> Optim.OACCEL(),
)

# initial values and bounds of the kernel parameters, those of GP_coffe_vs_nap.jl
function kernel_setup(kernel_name, num_layers, d)
    kernel_name == "matern" &&
        return (params0=(ψ=fill(PERIODICITY_P / 2, d),), lower=(ψ=0.0,), upper=(ψ=10.0,))
    kernel_name == "zz" &&
        return (params0=(η_p=ones(d, num_layers), η_zz=ones(d - 1, num_layers)),
                lower=(η_p=0.0, η_zz=0.0), upper=(η_p=2.0, η_zz=2.0))
    throw(ArgumentError("unknown kernel $(kernel_name), use one of $(KERNELS)"))
end

#################
# Cases
#################

# Every combination of the given values: the ZZ kernel's for every number of layers, the
# Matérn kernel's once, with num_layers = missing.
function select_cases(; kernels=KERNELS, layers=LAYERS, optimizers=OPTIMIZERS,
                      iterations=ITERATIONS, seeds=SEEDS)
    for k in kernels
        k in ("matern", "zz") || throw(ArgumentError("unknown kernel $(k), use matern or zz"))
    end
    for o in optimizers
        haskey(OPTIMIZER_METHODS, o) || throw(ArgumentError(
            "unknown optimizer $(o), use one of $(sort(collect(keys(OPTIMIZER_METHODS))))"))
    end
    all(>=(1), layers) || throw(ArgumentError("the number of layers must be at least 1"))
    return [(kernel=k, num_layers=l, optimizer=o, iterations=it, seed=s)
            for k in kernels for l in (k == "zz" ? layers : [missing])
            for o in optimizers for it in iterations for s in seeds]
end

case_key(c) = (c.kernel, c.num_layers, c.optimizer, c.iterations, c.seed)

#################
# CSV file
#################

# Appends one row, with the header if the file is new. A file with other columns, e.g. from
# an older version of this script, is left alone.
function append_row(path, row::NamedTuple)
    if isfile(path) && filesize(path) > 0
        header = join(string.(keys(row)), ",")
        open(readline, path) == header || error(
            "$(path) has other columns than this benchmark writes; pass another --csv")
        CSV.write(path, [row]; append=true)
    else
        CSV.write(path, [row])
    end
    return row
end

# the cases with a successful row in the file
function done_cases(path)
    isfile(path) && filesize(path) > 0 || return Set()
    return Set(case_key(r) for r in CSV.File(path; stringtype=String) if r.status == "ok")
end

#################
# Runs
#################

# Julia compiles the kernel builder the first time it runs, which took 7.5 s of the first
# build against 0.2 s after this throwaway 2-qubit kernel has been built and evaluated.
const BUILDER_WARMED_UP = Ref(false)

function warm_up_builder(X)
    BUILDER_WARMED_UP[] && return
    k = ZZFeatureMapKernel(2; num_layers=1)
    p = (η_p=ones(2, 1), η_zz=ones(1, 1))
    correlation_gradient(k, X[1:2, 1:2]; p...)
    k(X[1:2, 1:2], X[1:2, 3:4]; p...)
    BUILDER_WARMED_UP[] = true
end

# ZZ kernels are built once per number of layers. Their generated functions are compiled
# right away, so that neither counts as training time.
const BUILT_KERNELS = Dict{Any, NamedTuple}()

function get_kernel(kernel_name, num_layers, X)
    return get!(BUILT_KERNELS, (kernel_name, num_layers)) do
        kernel_name == "matern" && return (kernel=matern52_correlation, time_build=0.0, time_compile=0.0)
        warm_up_builder(X)
        time_build = @elapsed k = ZZFeatureMapKernel(size(X, 1); num_layers=num_layers)
        params0 = kernel_setup(kernel_name, num_layers, size(X, 1)).params0
        time_compile = @elapsed correlation_gradient(k, X[:, 1:2]; params0...)
        (kernel=k, time_build=time_build, time_compile=time_compile)
    end
end

# The training call of a run, with warnings off: not converging within the iteration limit
# is expected here, and the CSV records it. Optim's result goes into the Ref `result`.
train(kernel, setup, optimizer, X, y; iterations, time_limit, result) =
    Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Error)) do
        gp_train(kernel, X, y; setup..., nugget=NUGGET, method=OPTIMIZER_METHODS[optimizer](),
                 options=Optim.Options(iterations=iterations, time_limit=time_limit), result=result)
    end

# A short run on a few samples, and a prediction, before the first timed run of each kernel
# type and optimizer, so that compiling gp_train, the optimizer and the prediction does not
# count as training or prediction time either. It makes exactly the call of the timed run and
# prints where it prints, since any difference leaves some compiling to the timed run.
const WARMED_UP = Set{Any}()

function warm_up(kernel, setup, optimizer, X, y; time_limit)
    (typeof(kernel), optimizer) in WARMED_UP && return
    println("    warm-up of ", optimizer, " for ", nameof(typeof(kernel)))
    try
        gp = train(kernel, setup, optimizer, X[:, 1:10], y[1:10];
                   iterations=2, time_limit=time_limit, result=Ref{Any}())
        gp === nothing || gp_classify(gp, X[:, 11:20])
    catch
        # the timed run fails the same way and records the error
    end
    push!(WARMED_UP, (typeof(kernel), optimizer))
end

# The results of a run besides `converged` and `stopped_by`, in the order of the CSV columns.
# jit_train and jit_predict are the seconds of time_train and time_predict Julia spent
# compiling, about 0 when the warm-ups did their job.
const RESULT_KEYS = (:n_params, :iterations_run, :f_calls, :g_calls,
                     :loglik_initial, :loglik, :train_accuracy, :test_accuracy,
                     :time_build, :time_compile, :time_train, :time_predict,
                     :jit_train, :jit_predict, :params)

# Writes the results into `r` as soon as they are known, so that a run that fails keeps
# those it got to.
function run_case!(r, case, Xtr, ytr, Xte, yte; time_limit)
    k = get_kernel(case.kernel, case.num_layers, Xtr)
    r[:time_build], r[:time_compile] = k.time_build, k.time_compile
    setup = kernel_setup(case.kernel, case.num_layers, size(Xtr, 1))
    r[:n_params] = length(flatten_params(setup.params0))
    warm_up(k.kernel, setup, case.optimizer, Xtr, ytr; time_limit=time_limit)
    gp0 = gp_fit(k.kernel, Xtr, ytr, setup.params0, NUGGET)
    gp0 === nothing || (r[:loglik_initial] = gp0.loglik)
    Random.seed!(case.seed)
    res = Ref{Any}()
    t = @timed train(k.kernel, setup, case.optimizer, Xtr, ytr;
                     iterations=case.iterations, time_limit=time_limit, result=res)
    gp = t.value
    r[:time_train], r[:jit_train] = t.time, t.compile_time
    r[:converged] = Optim.converged(res[])
    r[:stopped_by] = string(Symbol(Optim.termination_code(res[])))
    r[:iterations_run], r[:f_calls], r[:g_calls] = Optim.iterations(res[]), Optim.f_calls(res[]), Optim.g_calls(res[])
    gp === nothing && error("R + nugget*I is not positive definite at the optimizer's result")
    r[:loglik] = gp.loglik
    r[:params] = join(flatten_params(gp.params), " ")
    accuracy(y_pred, y) = sum(y_pred .== y) / length(y)
    t = @timed gp_classify(gp, Xte)
    y_pred = t.value
    r[:time_predict], r[:jit_predict] = t.time, t.compile_time
    r[:test_accuracy] = accuracy(y_pred, yte)
    r[:train_accuracy] = accuracy(gp_classify(gp, Xtr), ytr)
    return r
end

# Runs the cases one after another on the train/test split of GP_coffe_vs_nap.jl and appends a
# row to `csv` after each.
function run_benchmarks(cases; csv=CSV_PATH, resume=false, time_limit=Inf)
    done = resume ? done_cases(csv) : Set()
    Xtr, ytr, Xte, yte = X[:, idx_train], y[idx_train], X[:, idx_test], y[idx_test]
    for (i, case) in enumerate(cases)
        progress = "[$(i)/$(length(cases))] $(case)"
        if case_key(case) in done
            println(progress, ": skipped, already in ", csv)
            continue
        end
        println(progress)
        r = Dict{Symbol, Any}(k => missing for k in (:converged, :stopped_by, RESULT_KEYS...))
        status, err = try
            run_case!(r, case, Xtr, ytr, Xte, yte; time_limit=time_limit)
            "ok", ""
        catch e
            e isa InterruptException && rethrow()
            @error "run failed" case exception=(e, catch_backtrace())
            "failed", first(replace(sprint(showerror, e), '\n' => " "), 1000)
        end
        # a run that failed before the optimizer returned did not converge, it stopped by the error
        converged, stopped_by = coalesce(r[:converged], false), coalesce(r[:stopped_by], "Error")
        row = (timestamp=string(Dates.now()), case..., status=status, converged=converged,
               stopped_by=stopped_by, error=err,
               gradient=OPTIMIZER_METHODS[case.optimizer]() isa Optim.FirstOrderOptimizer,
               n_train=length(ytr), n_test=length(yte), nugget=NUGGET, time_limit=time_limit,
               threads=Threads.nthreads(), NamedTuple{RESULT_KEYS}(Tuple(r[k] for k in RESULT_KEYS))...)
        append_row(csv, row)
        if status == "ok"
            println("    stopped by ", stopped_by, ", log-likelihood ", round(r[:loglik]; digits=3),
                    ", test accuracy ", r[:test_accuracy], ", training ", round(r[:time_train]; digits=1), " s")
        else
            println("    failed, stopped by ", stopped_by)
        end
    end
end

#################
# Command line
#################

if abspath(PROGRAM_FILE) == @__FILE__
    opts = Dict{String, String}()
    for a in ARGS
        startswith(a, "--") || error("unknown argument $(a), see the top of $(@__FILE__)")
        key, value = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "")
        key in ("kernels", "layers", "optimizers", "iterations", "seeds", "ids", "list",
                "resume", "csv", "time-limit") || error("unknown option --$(key), see the top of $(@__FILE__)")
        opts[key] = value
    end
    strings(key, default) = haskey(opts, key) ? String.(split(opts[key], ',')) : default
    ints(key, default) = haskey(opts, key) ? parse.(Int, split(opts[key], ',')) : default

    cases = select_cases(kernels=strings("kernels", KERNELS), layers=ints("layers", LAYERS),
                         optimizers=strings("optimizers", OPTIMIZERS),
                         iterations=ints("iterations", ITERATIONS), seeds=ints("seeds", SEEDS))
    ids = ints("ids", collect(eachindex(cases)))
    if haskey(opts, "list")
        foreach(i -> println(lpad(i, 4), "  ", cases[i]), ids)
        exit()
    end
    run_benchmarks(cases[ids]; csv=get(opts, "csv", CSV_PATH), resume=haskey(opts, "resume"),
                   time_limit=parse(Float64, get(opts, "time-limit", "Inf")))
end
