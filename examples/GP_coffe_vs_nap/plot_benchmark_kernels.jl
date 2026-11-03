# Plots of the CSV file written by benchmark_kernels.jl, three figures saved next to it:
#   <csv name>_overview   test accuracy, log-likelihood and training time over the iteration
#                         limit, one column per kernel and one line per optimizer
#   <csv name>_tradeoff   test accuracy over training time, the same panels and lines
#   <csv name>_costs      the kernels' one-time build and compile times and prediction times
# Failed runs are left out and counted. Runs of the same kernel, optimizer and iteration limit
# (several seeds, or a case run again) are averaged, and their range drawn as a band.
#
#   julia --project=. examples/GP_coffe_vs_nap/plot_benchmark_kernels.jl
#   julia --project=. examples/GP_coffe_vs_nap/plot_benchmark_kernels.jl --csv=path --out=dir --format=pdf
# Options:
#   --csv=path        the CSV file, by default benchmark_kernels.csv next to this script
#   --out=dir         where to save the figures, by default the directory of the CSV file
#   --format=ext      png (default), pdf or svg
# From the REPL, include this file and call e.g.
#   figs = plot_benchmarks(load_results(CSV_PATH)); display(figs.overview)
haskey(ENV, "GKSwstype") || (ENV["GKSwstype"] = "100")   # GR without a window
import CSV
import DataFrames
import Plots
import Statistics

const CSV_PATH = joinpath(@__DIR__, "benchmark_kernels.csv")

#################
# Style
#################

# The 13 optimizers benchmark_kernels.jl knows are more than the 8 colors that stay
# distinguishable, so the color marks an optimizer's family (Optim's grouping; the larger
# families get the higher-contrast first slots of the palette) and the marker and line style
# its member within the family. Every optimizer keeps its look in every figure, whichever
# others a CSV file contains.
const OPTIMIZER_FAMILIES = [
    (name="gradient-free", color="#2a78d6", members=["NelderMead", "ParticleSwarm", "SimulatedAnnealing"]),
    (name="accelerated", color="#eb6834",
     members=["AcceleratedGradientDescent", "MomentumGradientDescent", "Adam", "AdaMax"]),
    (name="quasi-Newton", color="#1baf7a", members=["LBFGS", "BFGS"]),
    (name="gradient descent, CG", color="#eda100", members=["GradientDescent", "ConjugateGradient"]),
    (name="nonlinear GMRES", color="#e87ba4", members=["NGMRES", "OACCEL"]),
]
const MEMBER_MARKERS = [:circle, :rect, :diamond, :utriangle]
const MEMBER_LINESTYLES = [:solid, :dash, :dot, :dashdot]
const OPTIMIZER_ORDER = reduce(vcat, f.members for f in OPTIMIZER_FAMILIES)

const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const INK_MUTED = "#898781"
const GRID = "#e1e0d9"
const AXIS = "#c3c2b7"
const BAR_DARK = "#2a78d6"    # one blue ramp for the bars of the costs figure
const BAR_LIGHT = "#9ec5f4"

function optimizer_style(name)
    for f in OPTIMIZER_FAMILIES
        i = findfirst(==(name), f.members)
        i === nothing || return (color=f.color, marker=MEMBER_MARKERS[i], linestyle=MEMBER_LINESTYLES[i])
    end
    return (color=INK_MUTED, marker=:circle, linestyle=:solid)
end

Plots.default(
    background_color=SURFACE, foreground_color_text=INK_SECONDARY,
    foreground_color_axis=AXIS, foreground_color_border=AXIS,
    foreground_color_guide=INK_SECONDARY, foreground_color_title=INK,
    foreground_color_grid=GRID, gridalpha=1.0, gridlinewidth=0.5, framestyle=:axes,
    foreground_color_legend=nothing, background_color_legend=nothing,
    titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=9,
    linewidth=2, markersize=4.5, markerstrokewidth=1.5, markerstrokecolor=SURFACE, dpi=150,
)

#################
# Data
#################

config_label(kernel, num_layers) =
    kernel == "matern" ? "Matérn 5/2" :
    kernel == "zz" ? "ZZ, $(num_layers) layer$(num_layers == 1 ? "" : "s")" :
    "$(kernel)$(ismissing(num_layers) ? "" : ", $(num_layers) layers")"

# The successful runs, with a column `config` naming the kernel and its number of layers.
function load_results(path)
    df = DataFrames.DataFrame(CSV.File(path; stringtype=String))
    n_failed = count(!=("ok"), df.status)
    n_failed > 0 && @warn "$(n_failed) of $(DataFrames.nrow(df)) runs in $(path) failed and are left out"
    df = df[df.status .== "ok", :]
    isempty(df) && error("$(path) has no successful runs")
    df.config = config_label.(df.kernel, df.num_layers)
    return df
end

# Matérn first, then the other kernels by name and number of layers.
function configs_of(df)
    keys = unique(zip(df.kernel, df.num_layers))
    sort!(keys; by=k -> (k[1] != "matern", k[1], coalesce(k[2], 0)))
    return [config_label(k...) for k in keys]
end

optimizers_of(df) = sort(unique(df.optimizer);
                         by=o -> something(findfirst(==(o), OPTIMIZER_ORDER), length(OPTIMIZER_ORDER) + 1))

# Mean, minimum and maximum of `col` over the runs of each kernel, optimizer and iteration
# limit.
function summarize(df, col)
    g = DataFrames.groupby(df, [:config, :optimizer, :iterations])
    s = DataFrames.combine(g, col => Statistics.mean => :mean, col => minimum => :lo,
                           col => maximum => :hi, DataFrames.nrow => :n)
    return sort!(s, :iterations)
end

# Axis limits covering all values, padded by 5 % of the range (a factor on a log axis).
function padded_limits(values, scale)
    a, b = extrema(values)
    if scale == :log10
        f = (b / a)^0.05
        return (a / f, b * f)
    end
    pad = 0.05 * max(b - a, 1e-3 * abs(b), 1e-3)
    return (a - pad, b + pad)
end

#################
# Figures
#################

# The rows of the legend: the families with optimizers in the data, then any optimizer
# without a family.
function legend_rows(optimizers)
    rows = [(f.name, filter(in(optimizers), f.members)) for f in OPTIMIZER_FAMILIES]
    others = filter(!in(OPTIMIZER_ORDER), optimizers)
    isempty(others) || push!(rows, ("other", others))
    return filter(r -> !isempty(r[2]), rows)
end

const LEGEND_ROW_HEIGHT = 22   # pixels

legend_height(optimizers) = LEGEND_ROW_HEIGHT * length(legend_rows(optimizers)) + 12

# A panel with only the legend, to put above a grid: a row per family, its name in muted ink,
# then a line and marker per optimizer.
function legend_panel(optimizers)
    rows = legend_rows(optimizers)
    ncols = maximum(length(members) for (_, members) in rows)
    # the family names take 0.6 of the width of an optimizer's column
    p = Plots.plot(; framestyle=:none, legend=false, grid=false, yflip=true,
                   xlims=(0, 0.6 + ncols), ylims=(0.4, length(rows) + 0.6))
    for (i, (family, members)) in enumerate(rows)
        Plots.annotate!(p, 0, i, Plots.text(family, 8, INK_MUTED, :left))
        for (j, o) in enumerate(members)
            st = optimizer_style(o)
            x = 0.6 + (j - 1)
            Plots.plot!(p, [x, x + 0.16], [i, i]; color=st.color, linestyle=st.linestyle, label="")
            Plots.scatter!(p, [x + 0.08], [i]; color=st.color, marker=st.marker, label="")
            Plots.annotate!(p, x + 0.21, i, Plots.text(o, 8, INK_SECONDARY, :left))
        end
    end
    return p
end

# The layout of a legend panel `legend_px` pixels high above a grid of r×c panels, in a
# figure `height_px` pixels high.
function legend_above_grid(legend_px, height_px, r, c)
    l = Plots.grid(2, 1; heights=[legend_px / height_px, 1 - legend_px / height_px])
    l[2, 1] = Plots.grid(r, c)
    return l
end

# Figures are at least this wide, so that the legend's longest names fit.
figure_width(n_panels) = max(300 * n_panels + 80, 1300)

# One line per optimizer through the rows of the summary `s` of `config`, x and y given by
# the functions `x(rows)` and `y(rows)`.
function optimizer_lines!(p, s, config, optimizers; x, band)
    for o in optimizers
        r = s[(s.config .== config) .& (s.optimizer .== o), :]
        isempty(r) && continue
        st = optimizer_style(o)
        ribbon = band && any(>(1), r.n) ? (r.mean .- r.lo, r.hi .- r.mean) : nothing
        Plots.plot!(p, x(r), r.mean; ribbon=ribbon, color=st.color, fillcolor=st.color,
                    fillalpha=0.15, marker=st.marker, linestyle=st.linestyle, label="")
    end
    return p
end

const OVERVIEW_ROWS = [
    (col=:test_accuracy, label="test accuracy", scale=:identity),
    (col=:loglik, label="log-likelihood", scale=:identity),
    (col=:time_train, label="training time [s]", scale=:log10),
]

function overview_figure(df, configs, optimizers)
    its = sort(unique(df.iterations))
    xlims = (its[1] / 1.4, its[end] * 1.4)
    panels = []
    for (i, row) in enumerate(OVERVIEW_ROWS)
        s = summarize(df, row.col)
        initial = Dict(c => collect(skipmissing(df.loglik_initial[df.config .== c])) for c in configs)
        bounds = [s.lo; s.hi]
        row.col == :loglik && append!(bounds, reduce(vcat, values(initial)))
        ylims = padded_limits(bounds, row.scale)
        for (j, c) in enumerate(configs)
            p = Plots.plot(; xscale=:log10, xticks=(its, string.(its)), xlims=xlims,
                           yscale=row.scale, ylims=ylims, legend=false,
                           title=i == 1 ? c : "", ylabel=j == 1 ? row.label : "",
                           xlabel=i == length(OVERVIEW_ROWS) ? "iteration limit" : "")
            # the likelihood at the initial parameters, where every optimizer starts
            if row.col == :loglik && !isempty(initial[c])
                l0 = Statistics.mean(initial[c])
                Plots.hline!(p, [l0]; color=INK_MUTED, linewidth=1, label="")
                Plots.annotate!(p, xlims[1], l0, Plots.text(" initial", 7, INK_MUTED, :left, :top))
            end
            push!(panels, optimizer_lines!(p, s, c, optimizers; x=r -> r.iterations, band=true))
        end
    end
    n = length(configs)
    lh = legend_height(optimizers)
    h = 720 + lh
    return Plots.plot(legend_panel(optimizers), panels...;
                      layout=legend_above_grid(lh, h, length(OVERVIEW_ROWS), n),
                      size=(figure_width(n), h), left_margin=4Plots.mm, bottom_margin=2Plots.mm)
end

# Test accuracy against training time: which optimizer reaches which accuracy how fast. The
# points of a line are the iteration limits, in increasing order from left to right.
function tradeoff_figure(df, configs, optimizers)
    acc, t = summarize(df, :test_accuracy), summarize(df, :time_train)
    s = DataFrames.innerjoin(acc, t[:, [:config, :optimizer, :iterations, :mean]];
                             on=[:config, :optimizer, :iterations], renamecols="" => "_time")
    sort!(s, :iterations)
    xlims, ylims = padded_limits(s.mean_time, :log10), padded_limits([s.lo; s.hi], :identity)
    panels = [optimizer_lines!(
                  Plots.plot(; xscale=:log10, xlims=xlims, ylims=ylims, legend=false, title=c,
                             xlabel="training time [s]", ylabel=j == 1 ? "test accuracy" : ""),
                  s, c, optimizers; x=r -> r.mean_time, band=false)
              for (j, c) in enumerate(configs)]
    n = length(configs)
    lh = legend_height(optimizers)
    h = 360 + lh
    return Plots.plot(legend_panel(optimizers), panels...; layout=legend_above_grid(lh, h, 1, n),
                      size=(figure_width(n), h), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
end

# The costs that do not depend on the optimizer: building and compiling a kernel once, and
# predicting the test set. Medians over the runs, since the first run of a kernel in a
# process may include compiling the prediction.
function costs_figure(df, configs)
    med(col) = [Statistics.median(df[df.config .== c, col]) for c in configs]
    build, compile, predict = med(:time_build), med(:time_compile), med(:time_predict)
    total = build .+ compile
    x = 1:length(configs)
    setup = Plots.bar(x, total; color=BAR_LIGHT, linecolor=SURFACE, bar_width=0.6,
                      label="compile", title="one-time setup", ylabel="time [s]",
                      xticks=(x, configs), xgrid=false,
                      ylims=(0, 1.15 * maximum(total; init=1.0)), legend=:topleft)
    Plots.bar!(setup, x, build; color=BAR_DARK, linecolor=SURFACE, bar_width=0.6, label="build")
    Plots.annotate!(setup, [(xi, ti, Plots.text(string(round(ti; digits=1), " s"), 8, INK_SECONDARY, :bottom))
                            for (xi, ti) in zip(x, total)])
    n_test = first(df.n_test)
    pred = Plots.bar(x, 1e3 .* predict; color=BAR_DARK, linecolor=SURFACE, bar_width=0.6,
                     label="", title="prediction of the $(n_test) test points", ylabel="time [ms]",
                     xticks=(x, configs), xgrid=false, ylims=(0, 1.15e3 * maximum(predict)))
    Plots.annotate!(pred, [(xi, 1e3 * pi_, Plots.text(string(round(1e3 * pi_; digits=1), " ms"), 8, INK_SECONDARY, :bottom))
                           for (xi, pi_) in zip(x, predict)])
    return Plots.plot(setup, pred; layout=(1, 2), size=(1000, 380),
                      left_margin=4Plots.mm, bottom_margin=4Plots.mm)
end

# The best test accuracy of every kernel, and the optimizer and iteration limit reaching it
# fastest, as a table.
function best_runs(df, configs)
    acc, t = summarize(df, :test_accuracy), summarize(df, :time_train)
    ll = summarize(df, :loglik)
    s = DataFrames.DataFrame(config=acc.config, optimizer=acc.optimizer, iterations=acc.iterations,
                             test_accuracy=acc.mean)
    s = DataFrames.innerjoin(s, DataFrames.select(ll, :config, :optimizer, :iterations, :mean => :loglik),
                             DataFrames.select(t, :config, :optimizer, :iterations, :mean => :time_train);
                             on=[:config, :optimizer, :iterations])
    best = map(configs) do c
        r = s[s.config .== c, :]
        r = r[r.test_accuracy .== maximum(r.test_accuracy), :]
        r[argmin(r.time_train), :]
    end
    return DataFrames.DataFrame(best)
end

function plot_benchmarks(df)
    configs, optimizers = configs_of(df), optimizers_of(df)
    return (overview=overview_figure(df, configs, optimizers),
            tradeoff=tradeoff_figure(df, configs, optimizers),
            costs=costs_figure(df, configs),
            best=best_runs(df, configs))
end

#################
# Command line
#################

if abspath(PROGRAM_FILE) == @__FILE__
    opts = Dict{String, String}()
    for a in ARGS
        startswith(a, "--") || error("unknown argument $(a), see the top of $(@__FILE__)")
        key, value = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "")
        key in ("csv", "out", "format") || error("unknown option --$(key), see the top of $(@__FILE__)")
        opts[key] = value
    end
    csv = get(opts, "csv", CSV_PATH)
    out = get(opts, "out", dirname(abspath(csv)))
    format = get(opts, "format", "png")
    format in ("png", "pdf", "svg") || error("unknown format $(format), use png, pdf or svg")

    df = load_results(csv)
    figs = plot_benchmarks(df)
    mkpath(out)
    name = splitext(basename(csv))[1]
    for key in (:overview, :tradeoff, :costs)
        path = joinpath(out, "$(name)_$(key).$(format)")
        Plots.savefig(getfield(figs, key), path)
        println("saved ", path)
    end
    println("\nbest test accuracy of every kernel, reached fastest by:")
    show(stdout, figs.best; summary=false, eltypes=false)
    println()
end
