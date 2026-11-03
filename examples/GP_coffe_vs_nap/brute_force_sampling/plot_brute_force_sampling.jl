# Plots of the CSV files written by brute_force_sampling.jl, one per number of layers, four
# figures saved next to them:
#   <prefix>_distributions        the distributions of the log-likelihood and the test accuracy
#                                 over the samples, one line per number of layers
#   <prefix>_loglik_vs_accuracy   test accuracy against log-likelihood, a 2D histogram per
#                                 number of layers: does a likelier GP classify better?
#   <prefix>_best_so_far          the highest log-likelihood among the first n samples and the
#                                 test accuracy of that sample: what a smaller scan would find
#   <prefix>_sensitivity          first-order sensitivity indices of the log-likelihood and the
#                                 test accuracy to every weight
# and a table per number of layers. Only successful samples are used, the others are counted.
#
# The files are large (up to some GB), so each is read once, reduced to what the figures need
# and dropped: only one is in memory at a time. Reading and the sensitivity indices use all
# threads. Run with QCSym's environment, which has Plots:
#   julia --project=. -t 32 examples/GP_coffe_vs_nap/brute_force_sampling/plot_brute_force_sampling.jl
#   julia --project=. -t 32 examples/GP_coffe_vs_nap/brute_force_sampling/plot_brute_force_sampling.jl --layers=1,2 --format=pdf
# Options:
#   --csv=prefix      read <prefix>_L<layers>.csv, by default brute_force_sampling_L<layers>.csv
#                     next to this script, as brute_force_sampling.jl writes them
#   --layers=1,2,3    only these numbers of layers, by default every file there is
#   --out=dir         where to save the figures, by default the directory of the CSV files
#   --format=ext      png (default), pdf or svg
# From the REPL, include this file and read the files once, then plot as often as needed:
#   scans = load_scans(CSV_PREFIX); figs = plot_scans(scans); display(figs.distributions)
haskey(ENV, "GKSwstype") || (ENV["GKSwstype"] = "100")   # GR without a window
import CSV
import DataFrames
import Plots
import Statistics

const CSV_PREFIX = joinpath(@__DIR__, "brute_force_sampling")

# the inputs of GP_coffe_vs_nap.jl in the order of the columns of data_in, i.e. of the qubits
const FEATURES = ["age", "occupation", "sleep", "duration", "alertness before",
                  "alertness after", "productivity", "mood", "side effects"]

const LOGLIK_BINS = 300        # bins of the log-likelihood histograms, over all numbers of layers
const HEATMAP_BINS = 80        # log-likelihood bins of a 2D histogram
const MIN_BIN_COUNT = 100      # samples a bin needs for its mean test accuracy to be drawn
const SENSITIVITY_BINS = 32    # bins of a weight for the conditional means of its index
const CURVE_POINTS = 400       # log-spaced points of a best-so-far curve

#################
# Style, that of plot_benchmark_kernels.jl
#################

const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const INK_MUTED = "#898781"
const GRID = "#e1e0d9"
const AXIS = "#c3c2b7"
const HIGHLIGHT = "#eb6834"    # the single marked sample in the 2D histograms

# The numbers of layers are ordered, so their colors are an ordinal ramp, light to dark: steps
# 250 to 700 of the blue ramp.
const BLUE_ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                      "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# steps 100 to 700 for the heatmaps, the lightest for the lowest values
const SEQUENTIAL = Plots.cgrad(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                "#184f95", "#0d366b"])

# The ramp is spread over 1 to at least 6 layers, so a number of layers keeps its color
# whichever others are plotted.
function layer_color(num_layers, max_layers)
    t = (num_layers - 1) / (max(max_layers, 6) - 1)
    return BLUE_ORDINAL[1 + round(Int, t * (length(BLUE_ORDINAL) - 1))]
end

colors_of(scans) = [layer_color(s.num_layers, maximum(x.num_layers for x in scans)) for s in scans]

layers_label(num_layers) = "$(num_layers) layer$(num_layers == 1 ? "" : "s")"

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

csv_path(prefix, num_layers) = "$(prefix)_L$(num_layers).csv"

# the numbers of layers with a file <prefix>_L<layers>.csv
function available_layers(prefix)
    dir, start = dirname(abspath(prefix)), basename(prefix) * "_L"
    isdir(dir) || return Int[]
    layers = [tryparse(Int, chop(f; head=length(start), tail=4)) for f in readdir(dir)
              if startswith(f, start) && endswith(f, ".csv")]
    return sort!(Int[l for l in layers if l !== nothing])
end

# kind (:p or :zz), qubit and layer of the weight in the column eta_<kind>_<qubit>_<layer>
function parse_weight(name)
    m = match(r"^eta_(p|zz)_(\d+)_(\d+)$", name)
    m === nothing && error("unexpected weight column $(name)")
    return (kind=Symbol(m[1]), qubit=parse(Int, m[2]), layer=parse(Int, m[3]))
end

# First-order sensitivity indices S_j = Var(E[y | w_j]) / Var(y) of y to each weight w_j, with
# the conditional means taken over SENSITIVITY_BINS bins of equal width over w_j's range. The
# share of the variance between the bins overestimates S_j by about f = (bins - 1)/(n - 1) for
# n samples, which a weight without any effect gets from noise alone; it is removed as in
# ANOVA's ε², (share - f)/(1 - f), so an index may come out slightly negative.
function sensitivity_indices(W, y)
    ȳ = Statistics.mean(y)
    V = sum(v -> abs2(v - ȳ), y)
    S = zeros(length(W))
    V == 0 && return S
    f = (SENSITIVITY_BINS - 1) / (length(y) - 1)
    Threads.@threads for j in eachindex(W)
        lo, hi = extrema(W[j])
        n, s = zeros(Int, SENSITIVITY_BINS), zeros(SENSITIVITY_BINS)
        for (w, v) in zip(W[j], y)
            b = clamp(floor(Int, (w - lo) / (hi - lo) * SENSITIVITY_BINS) + 1, 1, SENSITIVITY_BINS)
            n[b] += 1
            s[b] += v
        end
        share = sum(n[b] * abs2(s[b] / n[b] - ȳ) for b in 1:SENSITIVITY_BINS if n[b] > 0) / V
        S[j] = (share - f) / (1 - f)
    end
    return S
end

# One scan reduced to what the figures need: the results of its successful samples and the
# sensitivity indices of its weights. The weights themselves are dropped with the file.
function load_scan(path)
    f = CSV.File(path; stringtype=String)
    length(f) == 0 && error("$(path) has no rows")
    status = collect(f.status)
    ok = findall(==("ok"), status)
    if length(ok) < length(f)
        failed = filter(!=("ok"), unique(status))
        @warn "$(length(f) - length(ok)) of $(length(f)) samples in $(path) failed and are left out: " *
              join(["$(count(==(s), status)) $(s)" for s in failed], ", ")
    end
    # a sample in the file more than once (a scan appended again) counts once
    sample = collect(f.sample)[ok]
    first_rows = unique(i -> sample[i], eachindex(sample))
    length(first_rows) < length(ok) &&
        @warn "$(length(ok) - length(first_rows)) samples are in $(path) more than once, only their first rows are used"
    ok = ok[first_rows]
    isempty(ok) && error("$(path) has no successful samples")

    col(name) = Vector{Float64}(getproperty(f, name)[ok])
    names = filter(n -> startswith(n, "eta_"), string.(propertynames(f)))
    W = [col(Symbol(n)) for n in names]
    loglik, test_accuracy = col(:loglik), col(:test_accuracy)
    return (num_layers=first(f.num_layers), sampler=first(f.sampler), path=path,
            n_failed=count(!=("ok"), status), sample=sample[first_rows], loglik=loglik,
            test_accuracy=test_accuracy, train_accuracy=col(:train_accuracy), time=col(:time),
            weights=parse_weight.(names),
            bounds=(minimum(minimum, W), maximum(maximum, W)),
            sensitivity=(loglik=sensitivity_indices(W, loglik),
                         test_accuracy=sensitivity_indices(W, test_accuracy)))
end

function load_scans(prefix=CSV_PREFIX; layers=available_layers(prefix))
    isempty(layers) && error("no files $(prefix)_L<layers>.csv")
    return map(layers) do num_layers
        path = csv_path(prefix, num_layers)
        t = @elapsed s = load_scan(path)
        println("read ", path, " in ", round(t; digits=1), " s")
        return s
    end
end

# The number of test points n, from the step 1/n between test accuracies, and the range of
# k in the accuracies k/n of all scans.
function accuracy_grid(scans)
    a = sort(unique(reduce(vcat, [unique(s.test_accuracy) for s in scans])))
    n = length(a) < 2 ? 100 : round(Int, 1 / minimum(diff(a)))
    return n, round(Int, a[1] * n):round(Int, a[end] * n)
end

#################
# Figures
#################

# A panel with only the legend, a line per number of layers, to put above the others.
function legend_panel(scans)
    p = Plots.plot(; framestyle=:none, legend=:top, legend_column=length(scans))
    for (s, c) in zip(scans, colors_of(scans))
        Plots.plot!(p, [NaN], [NaN]; color=c, label=layers_label(s.num_layers))
    end
    return p
end

# the fraction of the values in each bin of the range `edges`, the outer bins closed
function fractions(values, edges)
    n = zeros(length(edges) - 1)
    for v in values
        n[clamp(floor(Int, (v - first(edges)) / step(edges)) + 1, 1, length(n))] += 1
    end
    return n ./ length(values)
end

nan_zeros(v) = replace(v, 0.0 => NaN)   # empty bins leave a gap on a log axis

# How the log-likelihood and the test accuracy are distributed over the weight box, on a log
# scale so that the tails, where the best samples are, stay visible.
function distributions_figure(scans)
    colors = colors_of(scans)
    lo, hi = minimum(minimum(s.loglik) for s in scans), maximum(maximum(s.loglik) for s in scans)
    edges = range(lo, hi; length=LOGLIK_BINS + 1)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    n_test, ks = accuracy_grid(scans)
    acc_edges = range((ks[1] - 0.5) / n_test, (ks[end] + 0.5) / n_test; length=length(ks) + 1)
    ll = Plots.plot(; yscale=:log10, legend=false, title="log-likelihood",
                    xlabel="log-likelihood", ylabel="fraction of samples")
    acc = Plots.plot(; yscale=:log10, legend=false, title="test accuracy", xlabel="test accuracy")
    for (s, c) in zip(scans, colors)
        Plots.plot!(ll, centers, nan_zeros(fractions(s.loglik, edges));
                    seriestype=:stepmid, color=c, linewidth=1.5)
        Plots.plot!(acc, ks ./ n_test, nan_zeros(fractions(s.test_accuracy, acc_edges));
                    seriestype=:stepmid, color=c, linewidth=1.5)
    end
    layout = Plots.@layout [a{0.08h}; Plots.grid(1, 2)]
    return Plots.plot(legend_panel(scans), ll, acc; layout=layout, size=(1300, 480),
                      left_margin=6Plots.mm, bottom_margin=6Plots.mm)
end

# Panels in rows of at most three, the empty places filled.
function panel_grid(panels; columns=3)
    c = min(columns, length(panels))
    r = cld(length(panels), c)
    append!(panels, [Plots.plot(; framestyle=:none) for _ in 1:(r * c - length(panels))])
    return panels, r, c
end

# Test accuracy against log-likelihood, the fraction of samples in each cell on a log color
# scale, with the mean test accuracy per log-likelihood bin and the sample of the highest
# log-likelihood, the one maximum likelihood would choose.
function loglik_vs_accuracy_figure(scans)
    n_test, ks = accuracy_grid(scans)
    accuracies = ks ./ n_test
    hists = map(scans) do s
        lo, hi = extrema(s.loglik)
        w = (hi - lo) / HEATMAP_BINS
        C = zeros(Int, length(ks), HEATMAP_BINS)
        for (l, a) in zip(s.loglik, s.test_accuracy)
            C[round(Int, a * n_test) - ks[1] + 1, clamp(floor(Int, (l - lo) / w) + 1, 1, HEATMAP_BINS)] += 1
        end
        (C=C, centers=lo .+ w .* ((1:HEATMAP_BINS) .- 0.5), n=length(s.loglik))
    end
    zlims = (minimum(log10(1 / h.n) for h in hists), maximum(log10(maximum(h.C) / h.n) for h in hists))
    ylims = ((ks[1] - 1) / n_test, (ks[end] + 1) / n_test)
    panels, r, c = panel_grid(map(enumerate(zip(scans, hists))) do (j, (s, h))
        p = Plots.heatmap(h.centers, accuracies, [x == 0 ? NaN : log10(x / h.n) for x in h.C];
                          color=SEQUENTIAL, clims=zlims, colorbar=false, grid=false, ylims=ylims,
                          title=layers_label(s.num_layers),
                          xlabel="log-likelihood", ylabel=j % 3 == 1 ? "test accuracy" : "",
                          label="", legend=j == 1 ? :bottomright : false)
        counts = vec(sum(h.C; dims=1))
        means = vec(sum(h.C .* accuracies; dims=1)) ./ counts
        means[counts .< MIN_BIN_COUNT] .= NaN
        Plots.plot!(p, h.centers, means; color=INK, linewidth=1.5, label="mean test accuracy")
        i = argmax(s.loglik)
        Plots.scatter!(p, [s.loglik[i]], [s.test_accuracy[i]]; color=HIGHLIGHT, marker=:star5,
                       markersize=8, label="highest log-likelihood")
        return p
    end)
    layout = Plots.grid(1, 2; widths=[0.955, 0.045])
    layout[1, 1] = Plots.grid(r, c)
    return Plots.plot(panels..., log_colorbar(zlims, "fraction of samples"); layout=layout,
                      size=(1400, 400 * r), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

# A colorbar as a panel of its own, for panels sharing a color scale of log10 values in
# zlims: the colorbar of each panel would take a third of its width in GR.
function log_colorbar(zlims, title)
    z = range(zlims...; length=256)
    decades = ceil(Int, z[1]):floor(Int, z[end])
    return Plots.heatmap([0, 1], z, [z z]; color=SEQUENTIAL, clims=zlims, colorbar=false,
                         grid=false, showaxis=:y, xticks=false, ylims=zlims, ymirror=true,
                         yticks=(decades, ["10^{$(k)}" for k in decades]), ylabel=title,
                         right_margin=8Plots.mm)
end

# The highest log-likelihood among the first n samples, i.e. what a scan of n samples would
# have found (the first n points of a Sobol sequence are a design of their own), and the test
# accuracy of that sample.
function best_so_far_figure(scans)
    ll = Plots.plot(; xscale=:log10, legend=false, title="highest log-likelihood of the first n samples",
                    xlabel="samples n", ylabel="log-likelihood")
    acc = Plots.plot(; xscale=:log10, legend=false, title="test accuracy of that sample",
                     xlabel="samples n", ylabel="test accuracy")
    for (s, c) in zip(scans, colors_of(scans))
        order = sortperm(s.sample)
        l = s.loglik[order]
        best = similar(order)
        b = 1
        for i in eachindex(l)
            l[i] > l[b] && (b = i)
            best[i] = b
        end
        points = unique(round.(Int, exp10.(range(0, log10(length(l)); length=CURVE_POINTS))))
        n = s.sample[order][points]
        Plots.plot!(ll, n, l[best[points]]; seriestype=:steppost, color=c)
        Plots.plot!(acc, n, s.test_accuracy[order][best[points]]; seriestype=:steppost, color=c)
    end
    layout = Plots.@layout [a{0.08h}; Plots.grid(1, 2)]
    return Plots.plot(legend_panel(scans), ll, acc; layout=layout, size=(1300, 480),
                      left_margin=6Plots.mm, bottom_margin=6Plots.mm)
end

feature_name(i, d) = d == length(FEATURES) ? FEATURES[i] : "qubit $(i)"

weight_label(w, d) = w.kind == :p ? "η_p[$(w.qubit),$(w.layer)] ($(feature_name(w.qubit, d)))" :
    "η_zz[$(w.qubit),$(w.layer)] ($(feature_name(w.qubit, d)), $(feature_name(w.qubit + 1, d)))"

# The sensitivity indices as heatmaps, a row per gate (from the top, the P gates on the
# qubits, then the ZZ terms of neighboring qubits) and a column per layer of the circuit,
# grouped by the scans' numbers of layers; one heatmap for the log-likelihood and one for the
# test accuracy. The indices span orders of magnitude, so the color scale is logarithmic,
# from the noise level of the estimate, below which a weight has no measurable effect.
function sensitivity_figure(scans)
    d = maximum(w.qubit for s in scans for w in s.weights if w.kind == :p)
    # bottom to top: GR shifts a heatmap with yflip=true by a row when ylims are wider
    rows = reverse([[(:p, i) for i in 1:d]; [(:zz, i) for i in 1:d-1]])
    row_labels = [k == :p ? "P  $(feature_name(i, d))" :
                  "ZZ  $(feature_name(i, d)), $(feature_name(i + 1, d))" for (k, i) in rows]
    columns = [(j, l) for (j, s) in enumerate(scans) for l in 1:s.num_layers]
    groups = [findall(c -> c[1] == j, columns) for j in eachindex(scans)]
    noise = maximum((SENSITIVITY_BINS - 1) / length(s.loglik) for s in scans)
    panels = map([(:loglik, "log-likelihood"), (:test_accuracy, "test accuracy")]) do (target, label)
        M = fill(NaN, length(rows), length(columns))
        for (j, s) in enumerate(scans), (w, S) in zip(s.weights, getfield(s.sensitivity, target))
            M[findfirst(==((w.kind, w.qubit)), rows), findfirst(==((j, w.layer)), columns)] = max(S, noise)
        end
        # the strip above the top row holds the numbers of layers
        p = Plots.heatmap(1:length(columns), 1:length(rows), M; color=SEQUENTIAL,
                          clims=(noise, max(maximum(filter(!isnan, M)), 10 * noise)),
                          colorbar_scale=:log10, grid=false,
                          xlims=(0.5, length(columns) + 0.5), ylims=(0.5, length(rows) + 1.4),
                          xticks=(1:length(columns), [string(l) for (_, l) in columns]),
                          yticks=(1:length(rows), row_labels), xlabel="layer of the circuit",
                          title="first-order sensitivity of the $(label) to each weight",
                          colorbar_title="share of the variance")
        # gaps between the scans, and between the P and the ZZ weights
        Plots.vline!(p, [last(g) + 0.5 for g in groups[1:end-1]]; color=SURFACE, linewidth=5, label="")
        Plots.hline!(p, [d - 0.5]; color=SURFACE, linewidth=5, label="")
        for (g, s) in zip(groups, scans)
            Plots.annotate!(p, (first(g) + last(g)) / 2, length(rows) + 0.95,
                            Plots.text(layers_label(s.num_layers), 8, INK_SECONDARY, :center))
        end
        return p
    end
    return Plots.plot(panels...; layout=(2, 1), size=(1400, 1000),
                      left_margin=5Plots.mm, bottom_margin=5Plots.mm, right_margin=3Plots.mm)
end

# A row per scan: the samples, the cost of one, how the log-likelihood and the test accuracy
# are spread, the test accuracy maximum likelihood would choose, the best test accuracy of any
# sample (and how many reach it), their correlation, and how much of the log-likelihood's
# variance the weights explain alone (the sum of the first-order indices; the rest comes from
# their interactions).
function summary_table(scans)
    r(x, digits) = round(x; digits=digits)
    return DataFrames.DataFrame(map(scans) do s
        i = argmax(s.loglik)
        best = maximum(s.test_accuracy)
        (layers=s.num_layers, weights=length(s.weights), samples=length(s.loglik),
         failed=s.n_failed, ms_per_sample=r(1e3 * Statistics.median(s.time), 1),
         loglik_median=r(Statistics.median(s.loglik), 1), loglik_max=r(s.loglik[i], 1),
         test_acc_at_max=s.test_accuracy[i], test_acc_best=best,
         n_best=count(==(best), s.test_accuracy),
         correlation=r(Statistics.cor(s.loglik, s.test_accuracy), 3),
         first_order=r(sum(s.sensitivity.loglik), 3))
    end)
end

# the weights with the largest first-order indices of the log-likelihood, per scan
function top_weights(scans; n=3)
    d = maximum(w.qubit for s in scans for w in s.weights if w.kind == :p)
    return [s.num_layers => [(weight_label(s.weights[j], d), s.sensitivity.loglik[j])
                             for j in first(sortperm(s.sensitivity.loglik; rev=true), n)]
            for s in scans]
end

function plot_scans(scans)
    return (distributions=distributions_figure(scans),
            loglik_vs_accuracy=loglik_vs_accuracy_figure(scans),
            best_so_far=best_so_far_figure(scans),
            sensitivity=sensitivity_figure(scans),
            summary=summary_table(scans))
end

#################
# Command line
#################

if abspath(PROGRAM_FILE) == @__FILE__
    opts = Dict{String, String}()
    for a in ARGS
        startswith(a, "--") || error("unknown argument $(a), see the top of $(@__FILE__)")
        key, value = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "")
        key in ("csv", "layers", "out", "format") || error("unknown option --$(key), see the top of $(@__FILE__)")
        opts[key] = value
    end
    prefix = get(opts, "csv", CSV_PREFIX)
    layers = haskey(opts, "layers") ? parse.(Int, split(opts["layers"], ',')) : available_layers(prefix)
    out = get(opts, "out", dirname(abspath(prefix)))
    format = get(opts, "format", "png")
    format in ("png", "pdf", "svg") || error("unknown format $(format), use png, pdf or svg")

    scans = load_scans(prefix; layers=layers)
    figs = plot_scans(scans)
    mkpath(out)
    for key in (:distributions, :loglik_vs_accuracy, :best_so_far, :sensitivity)
        path = joinpath(out, "$(basename(prefix))_$(key).$(format)")
        Plots.savefig(getfield(figs, key), path)
        println("saved ", path)
    end
    lo, hi = minimum(s.bounds[1] for s in scans), maximum(s.bounds[2] for s in scans)
    println("\n", join(unique(s.sampler for s in scans), ", "), " samples of the weights in [",
            round(lo; digits=2), ", ", round(hi; digits=2), "]:")
    show(stdout, figs.summary; summary=false, eltypes=false)
    println("\n\nthe weights the log-likelihood depends on most (first-order index):")
    for (num_layers, ws) in top_weights(scans)
        println("  ", rpad(layers_label(num_layers), 10),
                join(["$(name) $(round(S; digits=3))" for (name, S) in ws], ", "))
    end
end
