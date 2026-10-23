import LinearAlgebra
import StaticArrays
import BenchmarkTools
import DataFrames
import Dates

include("src.jl")



@testset "Benchmark basic matmul (runs ~3 minutes)" begin
    println("Running basic matmul tests... (runs ~3 minutes, start time: $(Dates.format(Dates.now(), "HH:MM")))")

    blas_num_threads = LinearAlgebra.BLAS.get_num_threads()

    LinearAlgebra.BLAS.set_num_threads(1)
    NUM_ELTYPE = ComplexF64


    list_functions_sizes_types = [
        (mul_3loops_any!, 0, "any"),
        (mul_explicit_2!, 2, "any"),
        (mul_explicit_static_2!, 2, "static"),
        (mul_strassen_static_2!, 2, "static"),
        (mul_linalg_any!, 0, "any"),
        (mul_base_any!, 0, "any")
    ]

    function print_benchmark_data_nonitnerative(pre, bm)
        println(pre, bm, " ", bm.memory)
    end

    # a size of 0 means the function works for any size
    size_matches(s, C, A, B) = s == 0 || s == size(A, 1) == size(A, 2) == size(B, 1) == size(B, 2) == size(C, 1) == size(C, 2)
    # "any" functions accept both standard and static arrays, "static" ones only static arrays
    type_matches(t, array_t) = t == "any" || t == array_t

    function run_benchmarks(C, A, B, CS, AS, BS)
        df = DataFrames.DataFrame(func=String[], array_t=String[], N=Int[],
                                time_min_ns=Float64[], time_median_ns=Float64[], time_mean_ns=Float64[],
                                memory_bytes=Int[], allocs=Int[])
        for (func, s, t) in list_functions_sizes_types
            # println("\nRunning benchmark for function: $(func) with size: $(s) and type: $(t)")
            for (array_t, c, a, b) in (("standard", C, A, B), ("static", CS, AS, BS), ("static-standard", C, AS, B), ("standard-static", C, A, BS))
                if !type_matches(t, array_t) || !size_matches(s, c, a, b)
                    # println("    $(array_t) arrays: skipped")
                    continue
                end
                # println("    $(array_t) arrays")
                bm = BenchmarkTools.@benchmark $func($c, $a, $b) samples=100000
                #print_benchmark_data_nonitnerative("    ", bm)
                push!(df, (func=string(func), array_t=array_t, N=size(a, 1),
                        time_min_ns=minimum(bm).time, time_median_ns=BenchmarkTools.median(bm).time,
                        time_mean_ns=BenchmarkTools.mean(bm).time,
                        memory_bytes=bm.memory, allocs=bm.allocs))
            end
        end
        sort!(df, [:array_t,:time_min_ns])
        return df
    end

    function mul_arrays(N::Int64)
        A = rand(NUM_ELTYPE, N, N)
        B = rand(NUM_ELTYPE, N, N)
        C = zeros(NUM_ELTYPE, N, N)

        AS = StaticArrays.rand(StaticArrays.SMatrix{N, N, NUM_ELTYPE})
        BS = StaticArrays.rand(StaticArrays.MMatrix{N, N, NUM_ELTYPE})
        CS = StaticArrays.MMatrix{N, N, NUM_ELTYPE}(undef)

        return run_benchmarks(C, A, B, CS, AS, BS)
    end

    df = vcat(mul_arrays(2), mul_arrays(4), mul_arrays(8), mul_arrays(16))
    #df = DataFrames.groupby(df, [:N, :array_t])
    # println(df)
    # println([first(d)[[:func, :array_t, :N]] for d in df])
    println(df)

    LinearAlgebra.BLAS.set_num_threads(blas_num_threads)

    @testset "2x2 static static" begin
        d = filter(row -> row.N == 2 && row.array_t == "static", df)
        @test first(d)[:func] in ["mul_explicit_2!", "mul_explicit_static_2!"]
    end

    @testset "2x2 standard standard" begin
        d = filter(row -> row.N == 2 && row.array_t == "standard", df)
        @test first(d)[:func] == "mul_explicit_2!"
    end

    @testset "2x2 static standard" begin
        d = filter(row -> row.N == 2 && row.array_t == "static-standard", df)
        @test first(d)[:func] == "mul_explicit_2!"
    end

    @testset "2x2 standard static" begin
        d = filter(row -> row.N == 2 && row.array_t == "standard-static", df)
        @test first(d)[:func] == "mul_explicit_2!"
    end

    @testset "4x4 static static" begin
        d = filter(row -> row.N == 4 && row.array_t == "static", df)
        @test first(d)[:func] in ["mul_base_any!", "mul_3loops_any!"]
    end

    @testset "4x4 standard standard" begin
        d = filter(row -> row.N == 4 && row.array_t == "standard", df)
        @test first(d)[:func] == "mul_3loops_any!"
    end

    @testset "4x4 static standard" begin
        d = filter(row -> row.N == 4 && row.array_t == "static-standard", df)
        @test first(d)[:func] == "mul_linalg_any!"
    end

    @testset "4x4 standard static" begin
        d = filter(row -> row.N == 4 && row.array_t == "standard-static", df)
        @test first(d)[:func] in ["mul_linalg_any!", "mul_linalg_any!"]
    end

    @testset "8x8 static static" begin
        d = filter(row -> row.N == 8 && row.array_t == "static", df)
        @test first(d)[:func] == "mul_base_any!"
    end

    @testset "8x8 standard standard" begin
        d = filter(row -> row.N == 8 && row.array_t == "standard", df)
        @test first(d)[:func] == "mul_linalg_any!"
    end
    
    @testset "8x8 static standard" begin
        d = filter(row -> row.N == 8 && row.array_t == "static-standard", df)
        @test first(d)[:func] == "mul_linalg_any!"
    end

    @testset "8x8 standard static" begin
        d = filter(row -> row.N == 8 && row.array_t == "standard-static", df)
        @test first(d)[:func] == "mul_linalg_any!"
    end

    # for 16x16 linalg seems to be the fastest allways
end