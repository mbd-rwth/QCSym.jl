import LinearAlgebra
import StaticArrays
import BenchmarkTools

function mul_3loops_any!(C, A, B)
    for i in axes(A, 1)
        for j in axes(B, 2)
            C[i, j] = zero(eltype(C))
            for k in axes(A, 2)
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
end

function mul_explicit_2!(C, A, B)
    @inbounds C[1, 1] = A[1, 1] * B[1, 1] + A[1, 2] * B[2, 1]
    @inbounds C[2, 1] = A[2, 1] * B[1, 1] + A[2, 2] * B[2, 1]
    @inbounds C[1, 2] = A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2]
    @inbounds C[2, 2] = A[2, 1] * B[1, 2] + A[2, 2] * B[2, 2]
end

function mul_explicit_static_2!(C, A, B)
    ad, bd = A.data, B.data
    # .data is column-major: (X11, X21, X12, X22)
    @inbounds C.data = (ad[1]*bd[1]+ad[3]*bd[2], ad[2]*bd[1]+ad[4]*bd[2], ad[1]*bd[3]+ad[3]*bd[4], ad[2]*bd[3]+ad[4]*bd[4])
    return nothing
end

function mul_strassen_static_2!(C, A, B)
    ad, bd = A.data, B.data
    # .data is column-major: (X11, X21, X12, X22)
    @inbounds M = ((ad[1]+ad[4])*(bd[1]+bd[4]), (ad[2]+ad[4])*bd[1], ad[1]*(bd[3]-bd[4]), ad[4]*(bd[2]-bd[1]), (ad[1]+ad[3])*bd[4], (ad[2]-ad[1])*(bd[1]+bd[3]), (ad[3]-ad[4])*(bd[2]+bd[4]))
    @inbounds C.data=(M[1]+M[4]-M[5]+M[7], M[2]+M[4], M[3]+M[5], M[1]-M[2]+M[3]+M[6])
    return nothing
end

function mul_linalg_any!(C, A, B)
    #C .= A * B
    LinearAlgebra.mul!(C, A, B)
    #LinearAlgebra.BLAS.gemm!('N', 'N', 1.0+0.0im, A, B, 0.0+0.0im, C)
end

function mul_base_any!(C, A, B)
    C = A * B
end


