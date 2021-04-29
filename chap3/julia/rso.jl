# QR
using LinearAlgebra, BenchmarkTools

x = collect(1:0.01:5)
y = 2 .* x .+ 5 .+ randn(length(x))

X = hcat(repeat([1], length(x)), x)

function qr_linreg(X::U, y::V) where { T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T} }
    Q, R = qr(X)
    return inv(R) * Q' * y
end

function solve_linreg(X::U, y::V) where { T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T} }
    Q, R = qr(X)
    Q = Array(Q)
    return R \ (Q' * y)
end

function svd_linreg(X::U, y::V) where { T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T} }
    return pinv(X) * y
end

println("QR: ")
@btime qr_linreg($X, $y);

println("QR (Solve): ")
@btime solve_linreg($X, $y);

println("SVD: ")
@btime svd_linreg($X, $y);

