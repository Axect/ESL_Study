using Statistics, Plots, Distributions, Random, LaTeXStrings

function knn_lam(arr::T, x::S, λ::S) where { S <: Number, T <: AbstractVector{S} }
    dist = abs.(arr .- x)
    idx = sortperm(dist)
    dist_sorted = dist[idx]
    return idx[dist_sorted .< λ]
end

function gen_natural_local_estimate(x::T, λ::S) where { S <: Number, T <: AbstractVector{S} }
    function f(x0::S)
        x0_nearest = knn_lam(x, x0, λ)
        return length(x0_nearest) / ( length(x) * λ )
    end

    return f
end

function gen_parzen_estimate(x::T, kernel::F, lam::S) where { S <: Number, T <: AbstractVector{S}, F <: Function }
    function f(x0::S)
        eval_kernel = x -> kernel(x0, x)
        sum(eval_kernel.(x)) / ( length(x) * lam )
    end

    return f
end

function gen_gaussian_kernel(λ::S) where { S <: Number }
    function f(x0::S, x::S)
        return exp( - (x0 - x)^2 / (2 * λ^2) ) / sqrt(2 * π)
    end

    return f
end

function gen_kdc(f_vec::G, π_vec::T) where { S <: Number, T <: AbstractVector{S}, F <: Function, G <: AbstractVector{F} }
    function p_vec(x0::S)
        arr = π_vec .* (map.(f_vec, x0))
        s = sum(arr)
        return s > 0 ? arr ./ s : zeros(length(arr))
    end

    return p_vec
end

function idd_mul(f_vec::G) where { F <: Function, G <: AbstractVector{F} }
    function f(x0_vec::T) where { S <: Number, T <: AbstractVector{S} }
        return prod(map.(f_vec, x0_vec))
    end

    return f
end

# ==============================================================================
# Main Function
# ==============================================================================

Random.seed!(8407)

const λ = 5.0
const N = 100

function main()
    Pr_math1 = Normal(70, 10)
    Pr_math2 = Normal(40, 15)
    Pr_eng1 = Normal(50, 10)
    Pr_eng2 = Normal(80, 5)

    math1_score = rand(Pr_math1, N)
    math2_score = rand(Pr_math2, N)
    eng1_score = rand(Pr_eng1, N)
    eng2_score = rand(Pr_eng2, N)

    data1 = hcat(math1_score, eng1_score)
    data2 = hcat(math2_score, eng2_score)

    domain = 0.0:0.1:100.0

    if ARGS[1] == "kdc"
        # 1. KDC for math with Natural Local Estimate
        f_math1 = gen_natural_local_estimate(data1[:,1], λ)
        f_math2 = gen_natural_local_estimate(data2[:,1], λ)
        f_math_vec = [f_math1, f_math2]
        π_vec = [0.5, 0.5]
        p_vec_math = gen_kdc(f_math_vec, π_vec)
        p_nat_math = hcat(p_vec_math.(domain)...)
        
        # 2. KDC for math with Parzen Estimate
        kernel = gen_gaussian_kernel(λ)
        f_math1 = gen_parzen_estimate(data1[:,1], kernel, λ)
        f_math2 = gen_parzen_estimate(data2[:,1], kernel, λ)
        f_math_vec = [f_math1, f_math2]
        π_vec = [0.5, 0.5]
        p_vec_math = gen_kdc(f_math_vec, π_vec)
        p_parzen_math = hcat(p_vec_math.(domain)...)

        plot(domain, p_nat_math[1,:], label="Group1")
        plot!(domain, p_nat_math[2,:], label="Group2")
    end
end

main()