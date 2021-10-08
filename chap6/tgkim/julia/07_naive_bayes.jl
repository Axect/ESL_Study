using Statistics, Plots, Distributions, Random, LaTeXStrings, Distributed

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
    function p_vec(x0)
        # arr = π_vec .* ([map.(f_vec, x0)])
        arr = π_vec .* (map(f -> f(x0), f_vec))
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

function parallel_kde(x::S, domain::T, p_vec::F) where { S <: Number, T <: AbstractVector{S}, F <: Function }
    xs = repeat([x], length(domain))
    xys = [[a, b] for (a, b) in zip(xs, domain)]
    return p_vec.(xys)
end

function compute_boundary2d(total_vec)
    boundary = Vector{Float64}(undef, length(total_vec))
    for i in 1:length(total_vec)
        x_fixed = hcat(total_vec[i]...)
        temp = findall(abs.(x_fixed[1,:] .- 0.5) .< 1e-1)
        if isempty(temp)
            boundary[i] = 0
        else
            boundary[i] = temp |> median |> round |> Int
        end
    end
    return boundary
end

function refine_boundary(domain, boundary)
    mat = hcat(domain, boundary)
    refine_mat = mat[mat[:,2] .> 0, :]
    refine_mat[:,2] = domain[map(Int, refine_mat[:,2])]
    return refine_mat
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

    domain = collect(0.0:0.1:100.0)

    if ARGS[1] == "kdc"
        # 1. KDC for math with Natural Local Estimate
        f_math1 = gen_natural_local_estimate(data1[:,1], λ)
        f_math2 = gen_natural_local_estimate(data2[:,1], λ)
        f_math_vec = [f_math1, f_math2]
        π_vec = [0.5, 0.5]
        p_vec_math = gen_kdc(f_math_vec, π_vec)
        p_nat_math = hcat(p_vec_math.(domain)...)

        plot(domain, p_nat_math[1,:], label="Group1")
        plot!(domain, p_nat_math[2,:], label="Group2")
        savefig("figure/kdc_natural.png")
        
        # 2. KDC for math with Parzen Estimate
        kernel = gen_gaussian_kernel(λ)
        f_math1 = gen_parzen_estimate(data1[:,1], kernel, λ)
        f_math2 = gen_parzen_estimate(data2[:,1], kernel, λ)
        f_math_vec = [f_math1, f_math2]
        π_vec = [0.5, 0.5]
        p_vec_math = gen_kdc(f_math_vec, π_vec)
        p_parzen_math = hcat(p_vec_math.(domain)...)

        closeall()
        plot(domain, p_parzen_math[1,:], label="Group1")
        plot!(domain, p_parzen_math[2,:], label="Group2")
        savefig("figure/kdc_parzen.png")
        
    elseif ARGS[1] == "naive"
        # 1. Naive Bayes (Natural Estimate)
        f_math1 = gen_natural_local_estimate(data1[:,1], λ)
        f_eng1 = gen_natural_local_estimate(data1[:,2], λ)
        f_math2 = gen_natural_local_estimate(data2[:,1], λ)
        f_eng2 = gen_natural_local_estimate(data2[:,2], λ)
        f_idd1 = idd_mul([f_math1, f_eng1])
        f_idd2 = idd_mul([f_math2, f_eng2])
        
        f_nat = [f_idd1, f_idd2]
        π_vec = [0.5, 0.5]
        p_vec = gen_kdc(f_nat, π_vec)

        parallel_kde_nat(x) = parallel_kde(x, domain, p_vec)
        p_nat = pmap(parallel_kde_nat, domain)
        boundary = compute_boundary2d(p_nat)
        result = refine_boundary(domain, boundary)
        scatter(data1[:,1], data1[:,2], label="Group1")
        scatter!(data2[:,1], data2[:,2], label="Group2")
        plot!(result[:,1], result[:,2], label="Boundary")
        savefig("figure/naive_bayes_natural.png")

        # 2. Naive Bayes (Parzen Estimate)
        kernel = gen_gaussian_kernel(λ)
        f_math1 = gen_parzen_estimate(data1[:,1], kernel, λ)
        f_eng1 = gen_parzen_estimate(data1[:,2], kernel, λ)
        f_math2 = gen_parzen_estimate(data2[:,1], kernel, λ)
        f_eng2 = gen_parzen_estimate(data2[:,2], kernel, λ)
        f_idd1 = idd_mul([f_math1, f_eng1])
        f_idd2 = idd_mul([f_math2, f_eng2])

        f_parzen = [f_idd1, f_idd2]
        π_vec = [0.5, 0.5]
        p_vec = gen_kdc(f_parzen, π_vec)

        parallel_kde_parzen(x) = parallel_kde(x, domain, p_vec)
        p_parzen = pmap(parallel_kde_parzen, domain)
        boundary = compute_boundary2d(p_parzen)
        result = refine_boundary(domain, boundary)
        closeall()
        scatter(data1[:,1], data1[:,2], label="Group1")
        scatter!(data2[:,1], data2[:,2], label="Group2")
        plot!(result[:,1], result[:,2], label="Boundary")
        savefig("figure/naive_bayes_parzen.png")
    end
end

main()