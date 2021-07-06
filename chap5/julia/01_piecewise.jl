using LinearAlgebra, Statistics

function find_beta(X::U, y::V) where { T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T}}
    return pinv(X) * y
end

# data = [ x | y ]
function piecewise_constant(data::U, interval::V) where { T <: Number, U <: AbstractMatrix{T}, V <: AbstractVector{T} }
    beta = zeros(T, length(interval)+1);
    prev_node = interval[1];
    beta[1] = mean(data[data[:,1] .< prev_node, :]);
    for i in 2:length(interval)
        curr_node = interval[i];
        node_data = data[(data[:,1] .>= prev_node) .& (data[:,1] .< curr_node), :];
        beta[i] = mean(node_data);
        prev_node = interval[i];
    end
    beta[end] = mean(data[data[:,1] .>= prev_node, :]);

    return function(x::T)
        idx = 1;
        if x < interval[1]
            return beta[1]
        end

        for i in 2:length(interval)
            if x < interval[i]
                return beta[i]
            end
        end

        return beta[end]
    end
end

    