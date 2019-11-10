#=
--------------------------------------------------------------------------------------------------------------------------------
Base: time series
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    companion_form(Ψ::Array{Float64,2}, Σ::SymMatrix)

Construct the companion form parameters of a VAR(p) with coefficients Ψ and var-cov matrix of the residuals Σ.
"""
function companion_form(Ψ::Array{Float64,2}, Σ::SymMatrix)

    # Dimensions
    n = size(Σ,2);
    p = Int64(size(Ψ,2)/n);
    np_1 = n*(p-1);

    # Companion form VAR(p)
    C = [Ψ; Matrix(I, np_1, np_1) zeros(np_1, n)];
    V = Symmetric([Σ zeros(n, np_1); zeros(np_1, n*p)])::SymMatrix;

    # Return output
    return C, V;
end

"""
    ext_companion_form(Ψ::Array{Float64,2}, Σ::SymMatrix)

Construct the companion form parameters of a VAR(p) with coefficients Ψ and var-cov matrix of the residuals Σ. The companion form is extend with additional n entries.
"""
function ext_companion_form(Ψ::Array{Float64,2}, Σ::SymMatrix)

    # Dimensions
    n = size(Σ,2);
    p = Int64(size(Ψ,2)/n);
    np = n*p;

    # Companion form VAR(p)
    C = [Ψ zeros(n, n); Matrix(I, np, np) zeros(np, n)];
    V = Symmetric([Σ zeros(n, np); zeros(np, np+n)])::SymMatrix;

    # Return output
    return C, V;
end

#=
-------------------------------------------------------------------------------------------------------------------------------
Combinatorics and probability
-------------------------------------------------------------------------------------------------------------------------------
=#

"""
    no_combinations(n::Int64, k::Int64)

Compute the binomial coefficient of `n` observations and `k` groups, for big integers.

# Examples
```jldoctest
julia> no_combinations(1000000,100000)
7.333191945934207610471288280331309569215030711272858517142085449641265002716664e+141178
```
"""
no_combinations(n::Int64, k::Int64) = factorial(big(n))/(factorial(big(k))*factorial(big(n-k)));

"""
    rand_without_replacement(nT::Int64, d::Int64)

Draw `length(P)-d` elements from the positional vector `P` without replacement.
`P` is permanently changed in the process.

rand_without_replacement(n::Int64, T::Int64, d::Int64)

Draw `length(P)-d` elements from the positional vector `P` without replacement.
In the sampling process, no more than n-1 elements are removed for each point in time.
`P` is permanently changed in the process.

# Examples
```jldoctest
julia> rand_without_replacement(20, 5)
15-element Array{Int64,1}:
  1
  2
  3
  5
  7
  8
 10
 11
 13
 14
 16
 17
 18
 19
 20
```
"""
function rand_without_replacement(nT::Int64, d::Int64)

    # Positional vector
    P = collect(1:nT);

    # Draw without replacement d times
    for i=1:d
        deleteat!(P, findall(P.==rand(P)));
    end

    # Return output
    return setdiff(1:nT, P);
end

function rand_without_replacement(n::Int64, T::Int64, d::Int64)

    # Positional vector
    P = collect(1:n*T);

    # Full set of coordinates
    coord = [repeat(1:n, T) kron(1:T, convert(Array{Int64}, ones(n)))];

    # Counter
    coord_counter = convert(Array{Int64}, zeros(T));

    # Loop over d
    for i=1:d

        while true

            # New candidate draw
            draw = rand(P);
            coord_draw = @view coord[draw, :];

            # Accept the draw if all observations are not missing for time t = coord[draw, :][2]
            if coord_counter[coord_draw[2]] < n-1
                coord_counter[coord_draw[2]] += 1;

                # Draw without replacement
                deleteat!(P, findall(P.==draw));
                break;
            end
        end
    end

    # Return output
    return setdiff(1:n*T, P);
end
