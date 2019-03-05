#=
--------------------------------------------------------------------------------------------------------------------------------
Base
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    sum_skipmissing(X::JVector)

Compute the sum of the observed values in `X`.

    sum_skipmissing(X::JArray)

Compute the sum of the observed values `X` column wise.

# Examples
```jldoctest
julia> sum_skipmissing([1.0; missing; 3.0])
4.0

julia> sum_skipmissing([1.0 2.0; missing 3.0; 3.0 5.0])
3-element Array{Float64,1}:
 3.0
 3.0
 8.0
```
"""
sum_skipmissing(X::JVector) = sum(skipmissing(X));
sum_skipmissing(X::JArray) = vcat([sum_skipmissing(X[i,:]) for i=1:size(X,1)]...);

"""
    mean_skipmissing(X::JVector)

Compute the mean of the observed values in `X`.

    mean_skipmissing(X::JArray)

Compute the mean of the observed values in `X` column wise.

# Examples
```jldoctest
julia> mean_skipmissing([1.0; missing; 3.0])
2.0

julia> mean_skipmissing([1.0 2.0; missing 3.0; 3.0 5.0])
3-element Array{Float64,1}:
 1.5
 3.0
 4.0
```
"""
mean_skipmissing(X::JVector) = mean(skipmissing(X));
mean_skipmissing(X::JArray) = vcat([mean_skipmissing(X[i,:]) for i=1:size(X,1)]...);

"""
    std_skipmissing(X::JVector)

Compute the standard deviation of the observed values in `X`.

    std_skipmissing(X::JArray)

Compute the standard deviation of the observed values in `X` column wise.

# Examples
```jldoctest
julia> std_skipmissing([1.0; missing; 3.0])
1.4142135623730951

julia> std_skipmissing([1.0 2.0; missing 3.0; 3.0 5.0])
3-element Array{Float64,1}:
   0.7071067811865476
 NaN
   1.4142135623730951
```
"""
std_skipmissing(X::JVector) = std(skipmissing(X));
std_skipmissing(X::JArray) = vcat([std_skipmissing(X[i,:]) for i=1:size(X,1)]...);

"""
    is_vector_in_matrix(vect::AbstractVector, matr::AbstractMatrix)

Check whether the vector `vect` is included in the matrix `matr`.

# Examples
julia> is_vector_in_matrix([1;2], [1 2; 2 3])
true
"""
is_vector_in_matrix(vect::AbstractVector, matr::AbstractMatrix) = sum(sum(vect.==matr, dims=1).==length(vect)) > 0;


#=
--------------------------------------------------------------------------------------------------------------------------------
Transformations
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    standardize(X::Array{Float64,1})
    standardize(X::JVector)

Standardize data to have mean zero and unit variance.

    standardize(X::Array{Float64,2})
    standardize(X::JArray)

Standardize data to have mean zero and unit variance column wise.

# Examples
```jldoctest
julia> standardize([1.0; 1.5; 2.0; 2.5; 3.0])
5-element Array{Float64,1}:
 -1.26491
 -0.632456
  0.0
  0.632456
  1.26491

julia> standardize([1.0 3.5 1.5 4.0 2.0; 4.5 2.5 5.0 3.0 5.5])
2×5 Array{Float64,2}:
 -1.08173    0.849934  -0.695401   1.23627   -0.309067
  0.309067  -1.23627    0.695401  -0.849934   1.08173
```
"""
standardize(X::Array{Float64,1}) = (X.-mean(X))./std(X);
standardize(X::Array{Float64,2}) = (X.-mean(X,dims=2))./std(X,dims=2);
standardize(X::JVector) = (X.-mean_skipmissing(X))./std_skipmissing(X);
standardize(X::JArray) = (X.-mean_skipmissing(X))./std_skipmissing(X);

"""
    standardize_verbose(X::Array{Float64,1})
    standardize_verbose(X::JVector)

Standardize data to have mean zero and unit variance.

    standardize_verbose(X::Array{Float64,2})
    standardize_verbose(X::JArray)

Standardize data to have mean zero and unit variance column wise.

# Output
- Mean.
- Standard deviation.
- Standardized data.

# Examples
```jldoctest
julia> standardize_verbose([1.0; 1.5; 2.0; 2.5; 3.0])
(2.0, 0.7905694150420949, [-1.26491, -0.632456, 0.0, 0.632456, 1.26491])

julia> standardize_verbose([1.0 3.5 1.5 4.0 2.0; 4.5 2.5 5.0 3.0 5.5])
([2.4; 4.1], [1.29422; 1.29422], [-1.08173 0.849934 … 1.23627 -0.309067; 0.309067 -1.23627 … -0.849934 1.08173])
```
"""
standardize_verbose(X::Array{Float64,1}) = (mean(X), std(X), standardize(X));
standardize_verbose(X::Array{Float64,2}) = (mean(X,dims=2), std(X,dims=2), standardize(X));
standardize_verbose(X::JVector) = (mean_skipmissing(X), std_skipmissing(X), standardize(X));
standardize_verbose(X::JArray) = (mean_skipmissing(X), std_skipmissing(X), standardize(X));


#=
--------------------------------------------------------------------------------------------------------------------------------
Base: time series
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    lag(X::Array, p::Int64)
    lag(X::JArray, p::Int64)

Construct the data required to run a standard vector autoregression.

# Arguments
- `X`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p`: number of lags in the vector autoregression

# Output
- `X_{t}`
- `X_{t-1}`
"""
function lag(X::Array, p::Int64)

    # VAR(p) data
    X_t = X[:, 1+p:end];
    X_lagged = vcat([X[:, p-j+1:end-j] for j=1:p]...);

    # Return output
    return X_t, X_lagged;
end

function lag(X::JArray, p::Int64)

    # VAR(p) data
    X_t = X[:, 1+p:end];
    X_lagged = vcat([X[:, p-j+1:end-j] for j=1:p]...);

    # Return output
    return X_t, X_lagged;
end


"""
    companion_form(Ψ::Array{Float64,2}, Σ::Array{Float64})

Construct the companion form parameters of a VAR(p) with coefficients Ψ and var-cov matrix of the residuals Σ.
"""
function companion_form(Ψ::Array{Float64,2}, Σ::Array{Float64})

    # Dimensions
    n = size(Σ,2);
    p = Int64(size(Ψ,2)/n);
    np_1 = n*(p-1);

    # Companion form VAR(p)
    C = [Ψ; Matrix(I, np_1, np_1) zeros(np_1, n)];
    V = [Σ zeros(n, np_1); zeros(np_1, n*p)];

    # Return output
    return C, V;
end

"""
    companion_form(Ψ::Array{Float64,2}, Σ::Array{Float64})

Construct the companion form parameters of a VAR(p) with coefficients Ψ and var-cov matrix of the residuals Σ. The companion form is extend with additional n entries.
"""
function ext_companion_form(Ψ::Array{Float64,2}, Σ::Array{Float64})

    # Dimensions
    n = size(Σ,2);
    p = Int64(size(Ψ,2)/n);
    np = n*p;

    # Companion form VAR(p)
    C = [Ψ zeros(n, n); Matrix(I, np, np) zeros(np, n)];
    V = [Σ zeros(n, np); zeros(np, np+n)];

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
    rand_without_replacement!(P::Array{Int64,1}, d::Int64)

Draw `length(P)-d` elements from the positional vector `P` without replacement.
`P` is transformed into the output vector in the process and thus permanently changed.

# Examples
```jldoctest
julia> P=collect(1:20);
julia> rand_without_replacement!(P, 5);
julia> P
15-element Array{Int64,1}:
  1
  2
  3
  5
  6
  7
  8
  9
 10
 11
 13
 14
 16
 19
 20
```
"""
function rand_without_replacement!(P::Array{Int64,1}, d::Int64)
    for i=1:d
        deleteat!(P, findall(P.==rand(P)));
    end
end
