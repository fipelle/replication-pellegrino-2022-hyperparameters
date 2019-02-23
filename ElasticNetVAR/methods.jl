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
1×2 Array{Float64,2}:
 4.0  10.0
```
"""
sum_skipmissing(X::JVector) = sum(skipmissing(X));
sum_skipmissing(X::JArray) = hcat([sum_skipmissing(X[:,i]) for i=1:size(X,2)]...);

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
1×2 Array{Float64,2}:
 2.0  3.33333
```
"""
mean_skipmissing(X::JVector) = mean(skipmissing(X));
mean_skipmissing(X::JArray) = hcat([mean_skipmissing(X[:,i]) for i=1:size(X,2)]...);

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
1×2 Array{Float64,2}:
 1.41421  1.52753
```
"""
std_skipmissing(X::JVector) = std(skipmissing(X));
std_skipmissing(X::JArray) = hcat([std_skipmissing(X[:,i]) for i=1:size(X,2)]...);

"""
    isweird(X::JArray)

Give a BitArray with true entries for each item of `X` equal to "nan" or "inf".

# Examples
```jldoctest
julia> isweird([NaN; 1.0; Inf; 0.0; 2.0])
5-element BitArray{1}:
  true
 false
  true
 false
 false

julia> isweird([NaN 6.0; 1.0 3.0; Inf NaN; 0.0 12.5; 2.0 NaN])
5×2 BitArray{2}:
  true  false
 false  false
  true   true
 false  false
 false   true
```
"""
isweird(X::JArray) = (isnan.(X).+isinf.(X)).>=1;


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

julia> standardize([1.0 3.5; 1.5 4.0; 2.0 4.5; 2.5 5.0; 3.0 5.5])
5×2 Array{Float64,2}:
 -1.26491   -1.26491
 -0.632456  -0.632456
  0.0        0.0
  0.632456   0.632456
  1.26491    1.26491
```
"""
standardize(X::Array{Float64,1}) = (X.-mean(X))./std(X);
standardize(X::Array{Float64,2}) = (X.-mean(X,1))./std(X,1);
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

julia> standardize_verbose([1.0 3.5; 1.5 4.0; 2.0 4.5; 2.5 5.0; 3.0 5.5])
([2.0 4.5], [0.790569 0.790569], [-1.26491 -1.26491; -0.632456 -0.632456; … ; 0.632456 0.632456; 1.26491 1.26491])
```
"""
standardize_verbose(X::Array{Float64,1}) = (mean(X), std(X), standardize(X));
standardize_verbose(X::Array{Float64,2}) = (mean(X,1), std(X,1), standardize(X));
standardize_verbose(X::JVector) = (mean_skipmissing(X), std_skipmissing(X), standardize(X));
standardize_verbose(X::JArray) = (mean_skipmissing(X), std_skipmissing(X), standardize(X));


#=
--------------------------------------------------------------------------------------------------------------------------------
Base: time series
--------------------------------------------------------------------------------------------------------------------------------
=#

function lag(X::JArray, p::Int64)

    # VAR(p) data
    X_t = X[:, 1+p:end];
    X_lagged = vcat([X[:, p-j+1:end-j] for j=1:p]...);

    # Return output
    return X_t, X_lagged;
end


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
