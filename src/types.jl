# Estimation structures

"""
    EstimSettings(...)

Define an immutable structure used to initialise the estimation routine.

# Arguments
- `Y`: observed measurements (`nxT`)
- `Y_output`: observed measurements (`nxTT`) to use to construct the KalmanSettings output (`TT` can be different than `T`). It is not used for the estimation.
- `n`: Number of series
- `T`: Number of observations
- `p`: Number of lags
- `np`: n*p
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `Γ`: Diagonal matrix used to input the hyperparameters in the estimation - see Pellegrino (2019) for details
- `ε`: Small number (default: 1e-8)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual estimation algorithm (default: 2)
- `verb`: Verbose output (default: true)
"""
struct EstimSettings
    Y::Union{FloatArray, JArray{Float64}}
    Y_output::Union{FloatArray, JArray{Float64}}
    n::Int64
    T::Int64
    p::Int64
    np::Int64
    λ::Number
    α::Number
    β::Number
    Γ::DiagMatrix
    ε::Float64
    tol::Float64
    max_iter::Int64
    prerun::Int64
    verb::Bool
end

# Constructor for Γ
function build_Γ(n::Int64, p::Int64, λ::Number, β::Number)

    vec_Γ = Array{Float64,1}();
    for i=0:p-1
        push!(vec_Γ, (β^i).*ones(n)...);
    end

    return λ.*Diagonal(vec_Γ)::DiagMatrix;
end

# EstimSettings constructors
EstimSettings(Y::Union{FloatArray, JArray{Float64}}, p::Int64, λ::Number, α::Number, β::Number; ε::Float64=1e-8, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true) =
    EstimSettings(Y, Y, size(Y,1), size(Y,2), p, size(Y,1)*p, λ, α, β, build_Γ(size(Y,1), p, λ, β), ε, tol, max_iter, prerun, verb);

EstimSettings(Y::Union{FloatArray, JArray{Float64}}, Y_output::Union{FloatArray, JArray{Float64}}, p::Int64, λ::Number, α::Number, β::Number; ε::Float64=1e-8, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true) =
    EstimSettings(Y, Y_output, size(Y,1), size(Y,2), p, size(Y,1)*p, λ, α, β, build_Γ(size(Y,1), p, λ, β), ε, tol, max_iter, prerun, verb);

# Validation types

"""
    ValidationSettings(...)

Define an immutable structure used to define the validation settings.

The arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.

# Arguments
- `err_type`:
    - 1 In-sample error
    - 2 Out-of-sample error
    - 3 Block jackknife error
    - 4 Artificial jackknife error
- `Y`: observed measurements (`nxT`)
- `Y_output`: observed measurements (`nxTT`) to use to construct the KalmanSettings output (`TT` can be different than `T`). It is not used for the estimation.
- `n`: Number of series
- `T`: Number of observations
- `ε`: Small number (default: 1e-8)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual estimation algorithm (default: 2)
- `verb`: Verbose output (default: true)
- `verb_estim`: Verbose output for the estimation algorithm (default: true)
- `weights`: Weights for the forecast error. standardise_error has priority over weights. (default: ones(n))
- `standardise_error`: Divide the forecast error for the standard deviation of the data, computed on the presample (default: false)
- `t0`: weight associated to the LASSO component of the elastic-net penalty
- `subsample`: number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1.
- `max_samples`: if `C(n*T,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples. (used only for the artificial jackknife)
- `log_folder_path`: folder to store the log file. When this file is defined then the stdout is redirected to this file.
"""
struct ValidationSettings
    err_type::Int64
    Y::Union{FloatArray, JArray{Float64}}
    n::Int64
    T::Int64
    ε::Float64
    tol::Float64
    max_iter::Int64
    prerun::Int64
    verb::Bool
    verb_estim::Bool
    standardise_error::Bool
    weights::Union{FloatVector, Nothing}
    t0::Union{Int64, Nothing}
    subsample::Union{Float64, Nothing}
    max_samples::Union{Int64, Nothing}
    log_folder_path::Union{String, Nothing}
end

# Constructor for ValidationSettings
ValidationSettings(err_type::Int64, Y::Union{FloatArray, JArray{Float64}}; ε::Float64=1e-8, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, verb_estim::Bool=false, standardise_error::Bool=false, weights::Union{FloatVector, Nothing}=nothing, t0::Union{Int64, Nothing}=nothing, subsample::Union{Float64, Nothing}=nothing, max_samples::Union{Int64, Nothing}=nothing, log_folder_path::Union{String, Nothing}=nothing) =
    ValidationSettings(err_type, Y, size(Y,1), size(Y,2), ε, tol, max_iter, prerun, verb, verb_estim, standardise_error, weights, t0, subsample, max_samples, log_folder_path);

"""
    HyperGrid(...)

Define an immutable structure used to define the grid of hyperparameters used in validation(...).

The arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.

# Arguments
- `p`: Number of lags
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `draws`: number of draws used to construct the grid of candidates
"""
struct HyperGrid
    p::Array{Int64,1}
    λ::Array{<:Number,1}
    α::Array{<:Number,1}
    β::Array{<:Number,1}
    draws::Int64
end
