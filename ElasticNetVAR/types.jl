# Aliases (types)
const FloatVector  = Array{Float64,1};
const FloatArray   = Array{Float64};
const SymMatrix    = Symmetric{Float64,Array{Float64,2}};
const DiagMatrix   = Diagonal{Float64,Array{Float64,1}};
const JVector{T}   = Array{Union{Missing, T}, 1};
const JArray{T, N} = Array{Union{Missing, T}, N};

# Kalman structures

abstract type KalmanSettings end

"""
    ImmutableKalmanSettings(...)

Define an immutable structure that includes all the Kalman filter and smoother input.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + u_{t}``

Where ``e_{t} ~ N(0, R)`` and ``u_{t} ~ N(0, V)``.

# Arguments
- `Y`: observed measurements (`nxT`)
- `B`: Measurement equations' coefficients
- `R`: Covariance matrix of the measurement equations' error terms
- `C`: Transition equations' coefficients
- `V`: Covariance matrix of the transition equations' error terms
- `X0`: Mean vector for the states at time t=0
- `P0`: Covariance matrix for the states at time t=0
- `n`: Number of series
- `T`: Number of observations
- `m`: Number of latent states
- `compute_loglik`: Boolean (true for computing the loglikelihood in the Kalman filter)
- `store_history`: Boolean (true to store the history of the filter and smoother)
"""
struct ImmutableKalmanSettings <: KalmanSettings
    Y::JArray{Float64}
    B::FloatArray
    R::SymMatrix
    C::FloatArray
    V::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

# ImmutableKalmanSettings constructor
function ImmutableKalmanSettings(Y::JArray{Float64}, B::FloatArray, R::SymMatrix, C::FloatArray, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);
    X0 = zeros(m);
    P0 = Symmetric(reshape((I-kron(C, C))\V[:], m, m));

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

# ImmutableKalmanSettings constructor
function ImmutableKalmanSettings(Y::JArray{Float64}, B::FloatArray, R::SymMatrix, C::FloatArray, V::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

"""
    MutableKalmanSettings(...)

Define a mutable structure identical in shape to ImmutableKalmanSettings.

See the docstring of ImmutableKalmanSettings for more information.
"""
mutable struct MutableKalmanSettings <: KalmanSettings
    Y::JArray{Float64}
    B::FloatArray
    R::SymMatrix
    C::FloatArray
    V::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

# MutableKalmanSettings constructor
function MutableKalmanSettings(Y::JArray{Float64}, B::FloatArray, R::SymMatrix, C::FloatArray, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);
    X0 = zeros(m);
    P0 = Symmetric(reshape((I-kron(C, C))\V[:], m, m));

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

# MutableKalmanSettings constructor
function MutableKalmanSettings(Y::JArray{Float64}, B::FloatArray, R::SymMatrix, C::FloatArray, V::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

"""
    KalmanStatus(...)

Define an mutable structure to manage the status of the Kalman filter and smoother.

# Arguments
- `t`: Current point in time
- `loglik`: Loglikelihood
- `X_prior`: Latest a-priori X
- `X_post`: Latest a-posteriori X
- `P_prior`: Latest a-priori P
- `P_post`: Latest a-posteriori P
- `history_X_prior`: History of a-priori X
- `history_X_post`: History of a-posteriori X
- `history_P_prior`: History of a-priori P
- `history_P_post`: History of a-posteriori P
"""
mutable struct KalmanStatus
    t::Int64
    loglik::Union{Float64, Nothing}
    X_prior::Union{FloatVector, Nothing}
    X_post::Union{FloatVector, Nothing}
    P_prior::Union{SymMatrix, Nothing}
    P_post::Union{SymMatrix, Nothing}
    history_X_prior::Union{Array{FloatVector,1}, Nothing}
    history_X_post::Union{Array{FloatVector,1}, Nothing}
    history_P_prior::Union{Array{SymMatrix,1}, Nothing}
    history_P_post::Union{Array{SymMatrix,1}, Nothing}
end

# KalmanStatus constructors
KalmanStatus() = KalmanStatus(0, [nothing for i=1:9]...);

"""
    EstimSettings(...)

Define an immutable structure used to initialise the estimation routine.

# Arguments
- `Y`: observed measurements (`nxT`)
- `n`: Number of series
- `T`: Number of observations
- `T`: Number of lags
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
    Y::JArray{Float64,2}
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

# EstimSettings constructor
EstimSettings(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ε::Float64=1e-8, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true) =
    EstimSettings(Y, size(Y,1), size(Y,2), p, size(Y,1)*p, λ, α, β, build_Γ(size(Y,1), p, λ, β), ε, tol, max_iter, prerun, verb);
