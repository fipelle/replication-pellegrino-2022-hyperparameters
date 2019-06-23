__precompile__()

module ElasticNetVAR

	using LinearAlgebra
	using Statistics;
	using Distributions;
	using Distributed;

	const local_path = dirname(@__FILE__);


	# ---------------------------------------------------------------------------------------------------------------------------
	# Types
	# ---------------------------------------------------------------------------------------------------------------------------

	# Aliases (types)
	const FloatVector  = Array{Float64,1};
	const FloatArray   = Array{Float64};
	const SymMatrix    = Symmetric{Float64,Array{Float64,2}};
	const JVector{T}   = Array{Union{Missing, T}, 1};
    const JArray{T, N} = Array{Union{Missing, T}, N};

	# Kalman structures

	"""
		KalmanSettings(...)

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
	struct KalmanSettings
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

	# KalmanSettings constructor
	function KalmanSettings(Y::JArray{Float64}, B::FloatArray, R::SymMatrix, C::FloatArray, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

		# Compute default value for missing parameters
		n, T = size(Y);
		m = size(B,2);
		X0 = zeros(m);
		P0 = Symmetric(reshape((I-kron(C, C))\V[:], m, m));

		# Return KalmanSettings
		return KalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
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
	KalmanStatus() = KalmanStatus(1, [nothing for i=1:9]...);


	# ---------------------------------------------------------------------------------------------------------------------------
	# Functions
	# ---------------------------------------------------------------------------------------------------------------------------

	# Load
    include("$local_path/methods.jl");
	include("$local_path/coordinate_descent.jl");
	include("$local_path/kalman.jl");
	include("$local_path/kalman_new.jl");
	include("$local_path/ecm.jl");
	include("$local_path/jackknife.jl");
	include("$local_path/validation.jl");

	# Export
	export JVector, JArray, KalmanSettings, KalmanStatus;
	export mean_skipmissing, std_skipmissing, is_vector_in_matrix, sym, sym_inv, demean, lag, companion_form, ext_companion_form, no_combinations, rand_without_replacement!;
	export kalman;
	export kfilter!, kforecast, ksmoother;
	export coordinate_descent, ecm;
	export block_jackknife, artificial_jackknife;
	export select_hyperparameters, fc_err, jackknife_err;
end
