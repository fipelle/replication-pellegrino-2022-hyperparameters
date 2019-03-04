__precompile__()

module ElasticNetVAR

	using LinearAlgebra
	using Statistics;
	using Distributions;

	const local_path = dirname(@__FILE__);

	# Aliases (types)
	const FloatVector  = Array{Float64,1};
	const FloatArray   = Array{Float64};
	const JVector{T}   = Array{Union{Missing, T}, 1};
    const JArray{T, N} = Array{Union{Missing, T}, N};


	# ---------------------------------------------------------------------------------------------------------------------------
	# Functions
	# ---------------------------------------------------------------------------------------------------------------------------

	# Load
    include("$local_path/methods.jl");
	include("$local_path/coordinate_descent.jl");
	include("$local_path/kalman.jl");
	include("$local_path/ecm.jl");
	include("$local_path/validation.jl");

	# Export
	export JVector, JArray;
	export mean_skipmissing, std_skipmissing, standardize, lag, companion_form, rand_without_replacement!;
	export kalman;
	export coordinate_descent, ecm;
	export fc_err;
end
