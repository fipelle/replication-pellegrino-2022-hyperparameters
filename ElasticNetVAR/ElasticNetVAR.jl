__precompile__()

module ElasticNetVAR

	using LinearAlgebra
	using Statistics;
	using Distributions;
	using Distributed;

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
	include("$local_path/jackknife.jl");
	include("$local_path/validation.jl");

	# Export
	export JVector, JArray;
	export mean_skipmissing, std_skipmissing, is_vector_in_matrix, sym, sym_inv, demean, lag, companion_form, ext_companion_form, no_combinations, rand_without_replacement!;
	export kalman;
	export coordinate_descent, ecm;
	export block_jackknife, artificial_jackknife;
	export select_hyperparameters, fc_err, jackknife_err;
end
