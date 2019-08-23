__precompile__()

module ElasticNetVAR

	# Libraries
	using Distributed;
	using Dates, Logging;
	using LinearAlgebra, Distributions, Statistics;

	# Custom dependencies
	const local_path = dirname(@__FILE__);
	include("$local_path/types.jl");
	include("$local_path/methods.jl");
	include("$local_path/coordinate_descent.jl");
	include("$local_path/kalman.jl");
	include("$local_path/ecm.jl");
	include("$local_path/jackknife.jl");
	include("$local_path/validation.jl");

	# Export
	export JVector, JArray, ImmutableKalmanSettings, KalmanStatus, EstimSettings, ValidationSettings, HyperGrid;
	export mean_skipmissing, std_skipmissing, is_vector_in_matrix, sym, sym_inv, demean, lag, companion_form, ext_companion_form, no_combinations, rand_without_replacement;
	export kalman;
	export kfilter!, kforecast, ksmoother;
	export coordinate_descent, ecm;
	export coordinate_descent, build_Γ;
	export block_jackknife, artificial_jackknife, optimal_d;
	export select_hyperparameters, fc_err, jackknife_err;
end
