__precompile__()

module ElasticNetVAR

	# Libraries
	using Distributed, Dates, Logging;
	using LinearAlgebra, Distributions, Statistics;
	using TSAnalysis;

	# Patch (TODO: update TSAnalysis and remove block)
	check_bounds 	  = TSAnalysis.check_bounds;
	isnothing 		  = TSAnalysis.isnothing;
	error_info 		  = TSAnalysis.error_info;
	verb_message      = TSAnalysis.verb_message;
	interpolate 	  = TSAnalysis.interpolate;
	soft_thresholding = TSAnalysis.soft_thresholding;
	isconverged 	  = TSAnalysis.isconverged;
	FloatVector 	  = TSAnalysis.FloatVector;
	FloatArray 		  = TSAnalysis.FloatArray;
	SymMatrix 		  = TSAnalysis.SymMatrix;
	DiagMatrix 		  = TSAnalysis.DiagMatrix;

	# Aliases for TSAnalysis
	compute_J1 = TSAnalysis.compute_J1;
	backwards_pass = TSAnalysis.backwards_pass;

	# Custom dependencies
	const local_path = dirname(@__FILE__);
	include("$local_path/types.jl");
	include("$local_path/methods.jl");
	include("$local_path/coordinate_descent.jl");
	include("$local_path/ecm.jl");
	include("$local_path/jackknife.jl");
	include("$local_path/validation.jl");

	# Export
	export JVector, JArray, EstimSettings, ValidationSettings, HyperGrid;
	export companion_form, ext_companion_form, no_combinations, rand_without_replacement;
	export coordinate_descent, ecm;
	export coordinate_descent, build_Γ;
	export block_jackknife, artificial_jackknife, optimal_d;
	export select_hyperparameters, fc_err, jackknife_err;
end
