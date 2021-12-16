# Libraries
using Distributed;
@everywhere using MessyTimeSeriesOptim;
include("./get_h10_dataset.jl");
using FileIO, JLD;
using Random, Statistics;

#=
Load arguments passed through the command line
=#

# VAR or VMA
is_var = parse(Bool, ARGS[1]);

# Error estimator id
err_type = parse(Int64, ARGS[2]);

# Output folder
log_folder_path = ARGS[3];

if (length(ARGS) > 3 && err_type != 3) || (length(ARGS) > 4 && err_type == 3)
    error("Wrong number of arguments passed through the command line");
end

# Print number of workers
@info("$(nprocs()) workers ready!");

#=
Presample: 53 weeks (1999) -> compute weights
Selection sample: 52*2 weeks (2000 and 2001) -> select hyperparameters
=#

# Download presample and selection samples
df = get_h10_dataset(fred_tickers, mnemonics, transform, to_include, "1999-01-01", "2001-12-28");

# Extract relevant data
presample = df[1:53, 2:end] |> JMatrix{Float64};
selection_sample = df[54:end, 2:end] |> JMatrix{Float64};

# Remove NaNs
nan_to_missing!(presample);
nan_to_missing!(selection_sample);

# Transpose data
presample = permutedims(presample);
selection_sample = permutedims(selection_sample);

# Validation inputs: common for all err_types
gamma_bounds = ([4, 4], [0.01, 2.50], [0.0, 1.0], [1.0, 2.0]);
grid_prerun = HyperGrid(gamma_bounds..., 1);
grid = HyperGrid(gamma_bounds..., 1000);
weights = 1 ./ (std_skipmissing(presample).^2);
weights = weights[:];

# Validation inputs: common for all oos err_types
t0 = 52;
n, T = size(selection_sample);

# Subsample
if err_type < 3 # iis and oos
    subsample = 1.0; # not used in validation for these cases

elseif err_type == 3 # block jackknife
    subsample = parse(Float64, ARGS[4]);

elseif err_type == 4 # artificial jackknife
    d = optimal_d(n, T);
    subsample = d/(n*T);
end

# Validation inputs: specific for the artificial jackknife (not used in validation for the other cases)
max_samples = 1000;

# Validation settings
model_kwargs = (tol=1e-3, check_quantile=true, verb=false);

if is_var
    vs_prerun = ValidationSettings(err_type, selection_sample, true, VARSettings, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path, verb=false, model_kwargs=model_kwargs);
    vs = ValidationSettings(err_type, selection_sample, true, VARSettings, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path, model_kwargs=model_kwargs);
else
    vs_prerun = ValidationSettings(err_type, selection_sample, true, VMASettings, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path, verb=false, model_kwargs=model_kwargs);
    vs = ValidationSettings(err_type, selection_sample, true, VMASettings, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path, model_kwargs=model_kwargs);
end

# Test run to speed up compilation
_ = select_hyperparameters(vs_prerun, grid_prerun);

# Compute and print ETA
ETA = @elapsed select_hyperparameters(vs_prerun, grid_prerun);
ETA *= grid.draws;
ETA /= 3600;
println("ETA: $(round(ETA, digits=2)) hours");

# Actual run
candidates, errors = select_hyperparameters(vs, grid);

# Save output to JLD
save("$(log_folder_path)/err_type_$(err_type).jld", Dict("df" => df, "presample" => presample, "selection_sample" => selection_sample, "vs" => vs, "grid" => grid, "candidates" => candidates, "errors" => errors));