# Libraries
include("./../src/MessyTimeSeriesOptim.jl");
include("./get_h10_dataset.jl");
using Main.MessyTimeSeriesOptim;
using FileIO, JLD;
using Random, Statistics;

#=
Load arguments passed through the command line
=#

# VAR or VMA
is_var = parse(Bool, ARGS[1]);
log_folder_path = ARGS[2];

#= 
Load benchmark grid of candidates from any run
=#

candidates = load("./VAR_output/err_type_2.jld")["candidates"];

#=
Presample: 53 weeks (1999) -> compute weights
Out-of-sample sample: from the 54th week onwards (2000 to 2020) -> run pseudo out-of-sample exercise 
Test sample: from the 158th week onwards (2002 to 2020) -> compute realised error
=#

# Download estimation and test samples
df = get_h10_dataset(fred_tickers, mnemonics, transform, to_include, "1999-01-01", "2020-12-25");

# Extract relevant data
presample = df[1:53, 2:end] |> JMatrix{Float64};
oos_sample = df[54:end, 2:end] |> JMatrix{Float64};
test_sample = df[158:end, 2:end] |> JMatrix{Float64};
test_dates = df[158:end, :date];

# Periods not indicated as NBER recession in the test sample
nber_recessions = vcat(collect(Date(2007, 12, 07):Week(1):Date(2009, 05, 29)), Date(2020, 02, 07):Week(1):Date(2020, 03, 27));
test_periods_excl_recessions = [findfirst(test_dates .== x) for x in setdiff(test_dates, nber_recessions)];

# Remove NaNs
nan_to_missing!(presample);
nan_to_missing!(oos_sample);
nan_to_missing!(test_sample);

# Transpose data
presample = permutedims(presample);
oos_sample = permutedims(oos_sample);
test_sample = permutedims(test_sample);

# Compute benchmark weights
weights = 1 ./ (std_skipmissing(presample).^2);

#=
Out-of-sample exercise
=#

# Validation settings
model_kwargs = (tol=1e-3, check_quantile=true, verb=false);

# Dimensions
t0 = 158-54;
n, T = size(oos_sample);
grid_length = size(candidates, 2);

# One-step ahead forecast output
weighted_errors = zeros(size(candidates,2));
weighted_errors_excl_recessions = zeros(size(candidates,2));
forecast_per_series = Array{FloatMatrix,1}(undef, size(candidates,2));

# Setup data for estimation and forecast
estimation_sample = @view oos_sample[:, 1:t0];
estimation_sample_mean = mean_skipmissing(estimation_sample);
estimation_sample_std = std_skipmissing(estimation_sample);
zscored_oos_sample = (oos_sample .- estimation_sample_mean) ./ estimation_sample_std;

# Loop over each candidate vector of hyperparameters
for i in axes(candidates, 2)

    @info("Iteration $(i) out of $(grid_length)")

    # Current candidate vector of hyperparameters
    p_float, λ, α, β = candidates[:,i];
    p = Int64(p_float);

    # Select appropriate EstimSettings
    if is_var
        estim = VARSettings(zscored_oos_sample[:, 1:t0], p, λ, α, β; model_kwargs...);
    else
        estim = VMASettings(zscored_oos_sample[:, 1:t0], p, λ, α, β; model_kwargs...);
    end

    # Estimate model
    sspace = ecm(estim, output_sspace_data=zscored_oos_sample);

    # Run Kalman filter
    status = kfilter_full_sample(sspace);

    # Forecast
    X_prior = mapreduce(Xt -> sspace.B*Xt, hcat, status.history_X_prior);
    forecast_per_series[i] = (X_prior[:, t0+1:end] .* estimation_sample_std) .+ estimation_sample_mean;
    weighted_se = weights .* (test_sample .- forecast_per_series[i]).^2;
    weighted_errors[i] = MessyTimeSeriesOptim.compute_loss(weighted_se)[1];
    weighted_errors_excl_recessions[i] = MessyTimeSeriesOptim.compute_loss(weighted_se[:, test_periods_excl_recessions])[1];
end

# Save output to JLD
save("$(log_folder_path)/realised_error.jld", Dict("candidates" => candidates, "weighted_errors" => weighted_errors, "weighted_errors_excl_recessions" => weighted_errors_excl_recessions, "test_periods_excl_recessions" => test_periods_excl_recessions, "forecast_per_series" => forecast_per_series));