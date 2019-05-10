# Libraries
@everywhere include("./../../ElasticNetVAR/ElasticNetVAR.jl");
@everywhere using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;
using JLD2;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y'[1:20, 1:120] |> JArray{Float64,2};

# Options
p_grid=[1; 2; 3];
λ_grid=[0.2; 0.6; 1.0];
α_grid=[0.0; 0.5; 1.0];
β_grid=[1.0; 1.5; 2.0];
subsample = 0.2;
max_samples = 500;
t0 = 60;
tol = 1e-3;
max_iter = 1000;
prerun = 2;
verb = true;
standardize_Y = true;

# In-sample
iis_hyperparameters, iis_error_grid, iis_hyper_grid = select_hyperparameters(Y, p_grid, λ_grid, α_grid, β_grid, 1, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);
save("./results/iis_hyperparameters_20.jld2", Dict("hyper" => iis_hyperparameters, "err_grid" => iis_error_grid, "hyper_grid" => iis_hyper_grid));

# Out-of-sample
oos_hyperparameters, oos_error_grid, oos_hyper_grid = select_hyperparameters(Y, p_grid, λ_grid, α_grid, β_grid, 2, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);
save("./results/oos_hyperparameters_20.jld2", Dict("hyper" => oos_hyperparameters, "err_grid" => oos_error_grid, "hyper_grid" => oos_hyper_grid));

# Artificial jackknife
Random.seed!(1);
ajk_hyperparameters, ajk_error_grid, ajk_hyper_grid = select_hyperparameters(Y, p_grid, λ_grid, α_grid, β_grid, 3, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);
save("./results/ajk_hyperparameters_20.jld2", Dict("hyper" => ajk_hyperparameters, "err_grid" => ajk_error_grid, "hyper_grid" => ajk_hyper_grid));

# Block jackknife
bjk_hyperparameters, bjk_error_grid, bjk_hyper_grid = select_hyperparameters(Y, p_grid, λ_grid, α_grid, β_grid, 4, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);
save("./results/bjk_hyperparameters_20.jld2", Dict("hyper" => bjk_hyperparameters, "err_grid" => bjk_error_grid, "hyper_grid" => bjk_hyper_grid));