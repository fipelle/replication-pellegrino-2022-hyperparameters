# Libraries
@everywhere include("./../../ElasticNetVAR/ElasticNetVAR.jl");
@everywhere using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;
using JLD2;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y'[1:20, 1:204] |> JArray{Float64,2}; # up to Dec-2006

# Options
p_grid_0=[1; 6];
λ_grid_0=[0.01; 2.0];
α_grid_0=[0.0; 1.0];
β_grid_0=[1.0; 2.0];
rs_draws = 500;
subsample = 0.25;
max_samples = 500;
t0 = 96; # up to Dec-1997
tol = 1e-3;
max_iter = 1000;
prerun = 2;
verb = true;
log_folder = "$(dirname(@__FILE__))/log";
demean_Y = true;

# In-sample
Random.seed!(1);
iis_hyperparameters, iis_error_grid, iis_hyper_grid = select_hyperparameters(Y, p_grid_0, λ_grid_0, α_grid_0, β_grid_0, 1, rs=true, rs_draws=rs_draws, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, log_folder=log_folder, demean_Y=demean_Y);
save("./results/iis_hyperparameters.jld2", Dict("hyper" => iis_hyperparameters, "err_grid" => iis_error_grid, "hyper_grid" => iis_hyper_grid));

# Out-of-sample
Random.seed!(1);
oos_hyperparameters, oos_error_grid, oos_hyper_grid = select_hyperparameters(Y, p_grid_0, λ_grid_0, α_grid_0, β_grid_0, 2, rs=true, rs_draws=rs_draws, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, log_folder=log_folder, demean_Y=demean_Y);
save("./results/oos_hyperparameters.jld2", Dict("hyper" => oos_hyperparameters, "err_grid" => oos_error_grid, "hyper_grid" => oos_hyper_grid));

# Block jackknife
Random.seed!(1);
bjk_hyperparameters, bjk_error_grid, bjk_hyper_grid = select_hyperparameters(Y, p_grid_0, λ_grid_0, α_grid_0, β_grid_0, 4, rs=true, rs_draws=rs_draws, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, log_folder=log_folder, demean_Y=demean_Y);
save("./results/bjk_hyperparameters_25.jld2", Dict("hyper" => bjk_hyperparameters, "err_grid" => bjk_error_grid, "hyper_grid" => bjk_hyper_grid));

# Artificial jackknife
Random.seed!(1);
ajk_hyperparameters, ajk_error_grid, ajk_hyper_grid = select_hyperparameters(Y, p_grid_0, λ_grid_0, α_grid_0, β_grid_0, 3, rs=true, rs_draws=rs_draws, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, log_folder=log_folder, demean_Y=demean_Y);
save("./results/ajk_hyperparameters_25.jld2", Dict("hyper" => ajk_hyperparameters, "err_grid" => ajk_error_grid, "hyper_grid" => ajk_hyper_grid));
