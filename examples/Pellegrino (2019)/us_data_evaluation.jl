# Libraries
include("./../../ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using FileIO;
using JLD2;

include("./us_results_functions.jl");
using Random;
using DataFrames;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y' |> JArray{Float64,2};

# Options
t0 = 120; # after the hyperparameter selection (hence, after 120)
tol = 1e-3;
max_iter = 1000;
prerun = 2;
verb = true;
standardize_Y = true;


#=
---------------------------------------------------------------
Evaluation
---------------------------------------------------------------
=#

iis_γ = zeros(4, 4);
oos_γ = zeros(4, 4);
bjk_10_γ = zeros(4, 4);
bjk_20_γ = zeros(4, 4);
ajk_10_γ = zeros(4, 4);
ajk_20_γ = zeros(4, 4);

iis_err = zeros(4);
oos_err = zeros(4);
bjk_10_err = zeros(4);
bjk_20_err = zeros(4);
ajk_10_err = zeros(4);
ajk_20_err = zeros(4);

iis_real_err = zeros(4);
oos_real_err = zeros(4);
bjk_10_real_err = zeros(4);
bjk_20_real_err = zeros(4);
ajk_10_real_err = zeros(4);
ajk_20_real_err = zeros(4);

α_selected = [0 0.5 1]

for j=1:3

    # Exp. err. and γ
    iis_err[j], iis_γ[:,j] = get_us_results("./results/iis_hyperparameters.jld2", α_selected[j]);
    oos_err[j], oos_γ[:,j] = get_us_results("./results/oos_hyperparameters.jld2", α_selected[j]);
    bjk_10_err[j], bjk_10_γ[:,j] = get_us_results("./results/bjk_hyperparameters_10.jld2", α_selected[j]);
    bjk_20_err[j], bjk_20_γ[:,j] = get_us_results("./results/bjk_hyperparameters_20.jld2", α_selected[j]);
    ajk_10_err[j], ajk_10_γ[:,j] = get_us_results("./results/ajk_hyperparameters_10.jld2", α_selected[j]);
    ajk_20_err[j], ajk_20_γ[:,j] = get_us_results("./results/ajk_hyperparameters_20.jld2", α_selected[j]);

    # OOS results
    iis_real_err[j] = fc_err(Y, iis_γ[1,j] |> Int64, iis_γ[2,j], iis_γ[3,j], iis_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
    oos_real_err[j] = fc_err(Y, oos_γ[1,j] |> Int64, oos_γ[2,j], oos_γ[3,j], oos_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
    bjk_10_real_err[j] = fc_err(Y, bjk_10_γ[1,j] |> Int64, bjk_10_γ[2,j], bjk_10_γ[3,j], bjk_10_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
    bjk_20_real_err[j] = fc_err(Y, bjk_20_γ[1,j] |> Int64, bjk_20_γ[2,j], bjk_20_γ[3,j], bjk_20_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
    ajk_10_real_err[j] = fc_err(Y, ajk_10_γ[1,j] |> Int64, ajk_10_γ[2,j], ajk_10_γ[3,j], ajk_10_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
    ajk_20_real_err[j] = fc_err(Y, ajk_20_γ[1,j] |> Int64, ajk_20_γ[2,j], ajk_20_γ[3,j], ajk_20_γ[4,j], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
end

# Exp. err. and γ
iis_err[4], iis_γ[:,4] = get_us_results("./results/iis_hyperparameters.jld2");
oos_err[4], oos_γ[:,4] = get_us_results("./results/oos_hyperparameters.jld2");
bjk_10_err[4], bjk_10_γ[:,4] = get_us_results("./results/bjk_hyperparameters_10.jld2");
bjk_20_err[4], bjk_20_γ[:,4] = get_us_results("./results/bjk_hyperparameters_20.jld2");
ajk_10_err[4], ajk_10_γ[:,4] = get_us_results("./results/ajk_hyperparameters_10.jld2");
ajk_20_err[4], ajk_20_γ[:,4] = get_us_results("./results/ajk_hyperparameters_20.jld2");

# OOS results
iis_real_err[4] = fc_err(Y, iis_γ[1,4] |> Int64, iis_γ[2,4], iis_γ[3,4], iis_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
oos_real_err[4] = fc_err(Y, oos_γ[1,4] |> Int64, oos_γ[2,4], oos_γ[3,4], oos_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
bjk_10_real_err[4] = fc_err(Y, bjk_10_γ[1,4] |> Int64, bjk_10_γ[2,4], bjk_10_γ[3,4], bjk_10_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
bjk_20_real_err[4] = fc_err(Y, bjk_20_γ[1,4] |> Int64, bjk_20_γ[2,4], bjk_20_γ[3,4], bjk_20_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
ajk_10_real_err[4] = fc_err(Y, ajk_10_γ[1,4] |> Int64, ajk_10_γ[2,4], ajk_10_γ[3,4], ajk_10_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);
ajk_20_real_err[4] = fc_err(Y, ajk_20_γ[1,4] |> Int64, ajk_20_γ[2,4], ajk_20_γ[3,4], ajk_20_γ[4,4], iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);


hyperparameters_table = cat(dims=[3], iis_γ'[[1,3,2,4],:], oos_γ'[[1,3,2,4],:], bjk_10_γ'[[1,3,2,4],:], bjk_20_γ'[[1,3,2,4],:], ajk_10_γ'[[1,3,2,4],:], ajk_20_γ'[[1,3,2,4],:]);
error_table = cat(dims=[3], [iis_err[[1,3,2,4]] iis_real_err[[1,3,2,4]]],
                            [oos_err[[1,3,2,4]] oos_real_err[[1,3,2,4]]],
                            [bjk_10_err[[1,3,2,4]] bjk_10_real_err[[1,3,2,4]]],
                            [bjk_20_err[[1,3,2,4]] bjk_20_real_err[[1,3,2,4]]],
                            [ajk_10_err[[1,3,2,4]] ajk_10_real_err[[1,3,2,4]]],
                            [ajk_20_err[[1,3,2,4]] ajk_20_real_err[[1,3,2,4]]]);

hyperparameters_table = permutedims(hyperparameters_table, [3,2,1]);
error_table = permutedims(error_table, [3,2,1]);
