# Libraries
include("./../../ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;
using JLD2;

include("./us_results_functions.jl");
using Random;
using DataFrames;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y' |> JArray{Float64,2}; # full data

# Options
t0 = 205; # after the hyperparameter selection (hence, after 205)
tol = 1e-3;
max_iter = 1000;
prerun = 2;
verb = true;
demean_Y = true;


#=
---------------------------------------------------------------
Evaluation
---------------------------------------------------------------
=#

iis_γ = zeros(4, 3);
oos_γ = zeros(4, 3);
bjk_25_γ = zeros(4, 3);
ajk_25_γ = zeros(4, 3);

# Exp. err. and γ
lag_selected = [2,4];
for j=1:2
    iis_γ[:,j] = get_us_results("./results/iis_hyperparameters.jld2", lag_selected[j]);
    oos_γ[:,j] = get_us_results("./results/oos_hyperparameters.jld2", lag_selected[j]);
    bjk_25_γ[:,j] = get_us_results("./results/bjk_hyperparameters_25.jld2", lag_selected[j]);
    ajk_25_γ[:,j] = get_us_results("./results/ajk_hyperparameters_25.jld2", lag_selected[j]);
end

# Exp. err. and γ
iis_γ[:,3] = get_us_results("./results/iis_hyperparameters.jld2");
oos_γ[:,3] = get_us_results("./results/oos_hyperparameters.jld2");
bjk_25_γ[:,3] = get_us_results("./results/bjk_hyperparameters_25.jld2");
ajk_25_γ[:,3] = get_us_results("./results/ajk_hyperparameters_25.jld2");

fc_iis = get_reconstruction(Y, iis_γ[:,3]);
fc_oos = get_reconstruction(Y, oos_γ[:,3]);
fc_bjk = get_reconstruction(Y, bjk_25_γ[:,3]);
fc_ajk = get_reconstruction(Y, ajk_25_γ[:,3]);

using PlolyJS;
YY = Y.-mean_skipmissing(Y[:,1:204]);

sqerr_iis = sqrt.(mean(((YY - fc_iis).^2)[:, 205:end], dims=2));
sqerr_bjk = sqrt.(mean(((YY - fc_bjk).^2)[:, 205:end], dims=2));
sqerr_ajk = sqrt.(mean(((YY - fc_ajk).^2)[:, 205:end], dims=2));
sqerr_iis_excrisis = sqrt.(mean(((YY - fc_iis).^2)[:, 241:end], dims=2));
sqerr_bjk_excrisis = sqrt.(mean(((YY - fc_bjk).^2)[:, 241:end], dims=2));
sqerr_ajk_excrisis = sqrt.(mean(((YY - fc_ajk).^2)[:, 241:end], dims=2));

RMSFE_ajk_bjk = sqerr_ajk./sqerr_bjk;
RMSFE_ajk_iis = sqerr_ajk./sqerr_iis;
RMSFE_ajk_bjk_excrisis = sqerr_ajk_excrisis./sqerr_bjk_excrisis;
RMSFE_ajk_iis_excrisis = sqerr_ajk_excrisis./sqerr_iis_excrisis;

save("./out_tables.csv", DataFrame([RMSFE_ajk_bjk RMSFE_ajk_iis RMSFE_ajk_bjk_excrisis RMSFE_ajk_iis_excrisis]));
