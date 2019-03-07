# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y'[1:20,:] |> JArray{Float64,2};
Y_zscored=standardize(Y) |> JArray{Float64,2};

# Hyperparameters
p=2;
λ=0.5;
α=0.5;
β=1.3;

#@time iis_loss = fc_err(Y_zscored, p, λ, α, β, tol=1e-3, verb=true);
#@time oos_loss = fc_err(Y_zscored, p, λ, α, β, iis=false, t0=60, tol=1e-3, verb=true);
bjk_loss = jackknife_err(Y_zscored, p, λ, α, β, ajk=false, t0=120, tol=1e-3);
#@time ajk_loss = jackknife_err(Y_zscored, p, λ, α, β, ajk=true, t0=60, tol=1e-3, max_samples=200);
#@time ajk_loss_500 = jackknife_err(Y_zscored, p, λ, α, β, ajk=true, t0=60, tol=1e-3, max_samples=500);
