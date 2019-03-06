# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y'[1:20,end-120+1:end] |> JArray{Float64,2};
Y_zscored=standardize(Y) |> JArray{Float64,2};

# Hyperparameters
p=2;
λ=0.6;
α=0.5;
β=1.3;

@time iis_loss = fc_err(Y_zscored, p, λ, α, β, tol=1e-3, verb=true);
@time oos_loss = fc_err(Y_zscored, p, λ, α, β, iis=false, t0=60, tol=1e-3, verb=true);
@time bjk_loss = jackknife_err(Y_zscored, p, λ, α, β, ajk=false, t0=60, tol=1e-3);
