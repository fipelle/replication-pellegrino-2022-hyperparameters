# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y'[1:2,:] |> JArray{Float64,2};
Y_zscored=standardize(Y) |> JArray{Float64,2};

# Hyperparameters
p=4;
λ=0.5;
α=0.5;
β=1.5;

# Run
B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, Ψ̂_init, Σ̂_init = ecm(Y_zscored, p, λ, α, β);
