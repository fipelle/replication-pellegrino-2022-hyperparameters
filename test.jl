# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=standardize(Y);
Y=Y' |> JArray{Float64,2};

# Hyperparameters
p=5;
λ=0.5;
α=0.5;
β=1.5;

# Run
B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, Ψ̂_init, Σ̂_init = ecm(Y, p, λ, α, β);

# test if the standardization is really required
