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
β=1.5;

# Run
#B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, Ψ̂_init, Σ̂_init = ecm(Y_zscored, p, λ, α, β);
#err_iis(Y_zscored, p, λ, α, β)

@time for i=1:10
    err_oos(Y_zscored, p, λ, α, β, 100, tol=1e-4, verb=false);
end
