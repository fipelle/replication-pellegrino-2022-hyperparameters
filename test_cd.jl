# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;

# Data
Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=standardize(Y);
Y=Y[:,1:3]' |> JArray{Float64,2};

# Hyperparameters
p=3;
λ=2;
α=0.5;
β=10.0;

    # Interpolated data (used for the initialisation only)
    Y_init = copy(Y);
    n=3
    for i=1:n
        Y_init[i, ismissing.(Y_init[i, :])] .= Main.ElasticNetVAR.mean_skipmissing(Y_init[i, :]);
    end
    Y_init = Y_init |> Array{Float64};

    # VAR(p) data
    Y_init, X_init = Main.ElasticNetVAR.lag(Y_init, p);
