include("./../../ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;
using JLD2;
using LinearAlgebra

Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y' |> JArray{Float64,2}; # full data

const FloatVector  = Array{Float64,1};
const FloatArray   = Array{Float64};
const SymMatrix    = Symmetric{Float64,Array{Float64,2}};
const DiagMatrix   = Diagonal{Float64,Array{Float64,1}};

# ------------------------------
# from ecm.jl
# ------------------------------

p=2; λ=0.6; α=0.5; β=1.2;
@time out_old = ecm(Y, p, λ, α, β);


# ------------------------------
# New code
# ------------------------------

estim_settings = EstimSettings(Y, p, λ, α, β)
@time out_new = ecm(estim_settings);
