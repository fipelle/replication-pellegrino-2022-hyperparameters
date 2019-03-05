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
λ=0.5;
α=0.5;
β=1.2;

# Run
#B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, Ψ̂_init, Σ̂_init = ecm(Y_zscored, p, λ, α, β);
#err_iis(Y_zscored, p, λ, α, β)

#=
@time for i=1:10
    fc_err(Y_zscored, p, λ, α, β, tol=1e-4, verb=true);
end
=#

#iis_loss = fc_err(Y_zscored, p, λ, α, β, tol=1e-4, verb=true);
#oos_loss = fc_err(Y_zscored, p, λ, α, β, iis=false, t0=60, tol=1e-4, verb=true);
bjk_loss = jackknife_err(Y_zscored, p, λ, α, β, ajk=false, t0=100, tol=1e-4);
