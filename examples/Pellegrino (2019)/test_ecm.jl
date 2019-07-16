include("./../../ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using DataFrames;
using FileIO;
using JLD2;
using LinearAlgebra
using Debugger;

Y=DataFrame(load("./data/data.csv")) |> JArray{Float64,2};
Y=Y' |> JArray{Float64,2}; # full data
Y = Y[5:10, 1:200];

const FloatVector  = Array{Float64,1};
const FloatArray   = Array{Float64};
const SymMatrix    = Symmetric{Float64,Array{Float64,2}};
const DiagMatrix   = Diagonal{Float64,Array{Float64,1}};

p_grid=[2, 4]; λ_grid=[1e-4, 10]; α_grid=[0,1]; β_grid=[1,10];
vs = ValidationSettings(4, Y, t0=100, subsample=0.1, max_samples=5, log_folder_path=".");
hg = HyperGrid(p_grid, λ_grid, α_grid, β_grid, 5);

#Random.seed!(1);
#@time out_old = select_hyperparameters(Y, p_grid, λ_grid, α_grid, β_grid, 2, rs_draws=5, verb=false);

Random.seed!(1);
#breakpoint(fc_err, 11);
select_hyperparameters(vs, hg);
#bp add "validation.jl":140
#bp add "jackknife_err":1
