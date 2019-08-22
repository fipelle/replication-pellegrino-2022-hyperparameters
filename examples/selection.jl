# Libraries
include("./../ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using FileIO, XLSX, DataFrames, JLD2;

# Load data
data_raw = DataFrame(XLSX.readtable("./data/data.xlsx", "weekly_returns")...);
date = data_raw[3:end,1];
data = data_raw[3:end,2:end] |> JArray{Float64,2};
data = permutedims(data);
n, T = size(data);

# Set options for the selection problem
p_grid=[1, 4]; λ_grid=[1e-4, 4]; α_grid=[0, 1]; β_grid=[1, 4];
d = optimal_d(n, T);
vs = ValidationSettings(4, data, t0=416, subsample=d/(n*T), max_samples=5000, log_folder_path=".");
hg = HyperGrid(p_grid, λ_grid, α_grid, β_grid, 1000);

# Run algorithm
Random.seed!(1);
candidates, errors = select_hyperparameters(vs, hg);
