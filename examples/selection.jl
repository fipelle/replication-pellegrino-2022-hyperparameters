# Libraries
using Distributed;
@everywhere include("./../ElasticNetVAR/ElasticNetVAR.jl");
@everywhere using Main.ElasticNetVAR;
using Random;
using FileIO, XLSX, DataFrames, BSON;

# Load data
data_raw = DataFrame(XLSX.readtable("./data/data.xlsx", "FX")...);

# Data
data = data_raw[3:end, 3:end] |> JArray{Float64,2};

# Store dates
date = data_raw[3:end,2];

# Skip EA19 and New Zealand
data = data[:, [1;2;4;collect(6:size(data,2))]] |> JArray{Float64,2};

# Transpose data
data = permutedims(data);

# End presample
end_ps = 80; # 1979-12-31

# End validation sample
end_vs = 160; # 1999-12-31

# Data used for validation
data_validation = data[:, 1:end_vs];
n, T = size(data_validation);

# Select optimal d
@info "Running optimal_d(n, T)";
d = optimal_d(n, T);

# Set options for the selection problem
p_grid=[1, 5]; λ_grid=[1e-4, 5]; α_grid=[0, 1]; β_grid=[1, 5];
vs = ValidationSettings(4, data_validation, t0=end_ps, subsample=d/(n*T), max_samples=5000, log_folder_path=".");
hg = HyperGrid(p_grid, λ_grid, α_grid, β_grid, 1000);

# Run algorithm
Random.seed!(1);
@info "Running select_hyperparameters(vs, hg)";
candidates, errors = select_hyperparameters(vs, hg);

# Save to file
save("./res.bson", Dict("data" => data,
                        "date" => date,
                        "end_ps" => end_ps,
                        "end_vs" => end_vs,
                        "d" => d,
                        "vs" => vs,
                        "hg" => hg,
                        "candidates" => candidates,
                        "errors" => errors));
