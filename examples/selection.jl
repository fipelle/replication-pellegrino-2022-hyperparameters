# Libraries
using Distributed;
@everywhere include("./../ElasticNetVAR/ElasticNetVAR.jl");
@everywhere using Main.ElasticNetVAR;
using Random;
using FileIO, XLSX, DataFrames, JLD2;

# Load data
data_raw = DataFrame(XLSX.readtable("./data/data.xlsx", "weekly_returns")...);
date = data_raw[3:end,1];
data = data_raw[3:end,2:end] |> JArray{Float64,2};
data = permutedims(data);

# End presample
end_ps = 416; # 2006-12-29

# End validation sample
end_vs = 833; # 2014-12-24

# Data used for validation
data_validation = data[:, 1:end_vs];
n, T = size(data_validation);

# Select optimal d
@info "Running optimal_d(n, T)";
d = optimal_d(n, T);

# Set options for the selection problem
p_grid=[1, 4]; λ_grid=[1e-4, 4]; α_grid=[0, 1]; β_grid=[1, 4];
vs = ValidationSettings(4, data_validation, t0=end_ps, subsample=d/(n*T), max_samples=5000, log_folder_path=".");
hg = HyperGrid(p_grid, λ_grid, α_grid, β_grid, 1000);

# Run algorithm
Random.seed!(1);
@info "Running select_hyperparameters(vs, hg)";
candidates, errors = select_hyperparameters(vs, hg);

# Save to file
save("./res.jld2", Dict("data" => data,
                        "date" => date,
                        "end_ps" => end_ps,
                        "end_vs" => end_vs,
                        "d" => d,
                        "vs" => vs,
                        "hg" => hg,
                        "candidates" => candidates,
                        "errors" => errors));
