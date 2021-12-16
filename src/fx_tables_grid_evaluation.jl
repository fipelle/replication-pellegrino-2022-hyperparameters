# Libraries
using DataFrames, Dates, FileIO, JLD;
using Contour, DecisionTree, Random, StableRNGs;
using PGFPlotsX, LaTeXStrings;
using MessyTimeSeries, MessyTimeSeriesOptim, Statistics;

overall_mse  = Matrix{Float64}(undef, 5, 2);
model_prefix = "VAR";

# BJK specific options
t0 = 52;
adj_bjk = true;

for i=1:2
    
    f1 = load("./$(model_prefix)_output/realised_error.jld");
    
    if i == 1
        realised_errors = f1["weighted_errors"];
    else
        realised_errors = f1["weighted_errors_excl_recessions"];
    end

    for j=1:5

        # Raw output
        if j <= 3
            err_type = j;
        else
            err_type = j-1;
        end

        if err_type != 3
            f2 = load("./$(model_prefix)_output/err_type_$(err_type).jld");
        else
            f2 = load("./$(model_prefix)_output/3_$(10+(j-3)*10)pct/err_type_$(err_type).jld");
        end

        expected_errors = f2["errors"];

        if err_type == 3
            if adj_bjk
                if j==3
                    bjk_sample = block_jackknife(f2["selection_sample"], 0.1);
                else
                    bjk_sample = block_jackknife(f2["selection_sample"], 0.2);
                end

                T_validation = size(bjk_sample, 2)-t0;
                missings_after_t0 = [sum(sum(ismissing.(bjk_sample[:, t0+1:end, i]), dims=1) .== size(bjk_sample, 1)) for i=1:size(bjk_sample, 3)];
                expected_errors .*= mean(T_validation ./ (T_validation .- missings_after_t0));
            end
        end

        # Compute metrics
        overall_mse[j,i] = mean((realised_errors .- expected_errors).^2);
    end
end

println(DataFrame(round.(overall_mse ./ overall_mse[2,:]', digits=2), Symbol.(["all periods", "all periods excl. recessions"])));