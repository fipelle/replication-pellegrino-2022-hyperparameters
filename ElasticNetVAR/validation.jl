"""
    select_hyperparameters(Y::JArray{Float64,2}, p_grid_0::Array{Int64,1}, λ_grid_0::Array{<:Number,1}, α_grid_0::Array{<:Number,1}, β_grid_0::Array{<:Number,1}, err_type::Int64; rs::Bool=true, rs_draws::Int64=500, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, log_folder::String="~", demean_Y::Bool=true)

Select the tuning hyper-parameters for the elastic-net vector autoregression.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p_grid_0`: grid of candidates for the number of lags in the vector autoregression (grid bounds only for the random search algorithm)
- `λ_grid_0`: grid of candidates for the overall shrinkage hyper-parameter for the elastic-net penalty (grid bounds only for the random search algorithm)
- `α_grid_0`: grid of candidates for the weight associated to the LASSO component of the elastic-net penalty (grid bounds only for the random search algorithm)
- `β_grid_0`: grid of candidates for the additional shrinkage for distant lags (grid bounds only for the random search algorithm)
- `err_type`: (1) in-sample, (2) out-of-sample, (3) artificial jackknife, (4) block jackknife
- `rs::Bool`: True for random search (default: true)
- `rs_draws`: Number of draws used to construct the random search grid (default: 500)
- `subsample`: Number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1. (default: 0.20)
- `max_samples`: if `C(T*n,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples. (default: 500 - used only when ajk==true)
- `t0`: End of the estimation sample (default: 1)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual ECM estimation routine (default: 2)
- `verb`: Verbose output (default: true)
- `log_folder`: folder to store the log file (default: "~")
- `demean_Y`: demean data (default: true)

# References
Pellegrino (2019)
"""
function select_hyperparameters(Y::JArray{Float64,2}, p_grid_0::Array{Int64,1}, λ_grid_0::Array{<:Number,1}, α_grid_0::Array{<:Number,1}, β_grid_0::Array{<:Number,1}, err_type::Int64; rs::Bool=true, rs_draws::Int64=500, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, log_folder::String="~", demean_Y::Bool=true)

    # Construct grid of hyperparameters - random search algorithm
    if rs == true
        if length(p_grid) != 2 || length(λ_grid) != 2 || length(α_grid) != 2 || length(β_grid) != 2
            error("The grids include more than two entries. Random search algorithm: they must include only the bounds for the grids!")
        end
        error_grid = zeros(rs_draws);
        p_grid = rand(Uniform(p_grid_0[1], p_grid_0[2]), rs_draws);
        λ_grid = rand(Uniform(λ_grid_0[1], λ_grid_0[2]), rs_draws);
        α_grid = rand(Uniform(α_grid_0[1], α_grid_0[2]), rs_draws);
        β_grid = rand(Uniform(β_grid_0[1], β_grid_0[2]), rs_draws);

    # Use pre-defined grid of hyperparameters - grid search algorithm
    else
        error_grid = zeros(length(p_grid)*length(λ_grid)*length(α_grid)*length(β_grid));
        p_grid = copy(p_grid_0);
        λ_grid = copy(λ_grid_0);
        α_grid = copy(α_grid_0);
        β_grid = copy(β_grid_0);
    end

    hyper_grid = zeros(4, length(error_grid));

    iter = 1;
    for p=p_grid
        for λ=λ_grid
            for α=α_grid
                for β=β_grid
                    if verb == true
                        message = "select_hyperparameters (error estimator $err_type) > running iteration $iter (out of $(length(error_grid)))";
                        println(message);
                        open("$log_folder/status.txt", "a") do io
                            write(io, "$message\n")
                        end
                    end

                    # in-sample error
                    if err_type == 1
                        error_grid[iter] = fc_err(Y, p, λ, α, β, iis=true, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, demean_Y=demean_Y);

                    # out-of-sample error
                    elseif err_type == 2
                        error_grid[iter] = fc_err(Y, p, λ, α, β, iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, demean_Y=demean_Y);

                    # artificial jackknife error
                    elseif err_type == 3
                        error_grid[iter] = jackknife_err(Y, p, λ, α, β, ajk=true, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, demean_Y=demean_Y);

                    # block jackknife error
                    elseif err_type == 4
                        error_grid[iter] = jackknife_err(Y, p, λ, α, β, ajk=false, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, demean_Y=demean_Y);
                    end

                    # Update hyper_grid and iter
                    hyper_grid[:, iter] = [p, λ, α, β];
                    iter += 1;
                end
            end
        end
    end

    if verb == true
        println("");
    end

    # Return output
    ind_min = argmin(error_grid);
    return hyper_grid[:, ind_min], error_grid, hyper_grid;
end


"""
    fc_err(data::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; iis::Bool=false, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, demean_Y::Bool=true)

Return the in-sample / out-of-sample error.

# Arguments
- `data`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p`: number of lags in the vector autoregression
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `iis`: True (false) for the in-sample (out-of-sample) error (default: false)
- `t0`: End of the estimation sample (default: 1)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual ECM estimation routine (default: 2)
- `verb`: Verbose output (default: true)
- `demean_Y`: demean data (default: true)

# References
Pellegrino (2019)
"""
function fc_err(data::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; iis::Bool=false, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, demean_Y::Bool=true)

    # Initialise
    n, T = size(data);

    # In-sample
    if iis == true

        # Demean data
        if demean_Y == true
            Y = demean(data) |> JArray{Float64};
        else
            Y = copy(data) |> JArray{Float64};
        end

        # Estimate the penalised VAR
        B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

        # Run Kalman filter and smoother
        _, _, _, _, _, _, 𝔛p, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

        # Residuals
        resid = (𝔛p[1:size(Y,1), :] - Y).^2 |> JArray{Float64};
        ind_not_all_missings = sum(ismissing.(Y), dims=1) .!= size(Y,1);

    # Out-of-sample
    else

        # Run Kalman filter and smoother
        𝔛p = zeros(n, T-t0);

        for t=t0:T-1

            # Demean data
            if demean_Y == true
                Y = demean(data[:,1:t]) |> JArray{Float64};
            else
                Y = data[:,1:t] |> JArray{Float64};
            end

            # Estimate the penalised VAR
            if t == t0
                B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);
            end

            # Out-of-sample
            Y = [Y missing.*ones(n)];
            _, _, _, _, _, _, 𝔛p_t, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

            # Store new forecast
            𝔛p[:, t-t0+1] = 𝔛p_t[1:n, t+1];
        end

        # Demean data
        if demean_Y == true
            Y = demean(data) |> JArray{Float64};
        else
            Y = copy(data) |> JArray{Float64};
        end

        # Residuals
        resid = (𝔛p - Y[:, t0+1:end]).^2 |> JArray{Float64};
        ind_not_all_missings = sum(ismissing.(Y[:, t0+1:end]), dims=1) .!= size(Y,1);
    end

    # Removes t with missings only
    ind_not_all_missings = findall(ind_not_all_missings[:]);
    resid = resid[:, ind_not_all_missings];

    # Compute loss
    loss = mean([mean_skipmissing(resid[:,t]) for t=1:size(resid,2)]);

    # Return output
    return loss;
end


"""
    jackknife_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ajk::Bool=true, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, demean_Y::Bool=true)

Return the jackknife out-of-sample error.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p`: number of lags in the vector autoregression
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `ajk`: True (false) for the artificial (block) jackknife (default: true)
- `subsample`: Number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1. (default: 0.20)
- `max_samples`: if `C(T*n,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples. (default: 500 - used only when ajk==true)
- `t0`: End of the estimation sample (default: 1)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual ECM estimation routine (default: 2)
- `verb`: verbose output (default: true)
- `demean_Y`: demean data (default: true)

# References
Pellegrino (2019)
"""
function jackknife_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ajk::Bool=true, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, demean_Y::Bool=true)

    # Block jackknife
    if ajk == false
        jackknife_data = block_jackknife(Y, subsample);

    # Artificial jackknife
    else
        jackknife_data = artificial_jackknife(Y, subsample, max_samples);
    end

    # Number of jackknife samples
    samples = size(jackknife_data, 3);

    # Compute jackknife loss
    if verb == true
        println("jackknife_err > running $samples iterations on $(nworkers()) workers");
    end

    loss = 0.0;
    loss = @sync @distributed (+) for j=1:samples
        fc_err(jackknife_data[:,:,j], p, λ, α, β, iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, demean_Y=demean_Y)/samples;
    end

    if verb == true
        println("");
    end

    # Return output
    return loss;
end
