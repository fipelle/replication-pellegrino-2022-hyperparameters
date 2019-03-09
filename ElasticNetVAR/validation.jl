"""
    select_hyperparameters(Y::JArray{Float64,2}, p_grid::Array{Int64,1}, λ_grid::Array{Number,1}, α_grid::Array{Number,1}, β_grid::Array{Number,1}, err_type::Int64; subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

Select the tuning hyper-parameters for the elastic-net vector autoregression.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p_grid`: grid of candidates for the number of lags in the vector autoregression
- `λ_grid`: grid of candidates for the overall shrinkage hyper-parameter for the elastic-net penalty
- `α_grid`: grid of candidates for the weight associated to the LASSO component of the elastic-net penalty
- `β_grid`: grid of candidates for the additional shrinkage for distant lags (p>1)
- `err_type`: (1) in-sample, (2) out-of-sample, (3) artificial jackknife, (4) block jackknife
- `subsample`: Number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1. (default: 0.20)
- `max_samples`: if `C(T*n,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples. (default: 500 - used only when ajk==true)
- `t0`: End of the estimation sample (default: 1)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual ECM estimation routine (default: 2)
- `verb`: Verbose output (default: true)
- `standardize_Y`: Standardize data (default: true)

# References
Pellegrino (2019)
"""
function select_hyperparameters(Y::JArray{Float64,2}, p_grid::Array{Int64,1}, λ_grid::Array{Number,1}, α_grid::Array{Number,1}, β_grid::Array{Number,1}, err_type::Int64; subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

    error_grid = zeros(length(p_grid)*length(λ_grid)*length(α_grid)*length(β_grid));
    hyper_grid = zeros(4, length(p_grid)*length(λ_grid)*length(α_grid)*length(β_grid))

    iter = 1;
    for p=p_grid
        for λ=λ_grid
            for α=α_grid
                for β=β_grid
                    if verb == true
                        println("select_hyperparameters > running iteration $iter (out of $(length(error_grid)))");
                    end

                    # in-sample error
                    if err_type == 1
                        error_grid[iter] = fc_err(Y, p, λ, α, β, iis=true, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);

                    # out-of-sample error
                    elseif err_type == 2
                        error_grid[iter] = fc_err(Y, p, λ, α, β, iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y);

                    # artificial jackknife error
                    elseif err_type == 3
                        error_grid[iter] = jackknife_err(Y, p, λ, α, β, ajk=true, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);

                    # block jackknife error
                    elseif err_type == 4
                        error_grid[iter] = jackknife_err(Y, p, λ, α, β, ajk=false, subsample=subsample, max_samples=max_samples, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb, standardize_Y=standardize_Y);
                    end

                    # Update hyper_grid and iter
                    hyper_grid[:, iter] = [p, λ, α, β];
                    iter += 1;
                end
            end
        end
    end

    # Return output
    ind_min = argmin(error_grid);
    return hyper_grid[:, ind_min];
end


"""
    fc_err(data::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; iis::Bool=false, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

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
- `standardize_Y`: Standardize data (default: true)

# References
Pellegrino (2019)
"""
function fc_err(data::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; iis::Bool=false, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

    # Initialise
    n, T = size(data);
    Y = copy(data);

    # In-sample
    if iis == true

        # Standardize data
        if standardize_Y == true
            Y = standardize(data) |> JArray{Float64};
        end

        # Estimate the penalised VAR
        B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

        # Run Kalman filter and smoother
        _, _, _, _, _, _, 𝔛p, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

        # Residuals
        resid = (𝔛p[1:size(Y,1), :] - Y).^2;
        ind_not_all_missings = sum(ismissing.(Y), dims=1) .!= size(Y,1);

    # Out-of-sample
    else

        # Run Kalman filter and smoother
        𝔛p = zeros(n, T-t0);

        for t=t0:T

            # Standardize data
            if standardize_Y == true
                Y = standardize(data[:,1:t]) |> JArray{Float64};
            end

            # Estimate the penalised VAR
            if t == t0
                B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

            # Out-of-sample
            else
                _, _, _, _, _, _, 𝔛p_t, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);
                𝔛p[:, t-t0] = 𝔛p_t[1:n, t];
            end
        end

        # Residuals
        resid = (𝔛p - Y[:, T-t0:end]).^2;
        ind_not_all_missings = sum(ismissing.(Y[:, T-t0:end]), dims=1) .!= size(Y,1);
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
    jackknife_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ajk::Bool=true, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

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
- `verb`: Verbose output (default: true)
- `standardize_Y`: Standardize data (default: true)

# References
Pellegrino (2019)
"""
function jackknife_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ajk::Bool=true, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true, standardize_Y::Bool=true)

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
    loss = 0.0;
    for j=1:samples
        if verb == true
            println("jackknife_err > iteration $j (out of $samples)");
        end
        loss += fc_err(jackknife_data[:,:,j], p, λ, α, β, iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false, standardize_Y=standardize_Y)/samples;
    end

    if verb == true
        println("");
    end

    # Return output
    return loss;
end
