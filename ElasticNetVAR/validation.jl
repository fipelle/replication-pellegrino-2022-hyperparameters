"""
"""
function select_hyperparameters(Y::JArray{Float64,2}, p_grid::Array{Int64,1}, λ_grid::Array{Number,1}, α_grid::Array{Number,1}, β_grid::Array{Number,1}; tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true)

    error_grid = zeros(length(p_grid)*length(λ_grid)*length(α_grid)*length(β_grid));

    iter = 1;
    for p=p_grid
        for λ=λ_grid
            for α=α_grid
                for β=β_grid
                    error_grid[iter] = 0;# TBA
                    iter += 1;
                end
            end
        end
    end
end

"""
"""
function fc_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; iis::Bool=true, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true)

    # Initialise
    n, T = size(Y);

    # Estimate the penalised VAR

    # In-sample
    if iis == true
        B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

    # Out-of-sample
    else
        B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y[:,1:t0], p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);
    end

    # Run Kalman filter and smoother
    _, _, _, _, _, _, 𝔛p, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

    # Measure out-of-sample fit
    loss = 0.0;

    # In-sample
    if iis == true
        # Residuals
        resid = (𝔛p[1:size(Y,1), :] - Y).^2;

        # Removes t with missings only
        ind_not_all_missings = sum(ismissing.(Y), dims=1) .!= size(Y,1);
        ind_not_all_missings = findall(ind_not_all_missings[:]);
        resid = resid[:, ind_not_all_missings];

        # Compute loss
        loss += mean([mean_skipmissing(resid[:,t]) for t=1:size(resid,2)]);

    # Out-of-sample
    else
        # Residuals
        resid = (𝔛p[1:size(Y,1), t0+1:end] - Y[:, t0+1:end]).^2;

        # Removes t with missings only
        ind_not_all_missings = sum(ismissing.(Y[:, t0+1:end]), dims=1) .!= size(Y,1);
        ind_not_all_missings = findall(ind_not_all_missings[:]);
        resid = resid[:, ind_not_all_missings];

        # Compute loss
        loss += mean([mean_skipmissing(resid[:,t]) for t=1:size(resid,2)]);
    end

    # Return output
    return loss;
end


"""
"""
function jackknife_err(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; ajk::Bool=true, subsample::Float64=0.20, max_samples::Int64=500, t0::Int64=1, tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb::Bool=true)

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
        loss += fc_err(jackknife_data[:,:,j], p, λ, α, β, iis=false, t0=t0, tol=tol, max_iter=max_iter, prerun=prerun, verb=false)/samples;
    end

    if verb == true
        println("");
    end

    # Return output
    return loss;
end
