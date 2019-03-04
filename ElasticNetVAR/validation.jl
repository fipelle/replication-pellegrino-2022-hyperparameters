"""
"""
function select_hyperparameters(Y::JArray{Float64,2}, p_grid::Array{Int64,1}, λ_grid::Array{Number,1}, α_grid::Array{Number,1}, β_grid::Array{Number,1}; tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, verb=true)

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
function err_iis(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, verb=true)

    # Estimate the penalised vector autoregression
    B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y, p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

    # Run Kalman filter and smoother
    𝔛ŝ, _, _, _, _, _, _, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false);

    # Measure in-sample fit
    Ŷ = 𝔛ŝ[1:size(Y,1), :][:];
    Y_vec = Y[:];
    ind_obs = .~(ismissing.(Y_vec));
    loss = mean((Ŷ[ind_obs]-Y_vec[ind_obs]).^2);

    # Return output
    return loss;
end

"""
"""
function err_oos(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number, t0::Int64; tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, verb=true)

    # Initialise
    n, T = size(Y);
    loss = 0.0;

    # Estimate the penalised VAR
    B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y[:,1:t0], p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

    # Run Kalman filter and smoother
    _, _, _, _, _, _, 𝔛p, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

    # Measure out-of-sample fit
    Ŷ = 𝔛p[1:size(Y,1), t0+1:end][:];
    Y_vec = Y[:, t0+1:end][:];
    ind_obs = .~(ismissing.(Y_vec));
    loss = mean((Ŷ[ind_obs]-Y_vec[ind_obs]).^2);

    # Return output
    return loss;
end
