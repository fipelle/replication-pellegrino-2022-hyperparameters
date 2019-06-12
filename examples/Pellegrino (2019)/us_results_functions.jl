function get_us_results(output_path, lag_selected=[])
    out = load(output_path);

    # Select minimum error and corresponding vector of hyperparameters
    if ~isempty(lag_selected)
        ind_lag = findall(out["hyper_grid"][1,:] .== lag_selected);
    else
        ind_lag = collect(1:length(out["err_grid"]));
    end

    ind_γ = argmin(out["err_grid"][ind_lag]);
    γ = out["hyper_grid"][:, ind_lag[ind_γ]];

    return γ;
end


function get_reconstruction(data, γ; t0=204);

    p, λ, α, β = γ;
    p = Int64(p);
    n, T = size(data);

    # Run Kalman filter and smoother
    𝔛p = zeros(n, T-t0);

    Y = data.-mean_skipmissing(data[:,1:t0]) |> JArray{Float64};

    # Estimate the penalised VAR
    B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, _, _ = ecm(Y[:,1:t0], p, λ, α, β, tol=tol, max_iter=max_iter, prerun=prerun, verb=verb);

    # Out-of-sample
    _, _, _, _, _, _, 𝔛p_t, _, _ = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=false, kf_only_flag=true);

    return 𝔛p_t[1:n,:];
end
