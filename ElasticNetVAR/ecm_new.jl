"""
"""
envar_penalty(estim_settings::EstimSettings, Σ::SymMatrix, Ψ::FloatArray, Φ::FloatArray) = tr(inv(Σ)*(envar_penalty_ridge(estim_settings, Ψ) + envar_penalty_lasso(estim_settings, Ψ, Φ)))::Float64;
envar_penalty_ridge(estim_settings::EstimSettings, Ψ::FloatArray) = ((1-estim_settings.α)/2) .* Symmetric(Ψ*estim_settings.Γ*Ψ')::SymMatrix;
envar_penalty_lasso(estim_settings::EstimSettings, Ψ::FloatArray, Φ::FloatArray) = (estim_settings.α/2) .* Symmetric((Ψ .* sqrt.(Φ))*estim_settings.Γ*(Ψ .* sqrt.(Φ))')::SymMatrix;

"""
"""
function update_ecm_stats!(estim_settings::EstimSettings, Xs::FloatVector, Xs_old::FloatVector, Ps::SymMatrix, Ps_old::SymMatrix, E::FloatArray, F::FloatArray, G::FloatArray)

    # Views
    Xs_view = @view Xs[1:estim_settings.n];
    Ps_view = @view Ps[1:estim_settings.n,1:estim_settings.n];
    Xs_old_view = @view Xs_old[1:estim_settings.np];
    Ps_old_view = @view Ps_old[1:estim_settings.np,1:estim_settings.np];
    PPs_view = @view Ps[1:estim_settings.n,estim_settings.n+1:end];

    # Update ECM statistics
    E .+= Xs_view*Xs_view' + Ps_view;
    F .+= Xs_view*Xs_old_view' + PPs_view;
    G .+= Xs_old_view*Xs_old_view' + Ps_old_view;
end

"""
"""
function ksmoother_ecm(estim_settings::EstimSettings, kalman_settings::KalmanSettings, status::KalmanStatus)

    # Memory pre-allocation
    E = zeros(estim_settings.n, estim_settings.n);
    F = zeros(estim_settings.n, estim_settings.np);
    G = zeros(estim_settings.np, estim_settings.np);
    Xs = copy(status.X_post);
    Ps = copy(status.P_post);

    # Loop over t
    for t=status.t:-1:2

        # Pointers
        Xp = status.history_X_prior[t];
        Pp = status.history_P_prior[t];
        Xf_lagged = status.history_X_post[t-1];
        Pf_lagged = status.history_P_post[t-1];

        # Smoothed estimates for t-1
        J1 = compute_J1(Pf_lagged, Pp, kalman_settings);
        Xs_old = backwards_pass(Xf_lagged, J1, Xs, Xp);
        Ps_old = backwards_pass(Pf_lagged, J1, Ps, Pp);

        # Update ECM statistics
        update_ecm_stats!(estim_settings, Xs, Xs_old, Ps, Ps_old, E, F, G);

        # Update Xs and Ps
        Xs = copy(Xs_old);
        Ps = copy(Ps_old);
    end

    # Pointers
    Xp = status.history_X_prior[1];
    Pp = status.history_P_prior[1];

    # Compute smoothed estimates for t==0
    J1 = compute_J1(kalman_settings.P0, Pp, kalman_settings);
    X0 = backwards_pass(kalman_settings.X0, J1, Xs, Xp);
    P0 = backwards_pass(kalman_settings.P0, J1, Ps, Pp);

    # Update ECM statistics
    update_ecm_stats!(estim_settings, Xs, X0, Ps, P0, E, F, G);

    # Use Symmetric for E and G
    E_sym = Symmetric(E)::SymMatrix;
    G_sym = Symmetric(G)::SymMatrix;

    # Return ECM statistics
    return E_sym, F, G_sym;
end

"""
    ecm(estim_settings::EstimSettings)

Estimate an elastic-net VAR(p) using the ECM algorithm in Pellegrino (2019).

# Arguments
- `estim_settings`: settings used for the estimation

# References
Pellegrino (2019)
"""
function ecm(estim_settings::EstimSettings)

    # Check inputs
    check_bounds(estim_settings.p, 1);
    check_bounds(estim_settings.λ, 0);
    check_bounds(estim_settings.α, 0, 1);
    check_bounds(estim_settings.β, 1);
    check_bounds(estim_settings.max_iter, 3);
    check_bounds(estim_settings.max_iter, estim_settings.prerun);
    check_bounds(estim_settings.n, 2); # It supports only multivariate models (for now ...)

    # Initialise using the coordinate descent algorithm
    verb_message(estim_settings.verb, "ecm > initialisation");
    Ψ_init, Σ_init = coordinate_descent(estim_settings);

    #=
    The state vector includes additional n terms with respect to the standard VAR companion form representation.
    This is to estimate the lag-one covariance smoother as in Watson and Engle (1983).
    =#

    # State-space parameters
    B = [Matrix{Float64}(I, n, n) zeros(n, np)];
    R = Symmetric(Matrix{Float64}(I, n, n).*ε)::SymMatrix;
    C, V = ext_companion_form(Ψ_init, Σ_init);
    V = Symmetric(V)::SymMatrix; # Implement in *_companion_form (TBC)

    # kalman_settings
    kalman_settings = KalmanSettings(estim_settings.Y, B, R, C, V);

    # Initialise additional variables
    Ψ = C[1:n, 1:np];
    Σ = V[1:n, 1:n];
    Φ = 1 ./ (abs.(Ψ).+ε);

    # ECM controls
    pen_loglik_old = -Inf;
    pen_loglik_new = -Inf;

    # Run ECM
    for iter=1:max_iter

        # Run Kalman filter and smoother
        status = KalmanStatus();
        for t=1:kalman_settings.T
            kfilter!(kalman_settings, status);
        end

        if iter > prerun

            # New penalised loglikelihood
            pen_loglik_new = loglik - envar_penalty(estim_settings, Σ, Ψ, Φ);
            verb_message(estim_settings.verb, "ecm > iter=$(iter-prerun), penalised loglik=$(round(pen_loglik_new, digits=5))");

            # Stop when the ECM algorithm converges
            if iter > prerun+1
                if isconverged(pen_loglik_new, pen_loglik_old, estim_settings.tol, estim_settings.ε, true)
                    verb_message(estim_settings.verb, "ecm > converged!\n");
                    break;
                end
            end

            # Store current run information
            pen_loglik_old = copy(pen_loglik_new);

        else
            verb_message(estim_settings.verb, "ecm > prerun $iter (out of $prerun)");
        end

        E, F, G = ksmoother_ecm(estim_settings, kalman_settings, status);

        # VAR(p) coefficients
        Φ = 1 ./ (abs.(Ψ).+ε);
        for i=1:n
            C[i, 1:np] = sym_inv(G + Γ.*((1-α)*I + α.*Diagonal(Φ[i, :])))*F[i,:];
        end

        # Update Ψ
        Ψ = C[1:n, 1:np];

        # Covariance matrix of the VAR(p) residuals
        V[1:n, 1:n] = sym(E-F*Ψ'-Ψ*F'+Ψ*G*Ψ') + (1-α).*sym(Ψ*Γ*Ψ') + α.*sym((Ψ.*sqrt.(Φ))*Γ*(Ψ.*sqrt.(Φ))');
        V[1:n, 1:n] *= 1/T;

        # Update Σ
        Σ = V[1:n, 1:n];
    end

    return nothing; # TBC return something
end
