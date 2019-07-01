"""
"""
envar_penalty(estim_settings::EstimSettings, Σ::AbstractArray{Float64}, Ψ::SubArray{Float64}, Φ::FloatArray) = tr(inv(Σ)*(envar_penalty_ridge(estim_settings, Ψ) + envar_penalty_lasso(estim_settings, Ψ, Φ)))::Float64;
envar_penalty_ridge(estim_settings::EstimSettings, Ψ::SubArray{Float64}) = ((1-estim_settings.α)/2) .* Symmetric(Ψ*estim_settings.Γ*Ψ')::SymMatrix;
envar_penalty_lasso(estim_settings::EstimSettings, Ψ::SubArray{Float64}, Φ::FloatArray) = (estim_settings.α/2) .* Symmetric((Ψ .* sqrt.(Φ))*estim_settings.Γ*(Ψ .* sqrt.(Φ))')::SymMatrix;

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
    return E_sym, F, G_sym, X0, P0;
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
    kalman_settings = MutableKalmanSettings(estim_settings.Y,
                                            [Matrix{Float64}(I, estim_settings.n, estim_settings.n) zeros(estim_settings.n, estim_settings.np)],
                                            Symmetric(Matrix{Float64}(I, estim_settings.n, estim_settings.n).*estim_settings.ε)::SymMatrix,
                                            ext_companion_form(Ψ_init, Σ_init)...);

    # Initialise additional variables
    Ψ = @view kalman_settings.C[1:estim_settings.n, 1:estim_settings.np];
    Φ = @. 1 / (abs(Ψ) + estim_settings.ε);

    # ECM controls
    pen_loglik_old = -Inf;
    pen_loglik_new = -Inf;

    # Run ECM
    for iter=1:estim_settings.max_iter

        # Run Kalman filter and smoother
        status = KalmanStatus();
        for t=1:kalman_settings.T
            kfilter!(kalman_settings, status);
        end

        if iter > estim_settings.prerun

            # New penalised loglikelihood
            Σ = Symmetric(@view parent(kalman_settings.V)[1:estim_settings.n, 1:estim_settings.n]);
            pen_loglik_new = status.loglik - envar_penalty(estim_settings, Σ, Ψ, Φ);
            verb_message(estim_settings.verb, "ecm > iter=$(iter-estim_settings.prerun), penalised loglik=$(round(pen_loglik_new, digits=5))");

            # Stop when the ECM algorithm converges
            if iter > estim_settings.prerun+1
                if isconverged(pen_loglik_new, pen_loglik_old, estim_settings.tol, estim_settings.ε, true)
                    verb_message(estim_settings.verb, "ecm > converged!\n");
                    break;
                end
            end

            # Store current run information
            pen_loglik_old = copy(pen_loglik_new);

        else
            verb_message(estim_settings.verb, "ecm > prerun $iter (out of $(estim_settings.prerun))");
        end

        E, F, G, X0, P0 = ksmoother_ecm(estim_settings, kalman_settings, status);

        # VAR(p) coefficients
        Φ = @. 1 / (abs(Ψ) + estim_settings.ε);
        for i=1:estim_settings.n
            Φ_i = @view Φ[i, :];
            F_i = @view F[i,:];
            XX_i = Symmetric(G + estim_settings.Γ.*((1-estim_settings.α)*I + estim_settings.α.*Diagonal(Φ_i)))::SymMatrix;
            kalman_settings.C[i, 1:estim_settings.np] = inv(XX_i)*F_i;
        end

        # Covariance matrix of the VAR(p) residuals
        parent(kalman_settings.V)[1:estim_settings.n, 1:estim_settings.n] =
            Symmetric(E-F*Ψ'-Ψ*F'+Ψ*G*Ψ' + (1-estim_settings.α).*(Ψ*estim_settings.Γ*Ψ') + estim_settings.α.*((Ψ.*sqrt.(Φ))*estim_settings.Γ*(Ψ.*sqrt.(Φ))'))::SymMatrix;

        parent(kalman_settings.V)[1:estim_settings.n, 1:estim_settings.n] *= 1/estim_settings.T;

        # New initial conditions
        kalman_settings.X0 = copy(X0);
        kalman_settings.P0 = copy(P0);
    end

    # Return output
    out_kalman_settings = ImmutableKalmanSettings(estim_settings.Y,
                                                  kalman_settings.B, kalman_settings.R,
                                                  kalman_settings.C, kalman_settings.V,
                                                  kalman_settings.X0, kalman_settings.P0);
    return out_kalman_settings;
end
