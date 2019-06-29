"""
"""
envar_penalty(estim_settings::EstimSettings, Σ::SymMatrix, Ψ::FloatArray, Φ::FloatArray) = tr(inv(Σ)*(envar_penalty_ridge(estim_settings, Ψ) + envar_penalty_lasso(estim_settings, Ψ, Φ)))::Float64;
envar_penalty_ridge(estim_settings::EstimSettings, Ψ::FloatArray) = (1-estim_settings.α)/2.*Symmetric(Ψ*estim_settings.Γ*Ψ')::SymMatrix;
envar_penalty_lasso(estim_settings::EstimSettings, Ψ::FloatArray, Φ::FloatArray) = estim_settings.α/2.*Symmetric((Ψ .* sqrt.(Φ))*estim_settings.Γ*(Ψ .* sqrt.(Φ))')::SymMatrix;

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
        for t=1:settings.T
            kfilter!(settings, status);
        end
        Xs, Ps, X0s, P0s = ksmoother(settings, status); # TBC write new function to get the ecm statistics without saving the smoother history

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

        # ECM statistics (TBC - use the new ecm smoother function)
        E = zeros(n, n);
        F = zeros(n, np);
        G = zeros(np, np);

        for t=1:T
            E += Xs[1:n,t]*Xs[1:n,t]' + Ps[1:n,1:n,t];

            if t == 1
                F += Xs[1:n,t]*X0s[1:np]' + Ps[1:n,n+1:end,t];
                G += X0s[1:np]*X0s[1:np]' + P0s[1:np,1:np];

            else
                F += Xs[1:n,t]*Xs[1:np,t-1]' + Ps[1:n,n+1:end,t];
                G += Xs[1:np,t-1]*Xs[1:np,t-1]' + Ps[1:np,1:np,t-1];
            end
        end

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
