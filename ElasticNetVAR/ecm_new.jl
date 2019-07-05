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

        # Run Kalman filter
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
            pen_loglik_old = pen_loglik_new;

        else
            verb_message(estim_settings.verb, "ecm > prerun $iter (out of $(estim_settings.prerun))");
        end

        E, F, G, kalman_settings.X0, kalman_settings.P0 = ksmoother_ecm(estim_settings, kalman_settings, status);

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
            Symmetric(E-F*Ψ'-Ψ*F'+Ψ*G*Ψ' + (1-estim_settings.α).*(Ψ*estim_settings.Γ*Ψ') + estim_settings.α.*((Ψ.*sqrt.(Φ))*estim_settings.Γ*(Ψ.*sqrt.(Φ))'))::SymMatrix ./ estim_settings.T;
    end

    # Return output
    out_kalman_settings = ImmutableKalmanSettings(estim_settings.Y_output,
                                                  kalman_settings.B, kalman_settings.R,
                                                  kalman_settings.C, kalman_settings.V,
                                                  kalman_settings.X0, kalman_settings.P0);
    return out_kalman_settings;
end

"""
    envar_penalty(estim_settings::EstimSettings, Σ::AbstractArray{Float64}, Ψ::SubArray{Float64}, Φ::FloatArray)

Compute the value of the loglikelihood penalty.
"""
envar_penalty(estim_settings::EstimSettings, Σ::AbstractArray{Float64}, Ψ::SubArray{Float64}, Φ::FloatArray) = tr(inv(Σ)*(envar_penalty_ridge(estim_settings, Ψ) + envar_penalty_lasso(estim_settings, Ψ, Φ)))::Float64;

# Ridge and LASSO components
envar_penalty_ridge(estim_settings::EstimSettings, Ψ::SubArray{Float64}) = ((1-estim_settings.α)/2) .* Symmetric(Ψ*estim_settings.Γ*Ψ')::SymMatrix;
envar_penalty_lasso(estim_settings::EstimSettings, Ψ::SubArray{Float64}, Φ::FloatArray) = (estim_settings.α/2) .* Symmetric((Ψ .* sqrt.(Φ))*estim_settings.Γ*(Ψ .* sqrt.(Φ))')::SymMatrix;
