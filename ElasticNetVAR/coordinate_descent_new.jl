"""
    coordinate_descent(estim_settings::EstimSettings)

Estimate an elastic-net VAR(p) with the coordinate descent algorithm (Friedman et al., 2010) as in Pellegrino (2019).

# Arguments
- `estim_settings`: settings used for the estimation

# References
Friedman et al. (2010) and Pellegrino (2019)
"""
function coordinate_descent(estim_settings::EstimSettings)

    # Check inputs
    check_bounds(estim_settings.p, 1);
    check_bounds(estim_settings.λ, 0);
    check_bounds(estim_settings.α, 0, 1);
    check_bounds(estim_settings.β, 1);
    check_bounds(estim_settings.max_iter, 3);
    check_bounds(estim_settings.max_iter, estim_settings.prerun);
    check_bounds(estim_settings.n, 2); # It supports only multivariate models (for now ...)

    # Interpolate data
    data = interpolate(estim_settings.Y, estim_settings.n, estim_settings.T);

    # VAR(p) data
    Y, Y_lagged = lag(data, estim_settings.p);
    T_minus_p = estim_settings.T - estim_settings.p;

    # Memory pre-allocation
    Ψ = zeros(estim_settings.n, estim_settings.np);
    objfun_old = -Inf;
    objfun_new = -Inf;

    # Run coordinate_descent, in turn, for each target variable
    for i=1:estim_settings.n

        # Views
        Y_i = @view Y[i,:];
        Ψ_i = @view Ψ[i,:];

        # Loop until max_iter is reached
        for iter=1:estim_settings.max_iter

            # Reset objfun_new
            objfun_new = 0.0;

            # Loop over the predictors
            for j=1:estim_settings.np
                objfun_new += ijth_coordinate_update!(i, j, T_minus_p, estim_settings, Y_i, Y_lagged, Ψ);
            end

            # Update objfun_new
            objfun_new += (1/(2*T_minus_p))*sum((Y_i - Y_lagged'*Ψ_i).^2);
            verb_message(estim_settings.verb, "coordinate_descent ($i-th row) > iter=$(iter), objfun=$(round(objfun_new, digits=5))");

            # Stop when the algorithm converges
            if iter > 1
                if isconverged(objfun_new, objfun_old, estim_settings.tol, estim_settings.ε, false)
                    verb_message(estim_settings.verb, "coordinate_descent ($i-th row) > converged!");
                    break;
                end
            end

            # Store current run information
            objfun_old = copy(objfun_new);
        end
    end

    # Print empty line
    verb_message(estim_settings.verb, "");

    # Estimate var-cov matrix of the residuals
    V = Y - Ψ*Y_lagged;
    Σ = Symmetric((V*V')./T_minus_p)::SymMatrix;

    # Return output
    return Ψ, Σ;
end

"""
    ijth_coordinate_update!(i::Int64, j::Int64, T_minus_p::Int64, estim_settings::EstimSettings, Y_i::SubArray{Float64}, Y_lagged::FloatArray, Ψ::FloatArray)

Update the (i,j)-th element of Ψ the coordinate descent algorithm (Friedman et al., 2010) as in Pellegrino (2019).

# References
Friedman et al. (2010) and Pellegrino (2019)
"""
function ijth_coordinate_update!(i::Int64, j::Int64, T_minus_p::Int64, estim_settings::EstimSettings, Y_i::SubArray{Float64}, Y_lagged::FloatArray, Ψ::FloatArray)

    # Views
    Y_lagged_j = @view Y_lagged[j,:];

    # Partial residuals
    Ψ_i_ex_j = Ψ[i,:]; # Julia creates a copy of Ψ[i,:]
    Ψ_i_ex_j[j] = 0.0;
    V_ij = Y_i - Y_lagged'*Ψ_i_ex_j;

    # Scalar product between Y_lagged_j and V_ij divided by (T-p)
    scalar_Y_lagged_j = sum(V_ij .* Y_lagged_j)/T_minus_p;

    # Norm squared of Y_lagged_j divided by (T-p)
    normsq_Y_lagged_j = sum(Y_lagged_j .^ 2)/T_minus_p;

    # Soft-thresholding operator: update for the coefficients
    hyper_prod = estim_settings.λ*estim_settings.β^fld(j-1, estim_settings.n);
    Ψ[i, j] = soft_thresholding(scalar_Y_lagged_j, estim_settings.α*hyper_prod)/(normsq_Y_lagged_j + (1-estim_settings.α)*hyper_prod);

    # Return update for objfun_new
    update_objfun_new = hyper_prod*(0.5*(1-estim_settings.α)*Ψ[i,j]^2 + estim_settings.α*abs(Ψ[i,j]));
    return update_objfun_new;
end
