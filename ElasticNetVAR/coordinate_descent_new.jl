"""
    interpolate(Y::JArray{Float64}, n::Int64, T::Int64)

Interpolate each series in `Y`, in turn, by replacing missing observations with the sample average of the non-missing values.

# Arguments
- `Y`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations
"""
function interpolate(Y::JArray{Float64}, n::Int64, T::Int64)
    data = zeros(n, T);
    for i=1:n
        data[i, ismissing.(Y[i, :])] .= mean_skipmissing(Y[i, :]);
    end
    return data;
end

"""
    verb_message(verb::Bool, message::String)

Println `message` if `verb` is true.
"""
verb_message(verb::Bool, message::String) = verb ? println(message) : nothing;

"""
    soft_thresholding(z::Float64, ζ::Float64)

Soft thresholding operator.
"""
soft_thresholding(z::Float64, ζ::Float64) = sign(z)*max(abs(z)-ζ);

"""
    isconverged(new::Float64, old::Float64, tol::Float64, ε::Float64, increasing::Bool)

Check whether `new` is close enough to `old`.

# Arguments
- `new`: new objective or loss
- `old`: old objective or loss
- `tol`: tolerance
- `ε`: small Float64
- `increasing`: true if `new` increases, at each iteration, with respect to `old`
"""
isconverged(new::Float64, old::Float64, tol::Float64, ε::Float64, increasing::Bool) = increasing ? (new-old)./(abs(old)+ε) <= tol : -(new-old)./(abs(old)+ε) <= tol;

"""
    ijth_coordinate_update!(i::Int64, j::Int64, objfun_new::Float64, estim_settings::EstimSettings, Y_i::SubArray{Float64}, Y_lagged::Array{Float64,2}, Ψ̂::Array{Float64,2})

Update the (i,j)-th element of Ψ̂ the coordinate descent algorithm (Friedman et al., 2010) as in Pellegrino (2019).

# References
Friedman et al. (2010) and Pellegrino (2019)
"""
function ijth_coordinate_update!(i::Int64, j::Int64, objfun_new::Float64, estim_settings::EstimSettings, Y_i::SubArray{Float64}, Y_lagged::Array{Float64,2}, Ψ̂::Array{Float64,2})

    # Views
    Y_lagged_j = @view Y_lagged[j,:];

    # Partial residuals
    Ψ̂_i_ex_j = Ψ̂[i,:]; # Julia creates a copy of Ψ̂[i,:]
    Ψ̂_i_ex_j[j] = 0.0;
    V̂_ij = Y_i - Y_lagged'*Ψ̂ᵢ_ex_j;

    # Scalar product between Y_lagged_j and V̂_ij divided by (T-p)
    scalar_Y_lagged_j = sum(V̂_ij .* Y_lagged_j)/T_minus_p;

    # Norm squared of Y_lagged_j divided by (T-p)
    normsq_Y_lagged_j = sum(Y_lagged_j .^ 2)/T_minus_p;

    # Soft-thresholding operator: update for the coefficients
    hyper_prod = estim_settings.λ*estim_settings.β^fld(j-1, estim_settings.n);
    Ψ̂[i, j] = soft_thresholding(scalar_Y_lagged_j, estim_settings.α*hyper_prod)/(normsq_Y_lagged_j + (1-estim_settings.α)*hyper_prod);

    # Update objfun_new
    objfun_new += hyper_prod*(0.5*(1-estim_settings.α)*Ψ̂[i,j]^2 + estim_settings.α*abs(Ψ̂[i,j]));
end

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
    check_bounds(estim_settings.prerun, estim_settings.max_iter);
    check_bounds(estim_settings.n, 2); # It supports only multivariate models (for now ...)

    # Interpolate data
    data = interpolate(estim_settings.Y, estim_settings.n, estim_settings.T);

    # VAR(p) data
    Y, Y_lagged = lag(data, estim_settings.p);
    T_minus_p = estim_settings.T - estim_settings.p;

    # Memory pre-allocation
    Ψ̂ = zeros(estim_settings.n, estim_settings.np);
    objfun_old = -Inf;
    objfun_new = -Inf;

    # Run coordinate_descent, in turn, for each target variable
    for i=1:n

        # Views
        Y_i = @view Y[i,:];
        Ψ̂_i = @view Ψ̂[i,:];

        # Loop until max_iter is reached
        for iter=1:max_iter

            # Reset objfun_new
            objfun_new = 0.0;

            # Loop over the predictors
            for j=1:np
                ijth_coordinate_update!(i, j, objfun_new, estim_settings, Y_i, Y_lagged, Ψ̂);
            end

            # Update objfun_new
            objfun_new += (1/(2*T_minus_p))*sum((Y_i - X'*Ψ̂_i).^2);
            verb_message(verb, "coordinate_descent ($i-th row) > iter=$(iter), objfun=$(round(objfun_new, digits=5))");

            # Stop when the algorithm converges
            if iter > 1
                if isconverged(objfun_new, objfun_old, estim_settings.tol, estim_settings.ε, false)
                    verb_message(verb, "coordinate_descent ($i-th row) > converged!");
                    break;
                end
            end

            # Store current run information
            objfun_old = copy(objfun_new);
        end
    end

    # Print empty line
    verb_message(verb, "");

    # Estimate var-cov matrix of the residuals
    V̂ = Y - Ψ̂*Y_lagged;
    Σ̂ = Symmetric((V̂*V̂')./T_minus_p)::SymMatrix;

    # Return output
    return Ψ̂, Σ̂;
end
