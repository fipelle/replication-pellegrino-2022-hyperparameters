"""
"""
check_bounds(X::Number, LB::Number, UB::Number) = X < LB || X > UB ? throw(DomainError) : nothing
check_bounds(X::Number, LB::Number) = X < LB ? throw(DomainError) : nothing




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
    check_bounds(estim_settings.prerun, estim_settings.max_iter);
    check_bounds(estim_settings.n, 2); # It supports only multivariate models (for now ...)


    #=
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    ECM initialisation
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    # Interpolated data (used for the initialisation only)
    Y_init = copy(Y);
    for i=1:n
        Y_init[i, ismissing.(Y_init[i, :])] .= mean_skipmissing(Y_init[i, :]);
    end
    Y_init = Y_init |> Array{Float64};

    # Initialise using the coordinate descent algorithm
    if verb == true
        println("ecm > initialisation");
    end
    Ψ̂_init, Σ̂_init = coordinate_descent(Y_init, p, λ, α, β, tol=tol, max_iter=max_iter, verb=false);


    #=
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    Memory pre-allocation
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    #=
    The state vector includes additional n terms with respect to the standard VAR companion form representation.
    This is to estimate the lag-one covariance smoother as in Watson and Engle (1983).
    =#

    # State-space parameters
    B̂ = [Matrix{Float64}(I, n, n) zeros(n, np)];
    R̂ = Matrix{Float64}(I, n, n).*ε;
    Ĉ, V̂ = ext_companion_form(Ψ̂_init, Σ̂_init);

    # Initial conditions
    𝔛0̂ = zeros(np+n);
    P0̂ = reshape((I-kron(Ĉ, Ĉ))\V̂[:], np+n, np+n);
    P0̂ = sym(P0̂);

    # Initialise additional variables
    Ψ̂ = Ĉ[1:n, 1:np];
    Σ̂ = V̂[1:n, 1:n];
    Φ̂ᵏ = 1 ./ (abs.(Ψ̂).+ε);

    # ECM controls
    pen_loglik_old = -Inf;
    pen_loglik_new = -Inf;


    #=
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    ECM algorithm
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    # Run ECM
    for iter=1:max_iter

        # Run Kalman filter and smoother
        𝔛ŝ, Pŝ, _, 𝔛s_0̂, Ps_0̂, _, _, _, loglik = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=true);

        if iter > prerun

            # New penalised loglikelihood
            pen_loglik_new = loglik - 0.5*tr(sym_inv(Σ̂)*((1-α).*sym(Ψ̂*Γ*Ψ̂') + α.*sym((Ψ̂.*sqrt.(Φ̂ᵏ))*Γ*(Ψ̂.*sqrt.(Φ̂ᵏ))')));
            if verb == true
                println("ecm > iter=$(iter-prerun), penalised loglik=$(round(pen_loglik_new, digits=5))");
            end

            # Stop when the ECM algorithm converges
            if iter > prerun+1
                if (pen_loglik_new-pen_loglik_old)./(abs(pen_loglik_old)+ε) <= tol
                    if verb == true
                        println("ecm > converged!");
                        println("");
                    end
                    break;
                end
            end

            # Store current run information
            pen_loglik_old = copy(pen_loglik_new);

        elseif verb == true
            println("ecm > prerun $iter (out of $prerun)");
        end

        # Initial conditions
        𝔛0̂ = copy(𝔛s_0̂);
        P0̂ = copy(Ps_0̂);

        # ECM statistics
        Ê = zeros(n, n);
        F̂ = zeros(n, np);
        Ĝ = zeros(np, np);

        for t=1:T
            Ê += 𝔛ŝ[1:n,t]*𝔛ŝ[1:n,t]' + Pŝ[1:n,1:n,t];

            if t == 1
                F̂ += 𝔛ŝ[1:n,t]*𝔛0̂[1:np]' + Pŝ[1:n,n+1:end,t];
                Ĝ += 𝔛0̂[1:np]*𝔛0̂[1:np]' + P0̂[1:np,1:np];

            else
                F̂ += 𝔛ŝ[1:n,t]*𝔛ŝ[1:np,t-1]' + Pŝ[1:n,n+1:end,t];
                Ĝ += 𝔛ŝ[1:np,t-1]*𝔛ŝ[1:np,t-1]' + Pŝ[1:np,1:np,t-1];
            end
        end

        # VAR(p) coefficients
        Φ̂ᵏ = 1 ./ (abs.(Ψ̂).+ε);
        for i=1:n
            Ĉ[i, 1:np] = sym_inv(Ĝ + Γ.*((1-α)*I + α.*Diagonal(Φ̂ᵏ[i, :])))*F̂[i,:];
        end

        # Update Ψ̂
        Ψ̂ = Ĉ[1:n, 1:np];

        # Covariance matrix of the VAR(p) residuals
        V̂[1:n, 1:n] = sym(Ê-F̂*Ψ̂'-Ψ̂*F̂'+Ψ̂*Ĝ*Ψ̂') + (1-α).*sym(Ψ̂*Γ*Ψ̂') + α.*sym((Ψ̂.*sqrt.(Φ̂ᵏ))*Γ*(Ψ̂.*sqrt.(Φ̂ᵏ))');
        V̂[1:n, 1:n] *= 1/T;

        # Update Σ̂
        Σ̂ = V̂[1:n, 1:n];
    end

    # The output excludes the additional n terms required to estimate the lag-one covariance smoother as described above.
    return B̂[:,1:np], R̂, Ĉ[1:np,1:np], V̂[1:np,1:np], 𝔛0̂[1:np], P0̂[1:np,1:np], Ψ̂_init, Σ̂_init;
end
