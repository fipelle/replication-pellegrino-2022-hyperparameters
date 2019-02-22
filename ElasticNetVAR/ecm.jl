"""
"""
function ecm(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb=true)

    #=
    --------------------------------------------------------------------------------------------------------------------------------
    Settings
    --------------------------------------------------------------------------------------------------------------------------------
    =#

    # Check hyper-parameters
    if β < 1
        error("β ≥ 1");
    end

    if α < 0 || α > 1
        error("0 ≤ α ≤ 1");
    end

    if λ < 0
        error("λ ≥ 0");
    end

    # Check init_iter
    if max_iter < 3
        error("max_iter > 2");
    end

    if prerun >= max_iter
        error("prerun < max_iter");
    end

    # Dimensions
    n, T = size(Y);
    np = n*p;

    if n < 2
        error("This code is not compatible with univariate autoregressions");
    end

    # Gamma matrix
    Γ = [];
    for i=0:p-1
        if i == 0
            Γ = Matrix{Float64}(I, n, n);
        else
            Γ = cat(Γ, (β^i).*Matrix{Float64}(I, n, n), dims=[1,2]);
        end
    end
    Γ = (λ/np).*Γ;


    #=
    --------------------------------------------------------------------------------------------------------------------------------
    ECM initialisation
    --------------------------------------------------------------------------------------------------------------------------------
    =#

    # Interpolated data (used for the initialisation only)
    Y_init = copy(Y);
    for i=1:n
        Y_init[i, ismissing.(Y_init[i, :])] .= mean_skipmissing(Y_init[i, :]);
    end

    # VAR(p) data
    Y_init, X_init = lag(Y_init, p);

    # Estimate ridge VAR(p)
    Ψ̂_init = Y_init*X_init'/(X_init*X_init' + Γ);
    V̂_init = Y_init - Ψ̂_init*X_init;
    Σ̂_init = (V̂_init*V̂_init')./(T-p);

    # State-space parameters
    B̂ = [Matrix{Float64}(I, n, n) zeros(n, np-n)];
    R̂ = Matrix{Float64}(I, n, n).*eps();
    Ĉ, V̂ = companion_form_VAR(Ψ̂_init, Σ̂_init);

    # Initial conditions
    𝔛0̂ = zeros(np);
    P0̂ = reshape((Matrix(I, np^2, np^2)-kron(Ĉ, Ĉ))\V̂[:], np, np);

    # Initialise additional variables
    Φ̂ᵏ = 1 ./ (abs.(Ĉ[1:n, :]).+eps());


    #=
    --------------------------------------------------------------------------------------------------------------------------------
    ECM algorithm
    --------------------------------------------------------------------------------------------------------------------------------
    =#

    # ECM controls
    pen_loglik_old = -Inf;
    pen_loglik_new = -Inf;

    # Run ECM
    for iter=1:max_iter

        # Run Kalman filter and smoother
        𝔛ŝ, Pŝ, PPŝ, 𝔛s_0̂, Ps_0̂, _, _, _, loglik = kalman(Y, B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂; loglik_flag=true);

        if iter > prerun

            # New penalised loglikelihood
            pen_loglik_new = loglik - 0.5*tr(V̂[1:n, 1:n]\((1-α).*Ĉ[1:n, :]*Γ*Ĉ[1:n, :]' + α.*(Ĉ[1:n, :].*Φ̂ᵏ)*Γ*Ĉ[1:n, :]'));

            if verb == true
                println("ecm > iter=$(iter-prerun), penalised loglik=$(round(pen_loglik_new, digits=5))");
            end

            # Stop when the ECM algorithm converges
            if iter > prerun+1
                if (pen_loglik_new-pen_loglik_old)./(abs(pen_loglik_old)+eps()) <= tol
                    if verb == true
                        println("ecm > converged!");
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
                F̂ += 𝔛ŝ[1:n,t]*𝔛0̂' + PPŝ[1:n,:,t];
                Ĝ += 𝔛0̂*𝔛0̂' + P0̂;

            else
                F̂ += 𝔛ŝ[1:n,t]*𝔛ŝ[:,t-1]' + PPŝ[1:n,:,t];
                Ĝ += 𝔛ŝ[:,t-1]*𝔛ŝ[:,t-1]' + Pŝ[:,:,t-1];
            end
        end

        # VAR(p) coefficients
        Φ̂ᵏ = 1 ./ (abs.(Ĉ[1:n, :]).+eps());
        for i=1:n
            Ĉ[i,:] = (Ĝ + Γ.*((1-α).*Matrix(I, np, np) + α.*Φ̂ᵏ[i,:]*ones(1, np)))\F̂[i,:];
        end

        # Covariance matrix of the VAR(p) residuals
        V̂[1:n, 1:n] = (1/T).*(Ê-F̂*Ĉ[1:n,:]'-Ĉ[1:n,:]*F̂'+Ĉ[1:n,:]*Ĝ*Ĉ[1:n,:]' + Ĉ[1:n,:]*Γ*((1-α).*Ĉ[1:n,:] + α.*Ĉ[1:n,:].*Φ̂ᵏ)');

        # Remove possible source of numerical instabilities in V̂
        V̂[1:n, 1:n] *= 0.5;
        V̂[1:n, 1:n] += V̂[1:n, 1:n]';
    end

    # Replace very small numbers with zeros
    Ĉ[abs.(Ĉ) .< eps()] .= 0.0;
    V̂[abs.(V̂) .< eps()] .= 0.0;

    # Return output
    return B̂, R̂, Ĉ, V̂, 𝔛0̂, P0̂, Ψ̂_init, Σ̂_init;
end
