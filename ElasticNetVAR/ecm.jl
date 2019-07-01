"""
    ecm(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb=true)

Estimate an elastic-net VAR(p) using the ECM algorithm in Pellegrino (2019).

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p`: number of lags in the vector autoregression
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `tol`: tolerance used to check convergence (default: 1e-3)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual ECM estimation routine (default: 2)
- `verb`: Verbose output (default: true)

# References
Pellegrino (2019)
"""
function ecm(Y::JArray{Float64,2}, p::Int64, λ::Number, α::Number, β::Number; tol::Float64=1e-3, max_iter::Int64=1000, prerun::Int64=2, verb=true)

    #=
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    Settings
    -----------------------------------------------------------------------------------------------------------------------------------------------------
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

    # ε
    ε = 1e-8;

    # Gamma matrix
    Γ = [];
    for i=0:p-1
        if i == 0
            Γ = ones(n);
        else
            Γ = vcat(Γ, (β^i).*ones(n));
        end
    end
    Γ = Diagonal(λ.*Γ);


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

    save("./init_old.jld", Dict("Ψ_init" => Ψ̂_init, "Σ_init" => Σ̂_init));

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

        #=
        save("./first_round_old.jld", Dict("Φ" => Φ̂ᵏ, "C" => Ĉ, "V" => V̂, "E" => Ê, "F" => F̂, "G" => Ĝ));

        if iter == 2
            save("./second_round_old.jld", Dict("Φ" => Φ̂ᵏ, "C" => Ĉ, "V" => V̂, "E" => Ê, "F" => F̂, "G" => Ĝ));
            error("");
        end
        =#
    end

    # The output excludes the additional n terms required to estimate the lag-one covariance smoother as described above.
    return B̂[:,1:np], R̂, Ĉ[1:np,1:np], V̂[1:np,1:np], 𝔛0̂[1:np], P0̂[1:np,1:np], Ψ̂_init, Σ̂_init;
end
