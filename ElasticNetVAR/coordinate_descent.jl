"""
"""
function coordinate_descent(Y::Array{Float64,2}, X::Array{Float64,2}, λ::Number, α::Number, β::Number; tol::Float64=1e-4, max_iter::Int64=1000, verb=true)

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
    if max_iter < 0
        error("max_iter > 0");
    end

    # Dimensions
    n, T_minus_p = size(Y);
    np = size(X,1);

    # ε
    ε = 1e-8;


    #=
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    Execution
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    # Memory pre-allocation
    Ψ̂ = zeros(n, np);
    Σ̂ = zeros(n, n);
    objfun_old = -Inf;
    objfun_new = -Inf;

    # Run coordinate_descent(⋅), in turn, for each target variable
    for i=1:n

        # Loop until max_iter is reached
        for iter=1:max_iter

            # Reset objfun_new
            objfun_new = 0.0;

            # Loop over the predictors
            for j=1:np

                # Partial residuals
                Ψ̂ᵢ_ex_j = copy(Ψ̂[i,:]);
                Ψ̂ᵢ_ex_j[j] = 0.0;
                V̂ᵢⱼ = Y[i,:] - X'*Ψ̂ᵢ_ex_j;

                # Scalar product between Xⱼ and V̂ᵢⱼ divided by (T-p)
                XV̂ᵢⱼ = sum(V̂ᵢⱼ .* X[j,:])/T_minus_p;

                # Soft-thresholding operator: update for the coefficients
                hyper_prod = (λ/np)*β^floor((j-1)/n);
                thresh = hyper_prod*α;

                if XV̂ᵢⱼ > thresh
                    Ψ̂[i, j] = (XV̂ᵢⱼ-thresh)/(1 + hyper_prod*(1-α));
                elseif XV̂ᵢⱼ < -thresh
                    Ψ̂[i, j] = (XV̂ᵢⱼ+thresh)/(1 + hyper_prod*(1-α));
                else
                    Ψ̂[i, j] = 0.0;
                end

                # Update objfun_new
                objfun_new += hyper_prod*(0.5*(1-α)*Ψ̂[i,j]^2 + α*abs(Ψ̂[i,j]));
            end

            # Update objfun_new
            objfun_new += (1/(2*T_minus_p))*sum((Y[i,:] - X'*Ψ̂[i,:]).^2);

            if verb == true
                println("coordinate_descent ($i-th row) > iter=$(iter), objfun=$(round(objfun_new, digits=5))");
            end

            # Stop when the algorithm converges
            if iter > 1
                if (objfun_new-objfun_old)./(abs(objfun_old)+ε) <= tol
                    if verb == true
                        println("coordinate_descent ($i-th row) > converged!");
                    end
                    break;
                end
            end

            # Store current run information
            objfun_old = copy(objfun_new);
        end
    end

    # Print empty line
    if verb == true
        println("");
    end

    # Estimate var-cov matrix of the residuals
    V̂ = Y - Ψ̂*X;
    Σ̂ = (V̂*V̂')./T_minus_p;

    # Return output
    return Ψ̂, Σ̂;
end
