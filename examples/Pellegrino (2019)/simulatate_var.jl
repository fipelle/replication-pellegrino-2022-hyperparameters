# Libraries
include("./ElasticNetVAR/ElasticNetVAR.jl");
using Main.ElasticNetVAR;
using Random;
using Distributions;
using LinearAlgebra;

function simulate_stationary_var(T::Int64, n::Int64, p::Int64, sparsity::Float64; burnin::Int64=100)
    if sparsity < 0 || sparsity > 1
        error("0 ≤ sparsity ≤ 1");
    end

    # Variance-covariance matrix of the residuals
    Σ = Matrix(I, n, n) |> Array{Float64,2};

    # VAR(p) coefficients
    Ψ = randn(n, n*p);

    # Draw position for the zeros
    ind_zeros = collect(1:(n^2)*p) |> Array{Int64,1};
    rand_without_replacement!(ind_zeros, Int64(floor((1-sparsity)*(n^2)*p)));

    # Set sparse structure
    Ψ[ind_zeros] .= 0.0;

    # Calculate eigenvalues of C
    C = [];
    V = [];
    if n*p > n
        C, V = companion_form(Ψ, Σ);
    else
        C = copy(Ψ);
        V = copy(Σ);
    end

    max_eigvals_C = maximum(abs.(eigvals(C)));
    while max_eigvals_C >= 0.999
        C[1:n, :] *= 1/maximum(abs.(C));
        max_eigvals_C = maximum(abs.(eigvals(C)));
    end

    # Update Ψ
    Ψ = C[1:n, :];

    # Initialise simulated data
    MNShocks = MvNormal(zeros(n), Σ);
    Y = zeros(n, T+burnin+p);
    Y[:,1:p] = rand(MNShocks, p);

    # Run simulation
    for t=1+p:T+burnin+p
        for i=1:n
            for j=1:n
                Y[i, t] += sum(Y[j, t-p:t-1].*reverse(Ψ[i, j:n:end]));
            end
        end

        Y[:,t] += rand(MNShocks);
    end

    # Standardise data
    # TBA

    # Return output
    return Y[:, p+burnin+1:end], Ψ;
end
