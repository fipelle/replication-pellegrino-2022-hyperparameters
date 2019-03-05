"""
    kalman(Y::JArray{Float64}, B::FloatArray, R::FloatArray, C::FloatArray, V::FloatArray, 𝔛0::FloatVector, P0::FloatArray; loglik_flag::Bool=false, kf_only_flag::Bool=false, lag1_cov_flag::Bool=false)

Perform the Kalman filter and smoother recursions as in Shumway and Stoffer (2011, chapter 6).

# Model
The state space model used below is,

``Y_{t} = B*𝔛_{t} + e_{t}``

``𝔛_{t} = C*𝔛_{t-1} + u_{t}``

Where ``e_{t} ~ N(0, R)`` and ``u_{t} ~ N(0, V)``.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `B`: Measurement equations' coefficients
- `R`: Covariance matrix of the measurement equations' error terms
- `C`: Transition equations' coefficients
- `V`: Covariance matrix of the transition equations' error terms
- `𝔛0`: Mean vector for the states at time t=0
- `P0`: Covariance matrix for the states at time t=0
- `loglik_flag`: True to estimate the loglikelihood (default: false)
- `kf_only_flag`: True to run the Kalman filter only (default: false)
- `lag1_cov_flag`: True to estimate the lag-one covariance smoother (default: false)

# References
Shumway and Stoffer (2011, chapter 6).
"""
function kalman(Y::JArray{Float64}, B::FloatArray, R::FloatArray, C::FloatArray, V::FloatArray, 𝔛0::FloatVector, P0::FloatArray; loglik_flag::Bool=false, kf_only_flag::Bool=false, lag1_cov_flag::Bool=false)

    #=
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Initialisation
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    # Dimensions
    n, T = size(Y);
    m = size(C,1);

    #=
    A priori filtered estimates,
    𝔛p_{t} ≡ 𝔛_{t|t-1} ≝ E[𝔛_{t} | Y_{t-1}; θ]
    Pp_{t} ≡ P_{t|t-1} ≝ E[(𝔛_{t} - 𝔛p_{t})(𝔛_{t} - 𝔛p_{t})'| Y_{t-1}; θ]

    A posteriori filtered estimates,
    𝔛f_{t} ≡ 𝔛_{t|t} ≝ E[𝔛_{t} | Y_{t}; θ]
    Pf_{t} ≡ P_{t|t} ≝ E[(𝔛_{t} - 𝔛f_{t})(𝔛_{t} - 𝔛f_{t})'| Y_{t-1}; θ]

    Smoothed estimates,
    𝔛s_{t} ≡ 𝔛_{t|T} ≝ E[𝔛_{t} | Y_{T}; θ]
    Ps_{t} ≡ P_{t|T} ≝ E[(𝔛_{t} - 𝔛s_{t})(𝔛_{t} - 𝔛s_{t})' | Y_{T}; θ]
    PPs_{t} ≝ E[(𝔛_{t-1} - 𝔛s_{t-1})(𝔛_{t-2} - 𝔛s_{t-2})' | Y_{T}; θ]

    for t=1, ..., T and with,
    θ ≝ (vec(B)', vech(R)', vec(C)', vech(V)', vec(𝔛0)', vech(P0)')
    =#

    𝔛p = zeros(m, T);
    Pp = zeros(m, m, T);
    𝔛f = zeros(m, T);
    Pf = zeros(m, m, T);
    𝔛s = zeros(m, T);
    Ps = zeros(m, m, T);
    PPs = zeros(m, m, T);
    𝔛s_0 = zeros(m);
    Ps_0 = zeros(m, m);

    #=
    Log likelihood
    - This is not the conditional expectation of the likelihood in Shumway Stoffer (2011, pp. 340)
    =#
    loglik = 0.0;


    #=
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Kalman filter
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    # Loop over t=1,...,T
    for t=1:T

        # A priori estimates
        if t==1
            𝔛p[:,t] = C*𝔛0;
            Pp[:,:,t] = sym(C*P0*C') + V;
        else
            𝔛p[:,t] = C*𝔛f[:,t-1];
            Pp[:,:,t] = sym(C*Pf[:,:,t-1]*C') + V;
        end

        # Handle missing observations following the "zeroing" approach in Shumway and Stoffer (2011, pp. 345, eq. 6.79)
        Y_t = copy(Y[:,t]);
        B_t = copy(B);
        R_t = copy(R);
        missings_t = findall(ismissing.(Y_t));
        if length(missings_t) > 0
            Y_t[missings_t] .= 0.0;
            B_t[missings_t, :] .= 0.0;
            R_t[missings_t, missings_t] = Matrix(I, length(missings_t), length(missings_t));
        end

        # Forecast error
        ε_t = Y_t - B_t*𝔛p[:,t];
        Σ_t = sym(B_t*Pp[:,:,t]*B_t') + R_t;

        # Kalman gain
        K_t = Pp[:,:,t]*B_t'*sym_inv(Σ_t);

        # A posteriori estimates
        𝔛f[:,t] = 𝔛p[:,t] + K_t*ε_t;
        Pf[:,:,t] = Pp[:,:,t] - sym(K_t*B_t*Pp[:,:,t]);

        # Initialise lag-one covariance as in Shumway and Stoffer (2011, pp. 334)
        if t == T && lag1_cov_flag == true
            PPs[:,:,t] = C*Pf[:,:,t-1] - K_t*B_t*C*Pf[:,:,t-1];
        end

        # Log likelihood
        if loglik_flag == true
            try
                loglik -= 0.5*(logdet(Σ_t) + ε_t'*sym_inv(Σ_t)*ε_t);
            catch
                error("Determinant: $(det(Σ_t))");
            end
        end
    end


    #=
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Kalman smoother, as in Shumway and Stoffer (2011, pp. 330)
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    =#

    if kf_only_flag == false

        # At t=T the smoothed estimates are identical to the filtered (a posteriori)
        𝔛s[:,T] = copy(𝔛f[:,T]);
        Ps[:,:,T] = copy(Pf[:,:,T]);

        # Loop over t=T,...,1
        for t=T:-1:1

            if t > 1
                # J_{t-1}
                J1 = Pf[:,:,t-1]*C'/Pp[:,:,t];

                # Smoothed estimates for t-1
                𝔛s[:,t-1] = 𝔛f[:,t-1] + J1*(𝔛s[:,t]-𝔛p[:,t]);
                Ps[:,:,t-1] = Pf[:,:,t-1] + J1*(Ps[:,:,t]-Pp[:,:,t])*J1';

            else
                # J_{t-1}
                J1 = P0*C'/Pp[:,:,t];

                # Smoothed estimates for t-1
                𝔛s_0 = 𝔛0 + J1*(𝔛s[:,t]-𝔛p[:,t]);
                Ps_0 = P0 + J1*(Ps[:,:,t]-Pp[:,:,t])*J1';
            end

            # Lag-one covariance smoother as in Shumway and Stoffer (2011, pp. 334)
            if t >= 2 && lag1_cov_flag == true

                # J_{t-2}
                if t > 2
                    J2 = Pf[:,:,t-2]*C'/Pp[:,:,t-1];
                else
                    J2 = P0*C'/Pp[:,:,t-1];
                end

                # Lag-one covariance
                PPs[:,:,t-1] = Pf[:,:,t-1]*J2' + J1*(PPs[:,:,t] - C*Pf[:,:,t-1])*J2';
            end
        end
    end

    # Return output
    return 𝔛s, Ps, PPs, 𝔛s_0, Ps_0, 𝔛f, 𝔛p, Pf, loglik;
end
