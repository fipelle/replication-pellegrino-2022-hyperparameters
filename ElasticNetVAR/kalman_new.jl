"""
    kfilter!(settings::KalmanSettings, status::KalmanStatus)

Kalman filter: a-priori prediction and a-posteriori update.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + u_{t}``

Where ``e_{t} ~ N(0, R)`` and ``u_{t} ~ N(0, V)``.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function kfilter!(settings::KalmanSettings, status::KalmanStatus)

    # A-priori prediction
    kpredict!(typeof(status.X_prior), settings, status);

    # Handle missing observations
    ind_not_missings = find_observed_data(settings, status);

    # Ex-post update
    kupdate!(settings, status, ind_not_missings);

    # Update history of *_prior and *_post
    if settings.store_history == true
        push!(status.history_X_prior, status.X_prior);
        push!(status.history_X_post, status.X_post);
        push!(status.history_P_prior, status.P_prior);
        push!(status.history_P_post, status.P_post);
    end

    # Update status.t
    status.t += 1;
end

"""
    kpredict!(::Type{Nothing}, settings::KalmanSettings, status::KalmanStatus)

Kalman filter a-priori prediction for t==1.

# Arguments
- `::Type{Nothing}`: first prediction
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct

    kpredict!(::Type{FloatVector}, settings::KalmanSettings, status::KalmanStatus)

Kalman filter a-priori prediction.

# Arguments
- `::Type{FloatVector}`: standard prediction
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function kpredict!(::Type{Nothing}, settings::KalmanSettings, status::KalmanStatus)

    status.X_prior = kpredict(settings.X0, settings);
    status.P_prior = kpredict(settings.P0, settings);

    if settings.compute_loglik == true
        status.loglik = 0.0;
    end

    if settings.store_history == true
        status.history_X_prior = Array{FloatVector,1}();
        status.history_X_post = Array{FloatVector,1}();
        status.history_P_prior = Array{SymMatrix,1}();
        status.history_P_post = Array{SymMatrix,1}();
    end
end

function kpredict!(::Type{FloatVector}, settings::KalmanSettings, status::KalmanStatus)
    status.X_prior = kpredict(status.X_post, settings);
    status.P_prior = kpredict(status.P_post, settings);
end

"""
    kpredict(X::FloatVector, settings::KalmanSettings)

Kalman filter a-priori prediction for X.

# Arguments
- `X`: Last expected value of the states
- `settings`: KalmanSettings struct

    kpredict(X::SymMatrix, settings::KalmanSettings)

Kalman filter a-priori prediction for P.

# Arguments
- `P`: Last conditional covariance the states
- `settings`: KalmanSettings struct
"""
kpredict(X::FloatVector, settings::KalmanSettings) = settings.C * X;
kpredict(P::SymMatrix, settings::KalmanSettings) = Symmetric(settings.C * P * settings.C' + settings.V)::SymMatrix;

"""
    find_observed_data(settings::KalmanSettings, status::KalmanStatus)

Return position of the observed measurements at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function find_observed_data(settings::KalmanSettings, status::KalmanStatus)
    if status.t <= settings.T
        Y_t_all = @view settings.Y[:, status.t];
        ind_not_missings = findall(ismissing.(Y_t_all) .== false);
        if length(ind_not_missings) > 0
            return ind_not_missings;
        end
    end
end

"""
    kupdate!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Array{Int64,1})

Kalman filter a-posteriori update. Measurements are observed (or partially observed) at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Position of the observed measurements

    kupdate!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)

Kalman filter a-posteriori update. All measurements are not observed at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Empty array
"""
function kupdate!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Array{Int64,1})

    Y_t = @view settings.Y[ind_not_missings, status.t];
    B_t = @view settings.B[ind_not_missings, :];
    R_t = @view settings.R[ind_not_missings, ind_not_missings];

    # Forecast error
    ε_t = Y_t - B_t*status.X_prior;
    Σ_t = Symmetric(B_t*status.P_prior*B_t' + R_t)::SymMatrix;

    # Kalman gain
    K_t = status.P_prior*B_t'*inv(Σ_t);

    # A posteriori estimates
    status.X_post = status.X_prior + K_t*ε_t;
    status.P_post = Symmetric(status.P_prior - K_t*B_t*status.P_prior)::SymMatrix;

    # Update log likelihood
    if settings.compute_loglik == true
        update_loglik!(status, ε_t, Σ_t);
    end
end

function kupdate!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)
    status.X_post = copy(status.X_prior);
    status.P_post = copy(status.P_prior);
end

"""
    update_loglik!(status::KalmanStatus, ε_t::FloatVector, Σ_t::SymMatrix)

Update status.loglik.

# Arguments
- `status`: KalmanStatus struct
- `ε_t`: Forecast error
- `Σ_t`: Forecast error covariance
"""
function update_loglik!(status::KalmanStatus, ε_t::FloatVector, Σ_t::SymMatrix)
    status.loglik -= 0.5*(logdet(Σ_t) + ε_t'*inv(Σ_t)*ε_t);
end

"""
    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, h::Int64)

Forecast X up to h-steps ahead.

    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, P::Union{SymMatrix, Nothing}, h::Int64)

Forecast X and P up to h-steps ahead.
"""
function kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, h::Int64)

    history_X = Array{FloatVector,1}();

    for horizon=1:h
        X = kpredict(X, settings);
        push!(history_X, X);
    end

    return history_X;
end

function kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, P::Union{SymMatrix, Nothing}, h::Int64)

    history_X = Array{FloatVector,1}();
    history_P = Array{SymMatrix,1}();

    for horizon=1:h
        X = kpredict(X, settings);
        P = kpredict(P, settings);
        push!(history_X, X);
        push!(history_P, P);
    end

    return history_X, history_P;
end

function ksmooth()
end
