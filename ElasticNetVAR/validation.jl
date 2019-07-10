"""
    select_hyperparameters(validation_settings::ValidationSettings, γ_grid::HyperGrid)

Select the tuning hyper-parameters for the elastic-net vector autoregression.

# Arguments
- `validation_settings`: ValidationSettings struct
- `γ_grid`: HyperGrid struct

# References
Pellegrino (2019)
"""
function select_hyperparameters(validation_settings::ValidationSettings, γ_grid::HyperGrid)

    # Check inputs
    check_bounds(validation_settings.max_iter, 3);
    check_bounds(validation_settings.max_iter, validation_settings.prerun);
    check_bounds(validation_settings.n, 2); # It supports only multivariate models (for now ...)

    if length(γ_grid.p) != 2 || length(γ_grid.λ) != 2 || length(γ_grid.α) != 2 || length(γ_grid.β) != 2
        error("The grids include more than two entries. They must include only the lower and upper bounds for the grids!")
    end

    check_bounds(γ_grid.p[2], γ_grid.p[1]);
    check_bounds(γ_grid.λ[2], γ_grid.λ[1]);
    check_bounds(γ_grid.α[2], γ_grid.α[1]);
    check_bounds(γ_grid.β[2], γ_grid.β[1]);
    check_bounds(γ_grid.p[1], 1);
    check_bounds(γ_grid.λ[1], 0);
    check_bounds(γ_grid.α[1], 0, 1);
    check_bounds(γ_grid.α[2], 0, 1);
    check_bounds(γ_grid.β[1], 1);

    # Construct grid of hyperparameters - random search algorithm
    errors     = zeros(γ_grid.draws);
    candidates = zeros(4, γ_grid.draws);

    for draw=1:γ_grid.draws
        candidates[:,draw] = [rand(γ_grid.p[1]:γ_grid.p[2]),
                              rand(Uniform(γ_grid.λ[1], γ_grid.λ[2])),
                              rand(Uniform(γ_grid.α[1], γ_grid.α[2])),
                              rand(Uniform(γ_grid.β[1], γ_grid.β[2]))];
    end

    if ~isnothing(validation_settings.log_folder_path)
        open("$(validation_settings.log_folder_path)/status.txt", "w") do io
            write(io, "")
        end
    end

    for iter=1:γ_grid.draws

        # Retrieve candidate hyperparameters
        p, λ, α, β = candidates[:,iter];
        p = Int64(p);

        # Update log
        if validation_settings.verb == true
            message = "select_hyperparameters (error estimator $(validation_settings.err_type)) > running iteration $iter (out of $(γ_grid.draws), γ=($(round(p,digits=3)), $(round(λ,digits=3)), $(round(α,digits=3)), $(round(β,digits=3)))";
            println(message);
            if ~isnothing(validation_settings.log_folder_path)
                open("$(validation_settings.log_folder_path)/status.txt", "a") do io
                    write(io, "$message\n")
                end
            end
        end

        # Evaluate
        if validation_settings.err_type < 3
            errors[iter], inactive_sample = fc_err(validation_settings, p, λ, α, β);
            if inactive_sample == 1
                error("The validation sample is a matrix of missings!");
            end
        else
            errors[iter] = jackknife_err(validation_settings, p, λ, α, β);
        end
    end

    verb_message(validation_settings.verb, "");

    # Return output
    return candidates, errors;
end

"""
    fc_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number)

Compute the in-sample / out-of-sample error associated with the candidate hyperparameters

# Arguments
- `validation_settings`: ValidationSettings struct
- `p`: (candidate) number of lags in the vector autoregression
- `λ`: (candidate) overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: (candidate) weight associated to the LASSO component of the elastic-net penalty
- `β`: (candidate) additional shrinkage for distant lags (p>1)
- `jth_jackknife_data`: j-th jackknife sample (default: nothing)

# References
Pellegrino (2019)
"""
function fc_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number; jth_jackknife_data::Union{JArray{Float64}, Nothing}=nothing)

    t0 = validation_settings.T;

    # Out-of-sample error
    if validation_settings.err_type != 1
        t0 = validation_settings.t0;
    end

    # Jackknife out-of-sample
    if validation_settings.err_type > 2
        μ = mean_skipmissing(jth_jackknife_data[:, 1:t0]);
        σ = std_skipmissing(jth_jackknife_data[:, 1:t0]);
        Y = @. jth_jackknife_data[:, 1:t0] - μ;
        Y_output = @. jth_jackknife_data - μ;

    # Standard out-of-sample
    else
        μ = mean_skipmissing(validation_settings.Y[:, 1:t0]);
        σ = std_skipmissing(validation_settings.Y[:, 1:t0]);
        Y = @. validation_settings.Y[:, 1:t0] - μ;
        Y_output = @. validation_settings.Y - μ;
    end

    # Construct estim_settings
    estim_settings = EstimSettings(Y, Y_output, p, λ, α, β, ε=validation_settings.ε, tol=validation_settings.tol, max_iter=validation_settings.max_iter, prerun=validation_settings.prerun, verb=validation_settings.verb_estim);

    # Estimate penalised VAR
    kalman_settings = ecm(estim_settings);

    # Run Kalman filter
    status = KalmanStatus();
    for t=1:kalman_settings.T
        kfilter!(kalman_settings, status);
    end

    # Residuals
    forecast  = hcat(status.history_X_prior...)[1:validation_settings.n, :];
    std_resid = @. ((forecast - Y_output)/σ)^2;

    # In-sample error
    if validation_settings.err_type == 1
        return compute_loss(std_resid);

    # Out-of-sample error
    else
        return compute_loss(std_resid[:, t0+1:end]);
    end
end

"""
    compute_loss(std_resid::FloatArray)
    compute_loss(std_resid::Array{Missing})
    compute_loss(std_resid::JArray{Float64})

Compute loss function.
"""
compute_loss(std_resid::FloatArray) = [mean(mean(std_resid, dims=1), dims=2)[1], 0.0];
compute_loss(std_resid::Array{Missing}) = [0.0, 1.0];
function compute_loss(std_resid::JArray{Float64})
    loss = 0.0;
    inactive_periods = 0.0;
    for t=1:size(std_resid,2)
        if sum(.~ismissing.(std_resid[:,t])) > 0
            loss += mean_skipmissing(std_resid[:,t])
        else
            inactive_periods += 1.0;
        end
    end

    if inactive_periods == size(std_resid,2)
        return [0.0 1.0];
    else
        return [loss/(size(std_resid,2)-inactive_periods) 0.0];
    end
end

"""
    jackknife_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number)

Return the jackknife out-of-sample error.

# Arguments
- `validation_settings`: ValidationSettings struct
- `p`: (candidate) number of lags in the vector autoregression
- `λ`: (candidate) overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: (candidate) weight associated to the LASSO component of the elastic-net penalty
- `β`: (candidate) additional shrinkage for distant lags (p>1)

# References
Pellegrino (2019)
"""
function jackknife_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number)

    # Block jackknife
    if validation_settings.err_type == 3
        jackknife_data = block_jackknife(validation_settings.Y, validation_settings.subsample);

    # Artificial jackknife
    elseif validation_settings.err_type == 4
        jackknife_data = artificial_jackknife(validation_settings.Y, validation_settings.subsample, validation_settings.max_samples);

    else
        error("Wrong err_type for jackknife_err!");
    end

    # Number of jackknife samples
    samples = size(jackknife_data, 3);

    # Compute jackknife loss
    verb_message(validation_settings.verb_estim, "jackknife_err > running $samples iterations on $(nworkers()) workers");

    output_fc_err = zeros(2);
    output_fc_err = @sync @distributed (+) for j=1:samples
        fc_err(validation_settings, p, λ, α, β; jth_jackknife_data=jackknife_data[:,:,j]);
    end

    # Compute average jackknife loss
    loss, inactive_samples = output_fc_err;
    loss *= 1/(samples-inactive_samples);

    verb_message(validation_settings.verb_estim, "");

    # Return output
    return loss;
end
