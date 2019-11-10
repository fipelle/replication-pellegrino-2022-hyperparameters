"""
    compute_loss_weights(data_presample::SubArray{Union{Float64,Missing}}, n::Int64, standardise_error::Bool, weights::Nothing)
    compute_loss_weights(data_presample::SubArray{Union{Float64,Missing}}, n::Int64, standardise_error::Bool, weights::FloatVector)

Compute weights for the forecast error.
"""
compute_loss_weights(data_presample::SubArray{Union{Float64,Missing}}, n::Int64, standardise_error::Bool, weights::Nothing) = standardise_error ? std_skipmissing(data_presample).^2 : ones(n);
compute_loss_weights(data_presample::SubArray{Union{Float64,Missing}}, n::Int64, standardise_error::Bool, weights::FloatVector) = standardise_error ? std_skipmissing(data_presample).^2 : weights;

"""
    compute_loss(weighted_resid::AbstractArray{Float64})
    compute_loss(weighted_resid::AbstractArray{Missing})
    compute_loss(weighted_resid::AbstractArray{Union{Float64, Missing}})

Compute loss function.
"""
compute_loss(weighted_resid::AbstractArray{Float64}) = [mean(mean(weighted_resid, dims=1), dims=2)[1], 0.0];
compute_loss(weighted_resid::AbstractArray{Missing}) = [0.0, 1.0];
function compute_loss(weighted_resid::AbstractArray{Union{Float64, Missing}})
    loss = 0.0;
    inactive_periods = 0.0;
    T = size(weighted_resid,2);

    for t=1:T
        weighted_resid_t = @view weighted_resid[:,t];
        if sum(.~ismissing.(weighted_resid_t)) > 0
            loss += mean_skipmissing(weighted_resid_t);
        else
            inactive_periods += 1.0;
        end
    end

    if inactive_periods == T
        return [0.0, 1.0];
    else
        return [loss/(T-inactive_periods), 0.0];
    end
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
        data = jth_jackknife_data;

    # Standard in-sample or out-of-sample
    else
        data = validation_settings.Y;
    end

    # Data
    data_presample = @view data[:, 1:t0];
    μ = mean_skipmissing(data_presample);
    Y = @. data_presample - μ;
    Y_output = @. data - μ;

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

    # Compute weights
    w = compute_loss_weights(data_presample, validation_settings.n, validation_settings.standardise_error, validation_settings.weights);

    # Weighted error
    weighted_resid = @. w*(forecast - Y_output)^2;

    # In-sample error
    if validation_settings.err_type == 1
        return compute_loss(weighted_resid);

    # Out-of-sample error
    else
        weighted_resid_oos = @view weighted_resid[:, t0+1:end];
        return compute_loss(weighted_resid_oos);
    end
end

"""
    jackknife_err(validation_settings::ValidationSettings, jackknife_data::JArray{Float64, 3}, p::Int64, λ::Number, α::Number, β::Number)

Return the jackknife out-of-sample error.

# Arguments
- `validation_settings`: ValidationSettings struct
- `jackknife_data`: jackknife partitions
- `p`: (candidate) number of lags in the vector autoregression
- `λ`: (candidate) overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: (candidate) weight associated to the LASSO component of the elastic-net penalty
- `β`: (candidate) additional shrinkage for distant lags (p>1)

# References
Pellegrino (2019)
"""
function jackknife_err(validation_settings::ValidationSettings, jackknife_data::JArray{Float64, 3}, p::Int64, λ::Number, α::Number, β::Number)

    # Error management
    if validation_settings.err_type <= 2
        error("Wrong err_type for jackknife_err!");
    end

    # Number of jackknife samples
    samples = size(jackknife_data, 3);

    # Compute jackknife loss
    verb_message(validation_settings.verb_estim, "jackknife_err > running $samples iterations on $(nworkers()) workers");

    output_fc_err = @sync @distributed (+) for j=1:samples
        fc_err(validation_settings, p, λ, α, β, jth_jackknife_data=jackknife_data[:,:,j]);
    end

    # Compute average jackknife loss
    loss, inactive_samples = output_fc_err;
    if samples == inactive_samples
        error("All samples are inactive! Check the initial settings or try a different random seed.");
    end
    loss *= 1/(samples-inactive_samples);

    # Return output
    return loss;
end

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
        io = open("$(validation_settings.log_folder_path)/status.txt", "w+");
        global_logger(ConsoleLogger(io));
    end

    # Generate partitions for the block jackknife out-of-sample
    if validation_settings.err_type == 3
        jackknife_data = block_jackknife(validation_settings.Y, validation_settings.subsample);

    # Generate partitions for the artificial jackknife
    elseif validation_settings.err_type == 4
        jackknife_data = artificial_jackknife(validation_settings.Y, validation_settings.subsample, validation_settings.max_samples);
    end

    for iter=1:γ_grid.draws

        # Retrieve candidate hyperparameters
        p, λ, α, β = candidates[:,iter];
        p = Int64(p);

        # Update log
        if validation_settings.verb == true
            @info "$(round(now(), Dates.Second(1))) select_hyperparameters (error estimator $(validation_settings.err_type)) > running iteration $iter (out of $(γ_grid.draws)), γ=($(round(p,digits=3)), $(round(λ,digits=3)), $(round(α,digits=3)), $(round(β,digits=3)))";
            if ~isnothing(validation_settings.log_folder_path)
                flush(io);
            end
        end

        #=
        Some candidate hyperparameters the estimation of the VAR can be unstable. This generally happens when:
        - The candidates are extreme
        - validation_settings.subsample is high and there is not enough shrinkage

        A priori, there is not a simple way to construct a grid of candidates that does not result in errors in the ECM.
        The try-catch statement below handles this problem by skipping the candidate values that generate model instability.

        Note that this issue is more likely to happen with the block-jackknife.
        =#

        try

            # In-sample or standard out-of-sample
            if validation_settings.err_type <= 2
                errors[iter], inactive_sample = fc_err(validation_settings, p, λ, α, β);
                if inactive_sample == 1
                    error("The validation sample is a matrix of missings!");
                end

            # Jackknife out-of-sample
            else
                errors[iter] = jackknife_err(validation_settings, jackknife_data, p, λ, α, β);
            end

        catch error_iter

            # Extract info on the error
            error_ex, error_msg, error_stacktrace_array = error_info(error_iter);
            error_stacktrace = join([*(string(error_stacktrace_array[error_line]), "\n") for error_line=1:length(error_stacktrace_array)]);

            # Update log
            @error "$(round(now(), Dates.Second(1))) $error_msg \n $error_stacktrace";

            # The instability pops up in the update_loglik! function
            if isa(error_ex, DomainError) && occursin("logdet", "$error_stacktrace") && occursin("update_loglik!", "$error_stacktrace")
                errors[iter] = Inf;

            # Any other error
            else
                rethrow(error_iter);
            end
        end
    end

    # Return output
    return candidates, errors;
end
