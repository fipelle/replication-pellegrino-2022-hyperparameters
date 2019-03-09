"""
    block_jackknife(Y::JArray{Float64,2}, subsample::Float64)

Generate block jackknife samples as in Kunsch (1989).

This technique subsamples a time series dataset by removing, in turn, all the blocks of consecutive observations
with a given size.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `subsample`: block size as a percentage of the original sample size. It is bounded between 0 and 1.

# References
Kunsch (1989) and Pellegrino (2019)
"""
function block_jackknife(Y::JArray{Float64,2}, subsample::Float64)

    # Error management
    if subsample <= 0 || subsample >= 1
        error("0 < subsample < 1");
    end

    # Dimensions
    n, T = size(Y);

    # Block size
    block_size = Int64(ceil(subsample*T));

    # Number of block jackknifes samples - as in Kunsch (1989)
    samples = T-block_size+1;

    # Initialise jackknife_data
    jackknife_data = JArray{Float64,3}(undef, n, T, samples);

    # Loop over j=1, ..., samples
    for j=1:samples

        # Index of missings
        indʲ = collect(j:j+block_size-1);

        # Block jackknife data
        jackknife_data[:, :, j] = Y;
        jackknife_data[:, indʲ, j] .= missing;
    end

    # Return jackknife_data
    return jackknife_data;
end

"""
    artificial_jackknife(Y::JArray{Float64,2}, subsample::Float64, max_samples::Int64)

Generate artificial jackknife samples as in Pellegrino (2019).

The artificial delete-d jackknife is an extension of the delete-d jackknife for dependent data problems.
This technique replaces the actual data removal step with a fictitious deletion, which consists of
imposing `d`-dimensional (artificial) patterns of missing observations to the data. This approach
does not alter the data order nor destroy the correlation structure.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `subsample`: `d` as a percentage of the original sample size. It is bounded between 0 and 1.
- `max_samples`: if `C(T*n,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples.

# References
Pellegrino (2019)
"""
function artificial_jackknife(Y::JArray{Float64,2}, subsample::Float64, max_samples::Int64)

    # Error management
    if subsample <= 0 || subsample >= 1
        error("0 < subsample < 1");
    end

    # Dimensions
    n, T = size(Y);
    Tn = T*n;

    # d
    d = Int64(ceil(subsample*Tn));
    if d <= sqrt(Tn)
        error("The number of (artificial) missing observations is too small. d must be larger or equal to sqrt(Tn).");
    end

    # Get vec(Y)
    vec_Y = Y[:] |> JArray{Float64};

    # Initialise ind_missings
    samples = min(no_combinations(Tn, d), max_samples) |> Int64;
    ind_missings = zeros(d, samples) |> Array{Int64};

    # Initialise jackknife_data
    jackknife_data = JArray{Float64,3}(undef, n, T, samples);

    # Loop over j=1, ..., samples
    for j=1:samples

        if j == 1

            # Get index
            indʲ = collect(1:Tn);
            rand_without_replacement!(indʲ, Tn-d);

            # Store index
            ind_missings[:,j] = indʲ;

        elseif j > 1

            # Iterates until ind_missings[:,j] is neither a vector of zeros, nor already included in ind_missings
            while ind_missings[:,j] == zeros(d) || is_vector_in_matrix(ind_missings[:,j], ind_missings[:, 1:j-1])

                # Get index
                indʲ = collect(1:Tn);
                rand_without_replacement!(indʲ, Tn-d);

                # Store index
                ind_missings[:,j] = indʲ;
            end
        end

        # Add (artificial) missing observations
        vec_Yʲ = copy(vec_Y);
        vec_Yʲ[ind_missings[:,j]] .= missing;

        # Store data
        jackknife_data[:, :, j] = reshape(vec_Yʲ, n, T);
    end

    # Return jackknife_data
    return jackknife_data;
end
