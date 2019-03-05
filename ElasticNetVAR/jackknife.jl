"""
    block_jackknife(object::BlockJackknife)

Generate block jackknife samples on the basis of a BlockJackknife structure.

This technique subsamples a time series dataset by removing, in turn, all the blocks of consecutive observations
with a given size.

See also: Kunsch (1989).
"""
function block_jackknife(object::BlockJackknife)

    # Error management
    if (object.subsample <= 0) || (object.subsample >= 1)
        error("object.subsample must be between 0 and 1.");
    end

    # Dimensions
    if ndims(object.data) == 1
        const T = length(object.data);
        const n = 1;
    else
        const T, n = size(object.data);
    end

    # Block size
    const block_size = Int64(ceil(object.subsample*T));

    # Number of block jackknifes samples - as in Kunsch (1989)
    const samples = T-block_size+1;

    # Initialise jackknife_data
    if n == 1
        jackknife_data = Array{FloatVector,1}(samples);
    else
        jackknife_data = Array{FloatMatrix,1}(samples);
    end

    # Loop over b=1, ..., samples
    for b=1:samples
        indᵇ = collect(1:T);
        deleteat!(indᵇ, collect(b:b+block_size-1));

        if n == 1
            jackknife_data[b] = object.data[indᵇ];
        else
            jackknife_data[b] = object.data[indᵇ, :];
        end
    end

    # Return jackknife_data
    return jackknife_data;
end

"""
    artificial_jackknife(object::ArtificialJackknife)

Generate artificial jackknife samples on the basis of a ArtificialJackknife structure.

The artificial delete-d jackknife is an extension of the delete-d jackknife for dependent data problems.
This technique replaces the actual data removal step with a fictitious deletion, which consists of
imposing `d`-dimensional (artificial) patterns of missing observations to the data. This approach
does not alter the data order nor destroy the correlation structure.

See also: Pellegrino (2019).
"""
function artificial_jackknife(object::ArtificialJackknife)

    # Error management
    if (object.subsample <= 0) || (object.subsample >= 1)
        error("object.subsample must be between 0 and 1.");
    end

    # Dimensions
    if ndims(object.data) == 1
        const T = length(object.data);
        const n = 1;
    else
        const T, n = size(object.data);
    end

    const Tn = T*n;

    # d
    const d = Int64(ceil(object.subsample*Tn));
    if d <= sqrt(Tn)
        error("The number of (artificial) missing observations is too small. d must be larger or equal to sqrt(Tn).");
    end

    # Get vec(Y)
    const vec_Y = object.data[:];

    # Initialise ind_missings
    ind_missings = convert(Array{Int64,2}, zeros(d, min(C(Tn,d), object.max_samples)));

    # If C(Tn,d) is sufficiently small, it uses `combinations(...)` to generate the full set
    if C(Tn,d) <= object.max_samples
        ind_missings = hcat(collect(combinations(collect(1:Tn), d))...);

    # Else: draw without replacement from the full combinations set an `object.max_samples` number of indices
    else
        for b=1:object.max_samples

            if b == 1
                # Get index
                indᵇ = collect(1:Tn);
                rand_without_replacement!(indᵇ, Tn-d);

                # Store index
                ind_missings[:,b] = indᵇ;

            elseif b > 1
                # Iterates until ind_missings[:,b] is neither a vector of zeros, nor already included in ind_missings
                while (ind_missings[:,b] == zeros(d)) || (vector_in_matrix(ind_missings[:,b], ind_missings[:,1:b-1]))

                    # Get index
                    indᵇ = collect(1:Tn);
                    rand_without_replacement!(indᵇ, Tn-d);

                    # Store index
                    ind_missings[:,b] = indᵇ;
                end
            end
        end
    end

    # Initialise jackknife_data
    if n == 1
        jackknife_data = Array{FloatVector,1}(object.max_samples);
    else
        jackknife_data = Array{FloatMatrix,1}(object.max_samples);
    end

    # Loop over b=1, ..., object.max_samples
    for b=1:object.max_samples

        # Add (artificial) missing observations
        vec_Yᵇ = FloatVectorWithMissings(vec_Y);
        vec_Yᵇ[ind_missings[:,b]] = missing;

        # Store data
        if n == 1
            jackknife_data[b] = copy(vec_Yᵇ);
        else
            jackknife_data[b] = reshape(vec_Yᵇ, T, n);
        end
    end

    # Return jackknife_data
    return jackknife_data;
end
