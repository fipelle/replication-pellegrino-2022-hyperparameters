function get_us_results(output_path, α_selected=[])
    out = load(output_path);

    # Select minimum error and corresponding vector of hyperparameters
    if ~isempty(α_selected)
        ind_α = findall(out["hyper_grid"][3,:] .== α_selected);
    else
        ind_α = collect(1:length(out["err_grid"]));
    end

    ind_γ = argmin(out["err_grid"][ind_α]);
    γ = out["hyper_grid"][:, ind_α[ind_γ]];
    err = minimum(out["err_grid"][ind_α]);

    return err, γ;
end
