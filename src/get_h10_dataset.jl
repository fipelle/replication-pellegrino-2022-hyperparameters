using CSV, DataFrames, Dates, MessyTimeSeries;
using FredData: get_data, Fred;

"""
    get_h10_dataset(fred_tickers::Array{String,1}, mnemonics::Array{String,1}, transform::IntVector, to_include::IntVector, observation_start::String, observation_end::String)

Get H.10 table from the Federal Reserve Economic Data (FRED) at weekly frequency and in log-return.
"""
function get_h10_dataset(fred_tickers::Array{String,1}, mnemonics::Array{String,1}, transform::IntVector, to_include::IntVector, observation_start::String, observation_end::String)

    f = Fred();
    df = DataFrame();

    for (i, ticker) in enumerate(fred_tickers)

        # Download current series
        df_current = get_data(f, ticker, observation_start=observation_start, observation_end=observation_end, frequency="wef", aggregation_method="eop", units="cch").data::DataFrame;
        df_current = df_current[!,[:date, :value]];
        if transform[i] == 1
            df_current[!,:value] *= -1;
        end

        # Rename `:value`
        rename!(df_current, Dict(:value => mnemonics[i]));

        # Add series to `df`
        if i == 1
            df = df_current;
        else
            df = outerjoin(df, df_current, on=:date);
        end
    end

    # Use selected exchange rates
    countries_selection = names(df)[2:end][to_include .== 1];
    df = df[!, vcat(names(df)[1], countries_selection)];

    # Chronological order
    sort!(df, :date);

    # Return output
    return df;
end

# Load info from CSV
fred_tickers = CSV.read("./data/fred_h10_data.csv", DataFrame)[!,:fred_ticker] |> Vector{String};
mnemonics = CSV.read("./data/fred_h10_data.csv", DataFrame)[!,:mnemonic] |> Vector{String};
transform = CSV.read("./data/fred_h10_data.csv", DataFrame)[!,:transform];
to_include = CSV.read("./data/fred_h10_data.csv", DataFrame)[!,:include];