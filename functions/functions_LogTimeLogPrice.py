import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_log_log_graph_with_oscillator(
    df,
    future_year=2050,
    start_year=2010,
    start_month=10,
    start_day=24,
    ylim=(0.1, 1000000),
    yearly_labels_until=2030,
    every_nth_year=5,
    fontsize=6,
    rotation=45,
    title="LogTime vs LogPrice",
):
    """
    Plots a log-log graph of PriceUSD over time with extended OLS lines for bottoms and tops,
    and an oscillator subplot ranging from 0 to 100.

    Parameters:
    - df: DataFrame containing 'time' and 'PriceUSD' columns
    - future_year: Last year for the future dates extension (default: 2050)
    - start_year: Start year for the plot (default: 2010)
    - start_month: Start month for the plot (default: 10)
    - start_day: Start day for the plot (default: 24)
    - ylim: Tuple for y-axis limits (default: (0.1, 1000000))
    - yearly_labels_until: Year until which yearly labels are shown (default: 2030)
    - every_nth_year: Interval for labeling years after yearly_labels_until (default: 5)
    - fontsize: Font size for year labels (default: 6)
    - rotation: Rotation angle for year labels (default: 45)
    - title: Title of the plot (default: "LogTime vs LogPrice")
    """

    # Ensure the required columns are present
    df = df[["time", "PriceUSD"]]

    # Convert 'time' to datetime format
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Extend the dataframe to include daily data until future_year with NaN for PriceUSD
    future_dates = pd.date_range(
        start=df["time"].min(), end=f"{future_year}-12-31", freq="D"
    )
    df = (
        df.set_index("time")
        .reindex(future_dates)
        .reset_index()
        .rename(columns={"index": "time"})
    )

    # Add a column that goes from 1 to n
    df["index"] = range(1, len(df) + 1)

    # Define the bottom dates and corresponding indices
    bottoms_dates = pd.to_datetime(["2010-12-06", "2012-06-04", "2015-08-25", "2018-12-15", "2022-11-21"])
    bottoms_data = df[df["time"].isin(bottoms_dates)]

    # Perform OLS regression on log-log transformed data for bottoms
    X_bottoms = sm.add_constant(np.log(bottoms_data["index"].dropna()))
    y_bottoms = np.log(bottoms_data["PriceUSD"].dropna())
    model_bottoms = sm.OLS(y_bottoms, X_bottoms).fit()

    # Define the top dates and corresponding indices
    tops_dates = pd.to_datetime(["2011-06-08", "2013-11-30", "2017-12-17", "2021-11-10"])
    tops_data = df[df["time"].isin(tops_dates)]

    # Perform OLS regression on log-log transformed data for tops
    X_tops = sm.add_constant(np.log(tops_data["index"].dropna()))
    y_tops = np.log(tops_data["PriceUSD"].dropna())
    model_tops = sm.OLS(y_tops, X_tops).fit()

    # Filter out the necessary data for plotting
    filtered_df = df.dropna(subset=["PriceUSD"])

    # Predict the OLS lines over the entire range of indices
    extended_log_indices = sm.add_constant(np.log(df["index"]))
    extended_line_bottoms = np.exp(model_bottoms.predict(extended_log_indices))
    extended_line_tops = np.exp(model_tops.predict(extended_log_indices))

    # Calculate the oscillator
    df["BottomLine"] = extended_line_bottoms
    df["TopLine"] = extended_line_tops
    df["oscillator"] = (
        100
        * (np.log(df["PriceUSD"]) - np.log(df["BottomLine"]))
        / (np.log(df["TopLine"]) - np.log(df["BottomLine"]))
    )

    # Get the last value of the oscillator
    last_oscillator_value = df["oscillator"].dropna().iloc[-1]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )

    # Plot log-log graph with the new index column and hidden xticks on the first subplot
    ax1.plot(filtered_df["index"], filtered_df["PriceUSD"], label="PriceUSD")
    ax1.plot(
        df["index"],
        extended_line_bottoms,
        color="green",
        linestyle="--",
        label="Bottom Line",
    )
    ax1.plot(
        df["index"],
        extended_line_tops,
        color="red",
        linestyle="--",
        label="Top Line",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(title, fontsize=20, fontweight="bold")

    # Add top-left text with the last oscillator value
    ax1.text(
        0.012,
        0.82,
        f"Last Value: \n{last_oscillator_value:.1f}%",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Define y-axis ticks and grid for the first subplot
    yticks = [1, 10, 100, 1000, 10000, 100000, 1000000]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{tick:,.0f}" for tick in yticks])
    ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)

    # Set x-axis limits starting from start_date for the first subplot
    start_date_index = df[
        df["time"] == pd.Timestamp(f"{start_year}-{start_month:02d}-{start_day:02d}")
    ]["index"].values[0]
    ax1.set_xlim([start_date_index, df["index"].max()])

    # Set y-axis limits for the first subplot
    ax1.set_ylim(*ylim)

    # Hide x-axis grid and ticks for the first subplot
    ax1.xaxis.set_visible(False)

    # Set color of ticks in the first subplot to match the color of PriceUSD line
    ax1.tick_params(axis="y", colors="#1f52b4")

    # Create a map of index to year based on the 'time' column and extended years
    index_year_map = {index: year.year for index, year in zip(df["index"], df["time"])}

    # Add vertical lines and labels for every 1st January starting from 2011 to yearly_labels_until
    for year in range(2011, yearly_labels_until + 1):
        january_first = df[df["time"] == pd.Timestamp(f"{year}-01-01")]
        if not january_first.empty:
            january_first_index = january_first["index"].values[0]
            ax1.axvline(
                january_first_index,
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.3,
            )
            ax1.text(
                january_first_index + 1,
                0.15,
                str(year),
                rotation=rotation,
                verticalalignment="bottom",
                fontsize=fontsize,
                color="k",
            )

    # Add vertical lines and labels for every nth January after yearly_labels_until
    for year in range(
        yearly_labels_until + every_nth_year, future_year + 1, every_nth_year
    ):
        january_first = df[df["time"] == pd.Timestamp(f"{year}-01-01")]
        if not january_first.empty:
            january_first_index = january_first["index"].values[0]
            ax1.axvline(
                january_first_index,
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.3,
            )
            ax1.text(
                january_first_index + 1,
                0.15,
                str(year),
                rotation=rotation,
                verticalalignment="bottom",
                fontsize=fontsize,
                color="k",
            )

    # Plot the oscillator on the second subplot
    ax2.fill_between(df["index"], 0, 10, color="green", alpha=0.2)
    ax2.fill_between(df["index"], 90, 100, color="red", alpha=0.2)
    ax2.plot(
        df["index"], df["oscillator"], color="orange", label="Oscillator", alpha=0.7
    )
    ax2.set_ylim([0, 100])
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    # Set color of ticks in the second subplot to match the color of the oscillator line
    ax2.tick_params(axis="y", colors="orange")

    # Add vertical lines and labels for every 1st January starting from 2011 to yearly_labels_until
    for year in range(2011, yearly_labels_until + 1):
        january_first = df[df["time"] == pd.Timestamp(f"{year}-01-01")]
        if not january_first.empty:
            january_first_index = january_first["index"].values[0]
            ax2.axvline(
                january_first_index,
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.3,
            )
            ax2.text(
                january_first_index + 1,
                5,
                str(year),
                rotation=rotation,
                verticalalignment="bottom",
                fontsize=fontsize,
                color="k",
            )

    # Add vertical lines and labels for every nth January after yearly_labels_until
    for year in range(
        yearly_labels_until + every_nth_year, future_year + 1, every_nth_year
    ):
        january_first = df[df["time"] == pd.Timestamp(f"{year}-01-01")]
        if not january_first.empty:
            january_first_index = january_first["index"].values[0]
            ax2.axvline(
                january_first_index,
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.3,
            )
            ax2.text(
                january_first_index + 1,
                5,
                str(year),
                rotation=rotation,
                verticalalignment="bottom",
                fontsize=fontsize,
                color="k",
            )

    # Hide x-axis grid and ticks for the second subplot
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_visible(False)

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0)

    # Add legend to both subplots
    ax1.legend(fontsize=8)
    ax2.legend()

    plt.savefig(
        "../output/1b.LogTimeLogPrice.jpg",
        bbox_inches="tight",
        dpi=350,
    )
    plt.show()

    return df


def plot_final_log_graph_with_oscillator(df):
    """
    Plots a graph showing the Bitcoin price prediction along with an oscillator.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the Bitcoin price data.

    Returns:
    None
    """

    # Drop rows with 'PriceUSD' NaN before 2011
    df_before_2012 = df[df["time"] < "2011-01-01"].dropna(subset=["PriceUSD"])
    # Rows from 2011 onwards
    df_from_2012_onwards = df[df["time"] >= "2011-01-01"]
    # Concatenate the two DataFrames
    df = pd.concat([df_before_2012, df_from_2012_onwards])

    # Filter the dataframe to start from 2011-01-01
    df = df[df["time"] >= "2011-01-01"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [5, 1]}
    )

    # First plot
    ax1.plot(df.time, df.TopLine, label="TopLine", color="red", linestyle="--")
    ax1.plot(df.time, df.PriceUSD, label="PriceUSD", color="#1f52b4")
    ax1.plot(df.time, df.BottomLine, label="BottomLine", color="green", linestyle="--")
    ax1.set_yscale("log")
    ax1.set_ylim(0, 1_000_000)
    ax1.set_xlim(pd.Timestamp("2011-01-01"), df.time.max())
    ax1.grid(ls="--", alpha=0.3)
    ax1.set_title("AritmTime vs LogPrice", fontsize=20, fontweight="bold")
    ax1.legend()

    # Set yticks in non-scientific notation
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))

    # Secondo plot
    ax2.plot(df.time, df.oscillator, label="Oscillator", color="orange")
    ax2.set_ylim(0, 100)
    ax2.fill_between(df.time, 0, 10, color="green", alpha=0.2)
    ax2.fill_between(df.time, 90, 100, color="red", alpha=0.2)
    ax2.grid(ls="--", alpha=0.3)
    ax2.legend()

    # Adjust space between plots
    plt.subplots_adjust(hspace=0)

    plt.savefig(
        "../output/1c.AritmTimeLogPrice.jpg",
        bbox_inches="tight",
        dpi=350,
    )

    plt.show()
