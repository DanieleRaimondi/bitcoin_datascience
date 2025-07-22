import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm
import numpy as np
import sys

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_data


def load_bitcoin_data():
    """Load Bitcoin price data from a predefined fetch function."""
    btc = fetch_data("btc")[["time", "PriceUSD"]]
    btc["time"] = pd.to_datetime(btc["time"])
    return btc


def load_election_probabilities():
    """Load US election probability data from a CSV file."""
    us_data = pd.read_csv("../data/polymarket_US_election_24.csv")
    us_data = us_data.drop(columns=["Timestamp (UTC)"])
    us_data = us_data.rename(columns={"Date (UTC)": "time"})
    us_data["time"] = pd.to_datetime(
        us_data["time"], format="%m-%d-%Y %H:%M"
    ).dt.normalize()
    return us_data


def merge_data(us_data, btc_data):
    """Merge Bitcoin price and election probability data."""
    merged_data = pd.merge(
        us_data, btc_data[["time", "PriceUSD"]], on="time", how="inner"
    )
    merged_data["time_numeric"] = merged_data["time"].astype("int64") // 10**9
    return merged_data


def apply_loess_smoothing(data, column, time_column, frac=0.05):
    """Apply LOESS smoothing to a data column."""
    lowess = sm.nonparametric.lowess(data[column], data[time_column], frac=frac)
    return np.interp(data[time_column], lowess[:, 0], lowess[:, 1])


def plot_data(
    merged_data,
    include_trump=True,
    include_biden=True,
    include_harris=True,
    include_democrats=True,
    apply_loess=True,
    frac=0.05,
):
    """Create a multi-axis plot showing Bitcoin price and US election probabilities."""
    # Create the main figure and primary axis for Bitcoin price
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_ylabel("BTC Price (USD)", color="black")

    # Configure y-axis formatting
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.tick_params(axis="y", colors="black")

    # Plot raw Bitcoin price
    ax1.plot(
        merged_data["time"],
        merged_data["PriceUSD"],
        color="black",
        label="BTC Price",
        alpha=0.1 if apply_loess else 1.0,
    )

    # Apply and plot LOESS smoothing for Bitcoin price
    if apply_loess:
        smoothed_btc = apply_loess_smoothing(
            merged_data, "PriceUSD", "time_numeric", frac
        )
        ax1.plot(
            merged_data["time"],
            smoothed_btc,
            color="black",
            linewidth=1,
            linestyle="--",  # Added dashed line style
        )

    # Configuration for probability plots
    prob_configs = [
        (include_trump, "Donald Trump", "red"),
        (include_biden, "Joe Biden", "orange"),
        (include_harris, "Kamala Harris", "green"),
    ]

    # Initial offset for secondary axes
    offset_increment = 0.07
    current_offset = 1.0

    # Plot individual candidate probabilities
    for include, column, color in prob_configs:
        if include:
            ax = ax1.twinx()
            ax.spines["right"].set_position(("axes", current_offset))
            current_offset += offset_increment
            ax.spines["right"].set_visible(True)
            ax.set_ylabel(f"{column} Probability", color=color)
            ax.tick_params(axis="y", colors=color)

            ax.plot(
                merged_data["time"],
                merged_data[column],
                color=color,
                label=f"{column} Probability",
                alpha=0.1 if apply_loess else 1.0,
            )

            if apply_loess:
                smoothed_prob = apply_loess_smoothing(
                    merged_data, column, "time_numeric", frac
                )
                ax.plot(
                    merged_data["time"],
                    smoothed_prob,
                    color=color,
                    linewidth=1,
                    linestyle="--",  # Added dashed line style
                )
                ax.set_ylim(np.min(smoothed_prob) * 0.95, np.max(smoothed_prob) * 1.05)

    # Plot Democrat combined probability if requested
    if include_democrats:
        ax = ax1.twinx()
        ax.spines["right"].set_position(("axes", current_offset))
        ax.spines["right"].set_visible(True)
        ax.set_ylabel("Democrats Probability", color="blue")
        ax.tick_params(axis="y", colors="blue")

        democrat_prob = merged_data["Joe Biden"] + merged_data["Kamala Harris"]
        ax.plot(
            merged_data["time"],
            democrat_prob,
            color="blue",
            label="Democrats Probability",
            alpha=0.1 if apply_loess else 1.0,
        )

        if apply_loess:
            dem_data = pd.DataFrame(
                {
                    "democrat_prob": democrat_prob,
                    "time_numeric": merged_data["time_numeric"],
                }
            )
            smoothed_dem_prob = apply_loess_smoothing(
                dem_data, "democrat_prob", "time_numeric", frac
            )
            ax.plot(
                merged_data["time"],
                smoothed_dem_prob,
                color="blue",
                linewidth=1,
                linestyle="--",  # Added dashed line style
            )
            ax.set_ylim(
                np.min(smoothed_dem_prob) * 0.95, np.max(smoothed_dem_prob) * 1.05
            )

    plt.title(
        "BTC Price and US Presidential Election Winner 2024",
        fontweight="bold",
        fontsize=16,
    )
    plt.savefig("../output/7a.BTCvsUSELECTIONS.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def preprocess_data(btc_data):
    """
    Preprocess Bitcoin data and align with U.S. presidential administrations.

    Parameters:
    btc_data (DataFrame): DataFrame containing Bitcoin price data with 'time' and 'PriceUSD' columns

    Returns:
    DataFrame: Processed data with Bitcoin prices and presidential administration indicators
    """
    import pandas as pd
    import numpy as np

    # Create a copy of the Bitcoin data
    df = btc_data.copy()

    # Ensure time column is in datetime format
    df["time"] = pd.to_datetime(df["time"])

    # Define U.S. presidential elections and administrations
    elections = pd.DataFrame(
        {
            "election_date": pd.to_datetime(
                ["2008-11-04", "2012-11-06", "2016-11-08", "2020-11-03", "2024-11-05"]
            ),
            "president": ["Obama", "Obama", "Trump", "Biden", "Trump"],
            "party": [
                "Democratic",
                "Democratic",
                "Republican",
                "Democratic",
                "Republican",
            ],
        }
    )

    # Add column to identify which administration each Bitcoin price falls under
    df["administration"] = None
    df["party"] = None

    # Assign each Bitcoin price to the most recent presidential administration
    for i, row in df.iterrows():
        # Find the most recent election before this Bitcoin price observation
        prior_elections = elections[elections["election_date"] <= row["time"]]
        if not prior_elections.empty:
            most_recent = prior_elections.iloc[-1]
            df.at[i, "administration"] = most_recent["president"]
            df.at[i, "party"] = most_recent["party"]

    # Drop rows before the first election in our dataset
    df = df.dropna(subset=["administration"])

    # Add a log-scaled price column for visualization
    df["log_price"] = np.log10(df["PriceUSD"])

    return df, elections


def plot_data_winners(df, elections=None):
    """
    Create a log-scale plot of Bitcoin prices segmented by U.S. presidential administrations.

    Parameters:
    df (DataFrame): DataFrame with Bitcoin prices and administration data
    elections (DataFrame, optional): DataFrame with election dates and results

    Returns:
    None: Displays and saves the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle

    # If elections DataFrame is not provided, create a default one
    if elections is None:
        elections = pd.DataFrame(
            {
                "election_date": pd.to_datetime(
                    [
                        "2008-11-04",
                        "2012-11-06",
                        "2016-11-08",
                        "2020-11-03",
                        "2024-11-05",
                    ]
                ),
                "president": ["Obama", "Obama", "Trump", "Biden", "Trump"],
                "party": [
                    "Democratic",
                    "Democratic",
                    "Republican",
                    "Democratic",
                    "Republican",
                ],
            }
        )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot Bitcoin price on a logarithmic scale
    ax.semilogy(
        df["time"], df["PriceUSD"], "k-", linewidth=1.5, label="Bitcoin Price (USD)"
    )

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Format y-axis labels with commas
    from matplotlib.ticker import FuncFormatter

    def y_fmt(y, pos):
        if y < 1:
            return f"${y:.2f}"
        elif y < 1000:
            return f"${int(y)}"
        else:
            return f"${int(y):,}"

    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

    # Shade regions by administration party
    min_date = df["time"].min()
    max_date = df["time"].max()

    # Get unique time spans for each administration
    admin_spans = []
    for i, election in elections.iterrows():
        start_date = election["election_date"]
        if i < len(elections) - 1:
            end_date = elections.iloc[i + 1]["election_date"]
        else:
            end_date = max_date

        # Only include spans that overlap with our Bitcoin data
        if end_date >= min_date and start_date <= max_date:
            admin_spans.append(
                {
                    "start": max(start_date, min_date),
                    "end": min(end_date, max_date),
                    "party": election["party"],
                    "president": election["president"],
                }
            )

    # Add shaded regions for each administration
    for span in admin_spans:
        color = "blue" if span["party"] == "Democratic" else "red"
        alpha = 0.2
        rect = Rectangle(
            (mdates.date2num(span["start"]), 0),
            mdates.date2num(span["end"]) - mdates.date2num(span["start"]),
            1,
            transform=ax.get_xaxis_transform(),
            color=color,
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(rect)

    # Add vertical lines for election dates
    for i, election in elections.iterrows():
        if (
            election["election_date"] >= min_date
            and election["election_date"] <= max_date
        ):
            ax.axvline(
                x=election["election_date"], color="gray", linestyle="--", alpha=0.7
            )

            # Add president name and party label
            color = "blue" if election["party"] == "Democratic" else "red"
            ax.text(
                election["election_date"],
                ax.get_ylim()[1] * 0.00001,
                f"{election['president']} ({election['party'][0]})",
                rotation=90,
                verticalalignment="bottom",
                color=color,
            )

    # Set labels and title
    ax.set_title(
        "Bitcoin Price Trends During U.S. Presidential Administrations",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Bitcoin Price (USD, Log Scale)", fontsize=12)

    # Format x-axis date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Removed peak annotations as requested

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend with Republican and Democratic indicators
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.2, label="Republican Administration"),
        Patch(facecolor="blue", alpha=0.2, label="Democratic Administration"),
        plt.Line2D([0], [0], color="k", linewidth=1.5, label="Bitcoin Price"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig("../output/7b.US_Elections.jpg", bbox_inches="tight", dpi=350)
    plt.show()
