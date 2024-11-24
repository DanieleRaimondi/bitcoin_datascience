import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm
import numpy as np

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_data


def load_bitcoin_data():
    """
    Load Bitcoin price data from a predefined fetch function.

    Returns:
        pandas.DataFrame: Bitcoin price data with datetime index
    """
    btc = fetch_data("btc")[["time", "PriceUSD"]]
    btc["time"] = pd.to_datetime(btc["time"])
    return btc


def load_election_probabilities():
    """
    Load US election probability data from a CSV file.

    Returns:
        pandas.DataFrame: Election probability data with datetime index
    """
    us_data = pd.read_csv("../data/polymarket_US_election_24.csv")
    us_data = us_data.drop(columns=["Timestamp (UTC)"])
    us_data = us_data.rename(columns={"Date (UTC)": "time"})
    us_data["time"] = pd.to_datetime(
        us_data["time"], format="%m-%d-%Y %H:%M"
    ).dt.normalize()
    return us_data


def merge_data(us_data, btc_data):
    """
    Merge Bitcoin price and election probability data.

    Args:
        us_data (pandas.DataFrame): Election probability data
        btc_data (pandas.DataFrame): Bitcoin price data

    Returns:
        pandas.DataFrame: Merged dataset with time as index
    """
    merged_data = pd.merge(
        us_data, btc_data[["time", "PriceUSD"]], on="time", how="inner"
    )
    merged_data["time_numeric"] = merged_data["time"].astype("int64") // 10**9
    return merged_data


def apply_loess_smoothing(data, column, time_column, frac=0.05):
    """
    Apply LOESS smoothing to a data column.

    Args:
        data (pandas.DataFrame): Input dataframe
        column (str): Column to smooth
        time_column (str): Time column for smoothing
        frac (float, optional): Smoothing fraction. Defaults to 0.05.

    Returns:
        numpy.ndarray: Smoothed data interpolated to original time points
    """
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
    """
    Create a multi-axis plot showing Bitcoin price and US election probabilities.

    Args:
        merged_data (pandas.DataFrame): Merged dataset containing time, BTC price,
            and election probabilities.
        include_trump (bool): If True, includes Donald Trump's probability in the plot.
        include_biden (bool): If True, includes Joe Biden's probability in the plot.
        include_harris (bool): If True, includes Kamala Harris's probability in the plot.
        include_democrats (bool): If True, includes Democrats' combined probability in the plot.
        apply_loess (bool): If True, applies LOESS smoothing to all data.
        frac (float): Fraction of data used for LOESS smoothing. Defaults to 0.05.

    Returns:
        None. Displays a matplotlib plot.
    """
    # Create the main figure and primary axis for Bitcoin price
    fig, ax1 = plt.subplots(figsize=(12, 6))
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
        alpha=0.3 if apply_loess else 1.0,
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
            linewidth=2,
        )

    # Configuration for probability plots
    prob_configs = [
        (include_trump, "Donald Trump", "red"),
        (include_biden, "Joe Biden", "orange"),
        (include_harris, "Kamala Harris", "green"),
    ]

    # Initial offset for secondary axes
    offset_increment = 0.07  # Distance between axes
    current_offset = 1.0  # Start just after the primary axis

    # Plot individual candidate probabilities
    for include, column, color in prob_configs:
        if include:
            # Create additional y-axis
            ax = ax1.twinx()
            ax.spines["right"].set_position(("axes", current_offset))
            current_offset += offset_increment
            ax.spines["right"].set_visible(True)

            # Configure y-axis for this candidate
            ax.set_ylabel(f"{column} Probability", color=color)
            ax.tick_params(axis="y", colors=color)

            # Plot raw probability data
            ax.plot(
                merged_data["time"],
                merged_data[column],
                color=color,
                label=f"{column} Probability",
                alpha=0.3 if apply_loess else 1.0,
            )

            # Apply and plot LOESS smoothing
            if apply_loess:
                smoothed_prob = apply_loess_smoothing(
                    merged_data, column, "time_numeric", frac
                )
                ax.plot(
                    merged_data["time"],
                    smoothed_prob,
                    color=color,
                    linewidth=2,
                )

    # Plot Democrat combined probability if requested
    if include_democrats:
        # Create additional y-axis for Democrats
        ax = ax1.twinx()
        ax.spines["right"].set_position(("axes", current_offset))
        current_offset += offset_increment
        ax.spines["right"].set_visible(True)

        # Configure y-axis for Democrats
        ax.set_ylabel("Democrats Probability", color="blue")
        ax.tick_params(axis="y", colors="blue")

        # Calculate Democrat probability (Biden + Harris)
        democrat_prob = merged_data["Joe Biden"] + merged_data["Kamala Harris"]

        # Plot raw Democrat probability
        ax.plot(
            merged_data["time"],
            democrat_prob,
            color="blue",
            label="Democrats Probability",
            alpha=0.3 if apply_loess else 1.0,
        )

        # Apply and plot LOESS smoothing
        if apply_loess:
            smoothed_dem_prob = apply_loess_smoothing(
                pd.DataFrame(
                    {
                        "democrat_prob": democrat_prob,
                        "time_numeric": merged_data["time_numeric"],
                    }
                ),
                "democrat_prob",
                "time_numeric",
                frac,
            )
            ax.plot(
                merged_data["time"],
                smoothed_dem_prob,
                color="blue",
                linewidth=2,
            )

    plt.title(
        "BTC Price and US Presidential Election Winner 2024",
        fontweight="bold",
        fontsize=16,
    )
    plt.savefig("../output/8.BTCvsUSELECTIONS.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def main():
    """
    Main function to load data and create visualization.
    """
    # Append project functions path
    sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
    from fetch_data import fetch_data

    # Load and merge data
    btc_data = load_bitcoin_data()
    election_data = load_election_probabilities()
    merged_data = merge_data(election_data, btc_data)

    # Create visualization
    plot_data(merged_data)


if __name__ == "__main__":
    main()
