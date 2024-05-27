import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker
import statsmodels.api as sm
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def load_dataframe():
    """
    Loads and merges Bitcoin price data from a CSV file and MVRV ratio data from a JSON file.
    It processes these datasets by parsing dates, resampling MVRV data to get daily medians,
    and merging the two datasets based on their timestamps. Additionally, it calculates the
    standard deviation of MVRV over a rolling window and normalizes the MVRV values by this
    standard deviation.

    Returns:
    tuple: Contains three pandas DataFrames:
      - df: DataFrame with original Bitcoin price data.
      - mvrv: DataFrame with daily median MVRV values.
      - merged: DataFrame that merges the Bitcoin price and normalized MVRV data,
                including additional calculated metrics.
    """
    df = pd.read_csv("/Users/danieleraimondi/bitcoin_datascience/functions/data/btc.csv", parse_dates=["time"])
    df = pd.DataFrame(data=df)

    # Loading the JSON file
    with open("/Users/danieleraimondi/bitcoin_datascience/functions/data/mvrv.json", encoding="utf-8") as file:
        mvrv_data = json.load(file)
    # Converting the data into a DataFrame
    mvrv_df = pd.json_normalize(mvrv_data["mvrv"])
    # Converting Unix timestamps into datetime objects
    mvrv_df["Day"] = pd.to_datetime(mvrv_df["x"], unit="ms")
    mvrv_df.rename(columns={"y": "mvrv"}, inplace=True)
    # Selecting the 'Day' and 'mvrv' columns
    mvrv_df = mvrv_df[["Day", "mvrv"]]
    # Aggregating the data to get the median MVRV for each day
    mvrv = mvrv_df.resample("D", on="Day").median()
    # Transforming dates to YYYY-MM-DD
    mvrv.index = mvrv.index.normalize()
    mvrv.reset_index(inplace=True)

    merged = pd.merge(df[["time", "PriceUSD"]], mvrv, left_on="time", right_on="Day", how="left")

    del merged["Day"]

    merged = merged.fillna(method="ffill")
    merged = merged.dropna()
    merged["mvrvstd"] = merged["mvrv"].rolling(window=365 * 4).std()
    merged["mvrv_norm"] = merged["mvrv"] / merged["mvrvstd"]
    merged = merged[450:]

    return df, mvrv, merged


def plot_btcusd_vs_mvrv(merged, frac=0.02):
    """
    Plots the Bitcoin (BTC) USD price and the Market Value to Realized Value (MVRV) ratio over time.
    It uses a LOWESS (Locally Weighted Scatterplot Smoothing) to smooth the MVRV data. The BTC price is
    plotted on a logarithmic scale on the primary y-axis, and the MVRV ratio, along with its LOWESS
    smoothed line, is plotted on a secondary y-axis also on a logarithmic scale.

    Parameters:
    - merged (DataFrame): A pandas DataFrame that must contain 'time', 'PriceUSD', and 'mvrv' columns.
      'time' should be in a datetime format.
    - frac (float): The fraction of the data to use for each local regression, determining the smoothness
      of the LOWESS curve. Default is 0.02.

    Returns:
    None. The function creates and displays a matplotlib figure showing the two plots.
    """
    lowess = sm.nonparametric.lowess(merged.mvrv, merged.time, frac=frac)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot()

    ax.plot(merged.time, merged["PriceUSD"], color="grey")
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("BTCUSD Price", color="grey", fontsize=14)
    ax.set_yscale("log")
    plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))

    ax2 = ax.twinx()
    ax2.plot(merged.time, merged["mvrv"], color="blue", alpha=0.2)
    ax2.plot(
        merged.time, lowess[:, 1], color="blue", linewidth=2
    )  # Adjusted for lowess output
    ax2.set_ylabel("MVRV", color="blue", fontsize=14)
    ax2.set_yscale("log")
    ax2.set_yticks((0, 0.25, 0.5, 1, 2, 3, 5, 10))
    ax2.set_ylim(0, 10)

    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.suptitle("BTCUSD Price vs MVRV", fontsize=20, fontweight="bold")
    ax2.grid()

    ax2.fill_between(merged.time, 0.5, 2, color="green", alpha=0.1)
    ax2.fill_between(merged.time, 0.75, 1, color="green", alpha=0.6)
    ax2.fill_between(merged.time, 3, 10, color="red", alpha=0.1)
    ax2.fill_between(merged.time, 3, 3.5, color="red", alpha=0.6)

    plt.show()
    # fig.savefig("BTCUSD vs MVRV.jpg", format="jpeg", dpi=100, bbox_inches="tight")


def plot_btcusd_and_mvrv_oscillator(merged, frac=0.02, k=0.015):
    """
    Plots the Bitcoin (BTC) price in USD and the MVRV Oscillator over time. 
    The MVRV Oscillator is calculated using a LOWESS smoothing of the MVRV values and then applying 
    a percentile scaling. Ellipses are added at specified points to highlight particular dates.

    Parameters:
    - merged: DataFrame containing the columns 'time', 'mvrv', and 'PriceUSD'. 'time' should be in datetime format.
    - frac: The fraction of the data used when estimating each y-value in the LOWESS model. Default is 0.02.
    - k: The percentile for scaling the MVRV Oscillator. Default is 0.015.

    Returns:
    None. Displays a plot with two subplots: the log of Bitcoin price over time and the MVRV Oscillator.
    """
    # Calculate LOWESS for MVRV
    merged["mvrv_lowess"] = sm.nonparametric.lowess(
        merged["mvrv"], merged["time"], frac=frac
    )[:, 1]

    # Calculate MVRV Oscillator using Percentile Scaler
    merged["mvrv_OSC"] = (
        merged[["mvrv_lowess"]] - merged[["mvrv_lowess"]].quantile(k)
    ) / (merged[["mvrv_lowess"]].quantile(1 - k) - merged[["mvrv_lowess"]].quantile(k))
    merged["mvrv_OSC"] = merged["mvrv_OSC"].clip(
        lower=0, upper=1
    )  # Force min-max values

    # Plotting
    plt.figure(figsize=(22, 12), dpi=150)

    # Bitcoin Price plot
    ax1 = plt.subplot(5, 5, (1, 20))
    ax1.plot(
        merged.time,
        merged["PriceUSD"],
        label="Bitcoin vs $ Price",
        color="k",
        linewidth=2,
    )
    plt.yscale("log")
    plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))
    plt.title("Bitcoin MVRV Oscillator", fontsize=35, fontweight="bold")
    plt.grid(linewidth=0.7, linestyle="--")
    plt.legend(loc=2, prop={"size": 15})
    plt.xticks(alpha=0)
    plt.xlim(merged["time"].min(), merged["time"].max())

    # MVRV Oscillator plot
    ax2 = plt.subplot(5, 5, (21, 25))
    ax2.plot(
        merged.time,
        merged["mvrv_OSC"] * 100,
        label="MVRV Oscillator",
        color="blue",
        linewidth=2,
    )
    plt.axhline(y=30, color="b", linestyle="--", linewidth=1)
    plt.axhline(y=70, color="b", linestyle="--", linewidth=1)
    plt.fill_between(merged.time, 0, 30, color="green", alpha=0.2)
    plt.fill_between(merged.time, 70, 100, color="red", alpha=0.2)
    plt.yticks([0, 30, 50, 70, 100])
    plt.ylim(0, 100)
    plt.grid(linewidth=0.7, linestyle="--")
    plt.legend(loc=3, prop={"size": 15}, handlelength=0)
    plt.xlim(merged["time"].min(), merged["time"].max())

    # Add ellipses at specified points
    dates = [
        pd.to_datetime("2013-03-19"),
        pd.to_datetime("2013-12-05"),
        pd.to_datetime("2017-06-10"),
        pd.to_datetime("2017-11-26"),
        pd.to_datetime("2021-03-17"),
        pd.to_datetime("2021-10-25"),
        #pd.to_datetime("2024-05-11"),
    ]
    values = [96, 97, 86, 92, 96, 70, 84]
    ellipse_width = 90
    ellipse_height = 23

    for i, (date, value) in enumerate(zip(dates, values)):
        circle_date_index = merged[merged["time"] == date].index
        if not circle_date_index.empty:
            circle_x = merged.loc[circle_date_index[0], "time"]
            ellipse_color = (
                "orange" if i % 2 == 0 else "red"
            )  # Odd indices in orange, even in red
            ellipse = Ellipse(
                (circle_x, value),
                width=ellipse_width,
                height=ellipse_height,
                edgecolor=ellipse_color,
                facecolor="none",
                linewidth=3,
                clip_on=False,  # This allows the ellipse to extend beyond the plot boundaries
            )
            ax2.add_artist(ellipse)

    plt.savefig("../output/5.MVRV_Oscillator.jpg", bbox_inches="tight", dpi=350)
    plt.show()
