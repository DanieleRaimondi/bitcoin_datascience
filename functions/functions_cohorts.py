import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def plot_cohorts(data, price_column, coin):
    """
    Plots multiple charts for all relevant Bitcoin balance cohorts with LOESS smoothing, starting from the first day
    with a valued price. Each chart shows the number of addresses in the cohort, their LOESS smoothed curve, and
    the Bitcoin price on a logarithmic scale.

    Parameters:
    data (DataFrame): The dataframe containing the data.
    price_column (str): The column name for the price data.
    """
    # Filter data to start from the first day with a valued price
    data_filtered = data[data[price_column].notnull()].copy()
    data_filtered["time"] = pd.to_datetime(data_filtered["time"])
    data_filtered["time_numeric"] = data_filtered["time"].apply(lambda x: x.toordinal())

    # Defining all cohorts based on available columns
    cohorts = [
        ("AdrBalNtv1MCnt", f"9_more than 1M {coin}"),
        ("AdrBalNtv100KCnt", f"8_100K to 1M {coin}"),
        ("AdrBalNtv10KCnt", f"7_10K to 100K {coin}"),
        ("AdrBalNtv1KCnt", f"6_1K to 10K {coin}"),
        ("AdrBalNtv100Cnt", f"5_100 to 1K {coin}"),
        ("AdrBalNtv10Cnt", f"4_10 to 100 {coin}"),
        ("AdrBalNtv1Cnt", f"3_1 to 10 {coin}"),
        ("AdrBalNtv0.1Cnt", f"2_0.1 to 1 {coin}"),
        ("AdrBalNtv0.01Cnt", f"1_0.01 to 0.1 {coin}"),
        ("AdrBalNtv0.001Cnt", f"0_0.001 to 0.01 {coin}"),
    ]

    for cohort_column, cohort_label in cohorts:
        fig, ax1 = plt.subplots(figsize=(15, 5))

        # Number of Addresses
        ax1.set_ylabel("Number of Addresses", color="tab:orange")
        ax1.plot(
            data_filtered["time"],
            data_filtered[cohort_column],
            color="tab:orange",
            alpha=0.6,
            linewidth=1.5,
        )
        ax1.tick_params(axis="y", labelcolor="tab:orange")

        # LOESS Smoothing
        loess_model = sm.nonparametric.lowess(
            data_filtered[cohort_column], data_filtered["time_numeric"], frac=0.03
        )
        loess_time = [pd.Timestamp.fromordinal(int(time)) for time, _ in loess_model]
        loess_value = [value for _, value in loess_model]
        ax1.plot(
            loess_time,
            loess_value,
            color="red",
            label="LOESS Smoothed Curve",
            linewidth=0.75, alpha=0.5
        )

        # Bitcoin Price
        ax2 = ax1.twinx()
        ax2.set_ylabel("Price USD (log scale)", color="black")
        ax2.plot(
            data_filtered["time"],
            data_filtered[price_column],
            color="black",
            linewidth=1.5,
        )
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="black")

        # Customizing y-axis ticks to avoid scientific notation
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

        plt.title(f"{coin} Addresses with {cohort_label[2:]}")
        plt.grid(ls="--", alpha=0.5)

        cohort_filename = (
            f"{cohort_label.replace(' ', '_').replace(',', '').replace('.', '_')}.jpeg"
        )
        plt.savefig(f"../output/{coin}_Cohorts//{cohort_filename}", dpi=400)
        plt.show()


import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np


def plot_all_cohorts(data, price_column, coin):
    """
    Plots a single chart for all relevant Bitcoin balance cohorts with LOESS smoothing, starting from 2012.
    The chart shows the number of addresses in each cohort, their LOESS smoothed curves, and
    the Bitcoin price on a logarithmic scale. Only the price y-axis is visible.

    Parameters:
    data (DataFrame): The dataframe containing the data.
    price_column (str): The column name for the price data.
    coin (str): The name of the cryptocurrency (e.g., 'BTC' or 'ETH').
    """
    # Filter data to start from 2012 and from the first day with a valued price
    data_filtered = data[
        (data["time"] >= "2012-01-01") & (data[price_column].notnull())
    ].copy()
    data_filtered["time"] = pd.to_datetime(data_filtered["time"])
    data_filtered["time_numeric"] = data_filtered["time"].apply(lambda x: x.toordinal())

    # Combine the specified cohorts
    data_filtered["AdrBalNtv10KPlusCnt"] = (
        data_filtered["AdrBalNtv1MCnt"]
        + data_filtered["AdrBalNtv100KCnt"]
        + data_filtered["AdrBalNtv10KCnt"]
    )
    data_filtered["AdrBalNtv0.001to1Cnt"] = (
        data_filtered["AdrBalNtv0.1Cnt"]
        + data_filtered["AdrBalNtv0.01Cnt"]
        + data_filtered["AdrBalNtv0.001Cnt"]
    )

    # Defining all cohorts based on available columns
    cohorts = [
        ("AdrBalNtv10KPlusCnt", f"9_more than 10K {coin}"),
        ("AdrBalNtv1KCnt", f"6_1K to 10K {coin}"),
        ("AdrBalNtv100Cnt", f"5_100 to 1K {coin}"),
        ("AdrBalNtv10Cnt", f"4_10 to 100 {coin}"),
        ("AdrBalNtv1Cnt", f"3_1 to 10 {coin}"),
        ("AdrBalNtv0.001to1Cnt", f"2_0.001 to 1 {coin}"),
    ]

    # Create the main figure and axis
    fig, ax_price = plt.subplots(figsize=(15, 10))

    # Plot Bitcoin Price
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price USD (log scale)", color="black")
    ax_price.plot(
        data_filtered["time"],
        data_filtered[price_column],
        color="black",
        linewidth=1.5,
        label="Price",
    )
    ax_price.set_yscale("log")
    ax_price.tick_params(axis="y", labelcolor="black")
    ax_price.yaxis.set_major_formatter(ScalarFormatter())
    ax_price.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Create a list to store all axes
    axes = [ax_price]

    # Colors for different cohorts
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cohorts)))

    # Plot each cohort
    for i, (cohort_column, cohort_label) in enumerate(cohorts):
        # Create a new axis for each cohort
        ax = ax_price.twinx()
        ax.spines["right"].set_visible(False)  # Hide the right spine
        axes.append(ax)

        # Plot number of addresses
        ax.plot(
            data_filtered["time"],
            data_filtered[cohort_column],
            color=colors[i],
            alpha=0.6,
            linewidth=1.5,
            label=cohort_label[2:],
        )

        # LOESS Smoothing
        loess_model = sm.nonparametric.lowess(
            data_filtered[cohort_column], data_filtered["time_numeric"], frac=0.03
        )
        loess_time = [pd.Timestamp.fromordinal(int(time)) for time, _ in loess_model]
        loess_value = [value for _, value in loess_model]
        ax.plot(
            loess_time,
            loess_value,
            color=colors[i],
            linestyle="--",
            linewidth=0.75,
            alpha=0.1,
        )

        ax.set_yscale("log")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

    # Adjust the layout and add a title
    plt.title(f"All {coin} Address Cohorts and Price", fontsize=16)

    # Add a legend inside the chart at the bottom right
    lines = []
    labels = []
    for ax in axes:
        axline, axlabel = ax.get_legend_handles_labels()
        lines.extend(axline)
        labels.extend(axlabel)

    # Reverse the order of lines and labels to match the chart order
    lines = lines[::-1]
    labels = labels[::-1]

    plt.legend(
        lines,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.05),
        ncol=2,  # Adjust the number of columns as needed
        fontsize="small",  # Adjust font size as needed
        title="Cohorts",
        title_fontsize="small",
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"../output/{coin}_Cohorts//{coin}_All_Cohorts.jpeg", dpi=400, bbox_inches="tight")
    plt.show()
