import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def plot_cohorts(data, price_column, coin):
    """
    Plots multiple charts for all relevant balance cohorts, starting from the first day
    with a valued price. Each chart shows the number of addresses in the cohort and
    the price on a logarithmic scale.

    Parameters:
    data (DataFrame): The dataframe containing the data.
    price_column (str): The column name for the price data.
    coin (str): The cryptocurrency symbol.
    """
    data_filtered = data[data[price_column].notnull()].copy()
    data_filtered["time"] = pd.to_datetime(data_filtered["time"])

    cohorts = [
        ("AdrBalNtv1MCnt", f"More than 1M {coin}"),
        ("AdrBalNtv100KCnt", f"100K to 1M {coin}"),
        ("AdrBalNtv10KCnt", f"10K to 100K {coin}"),
        ("AdrBalNtv1KCnt", f"1K to 10K {coin}"),
        ("AdrBalNtv100Cnt", f"100 to 1K {coin}"),
        ("AdrBalNtv10Cnt", f"10 to 100 {coin}"),
        ("AdrBalNtv1Cnt", f"1 to 10 {coin}"),
        ("AdrBalNtv0.1Cnt", f"0.1 to 1 {coin}"),
        ("AdrBalNtv0.01Cnt", f"0.01 to 0.1 {coin}"),
        ("AdrBalNtv0.001Cnt", f"0.001 to 0.01 {coin}"),
    ]

    for cohort_column, cohort_label in cohorts:
        fig, ax1 = plt.subplots(figsize=(15, 6))

        ax1.set_ylabel("Number of Addresses", color="tab:orange", fontsize=12)
        ax1.plot(
            data_filtered["time"],
            data_filtered[cohort_column],
            color="tab:orange",
            linewidth=1.5,
            label="Addresses",
        )
        ax1.tick_params(axis="y", labelcolor="tab:orange")
        ax1.grid(ls="--", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Price USD (log scale)", color="black", fontsize=12)
        ax2.plot(
            data_filtered["time"],
            data_filtered[price_column],
            color="black",
            linewidth=1.5,
            label="Price",
        )
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.title(f"{coin} Addresses: {cohort_label}", fontsize=14, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        cohort_filename = f"{cohort_label.replace(' ', '_').replace(',', '')}.jpeg"
        plt.tight_layout()
        plt.savefig(
            f"../output/Cohorts_{coin}/{cohort_filename}", dpi=400, bbox_inches="tight"
        )
        plt.show()


def plot_all_cohorts(data, price_column, coin):
    """
    Plots a single chart for all relevant balance cohorts starting from 2012.
    Each cohort has its own y-axis on the right side with matching colors.
    The price is displayed on the left axis.

    Parameters:
    data (DataFrame): The dataframe containing the data.
    price_column (str): The column name for the price data.
    coin (str): The cryptocurrency symbol.
    """
    data_filtered = data[
        (data["time"] >= "2012-01-01") & (data[price_column].notnull())
    ].copy()
    data_filtered["time"] = pd.to_datetime(data_filtered["time"])

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

    cohorts = [
        ("AdrBalNtv10KPlusCnt", f"More than 10K {coin}"),
        ("AdrBalNtv1KCnt", f"1K to 10K {coin}"),
        ("AdrBalNtv100Cnt", f"100 to 1K {coin}"),
        ("AdrBalNtv10Cnt", f"10 to 100 {coin}"),
        ("AdrBalNtv1Cnt", f"1 to 10 {coin}"),
        ("AdrBalNtv0.001to1Cnt", f"0.001 to 1 {coin}"),
    ]

    fig, ax_price = plt.subplots(figsize=(18, 9))

    ax_price.set_ylabel("Price USD (log scale)", fontsize=11, color="black")

    ax_price.plot(
        data_filtered["time"],
        data_filtered[price_column],
        color="black",
        linewidth=1.5,
        label="Price",
        zorder=10,
    )
    ax_price.set_yscale("log")
    ax_price.tick_params(axis="y", labelsize=10, labelcolor="black")
    ax_price.yaxis.set_major_formatter(ScalarFormatter())
    ax_price.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_price.grid(ls="--", alpha=0.3, zorder=0)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(cohorts)))

    spacing = 0.02

    for i, (cohort_column, cohort_label) in enumerate(cohorts):
        ax = ax_price.twinx()

        ax.spines["right"].set_position(("axes", 1 + i * spacing))
        ax.spines["right"].set_color(colors[i])
        ax.spines["right"].set_linewidth(0.8)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.plot(
            data_filtered["time"],
            data_filtered[cohort_column],
            color=colors[i],
            linewidth=1,
            label=cohort_label,
            alpha=0.7,
        )

        ax.set_ylabel("")
        ax.tick_params(axis="y", labelcolor=colors[i], labelsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.title(
        f"All {coin} Address Cohorts and Price", fontsize=16, fontweight="bold", pad=20
    )

    lines = [ax_price.get_lines()[0]]
    labels = ["Price"]
    for i, (_, cohort_label) in enumerate(cohorts):
        lines.append(plt.Line2D([0], [0], color=colors[i], linewidth=2))
        labels.append(cohort_label)

    ax_price.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.95)

    fig.subplots_adjust(left=0.07, right=0.88)

    plt.savefig(
        f"../output/Cohorts_{coin}/{coin}_All_Cohorts.jpeg",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()
