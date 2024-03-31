import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


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
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Addresses", color="tab:orange")
        ax1.plot(
            data_filtered["time"],
            data_filtered[cohort_column],
            color="tab:orange",
            alpha=0.5,
            linewidth=1.5, 
        )
        ax1.tick_params(axis="y", labelcolor="tab:orange")

        # LOESS Smoothing
        loess_model = sm.nonparametric.lowess(
            data_filtered[cohort_column], data_filtered["time_numeric"], frac=0.05
        )
        loess_time = [pd.Timestamp.fromordinal(int(time)) for time, _ in loess_model]
        loess_value = [value for _, value in loess_model]
        ax1.plot(
            loess_time,
            loess_value,
            color="red",
            label="LOESS Smoothed Curve",
            linewidth=1, 
        )

        # Bitcoin Price
        ax2 = ax1.twinx()
        ax2.set_ylabel("Price USD (log scale)", color="black")
        ax2.plot(
            data_filtered["time"],
            data_filtered[price_column],
            color="black",
            linewidth=1, 
        )
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="black")

        plt.title(f"{coin} Addresses with {cohort_label[2:]}")
        plt.grid(True)

        cohort_filename = (f"{cohort_label.replace(' ', '_').replace(',', '').replace('.', '_')}.jpeg")
        plt.savefig(f"../output/{coin}_Cohorts//{cohort_filename}", dpi=400)
        plt.show()
