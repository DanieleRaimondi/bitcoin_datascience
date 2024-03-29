import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def plot_cohorts(data, price_column):
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
        ("AdrBalNtv10KCnt", "More than 10,000 BTC"),
        ("AdrBalNtv1KCnt", "1,000 to 10,000 BTC"),
        ("AdrBalNtv100Cnt", "100 to 1,000 BTC"),
        ("AdrBalNtv10Cnt", "10 to 100 BTC"),
        ("AdrBalNtv1Cnt", "1 to 10 BTC"),
        ("AdrBalNtv0.1Cnt", "0.1 to 1 BTC"),
        ("AdrBalNtv0.01Cnt", "0.01 to 0.1 BTC"),
        ("AdrBalNtv0.001Cnt", "0.001 to 0.01 BTC"),
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

        plt.title(f"Bitcoin Addresses with {cohort_label}")
        plt.legend()
        plt.grid(True)

        cohort_filename = (f"{cohort_label.replace(' ', '_').replace(',', '').replace('.', '')}.jpeg")
        plt.savefig(f"../output/BTC_Cohorts//{cohort_filename}", dpi=400)
        plt.show()
