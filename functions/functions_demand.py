import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_btc_data():
    """
    Loads Bitcoin price data from a CSV file, parses the 'time' column as dates, removes rows with missing values in the 'PriceUSD' column,
    and resets the index to 'time'.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded and prepared Bitcoin price data.
    """
    btc = pd.read_csv(
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
        parse_dates=["time"],
    )
    btc = btc.dropna(subset=["PriceUSD"]).reset_index(drop=True)[["time", "AdrActCnt", "TxCnt", "PriceUSD"]]
    btc.set_index("time", inplace=True)
    return btc


def add_loess(df, column, frac=0.03):
    """
    Applies the LOESS (Locally Weighted Scatterplot Smoothing) method to the specified column of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column name on which LOESS will be applied.
    frac (float, optional): The fraction of data points to use in each local regression. Defaults to 0.03.

    Returns:
    pd.Series: A pandas Series containing the LOESS smoothed values based on the specified column.
    """
    loess = sm.nonparametric.lowess
    loess_fit = loess(df[column], df.index, frac=frac)
    return pd.Series(loess_fit[:, 1], index=df.index, name=f"{column}_LOESS")


def generate_plot(df):
    """
    Generates a combined plot showing Price USD, Active Addresses, and Transaction Count.
    The plot includes LOESS regression lines for Transaction Count and Active Addresses.
    Saves the plot as "Demand.jpg" in the output folder.
    """
    # LOESS regressionaddition
    df["TxCnt_LOESS"] = add_loess(df, "TxCnt")
    df["AdrActCnt_LOESS"] = add_loess(df, "AdrActCnt")

    # combined plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # BTC Price
    ax1.plot(
        df.index,
        df["PriceUSD"],
        label="Price USD",
        color="blue",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Price USD", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Active Addresses
    ax2 = ax1.twinx()
    ax2.plot(
        df.index,
        df["AdrActCnt"],
        color="orange",
        alpha=0.5,
        lw=0.3,
    )
    ax2.plot(
        df.index,
        df["AdrActCnt_LOESS"],
        label="Active Addresses (LOESS)",
        color="orange",
        linewidth=2.5,
    )
    ax2.set_ylabel("Active Addresses", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    #ax2.set_yscale("log")

    # Transaction Count
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(df.index, df["TxCnt"], color="green", alpha=0.5, lw=0.3)
    ax3.plot(
        df.index,
        df["TxCnt_LOESS"],
        label="Transaction Count (LOESS)",
        color="green",
        linewidth=2.5,
    )
    ax3.set_ylabel("Transaction Count", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    #ax3.set_yscale("log")
    
    # Formatting
    fig.tight_layout()
    ax1.grid(True, ls= "--", alpha=0.5, lw = 1)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title(
        "BTCUSD vs Active Addresses and Transaction Count",
        fontweight="bold",
        fontsize=20,
    )
    plt.savefig("../output/6.Demand.jpg", bbox_inches="tight", dpi=350)
    plt.show()
