import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter


def load_dxy_data():
    """
    This function extracts data related to dollar index up to today's date and returns the closing prices as a DataFrame.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    data = yf.download("DX-Y.NYB", start="2000-01-01", end=today)
    dxy = data[["Close"]].rename(columns={"Close": "DXY"})

    return dxy


def load_btc_data():
    btc = pd.read_csv(
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
        parse_dates=["time"],
    )
    btc = btc.dropna(subset=["PriceUSD"]).reset_index(drop=True)[["time", "PriceUSD"]]
    btc.rename(columns={"time": "Date"}, inplace=True)
    btc.set_index("Date", inplace=True)
    return btc


# Funzione per plottare LOESS colorata basata sulla derivata
def plot_colored_loess(
    ax, df, column, deriv_column, color_up="green", color_down="red", lw=2
):
    for i in range(1, len(df)):
        if df[deriv_column].iloc[i] > 0:
            ax.plot(
                df.index[i - 1 : i + 1],
                df[column].iloc[i - 1 : i + 1],
                color=color_up,
                lw=lw,
            )
        elif df[deriv_column].iloc[i] < 0:
            ax.plot(
                df.index[i - 1 : i + 1],
                df[column].iloc[i - 1 : i + 1],
                color=color_down,
                lw=lw,
            )


def plot_colored_loess(
    ax, df, column, deriv_column, color_up="green", color_down="red", lw=2
):
    for i in range(1, len(df)):
        if df[deriv_column].iloc[i] > 0:
            ax.plot(
                df.index[i - 1 : i + 1],
                df[column].iloc[i - 1 : i + 1],
                color=color_up,
                lw=lw,
            )
        elif df[deriv_column].iloc[i] < 0:
            ax.plot(
                df.index[i - 1 : i + 1],
                df[column].iloc[i - 1 : i + 1],
                color=color_down,
                lw=lw,
            )


# Funzione principale per plottare i modelli
def plot_models(df, tops_dates, bottoms_dates, startbull_dates):
    """
    Plots colored LOESS based on the derivative.

    Parameters:
    ax: Axes object for plotting.
    df: DataFrame containing the data.
    column: Column name for the data to be plotted.
    deriv_column: Column name for the derivative data.
    color_up: Color for positive derivative values (default is "green").
    color_down: Color for negative derivative values (default is "red").
    lw: Line width for the plot (default is 2).
    """
    # LOESS derivative
    df["PriceUSD_LOESS_DERIV"] = df["PriceUSD_LOESS"].diff()
    df["DXY_LOESS_DERIV"] = df["DXY_LOESS"].diff()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot PriceUSD in log scale
    ax1.set_ylabel("PriceUSD", color="tab:blue")
    ax1.plot(
        df.index, df["PriceUSD"], color="tab:blue", label="PriceUSD", lw=1.5, alpha=0.5
    )
    plot_colored_loess(ax1, df, "PriceUSD_LOESS", "PriceUSD_LOESS_DERIV", lw=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")
    ax1.set_ylim([df["PriceUSD"].min(), df["PriceUSD"].max()])
    ax1.set_xlim([df.index.min(), df.index.max()])

    # Set formatter to prevent scientific notation
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(style="plain", axis="y")

    # Create a second y-axis to plot DXY
    ax2 = ax1.twinx()
    ax2.set_ylabel("DXY", color="tab:orange")
    ax2.plot(df.index, df["DXY"], color="tab:orange", label="DXY", lw=1.5, alpha=0.5)
    plot_colored_loess(ax2, df, "DXY_LOESS", "DXY_LOESS_DERIV", lw=2)
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Fill areas between tops and bottoms with red color
    for top, bottom in zip(tops_dates[:-1], bottoms_dates):
        ax1.axvspan(top, bottom, color="red", alpha=0.15)
        mid_date = top + (bottom - top) / 2
        ax1.text(
            mid_date,
            0.03,
            "down \ntrend",
            color="red",
            fontsize=8,
            ha="center",
            va="center",
            transform=ax1.get_xaxis_transform(),
        )

    # Fill areas between startbull_dates and the next top date with green color
    for startbull in startbull_dates:
        next_top_index = tops_dates[tops_dates > startbull][0]
        ax1.axvspan(startbull, next_top_index, color="green", alpha=0.15)
        mid_date = startbull + (next_top_index - startbull) / 2
        ax1.text(
            mid_date,
            0.03,
            "final \nbull",
            color="green",
            fontsize=8,
            ha="center",
            va="center",
            transform=ax1.get_xaxis_transform(),
        )

    # Fill remaining areas with orange color
    for i in range(len(bottoms_dates)):
        if i < len(startbull_dates) - 1:  # Ensure not to go out of range
            start = bottoms_dates[i]
            end = startbull_dates[i + 1]
            ax1.axvspan(start, end, color="orange", alpha=0.15)
            mid_date = start + (end - start) / 2
            ax1.text(
                mid_date,
                0.03,
                "Initial \nbull",
                color="orange",
                fontsize=8,
                ha="center",
                va="center",
                transform=ax1.get_xaxis_transform(),
            )

    # Title and show plot
    plt.title("PriceUSD vs DXY", fontweight="bold", fontsize=20)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.savefig("../output/3.DXY.jpg",bbox_inches="tight",dpi=350,)
    plt.show()


# Add LOESS curves
def add_loess(df, column, frac=0.05):
    loess = sm.nonparametric.lowess
    loess_fit = loess(df[column], df.index, frac=frac)
    return pd.Series(loess_fit[:, 1], index=df.index, name=f"{column}_LOESS")
