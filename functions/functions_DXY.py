import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter
import sys
import time  # Missing import

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_crypto_data


def load_dxy_data():
    """
    Extracts data related to dollar index up to today's date and returns the closing prices as a DataFrame.
    Enhanced with rate limit handling.
    """
    today = datetime.today().strftime("%Y-%m-%d")

    max_retries = 5
    base_delay = 10

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (attempt + 1) + random.uniform(0, 5)
                print(f"Rate limited. Waiting {delay:.1f} seconds...")
                time.sleep(delay)

            print(f"Downloading DXY data (attempt {attempt + 1}/{max_retries})...")

            data = yf.download(
                "DX-Y.NYB",
                start="2000-01-01",
                end=today,
                progress=False,
                auto_adjust=False,
            )

            if not data.empty and "Close" in data.columns:
                # Handle both single-level and multi-level column indexes
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data[["Close"]].rename(columns={"Close": "DXY"})
            else:
                print(
                    f"Attempt {attempt + 1}: Downloaded data is empty or missing 'Close' column"
                )
                if attempt < max_retries - 1:
                    continue

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception("Failed to download DXY data after all retries")

    raise Exception("Failed to download DXY data: All attempts returned empty data")


def load_btc_data():
    btc = fetch_crypto_data("btc")
    btc = btc.dropna(subset=["PriceUSD"]).reset_index(drop=True)[["time", "PriceUSD"]]
    btc.rename(columns={"time": "Date"}, inplace=True)
    btc.set_index("Date", inplace=True)
    return btc


def plot_colored_loess(
    ax, df, column, deriv_column, color_up="green", color_down="red", lw=2
):
    """
    Plots colored LOESS based on the derivative.
    """
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


def add_election_markers(ax):
    """
    Adds vertical lines and labels for US presidential elections.
    """
    elections = {
        "2012-11-06": "Obama Re-elected (Dem)",
        "2016-11-08": "Trump Elected (Rep)",
        "2020-11-03": "Biden Elected (Dem)",
        "2024-11-05": "Trump Re-elected (Rep)",
    }

    for date, label in elections.items():
        election_date = pd.to_datetime(date)
        ax.axvline(x=election_date, color="brown", linestyle="--", alpha=0.2)
        ax.text(
            election_date + pd.Timedelta(days=3),
            0.3,
            label,
            rotation=90,
            verticalalignment="top",
            transform=ax.get_xaxis_transform(),
            color="brown",
            fontsize=7,
        )


def plot_models(df, tops_dates, bottoms_dates, startbull_dates):
    """
    Creates the main plot with all components including market phases and election markers.
    """
    # Calculate LOESS derivatives
    df["PriceUSD_LOESS_DERIV"] = df["PriceUSD_LOESS"].diff()
    df["DXY_LOESS_DERIV"] = df["DXY_LOESS"].diff()

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Configure primary axis (PriceUSD)
    ax1.set_ylabel("PriceUSD", color="tab:blue")
    ax1.plot(
        df.index, df["PriceUSD"], color="tab:blue", label="PriceUSD", lw=1.5, alpha=0.5
    )
    plot_colored_loess(ax1, df, "PriceUSD_LOESS", "PriceUSD_LOESS_DERIV", lw=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")
    ax1.set_ylim([df["PriceUSD"].min(), df["PriceUSD"].max() * 1.25])
    ax1.set_xlim([df.index.min(), df.index.max()])
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(style="plain", axis="y")

    # Configure secondary axis (DXY)
    ax2 = ax1.twinx()
    ax2.set_ylabel("DXY", color="tab:orange")
    ax2.plot(df.index, df["DXY"], color="tab:orange", label="DXY", lw=1.5, alpha=0.5)
    plot_colored_loess(ax2, df, "DXY_LOESS", "DXY_LOESS_DERIV", lw=2)
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Add market phase markers
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

    for i in range(len(bottoms_dates)):
        if i < len(startbull_dates) - 1:
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

    # Add election markers
    add_election_markers(ax1)

    # Finalize plot
    plt.title("PriceUSD vs DXY", fontweight="bold", fontsize=20)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.savefig("../output/3b.DXY.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def add_loess(df, column, frac=0.035):
    """
    Adds LOESS smoothing to a data column.
    """
    loess = sm.nonparametric.lowess
    loess_fit = loess(df[column], df.index, frac=frac)
    return pd.Series(loess_fit[:, 1], index=df.index, name=f"{column}_LOESS")
