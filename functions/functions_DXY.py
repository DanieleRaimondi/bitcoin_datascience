import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter
import sys
import time
import requests
from io import StringIO

from .fetch_data import fetch_crypto_data


def load_dxy_data_fred():
    """
    Alternative DXY loader using FRED (Federal Reserve Economic Data).
    More reliable than Yahoo Finance for economic indices.
    """
    try:
        print("Downloading DXY data from FRED...")

        # FRED API endpoint for Trade Weighted US Dollar Index
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {
            "id": "DTWEXBGS",  # Broad trade-weighted dollar index
            "cosd": "2000-01-01",
            "coed": datetime.today().strftime("%Y-%m-%d"),
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = pd.read_csv(StringIO(response.text))

        # Clean and format data
        data.columns = ["Date", "DXY"]
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.set_index("Date")

        # Remove any rows with missing values
        data = data.dropna()

        # Convert to numeric
        data["DXY"] = pd.to_numeric(data["DXY"], errors="coerce")
        data = data.dropna()

        if len(data) > 100:
            print(f"Successfully loaded {len(data)} DXY data points from FRED")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"DXY range: {data['DXY'].min():.2f} to {data['DXY'].max():.2f}")
            return data
        else:
            raise Exception("Insufficient data from FRED")

    except Exception as e:
        print(f"FRED download failed: {str(e)}")
        return None


def load_dxy_data_yfinance():
    """
    Fallback DXY loader using yfinance with aggressive rate limit handling.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    max_retries = 8
    base_delay = 5

    symbols = ["DX-Y.NYB", "DXY=X", "^DXY"]

    for symbol in symbols:
        print(f"Trying yfinance symbol: {symbol}")

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2**attempt) + random.uniform(5, 15)
                    print(f"Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)

                print(
                    f"Downloading DXY data with {symbol} (attempt {attempt + 1}/{max_retries})..."
                )

                # Create new session for each attempt
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )

                # Try manual ticker approach first
                ticker = yf.Ticker(symbol, session=session)
                data = ticker.history(
                    start="2000-01-01",
                    end=today,
                    auto_adjust=False,
                    prepost=False,
                    actions=False,
                    timeout=30,
                )

                if data is not None and not data.empty:
                    print(
                        f"Downloaded {len(data)} rows with columns: {list(data.columns)}"
                    )

                    # Handle column structure
                    if "Close" in data.columns:
                        result = data[["Close"]].rename(columns={"Close": "DXY"})
                    elif len(data.columns) >= 4:
                        result = data.iloc[:, [3]].rename(
                            columns={data.columns[3]: "DXY"}
                        )
                    else:
                        print(f"Unexpected column structure: {data.columns}")
                        continue

                    # Validate and clean data
                    if len(result) > 100 and not result["DXY"].isna().all():
                        # Timezone fix
                        if result.index.tz is not None:
                            result.index = result.index.tz_localize(None)

                        print(
                            f"Successfully loaded {len(result)} DXY data points from {symbol}"
                        )
                        print(
                            f"Date range: {result.index.min()} to {result.index.max()}"
                        )
                        print(
                            f"DXY range: {result['DXY'].min():.2f} to {result['DXY'].max():.2f}"
                        )
                        return result
                    else:
                        print(f"Invalid data for {symbol}")

                else:
                    print(f"Attempt {attempt + 1}: No data returned for {symbol}")

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed for {symbol}: {error_msg}")

                if (
                    "rate limit" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                ):
                    rate_limit_delay = 60 + random.uniform(30, 60)
                    print(
                        f"Rate limit detected. Waiting {rate_limit_delay:.1f} seconds..."
                    )
                    time.sleep(rate_limit_delay)
                    continue

        print(f"Failed to get data from {symbol}, trying next symbol...")
        time.sleep(30 + random.uniform(10, 20))

    return None


def load_dxy_data():
    """
    Main DXY loader: Try FRED first, then fall back to yfinance if needed.
    """
    # Try FRED first (no rate limits)
    fred_data = load_dxy_data_fred()
    if fred_data is not None:
        return fred_data

    print("FRED failed, falling back to yfinance...")

    # Fallback to yfinance
    yf_data = load_dxy_data_yfinance()
    if yf_data is not None:
        return yf_data

    raise Exception("All DXY data sources failed")


def load_btc_data():
    """
    Load BTC data and ensure timezone-naive index for compatibility.
    """
    btc = fetch_crypto_data("btc")
    btc = btc.dropna(subset=["PriceUSD"]).reset_index(drop=True)[["time", "PriceUSD"]]
    btc.rename(columns={"time": "Date"}, inplace=True)
    btc.set_index("Date", inplace=True)

    # Timezone fix
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)

    # Ensure proper datetime index
    if not isinstance(btc.index, pd.DatetimeIndex):
        btc.index = pd.to_datetime(btc.index)
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

    # Add red area after the last top date until the end of the data
    last_top = tops_dates[-2]  # The last defined top (not 'today')
    last_date = df.index.max()
    ax1.axvspan(last_top, last_date, color="red", alpha=0.15)
    mid_date = last_top + (last_date - last_top) / 2
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


def add_loess(df, column, frac=0.03):
    """
    Adds LOESS smoothing to a data column.
    """
    loess = sm.nonparametric.lowess
    loess_fit = loess(df[column], df.index, frac=frac)
    return pd.Series(loess_fit[:, 1], index=df.index, name=f"{column}_LOESS")
