from fredapi import Fred
import os
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

load_dotenv()


def plot_btc_economics(
    include_10yr_interest=True,
    include_federal_funds=True,
    include_m2=False,
    include_inflation=False,
    include_sma=False,
    sma_period=365,
):
    """
    Plots Bitcoin price against interest rates, Federal Funds Rate, M2 money supply, and inflation rate based on user preferences.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'time' and 'PriceUSD' columns for Bitcoin prices.
    include_10yr_interest (bool): Whether to include the 10-year interest rates in the plot.
    include_federal_funds (bool): Whether to include the Federal Funds Rate in the plot.
    include_m2 (bool): Whether to include M2 money supply in the plot.
    include_inflation (bool): Whether to include inflation rate in the plot.
    include_sma (bool): Whether to include the SMA for the 10-year interest rates.
    sma_period (int): The period for the SMA (if included).
    """
    # Initialize connection to FRED with your API key
    fred = Fred(api_key=os.getenv("FRED_API"))

    # Load Bitcoin price data from online CSV
    df = pd.read_csv(
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
        parse_dates=["time"],
        low_memory=False,
    )
    # Convert 'time' column to datetime if not already
    df["time"] = pd.to_datetime(df["time"])
    df.dropna(inplace=True, subset=["PriceUSD"])
    # Define start and end dates
    end_date = datetime.now()
    start_date = df["time"].min()  # Use the start date from your DataFrame

    # Create the plot with one y-axis for Bitcoin price
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Bitcoin price (left y-axis, logarithmic scale)
    color = "tab:orange"
    ax1.semilogy(df["time"], df["PriceUSD"], color=color, label="BTC Price")
    ax1.tick_params(axis="y", labelcolor=color)

    # Format y-axis for Bitcoin price
    def format_btc_price(x, p):
        return f"{x:,.0f}"

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_btc_price))

    # Create a second y-axis for the 10-year interest rates if included
    if include_10yr_interest:
        symbol = "^TNX"
        interest_data = yf.download(symbol, start=start_date, end=end_date)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.plot(
            interest_data.index,
            interest_data["Close"],
            color=color,
            label="10-Year Interest Rate (%)",
            alpha=0.7,
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Add SMA for 10-year interest rates if included
        if include_sma:
            interest_data["SMA"] = (
                interest_data["Close"].rolling(window=sma_period).mean()
            )
            ax2.plot(
                interest_data.index,
                interest_data["SMA"],
                color="tab:cyan",
                linestyle="--",
                label=f"SMA {sma_period} days",
            )

    # Add a third y-axis for the Federal Funds Rate if included
    if include_federal_funds:
        ffr_data = fred.get_series("FEDFUNDS", start_date, end_date)

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 17))
        color = "tab:green"
        ax3.plot(
            ffr_data.index,
            ffr_data,
            color=color,
            label="Federal Funds Rate (%)",
            alpha=0.7,
        )
        ax3.tick_params(axis="y", labelcolor=color)

    # Add a fourth y-axis for M2 money supply (M2SL) if included
    if include_m2:
        m2_data = fred.get_series("M2SL", start_date, end_date)

        # Convert M2 from billions to trillions
        m2_data_trillions = m2_data / 1000

        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 35))
        color = "purple"
        ax4.plot(
            m2_data_trillions.index,
            m2_data_trillions,
            color=color,
            label="M2 Money Supply (Trillion USD)",
            alpha=0.7,
        )
        ax4.tick_params(axis="y", labelcolor=color)

        # Format y-axis for M2 in trillion USD
        def format_trillions(x, p):
            return f"{x:.0f}"

        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(format_trillions))

    # Add a fifth y-axis for the inflation rate if included
    if include_inflation:
        cpi_data = fred.get_series(
            "CPIAUCNS", start_date, end_date
        )  # Consumer Price Index
        inflation_rate = (
            cpi_data.pct_change(periods=12) * 100
        )  # Year-over-year percentage change

        ax5 = ax1.twinx()
        ax5.spines["right"].set_position(("outward", 58))
        color = "brown"
        ax5.plot(
            inflation_rate.index,
            inflation_rate,
            color=color,
            label="Inflation Rate (%)",
            alpha=0.7,
        )
        ax5.tick_params(axis="y", labelcolor=color)

    # Title and legend
    plt.title(
        "BTC Price vs (Interest Rates, M2 and Inflation Rate)",
        fontweight="bold",
        fontsize=18,
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

    # Set x-axis limits to reduce empty space
    ax1.set_xlim(start_date, end_date)

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # Remove x and y axis labels
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    if include_10yr_interest:
        ax2.set_ylabel("")
    if include_federal_funds:
        ax3.set_ylabel("")
    if include_m2:
        ax4.set_ylabel("")
    if include_inflation:
        ax5.set_ylabel("")

    # Adjust layout to reduce overall whitespace
    plt.tight_layout()
    plt.savefig("../output/3a.Economics.jpg", dpi=400)
    # Show the plot
    plt.show()
