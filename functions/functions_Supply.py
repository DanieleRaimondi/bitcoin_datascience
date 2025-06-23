import numpy as np  
from prophet import Prophet
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_data

def process_data():
    # Load Bitcoin data
    df = fetch_data("btc")

    # 'time' column in datetime format
    df["time"] = pd.to_datetime(df["time"])
    df["Supply"] = df["CapMrktCurUSD"] / df["PriceUSD"]

    forecast = forecast_supply(df, years=15)
    forecast = lost_coins_estimation(df, forecast)

    return df, forecast


# Funzione per stimare la percentuale di monete perse nel tempo
def estimate_lost_coins_percentage(btc_data,time_series, initial_loss_rate=0.025, decay_rate=0.985):
    """
    Stima la percentuale di monete perse nel tempo.
    :param time_series: Serie temporale delle date.
    :param initial_loss_rate: Tasso percentuale iniziale di monete perse per anno.
    :param decay_rate: Tasso di decrescita del tasso di perdita annuale.
    :return: Percentuale di monete perse nel tempo.
    """
    start_date = btc_data["time"].min()
    years = (time_series - start_date).dt.days / 365.25
    loss_rates = initial_loss_rate * (decay_rate**years)
    lost_coins_percentage = 1 - np.exp(-loss_rates * years)
    return lost_coins_percentage, loss_rates


def forecast_supply(btc_data, years=15):
    """
    Forecast the supply of Bitcoin for future dates using the Prophet model.

    Parameters:
    btc_data (DataFrame): Historical Bitcoin data containing columns "time" and "Supply".
    years (int): Number of years to forecast into the future (default is 15).

    Returns:
    DataFrame: A DataFrame containing the forecasted supply values for future dates.
    """
    # Preparing the DataFrame for Prophet
    df_prophet = btc_data[["time", "Supply"]].rename(columns={"time": "ds", "Supply": "y"})

    # Initialize the Prophet model with a specified carrying capacity
    prophet_model = Prophet(growth="logistic")

    # Set the carrying capacity to 21 million for all future dates
    df_prophet["cap"] = 21000000

    # Fit the model
    prophet_model.fit(df_prophet)

    # Create a DataFrame for future predictions, covering the same range as the original data
    future_dates = prophet_model.make_future_dataframe(periods=365 * years)
    future_dates["cap"] = 21000000

    # Predict
    forecast = prophet_model.predict(future_dates)
    return forecast


def lost_coins_estimation(btc_data,forecast):
    """
    Estimate the lost coins percentage over time and calculate the available supply based on the forecast.

    :param btc_data: Historical Bitcoin data.
    :param forecast: Forecast data including time series and predicted values.
    :return: Updated forecast DataFrame with additional columns for loss rates, lost coins percentage, and available supply.
    """
    forecast["loss_rates"] = estimate_lost_coins_percentage(btc_data, forecast["ds"])[1]
    forecast["lost_coins_percentage"] = estimate_lost_coins_percentage(
        btc_data, forecast["ds"]
    )[0]
    forecast["available_supply"] = forecast["yhat"] * (
        1 - forecast["lost_coins_percentage"]
    )
    return forecast


def plot_available_supply(btc_data, forecast):
    """
    Plot the forecasted values along with historical and available supply data.
    """
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(forecast.ds, forecast.yhat, label="BTC Supply - Forecast", color="orange")
    # Supply
    plt.plot(
        btc_data["time"],
        btc_data["Supply"],
        label="BTC Supply - Historical",
        linewidth=3,
        color="blue",
    )

    # Available Supply
    plt.plot(
        forecast["ds"],
        forecast["available_supply"],
        label="BTC Supply Available - Forecast",
        linewidth=2,
        color="green",
    )

    halving_dates = pd.to_datetime(
        ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"]
    )
    halving_labels = ["Halving 1", "Halving 2", "Halving 3", "Halving 4"]

    for idx, date in enumerate(halving_dates):
        plt.axvline(x=date, color="lightblue", linestyle="--", lw=1)
        plt.text(
            date,
            plt.gca().get_ylim()[1] * 0.23,
            halving_labels[idx],
            color="blue",
            rotation=90,
            alpha=0.5,
            verticalalignment="top",
            fontsize=7,
        )

    plt.axhline(
        y=21000000, color="red", linestyle="--", lw=1, label="Max Supply = 21,000,000"
    )

    plt.gca().get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.xlabel("")
    plt.ylabel("Supply")
    plt.title("BTC Available Supply Estimation", fontsize=20, fontweight="bold")
    plt.legend(fontsize=8)

    # Add text box with percentage of coins lost
    lost_coins_text = coins_lost_percentage(btc_data, forecast)
    plt.text(
        0.02,
        0.92,
        lost_coins_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.savefig("../output/4b.AvailableSupply.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def plot_m2_supply(btc_data, forecast, m2_data):
    """
    Plot the forecasted Bitcoin supply values along with historical and available supply data,
    and M2 Money Supply on a secondary axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bitcoin Supply Plot
    ax1.plot(forecast.ds, forecast.yhat, label="BTC Supply - Forecast", color="orange")
    ax1.plot(
        btc_data["time"],
        btc_data["Supply"],
        label="BTC Supply - Historical",
        linewidth=3,
        color="blue",
    )
    # ax1.plot(forecast["ds"], forecast["available_supply"], label="Forecast Available BTC Supply", linewidth=2, color="green")

    ax1.axhline(
        y=21000000,
        color="red",
        linestyle="--",
        lw=1,
        label="Max BTC Supply = 21,000,000",
    )

    # Halving dates
    halving_dates = pd.to_datetime(
        ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"]
    )
    halving_labels = ["Halving 1", "Halving 2", "Halving 3", "Halving 4"]

    for idx, date in enumerate(halving_dates):
        ax1.axvline(x=date, color="lightblue", linestyle="--", lw=1)
        ax1.text(
            date,
            ax1.get_ylim()[1] * 0.23,
            halving_labels[idx],
            color="blue",
            rotation=90,
            alpha=0.5,
            verticalalignment="top",
            fontsize=7,
        )

    ax1.set_xlabel("")
    ax1.set_ylabel("BTC Supply", color="blue")  # Set y-axis label color
    ax1.tick_params(axis="y", labelcolor="blue")  # Set y-axis ticks color
    ax1.set_title("BTC Supply and U.S. M2 Money Supply", fontsize=20, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # M2 Money Supply Plot (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.plot(m2_data.index, m2_data["M2SL"], label="M2 Money Supply", color="purple")
    ax2.set_ylabel(
        "M2 Money Supply (Trillions of Dollars)", color="purple"
    )  # Set y-axis label color
    ax2.tick_params(axis="y", labelcolor="purple")  # Set y-axis ticks color
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x/1000:.1f}T"))
    ax2.legend(loc="upper right", fontsize=8)

    plt.grid(ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        "../output/4a.BTCvsM2.jpg", bbox_inches="tight", dpi=350
    )
    plt.show()


def plot_m2_btcprice(btc_data, forecast):
    """
    Plot the forecasted Bitcoin supply values along with historical and available supply data,
    and Bitcoin price on a secondary axis with a logarithmic scale.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bitcoin Supply Plot
    ax1.plot(forecast.ds, forecast.yhat, label="BTC Supply - Forecast", color="orange")
    ax1.plot(
        btc_data["time"],
        btc_data["Supply"],
        label="BTC Supply - Historical",
        linewidth=3,
        color="blue",
    )

    ax1.axhline(
        y=21000000,
        color="red",
        linestyle="--",
        lw=1,
        label="Max BTC Supply = 21,000,000",
    )

    # Halving dates
    halving_dates = pd.to_datetime(
        ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"]
    )
    halving_labels = ["Halving 1", "Halving 2", "Halving 3", "Halving 4"]

    for idx, date in enumerate(halving_dates):
        ax1.axvline(x=date, color="lightblue", linestyle="--", lw=1)
        ax1.text(
            date,
            ax1.get_ylim()[1] * 0.23,
            halving_labels[idx],
            color="blue",
            rotation=90,
            alpha=0.5,
            verticalalignment="top",
            fontsize=7,
        )

    ax1.set_xlabel("")
    ax1.set_ylabel("BTC Supply", color="blue")  # Set y-axis label color
    ax1.tick_params(axis="y", labelcolor="blue")  # Set y-axis ticks color
    ax1.set_title("BTC Supply and BTC Price", fontsize=20, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # Bitcoin Price Plot (secondary y-axis with logarithmic scale)
    ax2 = ax1.twinx()
    ax2.plot(btc_data["time"], btc_data["PriceUSD"], label="BTC Price", color="green")
    ax2.set_ylabel("BTC Price (USD)", color="green")  # Set y-axis label color
    ax2.tick_params(axis="y", labelcolor="green")  # Set y-axis ticks color
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=0, top=400_000)  # Set the y-axis limit for the BTC price
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:.0f}"))
    ax2.legend(loc="upper right", fontsize=8)

    plt.grid(ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        "../output/1f.BTCvsSupply.jpg", bbox_inches="tight", dpi=350
    )
    plt.show()


def loss_rates_plot(forecast):
    """
    Plot the loss rates of the forecasted data.

    :param forecast: Forecast data including time series and predicted values.
    :return: None
    """
    plt.plot(forecast.ds, forecast.loss_rates, label="Forecast", color="orange")


def coins_lost_percentage(btc_data, forecast):
    """
    Calculate the percentage of lost coins based on the last available data in btc_data and the forecast.

    :param btc_data: Historical Bitcoin data containing timestamps.
    :param forecast: Forecast data with predicted values and available supply.
    :return: A formatted string indicating the percentage of coins lost, or a message if data is not available.
    """
    # l'ultima data disponibile in btc_data
    last_date = btc_data.dropna().iloc[-1]["time"]
    last_forecast = forecast[forecast["ds"] == last_date]

    if not last_forecast.empty:
        last_supply = last_forecast["yhat"].values[0]
        last_available_supply = last_forecast["available_supply"].values[0]

        coins_lost_percentage = round(100 * (1 - (last_available_supply / last_supply)), 2)

        # Restituire la stringa formattata
        return f"Estimated percentage of coins lost: {coins_lost_percentage} %"
    else:
        return "Data not available for the last date in btc_data."
