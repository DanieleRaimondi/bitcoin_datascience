import numpy as np  
from prophet import Prophet
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd


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
    forecast["loss_rates"] = estimate_lost_coins_percentage(btc_data, forecast["ds"])[1]
    forecast["lost_coins_percentage"] = estimate_lost_coins_percentage(
        btc_data, forecast["ds"]
    )[0]
    forecast["available_supply"] = forecast["yhat"] * (
        1 - forecast["lost_coins_percentage"]
    )
    return forecast


def final_plot(btc_data, forecast):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(forecast.ds, forecast.yhat, label="Forecast", color="orange")
    # Supply
    plt.plot(
        btc_data["time"],
        btc_data["Supply"],
        label="Historical Supply",
        linewidth=3,
        color="blue",
    )

    # Available Supply
    plt.plot(
        forecast["ds"],
        forecast["available_supply"],
        label="Forecast Available Supply",
        linewidth=2,
        color="green",
    )

    halving_dates = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-25"])
    halving_labels = ["Halving 1", "Halving 2", "Halving 3", "Halving 4"]

    for idx, date in enumerate(halving_dates):
        plt.axvline(x=date, color="lightblue", linestyle="--", lw=1)
        plt.text(
            date,
            plt.gca().get_ylim()[1]*0.23,
            halving_labels[idx],
            color="blue",
            rotation=90, alpha=0.5,
            verticalalignment="top", fontsize=7)

    plt.axhline(y=21000000, color="red", linestyle="--", lw=1, label="Max Supply = 21,000,000")

    plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    plt.xlabel("")
    plt.ylabel("Supply")
    plt.title("BTC Available Supply Estimation", fontsize=20)
    plt.legend(fontsize=8)
    plt.savefig("../output/AvailableSupply.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def loss_rates_plot(forecast):
    plt.plot(forecast.ds, forecast.loss_rates, label="Forecast", color="orange")


def coins_lost_percentage(btc_data, forecast):
    # l'ultima data disponibile in btc_data
    last_date = btc_data.dropna().iloc[-1]["time"]
    last_forecast = forecast[forecast["ds"] == last_date]

    if not last_forecast.empty:
        last_supply = last_forecast["yhat"].values[0]
        last_available_supply = last_forecast["available_supply"].values[0]

        coins_lost_percentage = round(100 * (1 - (last_available_supply / last_supply)), 2)

        # Restituire la stringa formattata
        return f"Percentage of coins lost: {coins_lost_percentage} %"
    else:
        return "Data not available for the last date in btc_data."