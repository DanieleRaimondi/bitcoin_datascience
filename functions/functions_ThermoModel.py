import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.optimize import minimize


def load_btc_data(train_frac = 0.9, last_date = False):
    """
    Loads Bitcoin price data from a CSV file, prepares it by removing NA values in the 'PriceUSD' column,
    and resets the index.

    Parameters:
    - file_path (str): The path to the CSV file containing the Bitcoin price data.

    Returns:
    - pd.DataFrame: A pandas DataFrame with the prepared Bitcoin price data.
    - datetime: The last date available in the dataset.
    """
    df = pd.read_csv("../data/btc.csv", parse_dates=["time"])
    df = df.dropna(subset=["PriceUSD"]).reset_index(drop=True)[["time", "PriceUSD"]]
    # Calculate the number of rows to load based on the fraction
    num_rows = int(len(df) * train_frac)
    df = df.iloc[:num_rows].reset_index(drop=True)[["time", "PriceUSD"]]
    # Get the last date available in the dataset
    final_date = df["time"].iloc[-1]

    if last_date:
        return df, final_date
    else:
        return df


def cubic_regression(
    df,
    upper=1.5,
    lower=-0.96,
    visualize_plot=True,
):
    """
    Performs cubic regression on Bitcoin price data, generates price bands, filters bands values,
    and conditionally visualizes the result with a custom plot format.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'time' and 'PriceUSD' columns.
    - upper (float): Modifier for the upper band.
    - fib618 (float): Modifier for the Fibonacci 61.8% band.
    - medium (float): Modifier for the medium band.
    - fib382 (float): Modifier for the Fibonacci 38.2% band.
    - lower (float): Modifier for the lower band.
    - visualize_plot (bool): If True, shows a plot of the data with bands.
    """
    
    # start_date = "01-01-2015"
    # df = df.loc[df['date']>=start_date] #wanna filter out first days?
    x = np.arange(1, len(df["PriceUSD"]) + 1)
    xfit = np.arange(1, len(df["PriceUSD"]) + 100)
    y = np.log(1 + df["PriceUSD"])  # prices: log(1+price)

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(x[:, np.newaxis], y)
    yfit = model.predict(xfit[:, np.newaxis])

    # PARAMETERS
    # print(model.steps[1][1].coef_)
    # print(model.steps[1][1].intercept_)
    # print("lnPriceUSD=",model.steps[1][1].intercept_,"+",model.steps[1][1].coef_[1],"Day","+",model.steps[1][1].coef_[2],"Day^2","+",model.steps[1][1].coef_[3],"Day^3")
    # print("PriceUSD=",math.exp(model.steps[1][1].intercept_),"*",math.exp(model.steps[1][1].coef_[1]),"Day","*",math.exp(model.steps[1][1].coef_[2]),"Day^2","*",math.exp(model.steps[1][1].coef_[3]),"Day^3")

    df["band_cubic"] = np.exp(yfit[1 : len(x) + 1]) - 1
    df["upper_cubic"] = np.exp(upper + yfit[1 : len(x) + 1]) - 1
    df["lower_cubic"] = np.exp(lower + yfit[1 : len(x) + 1]) - 1

    # Save the model's parameters
    linear_model = model.named_steps["linearregression"]
    model_parameters = {
        "degrees": 3,
        "coef": linear_model.coef_.tolist(),  
        "intercept": linear_model.intercept_
    }

    if visualize_plot:
        colors = plt.cm.bwr(np.linspace(0, 1, 5))
        BTC_TM = plt.figure(figsize=(30, 12), dpi=80)

        plt.plot(df["time"], df["upper_cubic"].where(df["upper_cubic"] >= 1, np.nan), label="Upper Band", color=colors[4], linewidth=7)
        plt.plot(df["time"], df["lower_cubic"].where(df["lower_cubic"] >= 1, np.nan), label="Lower Band", color=colors[0], linewidth=7)
        plt.plot(df["time"], df["PriceUSD"], label="Bitcoin vs $ Price", color="black", linewidth=2)

        plt.yscale("log")
        plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))

        plt.title("Bitcoin Cubic Thermomodel", fontsize=35, fontweight="bold")

        plt.grid(linewidth=0.7, linestyle="--")
        plt.legend(loc=4, prop={"size": 12})
        plt.show()

    return df, model_parameters


def logarithmic_regression(
    df,
    upper=1.9,
    lower=-1.9,
    visualize_plot=True,
):
    
    """
    Performs logarithmic regression on Bitcoin price data, generates price bands, filters bands values,
    and conditionally visualizes the result with a custom plot format.

    Parameters:
    - df (DataFrame): DataFrame containing Bitcoin price data with columns 'time' and 'PriceUSD'.
    - upper (float): Upper bound coefficient for the logarithmic model.
    - lower (float): Lower bound coefficient for the logarithmic model.
    - visualize_plot (bool): Whether to visualize the regression plot (default True).

    Returns:
    - df (DataFrame): DataFrame with additional columns for regression results.
    - model_parameters (dict): Dictionary containing the coefficients of the regression model.
    """
    
    # start_date = "01-01-2015"
    # df = df.loc[df['date']>=start_date]

    x = np.arange(1, len(df["PriceUSD"]) + 1)
    xfit = np.arange(1, len(df["PriceUSD"]) + 100)
    y = np.log(1 + df["PriceUSD"])  # prices: log(1+price)
    xlog = np.log(x)
    xlog_fit = np.log(xfit)

    model = LinearRegression()
    model.fit(xlog[:, np.newaxis], y)
    yfit = model.predict(xlog_fit[:, np.newaxis])

    df["band_log"] = np.exp(yfit[1 : len(x) + 1]) - 1
    df["upper_log"] = np.exp(upper + yfit[1 : len(x) + 1]) - 1
    df["lower_log"] = np.exp(lower + yfit[1 : len(x) + 1]) - 1

    # predict manual
    # fitz = np.polyfit(xlog, y, 1)
    # print(fitz)
    # df["log_predict_auto"] = np.exp(model.predict(xlog[:, np.newaxis])) - 1
    # df["log_predict_manuale"] = np.exp(fitz[1] + fitz[0] * xlog) - 1

    # Save the model's parameters
    model_parameters = {"coef": model.coef_[0], "intercept": model.intercept_}

    if visualize_plot:
        colors = plt.cm.bwr(np.linspace(0, 1, 5))
        BTC_TM = plt.figure(figsize=(30, 12), dpi=80)

        plt.plot(df["time"], df["upper_log"].where(df["upper_log"] >= 1, np.nan), label="Upper Band", color=colors[4], linewidth=7)
        plt.plot(df["time"], df["lower_log"].where(df["lower_log"] >= 1, np.nan), label="Lower Band", color=colors[0], linewidth=7)
        plt.plot(df["time"], df["PriceUSD"], label="Bitcoin vs $ Price", color="black", linewidth=2)

        plt.yscale("log")
        plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))

        plt.title("Bitcoin Log ThermoModel", fontsize=35, fontweight="bold")

        # plt.xlabel('Time', fontsize=18)
        # plt.ylabel('USD Price', fontsize=18)
        plt.grid(linewidth=0.7, linestyle="--")
        plt.legend(loc=4, prop={"size": 12})
        plt.show()

    return df, model_parameters


def calculate_mape(params, df, tops_dates, bottoms_dates):
    
    """
    Calculate Mean Absolute Percentage Error (MAPE) for the upper and lower bands.

    Parameters:
    - params (tuple): Tuple containing parameters for upper and lower bands decay and max weight.
    - df (DataFrame): DataFrame containing Bitcoin price data with columns 'time', 'PriceUSD', 'upper_cubic', 'upper_log', 'lower_cubic', and 'lower_log'.
    - tops_dates (array-like): Array-like object containing dates corresponding to tops.
    - bottoms_dates (array-like): Array-like object containing dates corresponding to bottoms.

    Returns:
    - total_mape (float): Total MAPE for the upper and lower bands.
    """
    
    upper_decay, upper_maxw, lower_decay, lower_maxw = params
    weights_upper = (
        (np.exp(np.linspace(-1.0, 0.0, len(df)))) ** upper_decay
    ) * upper_maxw
    weights_lower = (
        (np.exp(np.linspace(-1.0, 0.0, len(df)))) ** lower_decay
    ) * lower_maxw

    df["upper_cubiclog"] = (1 - weights_upper) * df["upper_cubic"] + weights_upper * df[
        "upper_log"
    ]
    df["lower_cubiclog"] = (1 - weights_lower) * df["lower_cubic"] + weights_lower * df[
        "lower_log"
    ]

    # Calculate MAPE for tops and bottoms
    tops_mape = np.mean(
        np.abs(
            df.loc[df["time"].isin(tops_dates), "PriceUSD"]
            - df.loc[df["time"].isin(tops_dates), "upper_cubiclog"]
        )
        / df.loc[df["time"].isin(tops_dates), "PriceUSD"]
    )
    bottoms_mape = np.mean(
        np.abs(
            df.loc[df["time"].isin(bottoms_dates), "PriceUSD"]
            - df.loc[df["time"].isin(bottoms_dates), "lower_cubiclog"]
        )
        / df.loc[df["time"].isin(bottoms_dates), "PriceUSD"]
    )

    # Combine MAPEs for optimization
    total_mape = tops_mape + bottoms_mape
    return total_mape


def optimize_params(df, tops_dates, bottoms_dates, initial_guess, bounds):

    """
    Optimize parameters for the MAPE calculation using scipy's minimize function.

    Parameters:
    - df (DataFrame): DataFrame containing Bitcoin price data with columns 'time', 'PriceUSD', 'upper_cubic', 'upper_log', 'lower_cubic', and 'lower_log'.
    - tops_dates (array-like): Array-like object containing dates corresponding to tops.
    - bottoms_dates (array-like): Array-like object containing dates corresponding to bottoms.
    - initial_guess (array-like): Array-like object containing initial guesses for the optimization.
    - bounds (sequence): A sequence of (min, max) pairs for each element in x.

    Returns:
    - optimized_params (array-like): Array-like object containing optimized parameters.
    """

    # Optimization
    result = minimize(
        calculate_mape,
        initial_guess,
        args=(df, tops_dates, bottoms_dates),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return result.x


def create_cubiclog(df, optimized_params):

    """
    Create cubiclog bands and oscillator based on optimized parameters.

    Parameters:
    - df (DataFrame): DataFrame containing Bitcoin price data with columns 'time', 'PriceUSD', 'upper_cubic', 'upper_log', 'lower_cubic', and 'lower_log'.
    - optimized_params (array-like): Array-like object containing optimized parameters.

    Returns:
    - df (DataFrame): DataFrame with additional columns for cubiclog bands and oscillator.
    """
    
    df["Oscillator_cubiclog"] = (np.log(df["PriceUSD"]) - np.log(df["lower_cubiclog"])) / (np.log(df["upper_cubiclog"]) - np.log(df["lower_cubiclog"]))
    df["Oscillator_cubiclog"] = df["Oscillator_cubiclog"].bfill()
    lowess = sm.nonparametric.lowess(df["Oscillator_cubiclog"], df.time, frac=0.00175)
    lowess = (lowess[:, 1] - lowess[:, 1].min()) / (lowess[:, 1].max() - lowess[:, 1].min())
    # Estrazione dei parametri ottimizzati
    upper_decay, upper_maxw, lower_decay, lower_maxw = optimized_params

    # Calcolo dei pesi
    df["weights_upper"] = ((np.exp(np.linspace(-1.0, 0.0, len(df)))) ** upper_decay) * upper_maxw
    df["weights_lower"] = ((np.exp(np.linspace(-1.0, 0.0, len(df)))) ** lower_decay) * lower_maxw

    # Combinazione delle bande cubiche e logaritmiche utilizzando i pesi
    df["upper_cubiclog"] = (1 - df["weights_upper"]) * df["upper_cubic"] + df["weights_upper"] * df["upper_log"]
    df["lower_cubiclog"] = (1 - df["weights_lower"]) * df["lower_cubic"] + df["weights_lower"] * df["lower_log"]
    df["medium_cubiclog"] = np.exp(0.5 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["sell_cubiclog"] = np.exp(0.9 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["buy_cubiclog"] = np.exp(0.1 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["75_cubiclog"] = np.exp(0.75 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["25_cubiclog"] = np.exp(0.25 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    
    # creo oscillatore
    # df['Oscillator_cubiclog'] = (df['PriceUSD']- df['lower_cubiclog'])/(df['upper_cubiclog']- df['lower_cubiclog'])
    df["Oscillator_cubiclog"] = (np.log(df["PriceUSD"]) - np.log(df["lower_cubiclog"])) / (np.log(df["upper_cubiclog"]) - np.log(df["lower_cubiclog"]))
    df["Oscillator_cubiclog"] = df["Oscillator_cubiclog"].bfill()

    # smusso l'oscillatore e lo riscalo
    lowess = sm.nonparametric.lowess(df["Oscillator_cubiclog"], df.time, frac=0.00175)
    lowess = (lowess[:, 1] - lowess[:, 1].min()) / (lowess[:, 1].max() - lowess[:, 1].min())
    df["lowess"] = lowess
    df["lowess"][:250] = np.nan  # forzo perchè qui prezzo era <1, non posso fare log.

    # Visualizzazione delle bande
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(df["time"],df["PriceUSD"],label="Bitcoin vs $ Price",color="black",linewidth=2,)
    plt.plot(df["time"],df["upper_cubic"],label="Upper Cubic Band",color="red",linewidth=1,)
    plt.plot(df["time"],df["upper_cubiclog"],label="Upper Ensemble Band",color="darkgreen",linewidth=4,)
    plt.plot(df["time"], df["upper_log"], label="Upper Log Band", color="blue", linewidth=1)
    plt.plot(df["time"],df["lower_cubic"],label="Lower Cubic Band",color="red",linewidth=1,)
    plt.plot(df["time"],df["lower_cubiclog"],label="Lower Ensemble Band",color="darkgreen",linewidth=4,)
    plt.plot(df["time"], df["lower_log"], label="Lower Log Band", color="blue", linewidth=1)

    plt.yscale("log")
    plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))
    plt.grid(linewidth=0.7, linestyle="--")
    plt.legend()
    plt.show()

    return df


def inference(df, cubic_model_params, log_model_params, optimized_params):
    """
    Applies cubic and logarithmic model parameters to the entire dataset to calculate and visualize
    ensemble upper and lower bands, including additional calculated bands and the oscillator.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the Bitcoin price data with 'time' and 'PriceUSD' columns.
    - cubic_model_params (dict): Parameters of the cubic model including coefficients and intercept.
    - log_model_params (dict): Parameters of the logarithmic model including coefficient and intercept.
    - optimized_params (list): Optimized parameters used for weighting the ensemble bands.

    The function calculates ensemble bands based on cubic and logarithmic models and visualizes them
    with the original Bitcoin price data on a logarithmic scale.
    """

    # Generate sequence of days for the entire dataset
    x = np.arange(1, len(df) + 1)

    # Calculate cubic bands
    y_cubic_fit = (np.polyval(cubic_model_params["coef"][::-1], x)+ cubic_model_params["intercept"])
    df["upper_cubic"] = (np.exp(y_cubic_fit + 1) - 1)  # Adjusted for +1 for the upper bound
    df["lower_cubic"] = (np.exp(y_cubic_fit - 1) - 1
    )  # Adjusted for -1 for the lower bound

    # Calculate logarithmic bands
    x_log = np.log(x)
    y_log_fit = log_model_params["coef"] * x_log + log_model_params["intercept"]
    df["upper_log"] = np.exp(y_log_fit + 2) - 1  # Adjusted for +2 for the upper bound
    df["lower_log"] = np.exp(y_log_fit - 2) - 1  # Adjusted for -2 for the lower bound

    # Apply optimized parameters to calculate ensemble bands
    upper_decay, upper_maxw, lower_decay, lower_maxw = optimized_params
    df["weights_upper"] = ((np.exp(np.linspace(-1.0, 0.0, len(df)))) ** upper_decay) * upper_maxw
    df["weights_lower"] = ((np.exp(np.linspace(-1.0, 0.0, len(df)))) ** lower_decay) * lower_maxw

    df["upper_cubiclog"] = (1 - df["weights_upper"]) * df["upper_cubic"] + df["weights_upper"] * df["upper_log"]
    df["lower_cubiclog"] = (1 - df["weights_lower"]) * df["lower_cubic"] + df["weights_lower"] * df["lower_log"]

    # Additional calculated bands for analysis
    df["medium_cubiclog"] = np.exp(0.5 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["sell_cubiclog"] = np.exp(0.9 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["buy_cubiclog"] = np.exp(0.1 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["75_cubiclog"] = np.exp(0.75 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))
    df["25_cubiclog"] = np.exp(0.25 * (np.log(df["upper_cubiclog"] / df["lower_cubiclog"]))+ np.log(df["lower_cubiclog"]))

    # Calculate oscillator based on log prices and smooth it using LOWESS
    df["Oscillator_cubiclog"] = (np.log(df["PriceUSD"]) - np.log(df["lower_cubiclog"])) / (np.log(df["upper_cubiclog"]) - np.log(df["lower_cubiclog"]))
    df["Oscillator_cubiclog"] = df["Oscillator_cubiclog"].bfill()  # Forward fill to handle any initial NaNs
    lowess = sm.nonparametric.lowess(df["Oscillator_cubiclog"], df.time, frac=0.00175)
    df["lowess"] = (lowess[:, 1] - lowess[:, 1].min()) / (lowess[:, 1].max() - lowess[:, 1].min())
    df["lowess"][:250] = (np.nan)  # Force NaN for the initial period if price was <1, not applicable for log

    # Visualization with logarithmic scale
    plt.figure(figsize=(14, 7))
    plt.plot(df["time"], df["PriceUSD"], label="Bitcoin Price", color="black")
    plt.plot(df["time"], df["upper_cubiclog"], label="Upper Ensemble Band", color="green")
    plt.plot(df["time"], df["lower_cubiclog"], label="Lower Ensemble Band", color="red")
    plt.fill_between(df["time"],df["upper_cubiclog"],df["lower_cubiclog"],color="lightgray",alpha=0.5,)
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.title("Bitcoin Price with Inferenced Bands")
    plt.xlabel("Date")
    plt.ylabel("Price USD (log scale)")
    plt.legend()
    plt.show()


def weight_plot(df):
    df["weights_upper"].plot()
    df["weights_lower"].plot()
    plt.legend()


def final_plot(df, last_date):
    lowess = sm.nonparametric.lowess(df["Oscillator_cubiclog"], df.time, frac=0.00175)
    lowess = (lowess[:, 1] - lowess[:, 1].min()) / (lowess[:, 1].max() - lowess[:, 1].min())
    colors = plt.cm.bwr(np.linspace(0, 1, 5))
    plt.figure(figsize=(38.4, 21.6))
    plt.subplot(5, 5, (1, 20))

    df["lower_cubiclog"] = df["lower_cubiclog"].where(df["lower_cubiclog"] >= 1.75, np.nan)  # pulisco la serie xk + bella graficamente
    df["medium_cubiclog"] = df["medium_cubiclog"].where(df["medium_cubiclog"] >= 1, np.nan)
    df["upper_cubiclog"] = df["upper_cubiclog"].where(df["upper_cubiclog"] >= 27, np.nan)
    df["25_cubiclog"] = df["25_cubiclog"].where(df["25_cubiclog"] >= 3.8, np.nan)
    df["75_cubiclog"] = df["75_cubiclog"].where(df["75_cubiclog"] >= 8, np.nan)

    # plt.scatter(df['time'], df['PriceUSD'], label = "BTCUSD Price", c=lowess, linewidth=1)
    plt.plot(df["time"], df["upper_cubiclog"], label="Upper Band", color="red", linewidth=6)
    plt.plot(df["time"], df["75_cubiclog"], label="3Q Band", color="red", linewidth=1)
    plt.plot(df["time"], df["medium_cubiclog"], label="Middle Band", color="grey", linewidth=1.5)
    plt.plot(df["time"], df["25_cubiclog"], label="1Q Band", color="blue", linewidth=1)
    plt.plot(df["time"], df["lower_cubiclog"], label="Lower Band", color="blue", linewidth=6)
    plt.plot(df["time"], df["PriceUSD"], label="BTCUSD Price", color="black", linewidth=3)

    plt.yscale("log")
    plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))

    plt.title("Bitcoin ThermoModel", fontsize=75, fontweight="bold")
    plt.grid(linewidth=0.5, linestyle="--")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.07), prop={"size": 18, "weight": "bold"})
    plt.xticks(alpha=0)
    plt.yticks(fontsize=14, weight="bold")
    plt.ylim(df["PriceUSD"].min(), df["upper_cubiclog"].max() + 5000)
    plt.xlim(df["time"].min(), df["time"].max())
    plt.axvline(dt.datetime(2012, 11, 28), color="blue", alpha=0.6)
    plt.text(df.time.iloc[876],1.25,"1st Halving",rotation=90,fontsize=20,color="blue",alpha=0.6,)
    plt.axvline(dt.datetime(2016, 7, 9), color="blue", alpha=0.6)
    plt.text(df.time.iloc[2198],1.25, "2nd Halving",rotation=90,fontsize=20,color="blue",alpha=0.6,)
    plt.axvline(dt.datetime(2020, 5, 11), color="blue", alpha=0.6)
    plt.text(df.time.iloc[3600],1.25,"3rd Halving",rotation=90,fontsize=20,color="blue",alpha=0.6,)
    plt.axvline(dt.datetime(2024, 4, 19), color="blue", alpha=0.6)
    plt.text(df.time.iloc[5000],1.25,"4th Halving",rotation=90,fontsize=20,color="blue",alpha=0.6,)

    # TEXTS
    plt.text(df.time.iloc[2000], 10, "@Daniele Raimondi", fontsize=75, color="grey", alpha=0.25)
    plt.text(
        df.time.iloc[435],
        2500,
        "Current value: \n{} %".format(round(100 * lowess[-1], 1)),
        fontsize=40,
        color="k",
        bbox=dict(facecolor="white"),
        ha="center",
        weight="bold",
    )
    plt.text(
        df.time.iloc[420],
        0.125,
        f"Expected top: {round(df['upper_cubiclog'].iloc[-1]/1000)*1000} $",
        fontsize=25,
        color="k",
        bbox=dict(facecolor="white"),
        ha="center",
    )

    a1 = round(100 * lowess[-1], 1)
    a2 = round(100 * lowess[-2], 1)
    a3 = round(100 * lowess[-8], 1)
    a4 = round(100 * lowess[-31], 1)
    a5 = round(100 * lowess[-366], 1)
    plt.text(
        df.time.iloc[320],
        375,
        f"Yesterday: {a2} %\n A week ago: {a3} %\n A month ago: {a4} %\n A year ago: {a5} %",
        fontsize=20,
        color="k",
        bbox=dict(facecolor="white"),
        ha="center",
    )

    # FILL
    plt.fill_between(
        df["time"], df["sell_cubiclog"], df["upper_cubiclog"], color=colors[4], alpha=0.3
    )
    plt.fill_between(
        df["time"], df["lower_cubiclog"], df["buy_cubiclog"], color=colors[0], alpha=0.3
    )

    ####### Highlighting the training and inference data sections
    cutoff_date = last_date
    one_year_before_cutoff = cutoff_date - pd.Timedelta(days=365)

    # Use fill_betweenx to specify the y-range for the shaded area
    plt.fill_betweenx(
        y=[0, 0.1],
        x1=df["time"].min(),
        x2=cutoff_date - pd.Timedelta(days=500),
        color="orange",
        alpha=0.3,
        label="Train Data",
    )
    plt.fill_betweenx(
        y=[0, 0.1],
        x1=cutoff_date - pd.Timedelta(days=500),
        x2=df["time"].max(),
        color="blue",
        alpha=0.1,
        label="Inference Data",
    )

    # Adding text annotations for "train data" and "inference data"
    midpoint_train = one_year_before_cutoff - pd.Timedelta(days=(one_year_before_cutoff - df["time"].min()).days / 2)
    midpoint_inference = df["time"].max() - pd.Timedelta(days=300)
    plt.text(
        midpoint_train,
        0.07,
        "Train Data",
        horizontalalignment="center",
        fontsize=16,
        color="black",
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        midpoint_inference,
        0.07,
        "Inference Data",
        horizontalalignment="center",
        fontsize=16,
        color="black",
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # LOGO
    from matplotlib.cbook import get_sample_data
    img = plt.imread(get_sample_data("/Users/danieleraimondi/btc_charts/btc_charts/utils/btc_logo.png"))
    plt.figimage(img, 375, 5300)
    ###############################################################

    plt.subplot(5, 5, (21, 25))

    plt.axhline(y=100, color=colors[4], linestyle="-", linewidth=2)
    plt.axhline(y=90, color=colors[3], linestyle="-", linewidth=1)
    plt.axhline(y=75, color=colors[3], linestyle="-", linewidth=0.75)
    plt.axhline(y=50, color="darkgrey", linestyle="-", linewidth=1.25)
    plt.axhline(y=25, color=colors[1], linestyle="-", linewidth=0.75)
    plt.axhline(y=10, color=colors[1], linestyle="-", linewidth=1)
    plt.axhline(y=0, color=colors[0], linestyle="-", linewidth=2)

    plt.fill_between([df["time"].min(), df["time"].max()], 90, 100, color=colors[4], alpha=0.3)
    plt.fill_between([df["time"].min(), df["time"].max()], 50, 90, color=colors[3], alpha=0.1)
    plt.fill_between([df["time"].min(), df["time"].max()], 10, 50, color=colors[1], alpha=0.1)
    plt.fill_between([df["time"].min(), df["time"].max()], 0, 10, color=colors[0], alpha=0.3)

    plt.yticks([0, 10, 25, 50, 75, 90, 100], fontsize=14, weight="bold")
    plt.xticks(fontsize=16, weight="bold")
    plt.ylim(0, 100)
    plt.xlim(df["time"].min(), df["time"].max())
    plt.grid(linewidth=0.7, linestyle="--")
    plt.text(df.time.iloc[665], 92, "SELL ZONE", fontsize=16, color="Red", weight="bold")
    plt.text(df.time.iloc[1900], 92, "SELL ZONE", fontsize=16, color="Red", weight="bold")
    plt.text(df.time.iloc[3350], 92, "SELL ZONE", fontsize=16, color="Red", weight="bold")
    plt.text(df.time.iloc[1280], 2, "BUY ZONE", fontsize=16, color="Blue", weight="bold")
    plt.text(df.time.iloc[2700], 2, "BUY ZONE", fontsize=16, color="Blue", weight="bold")
    plt.text(df.time.iloc[3960], 2, "BUY ZONE", fontsize=16, color="Blue", weight="bold")
    plt.axvline(dt.datetime(2012, 11, 28), color="blue", alpha=0.6)
    plt.axvline(dt.datetime(2016, 7, 9), color="blue", alpha=0.6)
    plt.axvline(dt.datetime(2020, 5, 11), color="blue", alpha=0.6)
    plt.text(df.time.iloc[2800], 50, "@Daniele Raimondi", fontsize=50, color="grey", alpha=0.25)

    plt.scatter(df["time"], df["lowess"] * 100, c=lowess, linewidth=0.75, cmap="RdBu_r")

    plt.savefig("../output/ThermoModel.jpg",bbox_inches="tight",dpi=350,)

    plt.show()
