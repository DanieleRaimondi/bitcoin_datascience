import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import statsmodels.api as sm
from matplotlib.cbook import get_sample_data


def plot_ensemble_corridor(
    loglog_path,
    thermomodel_path,
    slopes_growth_path,
    btc_logo_path,
    output_path,
    weights=[1, 1, 1],  # weights for [loglog, thermo, slopes_growth]
):
    """
    Version with weighted average for the final corridor calculation.
    weights: list of 3 numbers representing relative importance of each model
    """
    # Fetch Data from all models
    loglog = pd.read_csv(loglog_path, parse_dates=["time"])
    thermomodel = pd.read_csv(thermomodel_path, parse_dates=["time"])
    slopes_growth = pd.read_csv(slopes_growth_path, parse_dates=["date"])

    # Rename date column in slopes_growth to match other datasets
    slopes_growth = slopes_growth.rename(columns={"date": "time"})

    # Merge all datasets
    df = pd.merge(
        loglog,
        thermomodel[["time", "lower_cubiclog", "upper_cubiclog"]],
        on="time",
        how="inner",
    )
    df = pd.merge(
        df,
        slopes_growth[["time", "lower", "upper"]],
        on="time",
        how="inner",
        suffixes=("", "_slopes"),
    )

    # Calculate weighted average for final corridor
    w1, w2, w3 = weights
    total_weight = sum(weights)

    df["lower"] = (
        w1 * df["BottomLine"] + w2 * df["lower_cubiclog"] + w3 * df["lower"]
    ) / total_weight
    df["upper"] = (
        w1 * df["TopLine"] + w2 * df["upper_cubiclog"] + w3 * df["upper"]
    ) / total_weight

    # Calculate oscillator based on final corridor
    df["oscillator"] = (
        100
        * (np.log(df["PriceUSD"]) - np.log(df["lower"]))
        / (np.log(df["upper"]) - np.log(df["lower"]))
    )
    df = df.dropna()

    # Calculate LOWESS
    lowess = sm.nonparametric.lowess(df["oscillator"], df.time, frac=0.00175)
    lowess = (lowess[:, 1] - lowess[:, 1].min()) / (
        lowess[:, 1].max() - lowess[:, 1].min()
    )
    df["lowess"] = lowess

    # Set up plot with optimized dimensions
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 5))
    plt.figure(figsize=(24, 14))  

    # Main price plot
    plt.subplot(5, 5, (1, 20))
    plt.plot(
        df["time"],
        df["upper"],
        label=f"Upper Band (Weighted: {weights})",
        color="red",
        linewidth=8,  # Reduced from 12
    )
    plt.plot(
        df["time"],
        df["lower"],
        label=f"Lower Band (Weighted: {weights})",
        color="green",
        linewidth=8,  # Reduced from 12
    )
    plt.plot(
        df["time"], df["PriceUSD"], label="BTCUSD Price", color="black", linewidth=2
    )

    plt.yscale("log")
    plt.yticks(10 ** np.arange(6), 10 ** np.arange(6))
    plt.title(
        "Bitcoin Ensemble Model (Weighted)",
        fontsize=48,  # FIXED: was 5400!
        fontweight="bold",
    )
    plt.grid(linewidth=0.5, linestyle="--")
    plt.legend(loc="lower right", bbox_to_anchor=(0.99, 0.07), prop={"size": 14})
    plt.xticks(alpha=0)
    plt.yticks(fontsize=12, weight="bold")
    plt.ylim(0, df["upper"].max() + 50000)
    plt.xlim(df["time"].min(), df["time"].max())

    # Add text annotations
    plt.text(
        df.time.iloc[435],
        2500,
        f"Current value: \n{round(100 * lowess[-1], 1)} %",
        fontsize=28,  # Reduced from 40
        color="k",
        bbox=dict(facecolor="white"),
        ha="center",
        weight="bold",
    )
    plt.text(
        df.time.iloc[420],
        2,
        f"Expected top: {round(df['upper'].iloc[-1]/1000)*1000} $",
        fontsize=18,  # Reduced from 25
        color="k",
        bbox=dict(facecolor="white"),
        ha="center",
    )

    # Add logo (with error handling)
    try:
        img = plt.imread(get_sample_data(btc_logo_path))
        plt.figimage(img, 250, 3500)  # Adjusted position
    except:
        print("Logo file not found, skipping logo placement")

    # Oscillator plot
    plt.subplot(5, 5, (21, 25))
    plt.axhline(y=100, color=colors[4], linestyle="-", linewidth=2)
    plt.axhline(y=90, color=colors[3], linestyle="-", linewidth=1)
    plt.axhline(y=75, color=colors[3], linestyle="-", linewidth=0.75)
    plt.axhline(y=50, color="darkgrey", linestyle="-", linewidth=1.25)
    plt.axhline(y=25, color=colors[1], linestyle="-", linewidth=0.75)
    plt.axhline(y=10, color=colors[1], linestyle="-", linewidth=1)
    plt.axhline(y=0, color=colors[0], linestyle="-", linewidth=2)

    plt.fill_between(
        [df["time"].min(), df["time"].max()], 90, 100, color=colors[4], alpha=0.4
    )
    plt.fill_between(
        [df["time"].min(), df["time"].max()], 50, 90, color=colors[3], alpha=0.1
    )
    plt.fill_between(
        [df["time"].min(), df["time"].max()], 10, 50, color=colors[1], alpha=0.1
    )
    plt.fill_between(
        [df["time"].min(), df["time"].max()], 0, 10, color=colors[0], alpha=0.4
    )

    plt.yticks([0, 10, 25, 50, 75, 90, 100], fontsize=12, weight="bold")
    plt.xticks(fontsize=14, weight="bold")
    plt.ylim(0, 100)
    plt.xlim(df["time"].min(), df["time"].max())
    plt.grid(linewidth=0.7, linestyle="--")

    # Add text annotations to oscillator plot (with bounds checking)
    if len(df) > 450:
        plt.text(
            df.time.iloc[450], 92, "SELL ZONE", fontsize=12, color="k", weight="bold"
        )
    if len(df) > 1900:
        plt.text(
            df.time.iloc[1900], 92, "SELL ZONE", fontsize=12, color="k", weight="bold"
        )
    if len(df) > 3600:
        plt.text(
            df.time.iloc[3600], 92, "SELL ZONE", fontsize=12, color="k", weight="bold"
        )
    if len(df) > 700:
        plt.text(
            df.time.iloc[700], 2, "BUY ZONE", fontsize=12, color="k", weight="bold"
        )
    if len(df) > 2700:
        plt.text(
            df.time.iloc[2700], 2, "BUY ZONE", fontsize=12, color="k", weight="bold"
        )
    if len(df) > 4200:
        plt.text(
            df.time.iloc[4200], 2, "BUY ZONE", fontsize=12, color="k", weight="bold"
        )

    plt.scatter(
        df["time"], df["lowess"] * 100, c=lowess, linewidth=0.75, cmap="RdYlGn_r"
    )

    # Save and show the plot with optimized settings
    plt.savefig(output_path, bbox_inches="tight", dpi=200)  # Reduced DPI from 350
    plt.tight_layout()
    plt.show()