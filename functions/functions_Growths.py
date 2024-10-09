import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure
import sys

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_data

# Load and preprocess data
def load_data():
    df = fetch_data("btc")
    df = df[["time", "PriceUSD"]].dropna()
    return df


def plot_bitcoin_price_vs_sma(df):
    """
    Plots the Bitcoin price against its 209-week Simple Moving Average (SMA).
    Parameters:
    df (DataFrame): A pandas DataFrame containing 'time' and 'PriceUSD' columns.
    """
    # Resample to weekly data and calculate the SMA
    weekly = df.resample("W", on="time").mean()
    weekly = weekly.iloc[:, :1]
    weekly["209SMA"] = weekly["PriceUSD"].rolling(window=209).mean()
    weekly["Multiple"] = weekly["PriceUSD"] / weekly["209SMA"]

    def price_formatter(x, p):
        """Format y-axis labels to show large numbers in M (millions) and K (thousands)."""
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        elif x >= 1e3:
            return f"{x/1e3:.0f}K"
        else:
            return f"{x:.0f}"

    # Set the style for the plot
    plt.style.use("default")
    sns.set_palette("deep")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(24, 10), dpi=80, facecolor="white")
    ax.set_facecolor("white")

    # Plot Bitcoin price and its SMA
    ax.plot(weekly.index, weekly["PriceUSD"], label="PriceUSD", linewidth=2.5)
    ax.plot(
        weekly.index, weekly["209SMA"], label="209SMA", color="orange", linewidth=2.5
    )

    # Set y-axis to logarithmic scale
    ax.set_yscale("log")

    # Format y-axis labels
    ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))

    # Set y-axis limits based on the global min and max
    global_min = weekly["PriceUSD"].min()
    global_max = weekly["PriceUSD"].max()
    ax.set_ylim(global_min * 0.8, global_max * 1.2)  # Add 20% padding on both ends

    # Set x-axis limits to reduce empty space
    ax.set_xlim(weekly.index.min(), weekly.index.max())

    # Add title and legend
    ax.set_title("Bitcoin Price VS SMA (209)", fontsize=24, fontweight="bold", pad=20)
    ax.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0, 1))

    # Add labels to the axes
    ax.set_ylabel("Price USD", fontsize=20, labelpad=15)

    # Show grid lines
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase the size of tick labels
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig("../output/1g.BTC_SMA.jpg",bbox_inches="tight",dpi=350,)
    plt.show()


def load_and_preprocess_data():
    """
    Loads and preprocesses Bitcoin price data from a CSV file.
    Returns:
        DataFrame: A pandas DataFrame containing the time, PriceUSD, Epoch, Halving, and Counter columns.
    """
    # Load the data from the CSV file and parse the 'time' column as dates
    df = fetch_data("btc")
    df = df[["time", "PriceUSD"]].dropna()

    # Define epoch boundaries and their corresponding epoch numbers
    epoch_boundaries = [
        ("2012-11-28", 1),
        ("2016-07-09", 2),
        ("2020-05-11", 3),
        ("2024-04-20", 4),
        ("2028-04-20", 5),
    ]

    # Create an 'Epoch' column initialized to 0
    df["Epoch"] = 0
    for date, epoch in epoch_boundaries:
        # Assign epoch numbers based on the time values
        df.loc[df["time"] >= date, "Epoch"] = epoch

    # Create a 'Halving' column indicating whether the date is a halving date (1) or not (0)
    df["Halving"] = df["time"].isin([date for date, _ in epoch_boundaries]).astype(int)

    # Create a 'Counter' column that counts occurrences within each epoch
    df["Counter"] = df.groupby("Epoch").cumcount() + 1

    return df


def plot_bitcoin_halving_growth():
    """
    Plots the growth of Bitcoin price since each halving event.
    """
    # Load the data
    df = fetch_data("btc")
    df = df[["time", "PriceUSD"]].dropna()

    # Define epoch boundaries and their corresponding epoch numbers
    epoch_boundaries = [
        ("2012-11-28", 1),
        ("2016-07-09", 2),
        ("2020-05-11", 3),
        ("2024-04-20", 4),
        ("2028-04-20", 5),
    ]

    # Create an 'Epoch' column initialized to 0
    df["Epoch"] = 0
    for date, epoch in epoch_boundaries:
        # Assign epoch numbers based on the time values
        df.loc[df["time"] >= date, "Epoch"] = epoch

    # Create a 'Counter' column that counts occurrences within each epoch
    df["Counter"] = df.groupby("Epoch").cumcount() + 1

    # Prepare data for plotting
    epochs = []
    for epoch in range(5):
        halv = df[df["Epoch"] == epoch].copy()
        halv["idx"] = halv["PriceUSD"] / halv["PriceUSD"].iloc[0]  # Normalize prices
        epochs.append(halv)

    # Labels for the epochs
    epoch_labels = [
        "Epoch 1 (50 BTC)",
        "Epoch 2 (25 BTC)",
        "Epoch 3 (12.5 BTC)",
        "Epoch 4 (6.25 BTC)",
        "Epoch 5 (3.125 BTC)",
    ]

    # Price formatting function for y-axis labels
    def price_formatter(x, p):
        """Format y-axis labels to show large numbers in M (millions) and K (thousands)."""
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        elif x >= 1e3:
            return f"{x/1e3:.1f}K"
        else:
            return f"{x:.0f}"

    # Create the plot
    plt.style.use("default")
    sns.set_palette("deep")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(24, 10), dpi=80, facecolor="white")
    ax.set_facecolor("white")
    colors = sns.color_palette("deep", n_colors=len(epochs))
    global_min, global_max = np.inf, -np.inf

    # Plot each epoch's price growth
    for i, halv in enumerate(epochs):
        ax.plot(
            halv["Counter"],
            halv["idx"],
            label=epoch_labels[i],
            color=colors[i],
            linewidth=2.5,
        )
        global_min = min(global_min, halv["idx"].min())
        global_max = max(global_max, halv["idx"].max())

    # Set y-axis to logarithmic scale and format labels
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
    ax.set_ylim(global_min * 0.8, global_max * 1.2)

    # Add title and legend
    ax.set_title(
        "Bitcoin Growth since Halvings", fontsize=28, fontweight="bold", pad=20
    )
    ax.legend(fontsize=18, loc="upper left")  # Changed legend position to upper left

    # Set axis labels
    ax.set_xlabel("Days since epoch start", fontsize=20, labelpad=15)
    ax.set_ylabel("Price Index (Starting price = 1)", fontsize=20, labelpad=15)

    # Show grid lines
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Add explanatory text to the plot
    ax.text(
        0.02,
        0.02,
        "Each line represents Bitcoin price growth starting from 1 at the beginning of each epoch.",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig("../output/2c.EpochsGrowth.jpg",bbox_inches="tight",dpi=350,)
    plt.show()


def plot_bitcoin_price_epochs(
):
    """
    Plots Bitcoin prices for different epochs defined by halving events.
    """
    # Load the data
    df = fetch_data("btc")
    df = df[["time", "PriceUSD"]].dropna()

    # Define epoch boundaries and their corresponding epoch numbers
    epoch_boundaries = [
        ("2012-11-28", 1),
        ("2016-07-09", 2),
        ("2020-05-11", 3),
        ("2024-04-20", 4),
        ("2028-04-20", 5),
    ]

    # Create an 'Epoch' column initialized to 0
    df["Epoch"] = 0
    for date, epoch in epoch_boundaries:
        # Assign epoch numbers based on the time values (subtract 1 to range from 0 to 4)
        df.loc[df["time"] >= date, "Epoch"] = epoch - 1

    # Set up the figure for plotting
    plt.figure(figsize=(24, 10), dpi=80)

    # Plot prices for each epoch
    for epoch in range(5):
        epoch_data = df[df["Epoch"] == epoch]
        plt.plot(
            epoch_data["time"],
            epoch_data["PriceUSD"],
            label=f"Epoch {epoch + 1} ({50 / (2**epoch)} BTC)",
            linewidth=2.5,
        )

    # Set y-axis to logarithmic scale
    plt.yscale("log")

    # Set custom ticks for y-axis to display specific price ranges
    plt.yticks([1, 10, 100, 1000, 10000, 100000])

    # Remove scientific notation from y-axis labels
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())

    # Set title and labels for the axes
    plt.title("Bitcoin Price per Epoch", fontsize=24, fontweight="bold")
    plt.ylabel("Price USD", fontsize=15)

    # Set tick size for major ticks
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Show legend with a larger font size
    plt.legend(fontsize=13)

    # Display the plot
    plt.show()
