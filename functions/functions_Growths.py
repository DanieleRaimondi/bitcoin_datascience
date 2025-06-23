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
    Includes a trading oscillator based on the price/SMA multiplier.
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

    # Create a figure with two subplots - main plot larger, oscillator smaller
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(24, 16),
        dpi=80,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ============== UPPER SUBPLOT: Price vs SMA ==============
    ax1.set_facecolor("white")

    # Plot Bitcoin price and its SMA with better colors
    ax1.plot(
        weekly.index,
        weekly["PriceUSD"],
        label="Bitcoin Price",
        linewidth=3,
        color="#00D4AA",
        alpha=0.9,
    )
    ax1.plot(
        weekly.index,
        weekly["209SMA"],
        label="209-Week SMA",
        color="#FF6B35",
        linewidth=3,
        alpha=0.9,
    )

    # Set y-axis to logarithmic scale
    ax1.set_yscale("log")

    # Format y-axis labels
    ax1.yaxis.set_major_formatter(FuncFormatter(price_formatter))

    # Set y-axis limits based on the global min and max
    global_min = weekly["PriceUSD"].min()
    global_max = weekly["PriceUSD"].max()
    ax1.set_ylim(global_min * 0.8, global_max * 1.2)

    # Set x-axis limits to reduce empty space
    ax1.set_xlim(weekly.index.min(), weekly.index.max())

    # Add title and legend
    ax1.set_title(
        "BITCOIN PRICE vs 209-WEEK SMA",
        fontsize=28,
        fontweight="bold",
        pad=25,
        color="black",
    )
    ax1.legend(
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        facecolor="white",
        edgecolor="gray",
    )

    # Add labels to the axes
    ax1.set_ylabel("Price USD (Log Scale)", fontsize=18, labelpad=15, color="black")

    # Show grid lines
    ax1.grid(True, which="both", ls="-", alpha=0.2, color="gray")
    ax1.grid(True, which="major", ls="-", alpha=0.4, color="gray")

    # Style spines
    for spine in ax1.spines.values():
        spine.set_color("gray")
        spine.set_linewidth(1)

    # Increase the size of tick labels
    ax1.tick_params(axis="both", which="major", labelsize=14, colors="black")

    # ============== LOWER SUBPLOT: Trading Oscillator ==============
    ax2.set_facecolor("white")

    # Create trading zones with background colors
    multiplier_clean = weekly["Multiple"].dropna()

    # Define trading zones with specific color gradations
    ax2.axhspan(
        0.7, 1, alpha=0.4, color="darkgreen", label="Strong Accumulation (0.7x-1x)"
    )
    ax2.axhspan(1, 1.5, alpha=0.2, color="green", label="Accumulation (1x-1.5x)")

    # Red gradation zones
    ax2.axhspan(
        1.5, 2.5, alpha=0.15, color="orange", label="Light Distribution (1.5x-2.5x)"
    )
    ax2.axhspan(2.5, 4, alpha=0.25, color="red", label="Distribution (2.5x-4x)")
    ax2.axhspan(4, 7, alpha=0.35, color="darkred", label="Heavy Distribution (4x-7x)")
    ax2.axhspan(7, 20, alpha=0.5, color="maroon", label="Extreme Distribution (>7x)")

    # Plot the multiplier with gradient-like effect
    ax2.plot(
        weekly.index,
        weekly["Multiple"],
        color="#0066CC",
        linewidth=3,
        alpha=0.9,
        label="Price/SMA Oscillator",
    )

    # Add key horizontal reference lines
    levels = [0.7, 1, 1.5, 2.5, 4, 7, 10]
    colors = [
        "darkgreen",
        "#FF6B35",
        "#FFD700",
        "#FFA500",
        "#FF4444",
        "#CC0000",
        "#800000",
    ]
    styles = [":", "-", "--", "--", "--", ":", ":"]

    for level, color, style in zip(levels, colors, styles):
        ax2.axhline(y=level, color=color, linestyle=style, linewidth=2, alpha=0.8)

    # Add trading signals annotations
    current_multiple = multiplier_clean.iloc[-1] if len(multiplier_clean) > 0 else 1

    # Add text box with current signal
    if current_multiple < 1:
        signal_text = "🟢 STRONG ACCUMULATION"
        signal_color = "darkgreen"
    elif current_multiple < 1.5:
        signal_text = "🟢 ACCUMULATION ZONE"
        signal_color = "green"
    elif current_multiple < 2.5:
        signal_text = "🟠 LIGHT DISTRIBUTION"
        signal_color = "orange"
    elif current_multiple < 4:
        signal_text = "🔴 DISTRIBUTION ZONE"
        signal_color = "red"
    elif current_multiple < 7:
        signal_text = "🔴 HEAVY DISTRIBUTION"
        signal_color = "darkred"
    else:
        signal_text = "🔴 EXTREME DISTRIBUTION"
        signal_color = "maroon"

    # Set y-axis to logarithmic scale
    ax2.set_yscale("log")

    # Set limits starting from 0.7
    ax2.set_xlim(weekly.index.min(), weekly.index.max())
    ax2.set_ylim(0.7, 20)

    # Add current value and signal
    ax2.text(
        0.02,
        0.85,
        f"Current: {current_multiple:.2f}x",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        color="black",
    )

    ax2.text(
        0.02,
        0.65,
        signal_text,
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=signal_color, alpha=0.3),
        color="black",
    )

    # Set title and labels
    ax2.set_title(
        "TRADING OSCILLATOR (Price/SMA Ratio)",
        fontsize=20,
        fontweight="bold",
        pad=15,
        color="black",
    )
    ax2.set_xlabel("Time", fontsize=16, labelpad=10, color="black")
    ax2.set_ylabel("Multiplier", fontsize=16, labelpad=10, color="black")

    # Show grid lines
    ax2.grid(True, which="both", ls="-", alpha=0.2, color="gray")
    ax2.grid(True, which="major", ls="-", alpha=0.4, color="gray")

    # Style spines
    for spine in ax2.spines.values():
        spine.set_color("gray")
        spine.set_linewidth(1)

    # Tick parameters
    ax2.tick_params(axis="both", which="major", labelsize=12, colors="black")

    # Add y-axis labels for key levels
    ax2.set_yticks([0.7, 1, 1.5, 2.5, 4, 7, 10])
    ax2.set_yticklabels(["0.7x", "1x", "1.5x", "2.5x", "4x", "7x", "10x"])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(
        "../output/1g.BTC_SMA.jpg",
        bbox_inches="tight",
        dpi=350,
        facecolor="white",
    )
    plt.show()

    # Print trading analysis
    print(f"\n{'='*50}")
    print(f"🔹 BITCOIN TRADING OSCILLATOR ANALYSIS")
    print(f"{'='*50}")
    print(f"📊 Current Multiplier: {current_multiple:.3f}x")
    print(f"🎯 Current Signal: {signal_text}")
    print(f"📈 Historical Stats:")
    print(f"   • Mean: {multiplier_clean.mean():.3f}x")
    print(f"   • Median: {multiplier_clean.median():.3f}x")
    print(f"   • Max Peak: {multiplier_clean.max():.3f}x")
    print(f"   • Min Bottom: {multiplier_clean.min():.3f}x")
    print(f"\n📋 Zone Distribution:")
    below_1x = (multiplier_clean < 1).mean() * 100
    zone_1_15x = ((multiplier_clean >= 1) & (multiplier_clean < 1.5)).mean() * 100
    zone_15_25x = ((multiplier_clean >= 1.5) & (multiplier_clean < 2.5)).mean() * 100
    zone_25_4x = ((multiplier_clean >= 2.5) & (multiplier_clean < 4)).mean() * 100
    zone_4_7x = ((multiplier_clean >= 4) & (multiplier_clean < 7)).mean() * 100
    above_7x = (multiplier_clean >= 7).mean() * 100

    print(f"   🟢 Strong Accumulation (0.7x-1x): {below_1x:.1f}% of time")
    print(f"   🟢 Accumulation (1x-1.5x): {zone_1_15x:.1f}% of time")
    print(f"   🟠 Light Distribution (1.5x-2.5x): {zone_15_25x:.1f}% of time")
    print(f"   🔴 Distribution (2.5x-4x): {zone_25_4x:.1f}% of time")
    print(f"   🔴 Heavy Distribution (4x-7x): {zone_4_7x:.1f}% of time")
    print(f"   🔴 Extreme Distribution (>7x): {above_7x:.1f}% of time")
    print(f"{'='*50}")

    # Trading recommendations
    if current_multiple < 1:
        print(f"💡 RECOMMENDATION: Maximum allocation - Historical accumulation zone")
    elif current_multiple < 1.5:
        print(f"💡 RECOMMENDATION: High allocation - Good accumulation opportunity")
    elif current_multiple < 2.5:
        print(f"💡 RECOMMENDATION: Moderate allocation - Light distribution zone")
    elif current_multiple < 4:
        print(f"💡 RECOMMENDATION: Low allocation - Distribution zone")
    elif current_multiple < 7:
        print(f"💡 RECOMMENDATION: Consider taking profits - Heavy distribution zone")
    else:
        print(
            f"💡 RECOMMENDATION: Take profits aggressively - Extreme distribution zone"
        )


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
