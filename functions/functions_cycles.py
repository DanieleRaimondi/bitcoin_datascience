# Import necessary libraries for data handling and visualization.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# function to add annotations with dates to a plot.
def add_annotations_with_dates(ax, dates, label, color, alpha):
    """
    Add vertical lines and text annotations with dates to a plot

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object to add annotations to
        dates (list): List of objects representing the dates to annotate
        label (str): Label to annotate with
        color (str): Color of the annotations and vertical lines
        alpha (float): level of transparency
    """
    for date in dates:
        ax.axvline(
            date, color=color, linestyle="--", alpha=alpha
        )  # Draw vertical line at each date
        if date >= pd.to_datetime(
            "2012-01-01"
        ):  # Add text annotations only for dates after Jan 1, 2012
            ax.text(
                date,
                6,
                f'{label}\n{date.strftime("%Y-%m-%d")}',
                verticalalignment="top",
                color=color,
                fontsize=8,
                ha="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    boxstyle="round,pad=0.5",
                ),
            )


# function to annotate cycle phases on the cyclical pattern.
def cycle_annotation_phase(
    ax1,
    ax2,
    fig,
    cycle_dates,
    sin_derivative,
    zero_crossings,
    sin_minima,
    sin_maxima,
    halving_dates,
    tops_dates,
    bottoms_dates,
    next_peak_prediction,
    next_peak_prediction_lower,
    next_peak_prediction_upper,
):
    """
    Annotate cycle phases (maxima, minima, and zero crossings) on the cyclical pattern

    Parameters:
        ax1 (matplotlib.axes.Axes): The first Axes object
        ax2 (matplotlib.axes.Axes): The second Axes object
        fig (matplotlib.figure.Figure): The Figure object containing the subplots
        cycle_dates (array): Array of datetime objects representing the time series
        sin_derivative (array): Array of sine wave derivatives
        zero_crossings (array): Array of indices representing zero crossings
        sin_minima (array): Array of indices representing minima
        sin_maxima (array): Array of indices representing maxima
        halving_dates (list): List of halving dates
        tops_dates (list): List of cycle top dates
        bottoms_dates (list): List of cycle bottom dates
        next_peak_prediction (datetime): Next peak prediction date
        next_peak_prediction_lower (datetime): Next peak prediction date lower band
        next_peak_prediction_upper (datetime): Next peak prediction date upper band
    """
    # Annotate "25%" and "75%" phases for zero crossings, and "0%" and "50%" for bottoms and tops
    for i in zero_crossings:
        if cycle_dates[i] >= pd.to_datetime(
            "2012-01-01"
        ):  # Only annotate if date is after Jan 1, 2012.
            phase = "25%" if sin_derivative[i] > 0 else "75%"
            ax2.text(
                cycle_dates[i] + pd.Timedelta(days=90),
                0,
                phase,
                color="b",
                fontsize=10,
                ha="right",
                va="bottom",
            )

    for i in sin_minima:
        if cycle_dates[i] >= pd.to_datetime("2012-01-01"):
            ax2.text(
                cycle_dates[i] + pd.Timedelta(days=70),
                -0.9,
                "0%",
                color="b",
                fontsize=10,
                ha="right",
                va="bottom",
            )

    for i in sin_maxima:
        if cycle_dates[i] >= pd.to_datetime("2012-01-01"):
            ax2.text(
                cycle_dates[i] + pd.Timedelta(days=80),
                0.8,
                "50%",
                color="b",
                fontsize=10,
                ha="right",
                va="bottom",
            )

    # Set the X axis and Y axis limits and configure the date formatting for the X-axis
    ax2.set_xlim(pd.to_datetime("2012-01-01"), pd.to_datetime("2026-07-31"))
    ax1.set_ylim(3, 100_000)
    ax2.set_ylim(-1.02, 1.05)
    ax2.set_ylabel("Cycles", fontsize=13)
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to years
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Add vertical lines for all significant dates on the second subplot
    for date in halving_dates:
        ax2.axvline(date, color="orange", linestyle="--", alpha=0.2)
    for date in tops_dates:
        ax2.axvline(date, color="green", linestyle="--", alpha=0.5)
    for date in bottoms_dates:
        ax2.axvline(date, color="red", linestyle="--", alpha=0.5)
    ax2.axvline(next_peak_prediction, color="green", linestyle="--", alpha=0.5)
    ax2.fill_between(
        x=[next_peak_prediction_lower, next_peak_prediction_upper],
        y1=ax2.get_ylim()[0],
        y2=ax2.get_ylim()[1],
        color="green",
        alpha=0.15,
    )

    # Add "Cycle" labels for each major peak date and for the next top prediction
    cycle_labels = ["Cycle 1", "Cycle 2", "Cycle 3"]
    for i, date in enumerate(tops_dates):
        ax2.text(
            date,
            -0.95,
            cycle_labels[i],
            color="b",
            fontsize=13,
            ha="center",
            fontweight="bold",
        )

    # Add the label "Cycle 4" for the next top prediction
    ax2.text(
        next_peak_prediction,
        -0.95,
        "Cycle 4",
        color="b",
        fontsize=13,
        ha="center",
        fontweight="bold",
    )


def plot_bitcoin_cycles(
    btc_data,
    halving_dates,
    tops_dates,
    bottoms_dates,
    next_peak_prediction,
    next_peak_prediction_lower,
    next_peak_prediction_upper,
    cycle_dates,
    cycle_wave,
    sin_derivative,
    zero_crossings,
    sin_minima,
    sin_maxima,
    current,
):
    """
    Plot Bitcoin cycles with annotations and cycle phases

    Parameters:
        btc_data (pd.DataFrame): DataFrame containing Bitcoin price data
        halving_dates (list): List of halving dates
        tops_dates (list): List of top dates
        bottoms_dates (list): List of bottom dates
        next_peak_prediction (datetime): Next peak prediction date
        cycle_dates (array): Array of datetime objects representing the time series
        cycle_wave (array): Array representing the cyclical pattern
        sin_derivative (array): Array representing the derivative of the cycle
        zero_crossings (array): Array of indices representing zero crossings
        sin_minima (array): Array of indices representing bottoms
        sin_maxima (array): Array of indices representing tops
    """
    # Initialize the figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.subplots_adjust(hspace=0)  # Adjust space between subplots.

    # Plot Bitcoin price data on the first subplot.
    ax1.plot(btc_data["time"], btc_data["PriceUSD"], color="k", label="BTC Price USD")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price USD", fontsize=13)
    ax1.set_title(
        "Bitcoin Cycles",
        fontsize=25,
        fontweight="bold",
    )
    ax1.grid(axis="y", ls="--", alpha=0.4)

    # Add annotations with dates for halvings, tops, bottoms, and next peak prediction on the first subplot.
    add_annotations_with_dates(ax1, halving_dates, "HALVING", "orange", 0.2)
    add_annotations_with_dates(ax1, tops_dates, "TOP", "green", 0.5)
    add_annotations_with_dates(ax1, bottoms_dates, "BOTTOM", "red", 0.5)
    ax1.axvline(next_peak_prediction, color="green", linestyle="--", alpha=0.5)
    ax1.text(
        next_peak_prediction,
        6,
        f'NEXT TOP\n{next_peak_prediction_lower.strftime("%b")}-{next_peak_prediction_upper.strftime("%b")} {next_peak_prediction.strftime("%y")}',
        verticalalignment="top",
        color="green",
        fontsize=8,
        ha="center",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            boxstyle="round,pad=0.5",
        ),
    )

    # Fill between the dates
    ax1.fill_between(
        x=pd.date_range(
            start=next_peak_prediction_lower, end=next_peak_prediction_upper, freq="D"
        ),
        y1=ax1.get_ylim()[0],
        y2=ax1.get_ylim()[1],
        color="green",
        alpha=0.15,
    )

    # Add BTC logo
    img_path = "../utils/btc_logo.png"
    img = plt.imread(img_path)
    left, width = 0.004, 0.2
    bottom, height = 0.88, 0.094
    right = left + width
    top = bottom + height
    ax1.imshow(
        img,
        aspect="auto",
        extent=[left, right, bottom, top],
        transform=ax1.transAxes,
        zorder=-1,
    )

    # Add text in the upper left area of the first subplot
    ax1.text(
        datetime(2012, 3, 1),
        18_000,
        f"Current cycle position: {current}",
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white"),
    )

    # Prepare segments with conditional colors.
    points = np.array([mdates.date2num(cycle_dates), cycle_wave]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a list of colors based on conditions.
    colors = []
    norm = Normalize(vmin=np.min(cycle_wave), vmax=np.max(cycle_wave))
    cmap = plt.get_cmap("RdYlGn")

    for i in range(len(segments)):
        if i in sin_maxima:
            colors.append("green")  # Maxima in green
        elif i in zero_crossings and sin_derivative[i] < 0:
            colors.append("red")  # Flex points with negative derivative in red
        else:
            val = norm(cycle_wave[i])
            colors.append(cmap(val))

    lc = LineCollection(segments, colors=colors, linewidth=4)
    ax2.add_collection(lc)
    ax2.autoscale_view()

    # Fill between 50% and 75% phases in red with alpha=0.3
    for i in range(len(tops_dates)):
        if i < len(bottoms_dates):
            minima_indices = np.where(
                (cycle_dates > tops_dates[i]) & (cycle_dates < bottoms_dates[i])
            )[0]
        else:
            minima_indices = np.where(cycle_dates > tops_dates[i])[0]
        ax2.fill_between(
            x=cycle_dates[minima_indices],
            y1=-1.02,
            y2=1.05,
            color="red",
            alpha=0.15,
        )

    # Fill green areas
    ax2.fill_between(
        x=cycle_dates[cycle_dates < tops_dates[0]],
        y1=-1.02,
        y2=1.05,
        color="green",
        alpha=0.15,
    )
    for i in range(len(bottoms_dates) - 1):
        minima_indices = np.where(
            (cycle_dates > bottoms_dates[i]) & (cycle_dates < tops_dates[i + 1])
        )[0]
        ax2.fill_between(
            x=cycle_dates[minima_indices],
            y1=-1.02,
            y2=1.05,
            color="green",
            alpha=0.15,
        )

    # Fill last green area
    ax2.fill_between(
        x=cycle_dates[
            (cycle_dates > bottoms_dates[-1]) & (cycle_dates < next_peak_prediction)
        ],
        y1=-1.02,
        y2=1.05,
        color="green",
        alpha=0.15,
    )

    # Fill area to the right of the next top in red
    ax2.fill_between(
        x=cycle_dates[cycle_dates > next_peak_prediction],
        y1=-1.02,
        y2=1.05,
        color="red",
        alpha=0.15,
    )

    # Configure limits and formatting for the second subplot.
    cycle_annotation_phase(
        ax1,
        ax2,
        fig,
        cycle_dates,
        sin_derivative,
        zero_crossings,
        sin_minima,
        sin_maxima,
        halving_dates,
        tops_dates,
        bottoms_dates,
        next_peak_prediction,
        next_peak_prediction_lower,
        next_peak_prediction_upper,
    )
    ax1.text(
        btc_data.time.iloc[2800],
        100,
        "@Daniele Raimondi",
        fontsize=25,
        color="grey",
        alpha=0.25,
    )
    # Save the figure and display it.
    plt.savefig("../output/Cycles.jpeg", dpi=400)
    plt.show()


def calculate_cycle_percentage(btc_data, cycle_dates, sin_minima, sin_maxima):
    """
    Calculate the exact percentage phase of the cycle for the last date with a price value

    Parameters:
    - btc_data: DataFrame containing Bitcoin price data with a 'time' column
    - cycle_dates: Array of datetime objects representing the cycle time series
    - sin_minima: Array of indices representing the minima of the cycle
    - sin_maxima: Array of indices representing the maxima of the cycle

    Returns:
    - percentage_of_cycle: The exact percentage of the cycle for the last date with a price value
    """
    # Find the last price date in btc_data
    last_price_date = btc_data["time"].max()

    # Find the index of the last price date within cycle_dates
    last_price_index = np.where(cycle_dates == last_price_date)[0][0]

    # Find the last significant point (minima or maxima) before the last price date
    last_minima_before_last_price = sin_minima[sin_minima < last_price_index]
    last_maxima_before_last_price = sin_maxima[sin_maxima < last_price_index]
    last_significant_point_index = max(
        (
            last_minima_before_last_price[-1]
            if last_minima_before_last_price.size > 0
            else 0
        ),
        (
            last_maxima_before_last_price[-1]
            if last_maxima_before_last_price.size > 0
            else 0
        ),
    )

    # Calculate the distance from the last significant point to the last price date
    distance_from_last_significant_point = (
        last_price_index - last_significant_point_index
    )

    # Calculate the approximate cycle length using the average distance between maxima as a reference
    approx_cycle_length = (
        np.mean(np.diff(sin_maxima)) if sin_maxima.size > 1 else cycle_dates.size
    )

    # Calculate the exact percentage of the cycle for the last price date
    percentage_of_cycle = round(
        (distance_from_last_significant_point / approx_cycle_length) * 100, 1
    )

    return f"{percentage_of_cycle}%"
