import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import numpy as np
from fetch_data import fetch_data


def setup_style():
    plt.style.use("classic")
    return {
        "cycle_2016": "green",
        "cycle_2020": "red",
        "cycle_2024": "blue",
        "light_red": "#ff9999",
        "light_orange": "#ffcc99",
        "light_blue": "#99ccff",
        "light_green": "#99ff99",
    }


def get_price_data(df, halving_date, days_before=861, days_after=600):
    # Extend the period to include 2 more months of data before
    extended_days_before = days_before + 60

    period_data = df[
        (df["time"] >= halving_date - timedelta(days=extended_days_before))
        & (df["time"] <= halving_date + timedelta(days=days_after))
    ].copy()

    halving_price = float(
        period_data[period_data["time"].dt.date == halving_date.date()][
            "PriceUSD"
        ].iloc[0]
    )
    period_data["normalized_price"] = period_data["PriceUSD"] / halving_price
    period_data["price"] = period_data["PriceUSD"]

    # Shift the alignment date by 2 months
    shift = pd.Timedelta(days=60)
    period_data["aligned_date"] = (
        period_data["time"] - halving_date + pd.Timestamp("2024-04-19") + shift
    )

    return period_data


def setup_plot():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    return fig, ax


def format_axes(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.grid(True, which="major", ls="-", alpha=0.2, color="#666666")
    ax.grid(True, which="minor", ls=":", alpha=0.1, color="#999999")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")


def add_background_shading(ax, colors):
    # Shift shading periods by 2 months
    shift = pd.Timedelta(days=60)
    shading_periods = [
        ("2021-12-01", "2022-10-31", colors["light_red"]),
        ("2022-11-01", "2023-10-31", colors["light_orange"]),
        ("2023-11-01", "2024-10-31", colors["light_blue"]),
        ("2024-11-01", "2025-10-31", colors["light_green"]),
        ("2025-11-01", "2025-12-15", colors["light_red"]),
    ]

    for start_date, end_date, color in shading_periods:
        ax.axvspan(
            pd.Timestamp(start_date) + shift,
            pd.Timestamp(end_date) + shift,
            color=color,
            alpha=0.3,
        )


def add_halving_lines(ax):
    halving_date = pd.Timestamp("2024-04-19")
    ax.axvline(
        halving_date,
        color="grey",
        linewidth=2,
        alpha=0.8,
        label="Halving Dates",
    )

    # label at 5% range
    y_min, y_max = ax.get_ylim()
    y_position = y_min + 0.05 * (y_max - y_min)
    ax.text(
        halving_date,
        y_position,
        "Halving Dates",
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="grey",
    )


def add_vertical_lines(ax, halving_date):
    # Dates for bottoms and tops (original dates)
    bottom_dates = [
        pd.Timestamp("2015-01-14"),
        pd.Timestamp("2018-12-15"),
        pd.Timestamp("2022-11-21"),
    ]

    top_dates = [
        pd.Timestamp("2017-12-17"),
        pd.Timestamp("2021-11-10"),
    ]

    # Compute time shift - align according with 2024 halving date
    shift = pd.Timedelta(days=60)  # Shift 2 months
    alignment_reference = pd.Timestamp("2024-04-19") + shift

    # Align dates according to halving cycles
    def align_date(date, halving_date):
        return alignment_reference + (date - halving_date)

    # Align bottom/top dates
    aligned_bottom_dates = (
        [align_date(date, pd.Timestamp("2016-07-09")) for date in bottom_dates[:1]]
        + [align_date(date, pd.Timestamp("2020-05-11")) for date in bottom_dates[1:2]]
        + [align_date(date, pd.Timestamp("2024-04-19")) for date in bottom_dates[2:]]
    )

    aligned_top_dates = [
        align_date(date, pd.Timestamp("2016-07-09")) for date in top_dates[:1]
    ] + [align_date(date, pd.Timestamp("2020-05-11")) for date in top_dates[1:]]

    # Bottom vertical lines with adjusted labels
    for i, aligned_date in enumerate(aligned_bottom_dates):
        color = "green" if i == 0 else "red" if i == 1 else "blue"
        ax.axvline(aligned_date, color=color, linestyle="--", alpha=0.5)

        # Vertical offset adjustments for 2016 and 2020 labels
        x_offset = -30 if i == 0 else (-15 if i == 1 else -15) # green,red,blue
        y_offset = -20 if i == 0 else (-10 if i == 1 else -32)  # green,red,blue
        label_position = ax.get_ylim()[0] * 1.5

        ax.annotate(
            "Bottom",
            xy=(aligned_date, label_position),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=10,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0),
        )

    # Top vertical lines
    for i, aligned_date in enumerate(aligned_top_dates):
        color = "green" if i == 0 else "red"
        ax.axvline(aligned_date, color=color, linestyle="--", alpha=0.5)

        ax.annotate(
            "Top",
            xy=(aligned_date, ax.get_ylim()[1] * 0.6),
            xytext=(-8, -20 if i % 2 == 0 else -69),
            textcoords="offset points",
            fontsize=10,
            color=color,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0),
        )


def visualize_plot():
    colors = setup_style()
    df = fetch_data("btc")

    halving_dates = {
        "2016": pd.Timestamp("2016-07-09"),
        "2020": pd.Timestamp("2020-05-11"),
        "2024": pd.Timestamp("2024-04-19"),
    }

    cycle_data = {
        year: get_price_data(df, date) for year, date in halving_dates.items()
    }

    halving_price_2024 = float(
        cycle_data["2024"][
            cycle_data["2024"]["time"].dt.date == halving_dates["2024"].date()
        ]["price"].iloc[0]
    )

    for data in cycle_data.values():
        data["scaled_price"] = data["normalized_price"] * halving_price_2024

    fig, ax = setup_plot()

    add_background_shading(ax, colors)

    for year, color_key in zip(
        ["2016", "2020", "2024"], ["cycle_2016", "cycle_2020", "cycle_2024"]
    ):
        ax.plot(
            cycle_data[year]["aligned_date"],
            cycle_data[year]["scaled_price"],
            color=colors[color_key],
            label=f"{year} Halving Cycle",
            linewidth=2,
            alpha=0.8,
        )

    format_axes(ax)
    ax.set_title(
        "Bitcoin Halving Cycles Comparison: Scaled Price Movement\n",
        fontsize=20,
        fontweight="bold",
        pad=4,
    )
    ax.set_ylabel("2024 Cycle Price (Log Scale)", fontsize=12, labelpad=1)
    ax.set_xlabel("2024 Cycle Time", fontsize=12, labelpad=1)
    ax.legend(fancybox=True, shadow=True, loc="upper left", fontsize=10, framealpha=0.9)

    # Shift line
    shift = pd.Timedelta(days=75)
    start_date = pd.Timestamp("2022-01-01") + shift
    end_date = pd.Timestamp("2026-01-01") + shift
    cycle_length = pd.Timestamp("2025-12-01") - pd.Timestamp("2022-12-01")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    omega = np.pi / (cycle_length.days)
    time_shift = (pd.Timestamp("2022-12-01") - start_date).days

    sine_wave = -1 * np.sin(omega * (np.arange(len(dates)) + time_shift))
    sine_wave = (
        2 * (sine_wave - sine_wave.min()) / (sine_wave.max() - sine_wave.min()) - 1
    )

    ax2 = ax.twinx()
    ax2.plot(dates, sine_wave, color="orange", linewidth=30, alpha=0.3)
    ax2.set_ylabel("Sinusoidal Amplitude", fontsize=10)
    ax2.grid(False)
    ax2.set_yticklabels([])
    ax2.set_ylabel("")

    # Set x-axis limits with shift
    ax.set_xlim(start_date, end_date)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right", fontsize=10)

    max_price = max(data["scaled_price"].max() for data in cycle_data.values())
    min_price = min(data["scaled_price"].min() for data in cycle_data.values())
    ax.set_ylim(bottom=min_price * 0.9, top=max_price * 1.1)

    add_halving_lines(ax)
    add_vertical_lines(ax, halving_dates["2024"])

    plt.margins(x=0.02)
    plt.tight_layout()

    # Save the figure and display it.
    plt.savefig("../output/2b.CyclesNorm.jpg", dpi=400)
    plt.show()
