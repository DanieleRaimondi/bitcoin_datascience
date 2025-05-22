import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import sys

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_data


def slopes_growth_model(
    title="Slopes Growth Model - Tops and Bottoms",
    cycles_ahead=1,
    first_top_date="2025-11-01",
    first_bottom_date="2026-11-01",
    show_all_forecasts=True,
    convergence_target=1.0,
    decay_rate_top=0.75,
    decay_rate_bottom=0.8,
    show_legend=False,
):
    """
    Bitcoin Slopes Growth Model with Exponential Decay Convergence

    Analyzes Bitcoin price cycles using regression lines connecting historical peaks and bottoms,
    then forecasts future cycles using exponential decay toward a convergence target.
    The model implements diminishing returns theory where growth rates naturally decline over time.

    Parameters
    ----------
    title : str, default "Slopes Growth Model - Tops and Bottoms"
        Chart title for the plot

    cycles_ahead : int, default 1
        Number of future cycles to predict (each cycle = 4 years apart)

    first_top_date : str, default "2025-11-01"
        Date for the first predicted peak (format: "YYYY-MM-DD")

    first_bottom_date : str, default "2026-11-01"
        Date for the first predicted bottom (format: "YYYY-MM-DD")

    show_all_forecasts : bool, default True
        Whether to show all forecast lines in legend
        - If True: Shows all predicted cycles in legend
        - If False: Shows only first cycle predictions in legend (cleaner for many cycles)

    convergence_target : float, default 1.0
        Target slope value that both tops and bottoms converge toward (market maturity level)

    decay_rate_top : float, default 0.75
        Exponential decay rate for top slopes (0-1, lower = faster convergence)

    decay_rate_bottom : float, default 0.8
        Exponential decay rate for bottom slopes (0-1, lower = faster convergence)

    show_legend : bool, default False
        Whether to display the legend showing all cycle lines

    Returns
    -------
    None
        Displays the plot and saves data to CSV file

    Notes
    -----
    Historical Data Used:
    - Bitcoin peaks: 2011-06-08, 2013-11-30, 2017-12-17, 2021-11-10
    - Bitcoin bottoms: 2011-11-18, 2015-01-14, 2018-12-15, 2022-11-21

    Exponential Decay Formula:
    - new_slope = convergence_target + (current_slope - convergence_target) * decay_rate

    Output Files:
    - Chart: ../output/1e.SlopesGrowthModel.jpg
    - Data: ../data/slopes_growth_model_data.csv (date, upper, lower)

    Examples
    --------
    Basic usage with 1 cycle ahead:
    >>> slopes_growth_model()

    Extended forecast with 4 cycles:
    >>> slopes_growth_model(cycles_ahead=4, show_legend=True)

    Custom convergence settings:
    >>> slopes_growth_model(
    ...     cycles_ahead=3,
    ...     convergence_target=1.2,
    ...     decay_rate_top=0.6,
    ...     decay_rate_bottom=0.7
    ... )
    """

    # Parse first forecast dates to calculate xlim
    first_top_date_parsed = pd.to_datetime(first_top_date)
    first_bottom_date_parsed = pd.to_datetime(first_bottom_date)

    # Calculate xlim automatically based on cycles_ahead
    last_forecast_year = max(
        first_top_date_parsed.year + (cycles_ahead - 1) * 4,
        first_bottom_date_parsed.year + (cycles_ahead - 1) * 4,
    )
    xlim = (2011, last_forecast_year + 2)  # Add 2 years buffer for visualization

    # Load Bitcoin price data
    df = fetch_data("btc")

    # Create a range of dates and reindex the dataframe
    date_range = pd.date_range(start=df["time"].min(), end=f"{xlim[1]}-12-31", freq="D")
    df = (
        df.set_index("time")
        .reindex(date_range)
        .reset_index()
        .rename(columns={"index": "time"})
    )
    df["index"] = range(1, len(df) + 1)

    # Define important peak and bottom dates for Bitcoin price
    tops_dates = pd.to_datetime(
        ["2011-06-08", "2013-11-30", "2017-12-17", "2021-11-10"]
    )

    bottom_dates = pd.to_datetime(
        [
            "2011-11-18",
            "2015-01-14",
            "2018-12-15",
            "2022-11-21",
        ]
    )

    # Define rainbow colors - perfect spectral progression from red to violet
    # Historical cycles: Red → Orange → Yellow → Green → Blue → Indigo → Violet
    cycle_colors = [
        "#FF0000",  # Pure Red (Cycle 1)
        "#FF8000",  # Orange (Cycle 2)
        "#FFFF00",  # Yellow (Cycle 3)
        "#00FF00",  # Green (Cycle 4)
        "#0080FF",  # Blue (Cycle 5)
        "#4000FF",  # Indigo (Cycle 6)
        "#8000FF",  # Violet (Cycle 7)
    ]

    # Forecast colors: Deeper/darker versions following same rainbow progression
    # Maintaining spectral order but with richer saturation for distinction
    forecast_colors = [
        "#CC0000",  # Dark Red (Forecast 1)
        "#CC6600",  # Dark Orange (Forecast 2)
        "#B8B800",  # Dark Yellow (Forecast 3)
        "#00CC00",  # Dark Green (Forecast 4)
        "#0066CC",  # Dark Blue (Forecast 5)
        "#3300CC",  # Dark Indigo (Forecast 6)
        "#6600CC",  # Dark Violet (Forecast 7)
    ]

    def plot_regression_lines(ax, dates, label_prefix, linestyle="--", colors=None):
        """Plot regression lines for given dates and return slopes and lines for legend"""
        data = df[df["time"].isin(dates)]
        slopes = []
        lines = []

        for i in range(len(dates) - 1):
            color = colors[i] if colors else cycle_colors[i]
            subset = data.iloc[i : i + 2]
            X = sm.add_constant(np.log(subset["index"]))
            y = np.log(subset["PriceUSD"])

            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            slopes.append(round(slope, 2))

            line_xs = np.linspace(subset["index"].iloc[0], subset["index"].iloc[1], 100)
            line_ys = np.exp(model.predict(sm.add_constant(np.log(line_xs))))

            (line,) = ax.plot(
                line_xs,
                line_ys,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"Cycle {i+1} {label_prefix} (Slope: {slope:.2f})",
            )
            lines.append(line)
        return slopes, lines

    def predict_next_slope_exponential_decay(
        top_slopes,
        bottom_slopes,
        convergence_target=1.0,
        decay_rate_top=0.75,
        decay_rate_bottom=0.8,
    ):
        """Predict next slopes using exponential decay toward convergence"""

        # Current slopes
        current_top = top_slopes[-1]
        current_bottom = bottom_slopes[-1]

        # Calculate how far we are from convergence target
        distance_top = current_top - convergence_target
        distance_bottom = current_bottom - convergence_target

        # Apply exponential decay - each cycle reduces distance to target
        new_distance_top = distance_top * decay_rate_top
        new_distance_bottom = distance_bottom * decay_rate_bottom

        # Calculate new slopes
        predicted_slope_top = convergence_target + new_distance_top
        predicted_slope_bottom = convergence_target + new_distance_bottom

        # Apply minimal constraints
        predicted_slope_top = max(predicted_slope_top, 0.1)
        predicted_slope_bottom = max(predicted_slope_bottom, 0.05)

        return predicted_slope_top, predicted_slope_bottom

    def calculate_price_at_date(start_date, start_price, target_date, slope, df):
        """Calculate price at target date given start conditions and slope"""
        start_index = df[df["time"] == start_date]["index"].iloc[0]
        target_index = df[df["time"] == target_date]["index"].iloc[0]

        intercept = np.log(start_price) - slope * np.log(start_index)
        predicted_price = np.exp(slope * np.log(target_index) + intercept)

        return predicted_price, start_index, target_index, intercept

    def plot_forecast_line(
        ax, start_date, start_price, target_date, slope, df, color, linestyle, label
    ):
        """Plot a single forecast line and return the line object"""
        predicted_price, start_index, target_index, intercept = calculate_price_at_date(
            start_date, start_price, target_date, slope, df
        )

        line_xs = np.linspace(start_index, target_index, 100)
        line_ys = np.exp(slope * np.log(line_xs) + intercept)

        (line,) = ax.plot(
            line_xs, line_ys, linestyle=linestyle, color=color, linewidth=3, label=label
        )
        return line, predicted_price, target_index

    # Create single plot
    fig, ax = plt.subplots(figsize=(16, 10))
    # Plot Bitcoin price without label for legend
    ax.plot(df["index"], df["PriceUSD"], color="#1E88E5", linewidth=2)

    # ADD HISTORICAL BUBBLES FIRST - so they're visible
    for date in tops_dates:
        price = df[df["time"] == date]["PriceUSD"].iloc[0]
        index = df[df["time"] == date]["index"].iloc[0]
        ax.scatter(
            index, price, color="green", s=100, zorder=15, alpha=0.8, edgecolors="black"
        )

    for date in bottom_dates:
        price = df[df["time"] == date]["PriceUSD"].iloc[0]
        index = df[df["time"] == date]["index"].iloc[0]
        ax.scatter(
            index, price, color="red", s=100, zorder=15, alpha=0.8, edgecolors="black"
        )

    # Plot historical tops and bottoms
    slopes_tops, lines_tops = plot_regression_lines(
        ax, tops_dates, "Top", "--", cycle_colors
    )
    slopes_bottoms, lines_bottoms = plot_regression_lines(
        ax, bottom_dates, "Bottom", "-.", cycle_colors
    )

    # Initialize lists for forecast lines and prediction points
    forecast_lines_tops = []
    forecast_lines_bottoms = []
    prediction_points = []
    predicted_slopes_info = []

    # Generate forecasts for requested cycles
    current_top_slopes = slopes_tops.copy()
    current_bottom_slopes = slopes_bottoms.copy()

    for cycle in range(cycles_ahead):
        # Calculate target dates for this cycle
        years_ahead = cycle * 4
        target_top_date = first_top_date_parsed + pd.DateOffset(years=years_ahead)
        target_bottom_date = first_bottom_date_parsed + pd.DateOffset(years=years_ahead)

        # Predict slopes using exponential decay toward convergence
        predicted_slope_top, predicted_slope_bottom = (
            predict_next_slope_exponential_decay(
                current_top_slopes,
                current_bottom_slopes,
                convergence_target,
                decay_rate_top,
                decay_rate_bottom,
            )
        )

        predicted_slopes_info.append(
            {
                "cycle": cycle + 1,
                "top_slope": predicted_slope_top,
                "bottom_slope": predicted_slope_bottom,
            }
        )

        # Get starting points for this cycle
        if cycle == 0:
            # First cycle starts from last historical points
            start_top_date = tops_dates[-1]
            start_top_price = df[df["time"] == start_top_date]["PriceUSD"].iloc[0]
            start_bottom_date = bottom_dates[-1]
            start_bottom_price = df[df["time"] == start_bottom_date]["PriceUSD"].iloc[0]
        else:
            # Subsequent cycles start from previous predictions
            start_top_date = prev_target_top_date
            start_top_price = prev_predicted_top_price
            start_bottom_date = prev_target_bottom_date
            start_bottom_price = prev_predicted_bottom_price

        # Create labels for this cycle
        cycle_label = f"Cycle {cycle+1}" if cycle > 0 else ""
        top_label = (
            f"Predicted Top {cycle_label}(Slope: {predicted_slope_top:.2f})".strip()
        )
        bottom_label = f"Predicted Bottom {cycle_label}(Slope: {predicted_slope_bottom:.2f})".strip()

        # Plot forecast lines
        top_line, predicted_top_price, target_top_index = plot_forecast_line(
            ax,
            start_top_date,
            start_top_price,
            target_top_date,
            predicted_slope_top,
            df,
            forecast_colors[cycle % len(forecast_colors)],
            "--",
            top_label,
        )

        bottom_line, predicted_bottom_price, target_bottom_index = plot_forecast_line(
            ax,
            start_bottom_date,
            start_bottom_price,
            target_bottom_date,
            predicted_slope_bottom,
            df,
            forecast_colors[cycle % len(forecast_colors)],
            "-.",
            bottom_label,
        )

        forecast_lines_tops.append(top_line)
        forecast_lines_bottoms.append(bottom_line)

        # Add prediction points (bubbles)
        ax.scatter(
            target_top_index,
            predicted_top_price,
            color="green",
            s=100,
            zorder=5,
            alpha=0.8,
            edgecolors="black",
        )
        ax.scatter(
            target_bottom_index,
            predicted_bottom_price,
            color="red",
            s=100,
            zorder=5,
            alpha=0.8,
            edgecolors="black",
        )

        # Store prediction info for text box
        prediction_points.append(
            {
                "type": "top",
                "date": target_top_date,
                "price": predicted_top_price,
                "cycle": cycle + 1,
            }
        )
        prediction_points.append(
            {
                "type": "bottom",
                "date": target_bottom_date,
                "price": predicted_bottom_price,
                "cycle": cycle + 1,
            }
        )

        # Update slopes for next iteration
        current_top_slopes.append(predicted_slope_top)
        current_bottom_slopes.append(predicted_slope_bottom)

        # Store for next cycle
        prev_target_top_date = target_top_date
        prev_predicted_top_price = predicted_top_price
        prev_target_bottom_date = target_bottom_date
        prev_predicted_bottom_price = predicted_bottom_price

    # Create prediction text
    prediction_text = "PREDICTIONS:\n"
    for point in prediction_points:
        rounded_price = int(point["price"] // 1000 * 1000)
        point_type = "Peak" if point["type"] == "top" else "Bottom"
        prediction_text += (
            f"{point_type} {point['date'].strftime('%Y-%m-%d')}: ${rounded_price:,}\n"
        )

    ax.text(
        0.98,
        0.02,
        prediction_text.strip(),
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9, pad=10),
    )

    # Create legend based on show_all_forecasts parameter
    if cycles_ahead == 1 or show_all_forecasts:
        # Show all forecast lines
        forecast_lines_to_show = forecast_lines_bottoms + forecast_lines_tops
    else:
        # Show only first cycle forecasts
        forecast_lines_to_show = [forecast_lines_bottoms[0], forecast_lines_tops[0]]

    # Combine all lines for the legend
    all_lines = lines_bottoms + forecast_lines_to_show + lines_tops
    all_labels = [line.get_label() for line in all_lines]

    # Create the legend only if requested
    if show_legend:
        legend = ax.legend(
            all_lines,
            all_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
            ncol=1,
        )

    # Style the plot
    ax.set_yscale("log")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Set up axes
    year_ticks = [
        df[df["time"].dt.year == year].iloc[0]["index"]
        for year in range(xlim[0], xlim[1] + 1, 2)
    ]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(range(xlim[0], xlim[1] + 1, 2), rotation=45, fontsize=12)
    ax.set_xlim(
        df[df["time"].dt.year == xlim[0]].iloc[0]["index"],
        df[df["time"].dt.year == xlim[1]].iloc[0]["index"],
    )
    ax.set_ylim(0.5, 1000000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Print slope analysis
    print("SLOPE ANALYSIS:")
    print(f"Historical top slopes: {slopes_tops}")
    print(f"Historical bottom slopes: {slopes_bottoms}")
    print(f"Predicted slopes for {cycles_ahead} cycle(s) (with corridor constraints):")

    for slope_info in predicted_slopes_info:
        print(
            f"  Cycle {slope_info['cycle']} - Top: {slope_info['top_slope']:.3f}, Bottom: {slope_info['bottom_slope']:.3f}"
        )

    print(
        f"Average slope decline (historical tops): {np.mean(np.diff(slopes_tops)):.3f}"
    )
    print(
        f"Average slope decline (historical bottoms): {np.mean(np.diff(slopes_bottoms)):.3f}"
    )
    print(
        f"Exponential decay model: convergence_target={convergence_target}, decay_rates=(top:{decay_rate_top}, bottom:{decay_rate_bottom})"
    )

    # Show slope progression with convergence analysis
    print("\nSlope progression (Exponential Decay Model):")
    print("Historical tops:", [round(s, 3) for s in slopes_tops])
    print(
        "Predicted tops: ",
        [
            round(predicted_slopes_info[i]["top_slope"], 3)
            for i in range(len(predicted_slopes_info))
        ],
    )
    print("Historical bottoms:", [round(s, 3) for s in slopes_bottoms])
    print(
        "Predicted bottoms: ",
        [
            round(predicted_slopes_info[i]["bottom_slope"], 3)
            for i in range(len(predicted_slopes_info))
        ],
    )

    # Show convergence progress
    if len(predicted_slopes_info) > 0:
        gap_reduction = []
        last_historical_gap = abs(slopes_tops[-1] - slopes_bottoms[-1])

        for i, info in enumerate(predicted_slopes_info):
            current_gap = abs(info["top_slope"] - info["bottom_slope"])
            gap_reduction.append(current_gap)

        print(f"\nConvergence analysis:")
        print(f"Last historical gap: {last_historical_gap:.3f}")
        print(f"Predicted gaps: {[round(gap, 3) for gap in gap_reduction]}")
        print(
            f"Gap reduction: {((last_historical_gap - gap_reduction[-1]) / last_historical_gap * 100):.1f}% over {cycles_ahead} cycles"
        )

    # Extract and save data to CSV
    save_data_to_csv(
        df,
        tops_dates,
        bottom_dates,
        predicted_slopes_info,
        first_top_date_parsed,
        first_bottom_date_parsed,
        slopes_tops,
        slopes_bottoms,
    )

    plt.tight_layout()
    plt.savefig("../output/1e.SlopesGrowthModel.jpg", bbox_inches="tight", dpi=350)
    plt.show()


def save_data_to_csv(
    df,
    tops_dates,
    bottom_dates,
    predicted_slopes_info,
    first_top_date_parsed,
    first_bottom_date_parsed,
    slopes_tops,
    slopes_bottoms,
):
    """Extract and save the upper and lower series data to CSV"""

    # Create a comprehensive date range for the entire period
    start_date = df["time"].min()
    end_date = df["time"].max()

    # Create daily date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    result_df = pd.DataFrame({"date": full_date_range})
    result_df["upper"] = np.nan
    result_df["lower"] = np.nan

    def calculate_line_values(
        start_date, start_price, end_date, end_price, slope, df_ref
    ):
        """Calculate values along a regression line"""
        start_idx = df_ref[df_ref["time"] == start_date]["index"].iloc[0]
        end_idx = df_ref[df_ref["time"] == end_date]["index"].iloc[0]

        # Calculate intercept
        intercept = np.log(start_price) - slope * np.log(start_idx)

        # Get all dates between start and end
        mask = (df_ref["time"] >= start_date) & (df_ref["time"] <= end_date)
        date_subset = df_ref[mask].copy()

        # Calculate values for each date
        values = np.exp(slope * np.log(date_subset["index"]) + intercept)

        return date_subset["time"].values, values

    # Fill historical top lines
    for i in range(len(tops_dates) - 1):
        start_date = tops_dates[i]
        end_date = tops_dates[i + 1]
        start_price = df[df["time"] == start_date]["PriceUSD"].iloc[0]
        end_price = df[df["time"] == end_date]["PriceUSD"].iloc[0]
        slope = slopes_tops[i]

        dates, values = calculate_line_values(
            start_date, start_price, end_date, end_price, slope, df
        )

        # Update result_df
        for date, value in zip(dates, values):
            idx = result_df[result_df["date"] == pd.to_datetime(date)].index
            if len(idx) > 0:
                current_val = result_df.loc[idx[0], "upper"]
                if pd.isna(current_val) or value > current_val:
                    result_df.loc[idx[0], "upper"] = value

    # Fill historical bottom lines
    for i in range(len(bottom_dates) - 1):
        start_date = bottom_dates[i]
        end_date = bottom_dates[i + 1]
        start_price = df[df["time"] == start_date]["PriceUSD"].iloc[0]
        end_price = df[df["time"] == end_date]["PriceUSD"].iloc[0]
        slope = slopes_bottoms[i]

        dates, values = calculate_line_values(
            start_date, start_price, end_date, end_price, slope, df
        )

        # Update result_df
        for date, value in zip(dates, values):
            idx = result_df[result_df["date"] == pd.to_datetime(date)].index
            if len(idx) > 0:
                current_val = result_df.loc[idx[0], "lower"]
                if pd.isna(current_val) or value < current_val:
                    result_df.loc[idx[0], "lower"] = value

    # Fill predicted lines
    current_top_slopes = slopes_tops.copy()
    current_bottom_slopes = slopes_bottoms.copy()

    for cycle, slope_info in enumerate(predicted_slopes_info):
        # Calculate target dates for this cycle
        years_ahead = cycle * 4
        target_top_date = first_top_date_parsed + pd.DateOffset(years=years_ahead)
        target_bottom_date = first_bottom_date_parsed + pd.DateOffset(years=years_ahead)

        # Get starting points for this cycle
        if cycle == 0:
            start_top_date = tops_dates[-1]
            start_top_price = df[df["time"] == start_top_date]["PriceUSD"].iloc[0]
            start_bottom_date = bottom_dates[-1]
            start_bottom_price = df[df["time"] == start_bottom_date]["PriceUSD"].iloc[0]
        else:
            start_top_date = prev_target_top_date
            start_top_price = prev_predicted_top_price
            start_bottom_date = prev_target_bottom_date
            start_bottom_price = prev_predicted_bottom_price

        # Calculate target prices
        start_top_idx = df[df["time"] == start_top_date]["index"].iloc[0]
        target_top_idx = df[df["time"] == target_top_date]["index"].iloc[0]
        intercept_top = np.log(start_top_price) - slope_info["top_slope"] * np.log(
            start_top_idx
        )
        target_top_price = np.exp(
            slope_info["top_slope"] * np.log(target_top_idx) + intercept_top
        )

        start_bottom_idx = df[df["time"] == start_bottom_date]["index"].iloc[0]
        target_bottom_idx = df[df["time"] == target_bottom_date]["index"].iloc[0]
        intercept_bottom = np.log(start_bottom_price) - slope_info[
            "bottom_slope"
        ] * np.log(start_bottom_idx)
        target_bottom_price = np.exp(
            slope_info["bottom_slope"] * np.log(target_bottom_idx) + intercept_bottom
        )

        # Fill predicted top line
        dates, values = calculate_line_values(
            start_top_date,
            start_top_price,
            target_top_date,
            target_top_price,
            slope_info["top_slope"],
            df,
        )

        for date, value in zip(dates, values):
            idx = result_df[result_df["date"] == pd.to_datetime(date)].index
            if len(idx) > 0:
                current_val = result_df.loc[idx[0], "upper"]
                if pd.isna(current_val) or value > current_val:
                    result_df.loc[idx[0], "upper"] = value

        # Fill predicted bottom line
        dates, values = calculate_line_values(
            start_bottom_date,
            start_bottom_price,
            target_bottom_date,
            target_bottom_price,
            slope_info["bottom_slope"],
            df,
        )

        for date, value in zip(dates, values):
            idx = result_df[result_df["date"] == pd.to_datetime(date)].index
            if len(idx) > 0:
                current_val = result_df.loc[idx[0], "lower"]
                if pd.isna(current_val) or value < current_val:
                    result_df.loc[idx[0], "lower"] = value

        # Store for next cycle
        prev_target_top_date = target_top_date
        prev_predicted_top_price = target_top_price
        prev_target_bottom_date = target_bottom_date
        prev_predicted_bottom_price = target_bottom_price

    # Clean up the data - remove rows where both upper and lower are NaN
    result_df = result_df.dropna(subset=["upper", "lower"], how="all")

    # Forward fill to create continuous lines
    result_df["upper"] = result_df["upper"].fillna(method="ffill")
    result_df["lower"] = result_df["lower"].fillna(method="ffill")

    # Create the data directory if it doesn't exist
    import os

    os.makedirs("../data", exist_ok=True)

    # Save to CSV
    csv_path = "../data/slopes_growth.csv"
    result_df.to_csv(csv_path, index=False)

    print(f"\nData saved to: {csv_path}")
    print(
        f"CSV contains {len(result_df)} rows from {result_df['date'].min().strftime('%Y-%m-%d')} to {result_df['date'].max().strftime('%Y-%m-%d')}"
    )
    print(
        f"Upper range: ${result_df['upper'].min():.0f} - ${result_df['upper'].max():.0f}"
    )
    print(
        f"Lower range: ${result_df['lower'].min():.0f} - ${result_df['lower'].max():.0f}"
    )
