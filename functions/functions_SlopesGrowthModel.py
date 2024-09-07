import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


def slopes_growth_model(xlim=(2011, 2026), title="Slopes Growth Model"):
    # Load Bitcoin price data from online CSV
    df = pd.read_csv(
        "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
        parse_dates=["time"],
        usecols=["time", "PriceUSD"],
    )

    # Create a range of dates and reindex the dataframe
    date_range = pd.date_range(start=df["time"].min(), end=f"{xlim[1]}-12-31", freq="D")
    df = (
        df.set_index("time")
        .reindex(date_range)
        .reset_index()
        .rename(columns={"index": "time"})
    )
    df["index"] = range(1, len(df) + 1)

    # Define important peak dates for Bitcoin price
    tops_dates = pd.to_datetime(
        ["2011-06-08", "2013-11-30", "2017-12-17", "2021-11-10"]
    )

    def plot_regression_lines(ax, dates, cmap, label_prefix):
        """Plot regression lines for given dates and return slopes"""
        data = df[df["time"].isin(dates)]
        slopes = []
        colors = cmap(np.linspace(0, 1, len(dates) - 1))

        for i, color in enumerate(colors):
            subset = data.iloc[i : i + 2]
            X = sm.add_constant(np.log(subset["index"]))
            y = np.log(subset["PriceUSD"])

            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            slopes.append(round(slope, 2))

            line_xs = np.linspace(subset["index"].iloc[0], subset["index"].iloc[1], 100)
            line_ys = np.exp(model.predict(sm.add_constant(np.log(line_xs))))

            ax.plot(
                line_xs,
                line_ys,
                color=color,
                linestyle="--",
                label=f"{label_prefix} Line {i+1} (Slope: {slope:.2f})",
            )
        return slopes

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df["index"], df["PriceUSD"], label="PriceUSD", color="#1E88E5", linewidth=2)

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#FFA726", "#EF5350", "#AB47BC", "#66BB6A"]
    )
    slopes = plot_regression_lines(ax, tops_dates, cmap, "Top")

    # Predict next slope
    X = np.arange(len(slopes)).reshape(-1, 1)
    y = np.array(slopes)
    model = sm.OLS(y, sm.add_constant(X)).fit()
    new_slope = model.params[0] + model.params[1] * len(slopes)

    # Calculate and plot predicted future line
    last_top_date = tops_dates[-1]
    last_top_index = df[df["time"] == last_top_date]["index"].iloc[0]
    target_date = pd.to_datetime("2025-11-01")
    target_index = df[df["time"] == target_date]["index"].iloc[0]

    intercept_new_line = np.log(
        df[df["time"] == last_top_date]["PriceUSD"].iloc[0]
    ) - new_slope * np.log(last_top_index)
    new_line_xs = np.linspace(last_top_index, target_index, 100)
    new_line_ys = np.exp(new_slope * np.log(new_line_xs) + intercept_new_line)

    ax.plot(
        new_line_xs,
        new_line_ys,
        linestyle="--",
        color="#FF4081",
        linewidth=2,
        label=f"Predicted Line (Slope: {new_slope:.2f})",
    )

    # Add estimated value text
    value_at_target_date = np.exp(new_slope * np.log(target_index) + intercept_new_line)
    rounded_value = int(value_at_target_date // 1000 * 1000)
    estimated_value_text = (
        f"Estimated value for {target_date.strftime('%Y-%m-%d')}: \n${rounded_value:,}"
    )
    ax.text(
        0.98,
        0.02,
        estimated_value_text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=15,
        bbox=dict(facecolor="white", edgecolor="black", alpha=1, pad=5),
    )

    # Style the plot
    ax.set_yscale("log")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)

    year_ticks = [
        df[df["time"].dt.year == year].iloc[0]["index"]
        for year in range(xlim[0], xlim[1] + 1, 2)
    ]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(range(xlim[0], xlim[1] + 1, 2), rotation=45, fontsize=10)

    ax.set_xlim(
        df[df["time"].dt.year == xlim[0]].iloc[0]["index"],
        df[df["time"].dt.year == xlim[1]].iloc[0]["index"],
    )
    ax.set_ylim(0.5, 150000)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig("../output/1e.SlopesGrowthModel.jpg",bbox_inches="tight",dpi=350,)
    plt.show()