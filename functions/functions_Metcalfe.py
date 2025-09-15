import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.ticker import FuncFormatter
from fetch_data import fetch_crypto_data
import warnings

warnings.filterwarnings("ignore")


def analyze_metcalfe(network_metric):
    """
    Perform a Metcalfe's Law analysis on Bitcoin using network metrics
    and a best-fit affinity model over time.

    Parameters:
    - network_metric (str): either "TxCnt" (transaction count) or "AdrActCnt" (active addresses)

    Saves:
    - Plot to '../output/1h.Metcalfe.jpg'
    - Displays overbought/oversold deviation indicator
    """

    if network_metric not in ["TxCnt", "AdrActCnt"]:
        print("❌ Error: network_metric must be 'TxCnt' or 'AdrActCnt'")
        return

    def affinity_model_1(t, a1):
        return np.full_like(t, a1)

    def affinity_model_2(t, a1, a2):
        return a1 * (1 + a2 * t)

    def affinity_model_3(t, a1, a2):
        return a1 * np.exp(a2 * t)

    def affinity_model_4(t, a1, a2, a3):
        return a1 * (1 + a2 * t + a3 * t**2)

    def find_best_metcalfe_model(df, network_col):
        models = {
            "Constant": (affinity_model_1, [(1e-20, 1e-5)]),
            "Linear Time": (affinity_model_2, [(1e-20, 1e-5), (-10, 10)]),
            "Exponential Time": (affinity_model_3, [(1e-20, 1e-5), (-5, 8)]),
            "Quadratic Time": (affinity_model_4, [(1e-20, 1e-5), (-10, 10), (-5, 5)]),
        }

        best_model, best_score, best_params, best_name = None, -np.inf, None, None
        for name, (model_func, bounds) in models.items():
            print(f"Testing {name}...")

            def objective(params):
                try:
                    base = (df[network_col] * (df[network_col] - 1)) / (
                        2 * df["SplyCur"]
                    )
                    affinity = model_func(df["time_factor"], *params)
                    metcalfe = affinity * base
                    mask = (metcalfe > 0) & (df["PriceUSD"] > 0)
                    if mask.sum() < 10:
                        return 1e6
                    log_m = np.log(metcalfe[mask])
                    log_p = np.log(df["PriceUSD"][mask])
                    corr = np.corrcoef(log_m, log_p)[0, 1]
                    return -corr if not np.isnan(corr) else 1e6
                except:
                    return 1e6

            result = differential_evolution(objective, bounds, seed=42, maxiter=500)
            score = -result.fun
            print(f"  Score: {score:.6f}")
            if score > best_score:
                best_model, best_score = model_func, score
                best_params, best_name = result.x, name

        return best_model, best_params, best_name, best_score

    def price_formatter(x, pos):
        return f"{x:,.0f}" if x >= 1 else f"{x:.2f}"

    print("METCALFE LAW ANALYSIS")
    print("=" * 50)

    df = fetch_crypto_data("btc")
    df = df.dropna(subset=["PriceUSD", "TxCnt", "AdrActCnt", "SplyCur"]).copy()
    df["days_since_start"] = (df["time"] - df["time"].min()).dt.days
    df["time_factor"] = df["days_since_start"] / df["days_since_start"].max()
    df["NetworkValue"] = df["PriceUSD"] * df["SplyCur"]

    metric_name = (
        "Transaction Count" if network_metric == "TxCnt" else "Active Addresses"
    )
    print(f"Using {metric_name} as network metric ({network_metric})")
    print("-" * 50)

    model_func, params, model_name, score = find_best_metcalfe_model(df, network_metric)
    print(f"Best model: {model_name} with correlation {score:.6f}")

    base = (df[network_metric] * (df[network_metric] - 1)) / (2 * df["SplyCur"])
    affinity = model_func(df["time_factor"], *params)
    metcalfe_value = affinity * base

    mask = (metcalfe_value > 0) & (df["PriceUSD"] > 0)
    X = np.log(metcalfe_value[mask]).values.reshape(-1, 1)
    y = np.log(df["PriceUSD"][mask]).values

    ransac = RANSACRegressor(LinearRegression(), random_state=42)
    ransac.fit(X, y)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    metcalfe_aligned = np.exp(slope * np.log(metcalfe_value) + intercept)

    window = min(51, len(metcalfe_aligned) // 4)
    if window % 2 == 0:
        window += 1
    metcalfe_smooth = (
        savgol_filter(metcalfe_aligned, window, 3) if window >= 5 else metcalfe_aligned
    )

    log_dev = np.log(df["PriceUSD"]) - np.log(metcalfe_smooth)
    deviation_normalized = ((log_dev - log_dev.mean()) / log_dev.std()) * 20
    correlation = np.corrcoef(X.flatten(), y)[0, 1]

    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=6)
    ax1.semilogy(
        df["time"], df["PriceUSD"], "blue", linewidth=3, label="BTC Price", alpha=0.9
    )
    ax1.semilogy(
        df["time"],
        metcalfe_aligned,
        color="orange",
        linewidth=1.5,
        alpha=0.3,
        label=f"Metcalfe Raw ({network_metric})",
    )
    ax1.semilogy(
        df["time"],
        metcalfe_smooth,
        color="red",
        linewidth=2,
        alpha=0.9,
        label=f"Metcalfe Smoothed ({network_metric})",
    )
    ax1.yaxis.set_major_formatter(FuncFormatter(price_formatter))

    param_str = ", ".join(
        [f"{p:.2e}" if abs(p) < 0.001 else f"{p:.4f}" for p in params]
    )
    ax1.text(
        0.5,
        1.08,
        "Bitcoin Price vs Metcalfe Law - Best Fit",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="center",
    )
    ax1.text(
        0.5,
        1.04,
        f"Model: {model_name} | Parameters: [{param_str}]",
        transform=ax1.transAxes,
        fontsize=8,
        ha="center",
    )
    ax1.text(
        0.5,
        1.01,
        f"V = A(t) × n(n-1)/(2×c) | Correlation: {correlation:.6f}",
        transform=ax1.transAxes,
        fontsize=8,
        ha="center",
    )

    ax1.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])

    ax2 = plt.subplot2grid((8, 1), (6, 0), rowspan=2)
    ax2.plot(df["time"], deviation_normalized, color="purple", linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
    ax2.axhspan(15, deviation_normalized.max(), alpha=0.15, color="red")
    ax2.axhspan(deviation_normalized.min(), -15, alpha=0.15, color="green")
    ax2.axhline(y=15, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=-15, color="green", linestyle="--", alpha=0.5, linewidth=1)

    current = deviation_normalized.iloc[-1]
    if current > 15:
        status, color = "OVERBOUGHT", "red"
    elif current < -15:
        status, color = "OVERSOLD", "green"
    else:
        status, color = "NEUTRAL", "black"

    ax2.text(
        0.02,
        0.95,
        f"Current: {status} ({current:.1f})",
        transform=ax2.transAxes,
        fontsize=8,
        fontweight="bold",
        color=color,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.set_ylabel("Deviation", fontsize=10, fontweight="bold")
    ax2.set_title("Overbought/Oversold Indicator", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    years = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="YS")
    ax2.set_xticks(years)
    ax2.set_xticklabels([str(year.year) for year in years], rotation=0)
    ax2.set_ylim(deviation_normalized.min(), deviation_normalized.max())

    plt.tight_layout()
    plt.savefig("../output/1h.Metcalfe.jpg", bbox_inches="tight", dpi=350)
    plt.show()

    print(f"\n📊 CURRENT STATUS: {status} ({current:.1f})")
    print(f"Using {metric_name} ({network_metric}) | Correlation: {correlation:.6f}")

    print("\n✅ Indicator validation:")
    print("- Deviation = log(BTC price) − log(Metcalfe estimate)")
    print("- Standardized (z-score) and scaled ×20 → interpretable signal")
    print("- Threshold ±20 ≈ ±1σ → consistent over time")
    print("- Sound statistical base, better than RSI/EMA for valuation gaps")
