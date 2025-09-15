import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from fredapi import Fred
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.impute import KNNImputer
from scipy.signal import argrelextrema
import warnings

from fetch_data import fetch_crypto_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Cycle constants
HALVING_DATES = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"])
TOPS_DATES = pd.to_datetime(["2011-06-08", "2013-11-30", "2017-12-17", "2021-11-10"])
BOTTOMS_DATES = pd.to_datetime(["2011-11-19", "2015-01-14", "2018-12-15", "2022-11-21"])


def load_btc_data():
    """Load Bitcoin data from CoinMetrics"""
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    df = pd.read_csv(url, parse_dates=["time"], low_memory=False)
    df = df.sort_values("time").dropna(subset=["PriceUSD"]).reset_index(drop=True)

    # Drop unnecessary columns
    cols_to_drop = [
        "principal_market_price_usd",
        "principal_market_usd",
        "ReferenceRate",
        "ReferenceRateETH",
        "ReferenceRateEUR",
        "ReferenceRateUSD",
        "CapMrktEstUSD",
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    print(f"✅ Bitcoin data loaded: {len(df)} rows")
    return df


def load_market_data(start_date):
    """Load market data from Yahoo Finance"""
    end_date = datetime.today().strftime("%Y-%m-%d")
    market_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    tickers = {
        "^GSPC": "SP500",
        "^IXIC": "NASDAQ",
        "^VIX": "VIX",
        "DX-Y.NYB": "DXY",
        "^TNX": "US10Y",
        "GC=F": "GOLD",
        "CL=F": "OIL",
        "BTC-USD": "BTC_VOLUME",
    }
    successful = 0
    for ticker, name in tickers.items():
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                timeout=10,
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                col = "Volume" if name == "BTC_VOLUME" else "Close"
                if col in data.columns:
                    market_data[name] = data[col]
                    successful += 1
            time.sleep(0.2)
        except Exception:
            continue
    print(f"✅ Market data loaded: {successful}/{len(tickers)} tickers")
    return market_data.ffill().bfill()


def load_fred_data(start_date):
    """Load FRED economic data"""
    fred_api_key = os.getenv("FRED_API")
    if not fred_api_key:
        print("⚠️ FRED API key not found")
        return None

    fred = Fred(api_key=fred_api_key)
    series = {
        "FEDFUNDS": "FED_FUNDS_RATE",
        "CPIAUCSL": "CPI",
        "M2SL": "M2_MONEY_SUPPLY",
        "UNRATE": "UNEMPLOYMENT",
        "T10Y2Y": "YIELD_SPREAD",
        "UMCSENT": "CONSUMER_SENTIMENT",
    }
    end_date = datetime.today().strftime("%Y-%m-%d")
    fred_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    loaded = 0
    for series_id, name in series.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            if not data.empty:
                fred_data[name] = data
                loaded += 1
            time.sleep(0.3)
        except Exception:
            continue

    # Calculate derived indicators
    if "CPI" in fred_data.columns:
        fred_data["INFLATION_RATE"] = fred_data["CPI"].pct_change(12) * 100
    if "M2_MONEY_SUPPLY" in fred_data.columns:
        fred_data["M2_YOY_GROWTH"] = fred_data["M2_MONEY_SUPPLY"].pct_change(12) * 100

    print(f"✅ FRED data loaded: {loaded}/{len(series)} series")
    return fred_data.ffill().bfill()


def load_thermomodel_data(data_folder):
    """Load ThermoModel data"""
    try:
        thermo_path = os.path.join(data_folder, "thermomodel.csv")
        if os.path.exists(thermo_path):
            thermo_df = pd.read_csv(thermo_path)
            thermo_df["time"] = pd.to_datetime(thermo_df["time"])
            thermo_df = thermo_df.sort_values("time").reset_index(drop=True)

            if (
                "lower_cubiclog" in thermo_df.columns
                and "upper_cubiclog" in thermo_df.columns
                and "PriceUSD" in thermo_df.columns
            ):
                thermo_df["thermo_dist_from_upper"] = (
                    thermo_df["PriceUSD"] - thermo_df["upper_cubiclog"]
                ) / thermo_df["upper_cubiclog"]
                thermo_df["thermo_dist_from_lower"] = (
                    thermo_df["PriceUSD"] - thermo_df["lower_cubiclog"]
                ) / thermo_df["lower_cubiclog"]

                band_range = thermo_df["upper_cubiclog"] - thermo_df["lower_cubiclog"]
                thermo_df["thermo_position_in_band"] = np.where(
                    band_range > 0,
                    (thermo_df["PriceUSD"] - thermo_df["lower_cubiclog"]) / band_range,
                    0.5,
                ).clip(0, 1)

                thermo_df["thermo_overbought"] = (
                    thermo_df["thermo_position_in_band"] > 0.8
                ).astype(int)
                thermo_df["thermo_oversold"] = (
                    thermo_df["thermo_position_in_band"] < 0.2
                ).astype(int)

                thermo_df["thermo_upper_roc"] = (
                    thermo_df["upper_cubiclog"].pct_change(30).fillna(0)
                )
                thermo_df["thermo_lower_roc"] = (
                    thermo_df["lower_cubiclog"].pct_change(30).fillna(0)
                )

            print("✅ ThermoModel data loaded successfully")
            return thermo_df
        else:
            print(f"⚠️ ThermoModel file not found: {thermo_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading ThermoModel: {e}")
        return None


def transform_cycle_data(cycles_df, window=100):
    """Transform cycle data using local maxima detection"""
    df_merged = cycles_df.copy()
    start_date = pd.to_datetime("2012-01-01")
    df_merged = df_merged[df_merged["time"] >= start_date].copy()

    cycle_vals = df_merged["cycle"].values

    # Find local maxima
    max_idx = argrelextrema(cycle_vals, np.greater, order=window)[0]

    # Add boundaries if needed
    if len(max_idx) > 0:
        if max_idx[0] > 0:
            max_idx = np.insert(max_idx, 0, 0)
        if max_idx[-1] < len(cycle_vals) - 1:
            max_idx = np.append(max_idx, len(cycle_vals) - 1)

    # Build transformed function
    transformed = np.full_like(cycle_vals, np.nan, dtype=np.float64)

    for i in range(len(max_idx) - 1):
        start = max_idx[i]
        end = max_idx[i + 1]
        seg_len = end - start + 1

        # 1/4 decline (bear): from top to bottom
        bear_len = int(seg_len * 0.25)
        bull_len = seg_len - bear_len

        # Linear decline from 1 to 0 (1/4 cycle)
        if bear_len > 0:
            transformed[start : start + bear_len] = np.linspace(
                1, 0, bear_len, endpoint=False
            )
        # Linear rise from 0 to 1 (3/4 cycle)
        if bull_len > 0:
            transformed[start + bear_len : end + 1] = np.linspace(0, 1, bull_len)

    df_merged["cycle_transformed"] = transformed
    print("✅ Cycle transformation applied")
    return df_merged


def load_cycles_data(data_folder, apply_transformation=True, window=100):
    """Load and optionally transform cycle data"""
    try:
        cycles_path = os.path.join(data_folder, "cycles.csv")
        if os.path.exists(cycles_path):
            cycles_df = pd.read_csv(cycles_path)
            cycles_df["time"] = pd.to_datetime(cycles_df["time"])
            cycles_df = cycles_df.sort_values("time").reset_index(drop=True)

            if "cycle" in cycles_df.columns:
                # Apply transformation if requested
                if apply_transformation:
                    cycles_df = transform_cycle_data(cycles_df, window)

                # Add standard cycle features
                cycles_df["cycle_smoothed"] = (
                    cycles_df["cycle"]
                    .rolling(window=7, center=True)
                    .mean()
                    .fillna(cycles_df["cycle"])
                )

                cycles_df["cycle_momentum"] = cycles_df["cycle"].diff(7).fillna(0)
                cycles_df["cycle_acceleration"] = (
                    cycles_df["cycle_momentum"].diff(7).fillna(0)
                )

                cycles_df["cycle_extreme_high"] = (cycles_df["cycle"] > 0.8).astype(int)
                cycles_df["cycle_high"] = (cycles_df["cycle"] > 0.6).astype(int)
                cycles_df["cycle_low"] = (cycles_df["cycle"] < 0.4).astype(int)
                cycles_df["cycle_extreme_low"] = (cycles_df["cycle"] < 0.2).astype(int)

                cycles_df["cycle_ma_30"] = cycles_df["cycle"].rolling(30).mean()
                cycles_df["cycle_ma_90"] = cycles_df["cycle"].rolling(90).mean()

                cycles_df["cycle_above_ma30"] = (
                    cycles_df["cycle"] > cycles_df["cycle_ma_30"]
                ).astype(int)
                cycles_df["cycle_above_ma90"] = (
                    cycles_df["cycle"] > cycles_df["cycle_ma_90"]
                ).astype(int)

            print("✅ Cycles data loaded successfully")
            return cycles_df
        else:
            print(f"⚠️ Cycles file not found: {cycles_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading cycles: {e}")
        return None


def add_btc_technicals(df):
    """Add Bitcoin technical indicators"""
    df["BTC_MA50"] = df["PriceUSD"].rolling(50).mean()
    df["BTC_MA200"] = df["PriceUSD"].rolling(200).mean()
    df["BTC_VOLATILITY"] = df["PriceUSD"].pct_change().rolling(30).std() * 100

    # RSI
    delta = df["PriceUSD"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["BTC_RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["PriceUSD"].ewm(span=12).mean()
    exp2 = df["PriceUSD"].ewm(span=26).mean()
    df["BTC_MACD"] = exp1 - exp2
    df["BTC_SIGNAL"] = df["BTC_MACD"].ewm(span=9).mean()

    return df


def load_data(
    start_date="2010-01-01",
    load_btc=True,
    load_markets=True,
    load_fred=True,
    load_thermomodel=True,
    load_cycles=True,
    data_folder="data",
    apply_cycle_transformation=True,
    cycle_window=100,
):
    """
    Load and merge all data sources without processing - raw data only.

    Parameters:
    -----------
    start_date : str
        Start date for data collection (YYYY-MM-DD format)
    load_btc : bool
        Load Bitcoin data from fetch_crypto_data("btc")
    load_markets : bool
        Load market data from Yahoo Finance
    load_fred : bool
        Load FRED economic data
    load_thermomodel : bool
        Load ThermoModel data from CSV
    load_cycles : bool
        Load Cycles data from CSV
    data_folder : str
        Folder path containing CSV files
    apply_cycle_transformation : bool
        Apply cycle transformation using local maxima
    cycle_window : int
        Window size for cycle transformation

    Returns:
    --------
    pd.DataFrame
        Combined raw DataFrame with all data sources merged by time
    """

    print("🚀 LOADING RAW DATA SOURCES")
    print("=" * 40)

    # Initialize base dataframe
    df = None

    # 1. Load Bitcoin data
    if load_btc:
        print("₿ Loading Bitcoin data...")
        try:
            btc_df = fetch_crypto_data("btc")
            btc_df["time"] = pd.to_datetime(btc_df["time"]).dt.date
            btc_df["time"] = pd.to_datetime(
                btc_df["time"]
            )  # Convert back to datetime but date only
            btc_df = btc_df.sort_values("time").reset_index(drop=True)

            # Filter by start date
            btc_df = btc_df[btc_df["time"] >= start_date].copy()

            # Drop unnecessary columns if they exist
            cols_to_drop = [
                "principal_market_price_usd",
                "principal_market_usd",
                "ReferenceRate",
                "ReferenceRateETH",
                "ReferenceRateEUR",
                "ReferenceRateUSD",
                "CapMrktEstUSD",
            ]
            btc_df = btc_df.drop(
                [c for c in cols_to_drop if c in btc_df.columns], axis=1
            )

            df = btc_df
            print(f"✅ Bitcoin data loaded: {len(df)} rows")

        except Exception as e:
            print(f"❌ Error loading Bitcoin data: {e}")
            return None

    # 2. Load market data
    if load_markets:
        print("📈 Loading market data...")
        try:
            market_data = load_market_data(start_date)
            if len(market_data.columns) > 0:
                df = pd.merge_asof(
                    df.sort_values("time"),
                    market_data.reset_index().rename(columns={"index": "time"}),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta("3D"),
                )
                print(f"✅ Market data merged: {len(market_data.columns)} indicators")
        except Exception as e:
            print(f"⚠️ Error loading market data: {e}")

    # 3. Load FRED data
    if load_fred:
        print("🏦 Loading FRED economic data...")
        try:
            fred_indicators = load_fred_data(start_date)
            if fred_indicators is not None and len(fred_indicators.columns) > 0:
                df = pd.merge_asof(
                    df.sort_values("time"),
                    fred_indicators.reset_index().rename(columns={"index": "time"}),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta("7D"),
                )
                print(f"✅ FRED data merged: {len(fred_indicators.columns)} indicators")
        except Exception as e:
            print(f"⚠️ Error loading FRED data: {e}")

    # 4. Load ThermoModel data
    if load_thermomodel:
        print("🌡️ Loading ThermoModel data...")
        try:
            thermo_df = load_thermomodel_data(data_folder)
            if thermo_df is not None:
                thermo_features = [
                    col for col in thermo_df.columns if col not in ["PriceUSD"]
                ]
                df = pd.merge_asof(
                    df.sort_values("time"),
                    thermo_df[thermo_features].sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta("1D"),
                )
                print(
                    f"✅ ThermoModel merged: {len([c for c in thermo_features if c != 'time'])} features"
                )
        except Exception as e:
            print(f"⚠️ Error loading ThermoModel data: {e}")

    # 5. Load Cycles data
    if load_cycles:
        print("🔄 Loading Cycles data...")
        try:
            cycles_df = load_cycles_data(
                data_folder, apply_cycle_transformation, cycle_window
            )
            if cycles_df is not None:
                df = pd.merge_asof(
                    df.sort_values("time"),
                    cycles_df.sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta("1D"),
                )
                cycles_features = [c for c in cycles_df.columns if c != "time"]
                print(f"✅ Cycles merged: {len(cycles_features)} features")
                if (
                    apply_cycle_transformation
                    and "cycle_transformed" in cycles_df.columns
                ):
                    print(f"   🔧 Cycle transformation applied (window={cycle_window})")
        except Exception as e:
            print(f"⚠️ Error loading Cycles data: {e}")

    # Add Bitcoin technical indicators
    if df is not None:
        df = add_btc_technicals(df)
        print("✅ Bitcoin technical indicators added")

    # Forward fill and backward fill missing values
    if df is not None:
        df = df.ffill().bfill().reset_index(drop=True)
        print(f"🎯 Final raw dataset shape: {df.shape}")
        print("=" * 40)

    return df


def _calc_halving_features(date):
    """Calculate halving-related features"""
    last_halving = HALVING_DATES[HALVING_DATES <= date].max()
    next_halving = HALVING_DATES[HALVING_DATES > date].min()
    days_since_last = (
        (date - last_halving).days if not pd.isnull(last_halving) else np.nan
    )
    days_until_next = (
        (next_halving - date).days
        if not pd.isnull(next_halving)
        else 1379 - days_since_last
    )
    return days_since_last, days_until_next


def _calc_days_since_last(date, event_dates):
    """Calculate days since last event"""
    last_event = event_dates[event_dates <= date].max()
    return (date - last_event).days if not pd.isnull(last_event) else np.nan


def feature_engineering(df):
    """Enhanced feature engineering"""
    df_new = df.copy()

    def safe_div(a, b, fill=0):
        return np.divide(
            a,
            b,
            out=np.full_like(a, fill, dtype=float),
            where=(b != 0) & (~np.isnan(b)),
        )

    # Halving cycle features
    halving_features = df_new["time"].apply(lambda x: _calc_halving_features(x))
    df_new["days_since_last_halving"] = [x[0] for x in halving_features]
    df_new["days_until_next_halving"] = [x[1] for x in halving_features]
    df_new["phase_in_halving_cycle"] = df_new["days_since_last_halving"] / 1379

    halving_cycle = np.zeros(len(df_new))
    for i, date in enumerate(df_new["time"]):
        halving_cycle[i] = sum(date >= halving_date for halving_date in HALVING_DATES)
    df_new["halving_cycle_number"] = halving_cycle
    df_new["phase_sin"] = np.sin(2 * np.pi * df_new["phase_in_halving_cycle"])
    df_new["phase_cos"] = np.cos(2 * np.pi * df_new["phase_in_halving_cycle"])

    df_new["days_since_last_top"] = df_new["time"].apply(
        lambda x: _calc_days_since_last(x, TOPS_DATES)
    )
    df_new["days_since_last_bottom"] = df_new["time"].apply(
        lambda x: _calc_days_since_last(x, BOTTOMS_DATES)
    )

    # Price features
    df_new["log_price"] = np.log(df_new["PriceUSD"].replace(0, np.nan))
    for window in [1, 7, 30, 90, 365]:
        df_new[f"returns_{window}d"] = df_new["PriceUSD"].pct_change(window).clip(-5, 5)

    # Moving averages
    for window in [14, 50, 200]:
        df_new[f"ma_{window}"] = df_new["PriceUSD"].rolling(window).mean()
    df_new["dist_from_ma50"] = (
        safe_div(df_new["PriceUSD"], df_new["ma_50"], 1) - 1
    ).clip(-3, 3)
    df_new["dist_from_ma200"] = (
        safe_div(df_new["PriceUSD"], df_new["ma_200"], 1) - 1
    ).clip(-3, 3)

    # RSI
    delta = df_new["PriceUSD"].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(14).mean()
    roll_down = down.abs().rolling(14).mean()
    rs = safe_div(roll_up, roll_down)
    df_new["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility
    for window in [14, 30, 90]:
        df_new[f"volatility_{window}d"] = (
            df_new["log_price"].diff().rolling(window).std().clip(0, 0.3)
        )

    # ATH features
    df_new["ath"] = df_new["PriceUSD"].expanding().max()
    df_new["dist_from_ath"] = safe_div(df_new["PriceUSD"], df_new["ath"], 0)

    # Time features
    df_new["month"] = df_new["time"].dt.month
    df_new["quarter"] = df_new["time"].dt.quarter
    df_new["sin_365"] = np.sin(2 * np.pi * df_new["time"].dt.dayofyear / 365)
    df_new["cos_365"] = np.cos(2 * np.pi * df_new["time"].dt.dayofyear / 365)

    # ENHANCED FEATURE ENGINEERING
    # Phase momentum and acceleration
    df_new["phase_momentum"] = df_new["phase_in_halving_cycle"].diff(7).fillna(0)
    df_new["phase_acceleration"] = df_new["phase_momentum"].diff(7).fillna(0)

    # Enhanced distance from ATH features
    df_new["dist_from_ath_diff_7d"] = df_new["dist_from_ath"].diff(7).fillna(0)
    df_new["dist_from_ath_diff_30d"] = df_new["dist_from_ath"].diff(30).fillna(0)
    df_new["dist_from_ath_roc_7d"] = df_new["dist_from_ath"].pct_change(7).fillna(0)
    df_new["dist_from_ath_roc_30d"] = df_new["dist_from_ath"].pct_change(30).fillna(0)

    # Days since events momentum
    df_new["days_since_top_momentum"] = df_new["days_since_last_top"].diff(7).fillna(0)
    df_new["days_since_bottom_momentum"] = (
        df_new["days_since_last_bottom"].diff(7).fillna(0)
    )

    # RSI momentum and acceleration
    df_new["rsi_momentum"] = df_new["rsi_14"].diff(7).fillna(0)
    df_new["rsi_acceleration"] = df_new["rsi_momentum"].diff(7).fillna(0)
    df_new["rsi_extreme"] = ((df_new["rsi_14"] > 80) | (df_new["rsi_14"] < 20)).astype(
        int
    )

    # Volatility regime changes
    df_new["volatility_regime_30d"] = (
        df_new["volatility_30d"] > df_new["volatility_30d"].rolling(90).mean()
    ).astype(int)
    df_new["volatility_momentum"] = df_new["volatility_30d"].diff(7).fillna(0)

    # Returns momentum and acceleration
    for window in [7, 30, 90]:
        df_new[f"returns_{window}d_momentum"] = (
            df_new[f"returns_{window}d"].diff(7).fillna(0)
        )
        df_new[f"returns_{window}d_acceleration"] = (
            df_new[f"returns_{window}d_momentum"].diff(7).fillna(0)
        )

    # MA cross features
    df_new["ma50_above_ma200"] = (df_new["ma_50"] > df_new["ma_200"]).astype(int)
    df_new["price_above_ma50"] = (df_new["PriceUSD"] > df_new["ma_50"]).astype(int)
    df_new["price_above_ma200"] = (df_new["PriceUSD"] > df_new["ma_200"]).astype(int)

    # MA distance momentum
    df_new["dist_from_ma50_momentum"] = df_new["dist_from_ma50"].diff(7).fillna(0)
    df_new["dist_from_ma200_momentum"] = df_new["dist_from_ma200"].diff(7).fillna(0)

    # Price regime features
    df_new["price_regime_bull"] = (df_new["returns_90d"] > 0.2).astype(int)
    df_new["price_regime_bear"] = (df_new["returns_90d"] < -0.2).astype(int)

    # Log price features
    df_new["log_price_diff_7d"] = df_new["log_price"].diff(7).fillna(0)
    df_new["log_price_diff_30d"] = df_new["log_price"].diff(30).fillna(0)
    df_new["log_price_momentum"] = df_new["log_price_diff_7d"].diff(7).fillna(0)

    # Enhanced thermo features if available
    if "thermo_position_in_band" in df_new.columns:
        df_new["thermo_momentum"] = df_new["thermo_position_in_band"].diff(7).fillna(0)
        df_new["thermo_acceleration"] = df_new["thermo_momentum"].diff(7).fillna(0)
        df_new["thermo_extreme"] = (
            (df_new["thermo_position_in_band"] > 0.9)
            | (df_new["thermo_position_in_band"] < 0.1)
        ).astype(int)

    # Enhanced cycle features if available
    if "cycle" in df_new.columns:
        df_new["cycle_diff_7d"] = df_new["cycle"].diff(7).fillna(0)
        df_new["cycle_diff_30d"] = df_new["cycle"].diff(30).fillna(0)
        df_new["cycle_momentum_enhanced"] = df_new["cycle_diff_7d"].diff(7).fillna(0)
        df_new["cycle_above_08"] = (df_new["cycle"] > 0.8).astype(int)
        df_new["cycle_below_02"] = (df_new["cycle"] < 0.2).astype(int)

    # Enhanced cycle transformed features if available
    if "cycle_transformed" in df_new.columns:
        df_new["cycle_transformed_diff_7d"] = (
            df_new["cycle_transformed"].diff(7).fillna(0)
        )
        df_new["cycle_transformed_diff_30d"] = (
            df_new["cycle_transformed"].diff(30).fillna(0)
        )
        df_new["cycle_transformed_momentum"] = (
            df_new["cycle_transformed_diff_7d"].diff(7).fillna(0)
        )
        df_new["cycle_transformed_above_08"] = (
            df_new["cycle_transformed"] > 0.8
        ).astype(int)
        df_new["cycle_transformed_below_02"] = (
            df_new["cycle_transformed"] < 0.2
        ).astype(int)

    print(f"✅ Feature engineering completed: {df_new.shape[1]} features")

    return (
        df_new.replace([np.inf, -np.inf], np.nan)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )


def generate_targets(df):
    """Generate target variables based on cycle tops"""
    horizons = {"Target_1M": 30, "Target_6M": 180, "Target_1Y": 365}
    for target_name in horizons.keys():
        df[target_name] = 0

    for top in TOPS_DATES:
        for target_name, days in horizons.items():
            mask = (df["time"] >= (top - pd.Timedelta(days=days))) & (df["time"] < top)
            df.loc[mask, target_name] = 1

    print("✅ Target variables generated")
    return df


def reduce_features(df, corr_threshold=0.9, nan_threshold=0.3):
    """Reduce features by removing low variance and highly correlated features"""
    excluded = ["PriceUSD", "Target_1M", "Target_6M", "Target_1Y", "time"]
    df_clean = df.copy()
    numeric_df = df_clean.select_dtypes(include=[np.number])

    # Remove zero variance features
    selector = VarianceThreshold(threshold=0)
    selector.fit(numeric_df)
    zero_var_cols = set(numeric_df.columns[~selector.get_support()]) - set(excluded)
    df_clean.drop(columns=zero_var_cols, inplace=True)

    # Remove features with too many NaN values
    nan_cols = set(df_clean.columns[df_clean.isnull().mean() > nan_threshold]) - set(
        excluded
    )
    df_clean.drop(columns=nan_cols, inplace=True)

    # Remove highly correlated features
    corr_matrix = df_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for column in upper_tri.columns:
        if column in excluded:
            continue
        correlated_cols = upper_tri.index[upper_tri[column] > corr_threshold].tolist()
        for correlated_col in correlated_cols:
            if correlated_col not in to_drop and correlated_col not in excluded:
                if abs(corr_matrix["PriceUSD"][column]) >= abs(
                    corr_matrix["PriceUSD"][correlated_col]
                ):
                    to_drop.add(correlated_col)
                else:
                    to_drop.add(column)
    df_clean.drop(columns=to_drop, inplace=True)
    print(f"🔧 Features reduced: {len(df.columns)} → {len(df_clean.columns)}")
    return df_clean


def create_robust_pipeline():
    """Create ML pipeline with preprocessing and ensemble classifier"""
    numeric_transformer = Pipeline(
        [("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())]
    )
    preprocessor = ColumnTransformer([("num", numeric_transformer, slice(None))])

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.8,
                bootstrap=True,
                oob_score=True,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                n_estimators=300,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                validation_fraction=0.1,
                n_iter_no_change=15,
            ),
        ),
    ]

    if XGBOOST_AVAILABLE:
        estimators.append(
            (
                "xgb",
                XGBClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    eval_metric="logloss",
                ),
            )
        )

    voting_classifier = VotingClassifier(estimators=estimators, voting="soft")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "feature_selector",
                SelectFromModel(
                    RandomForestClassifier(random_state=42, n_estimators=100),
                    threshold="0.5*median",
                ),
            ),
            ("classifier", voting_classifier),
        ]
    )


import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight


def create_stratified_time_splits(df, target_col, n_splits=5, min_positive_ratio=0.01):
    """
    FIXED VERSION: Create time-aware splits that GUARANTEE positive examples in validation
    """
    target = df[target_col]
    positive_indices = np.where(target == 1)[0]

    print(f"Creating fixed splits for {target_col}:")
    print(f"  Total samples: {len(df)}")
    print(f"  Positive samples: {len(positive_indices)}")
    print(f"  Positive ratio: {len(positive_indices)/len(df):.1%}")

    if len(positive_indices) < n_splits:
        n_splits = max(2, len(positive_indices) // 2)
        print(f"  Reducing to {n_splits} splits due to sparse positive examples")

    splits = []
    total_size = len(df)

    # NEW STRATEGY: Ensure each validation fold has positive examples
    # Divide positive examples across folds first
    pos_per_fold = len(positive_indices) // n_splits

    for i in range(n_splits):
        # Calculate validation boundaries to include positive examples
        start_pos_idx = i * pos_per_fold
        end_pos_idx = (
            (i + 1) * pos_per_fold if i < n_splits - 1 else len(positive_indices)
        )

        if start_pos_idx >= len(positive_indices):
            break

        # Get positive indices for this fold
        fold_positive_indices = positive_indices[start_pos_idx:end_pos_idx]

        if len(fold_positive_indices) == 0:
            continue

        # Define validation set around these positive examples
        val_start = max(0, fold_positive_indices[0] - 100)  # Context before
        val_end = min(total_size, fold_positive_indices[-1] + 100)  # Context after

        # Ensure minimum validation size
        if val_end - val_start < 50:
            val_center = (fold_positive_indices[0] + fold_positive_indices[-1]) // 2
            val_start = max(0, val_center - 25)
            val_end = min(total_size, val_center + 25)

        # Training set is everything before validation (time-series constraint)
        train_end = val_start
        train_indices = list(range(train_end))
        val_indices = list(range(val_start, val_end))

        # Ensure minimum training size
        if len(train_indices) < 100:
            continue

        # Check class distribution
        train_target = target.iloc[train_indices]
        val_target = target.iloc[val_indices]

        train_positive = (train_target == 1).sum()
        val_positive = (val_target == 1).sum()

        val_ratio = val_positive / len(val_indices) if len(val_indices) > 0 else 0

        print(f"  Split {i}: Train={len(train_indices)}, Val={len(val_indices)}")
        print(f"    Train pos: {train_positive}, Val pos: {val_positive}")
        print(f"    Val pos ratio: {val_ratio:.1%}")

        # Accept split only if both train and val have positive examples
        if (
            train_positive > 0
            and val_positive > 0
            and len(train_target.unique()) > 1
            and len(val_target.unique()) > 1
        ):
            splits.append((train_indices, val_indices))
        else:
            print(f"    Skipping split {i} (insufficient class diversity)")

    if len(splits) == 0:
        # FALLBACK: Create at least one valid split
        print("  Creating fallback split...")
        mid_point = total_size // 2
        train_indices = list(range(mid_point))
        val_indices = list(range(mid_point, total_size))

        train_target = target.iloc[train_indices]
        val_target = target.iloc[val_indices]

        if (
            len(train_target.unique()) > 1
            and len(val_target.unique()) > 1
            and (train_target == 1).sum() > 0
            and (val_target == 1).sum() > 0
        ):
            splits.append((train_indices, val_indices))
            print(
                f"  Fallback split: Train={len(train_indices)}, Val={len(val_indices)}"
            )

    return splits


def create_robust_pipeline_with_class_weights():
    """
    FIXED VERSION: Use VotingClassifier instead of StackingClassifier to avoid CV issues
    """
    # Enhanced preprocessing
    numeric_transformer = Pipeline(
        [("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())]
    )
    preprocessor = ColumnTransformer([("num", numeric_transformer, slice(None))])

    # Create diverse base models
    # Keep only the BEST performing models
    estimators = []

    # Gradient Boosting Conservative (good for 6M and 1Y)
    estimators.append(
        (
            "gb_conservative",
            GradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                n_estimators=200,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
            ),
        )
    )

    # Gradient Boosting Aggressive (excellent for 6M)
    estimators.append(
        (
            "gb_aggressive",
            GradientBoostingClassifier(
                random_state=42,
                learning_rate=0.1,
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.9,
            ),
        )
    )

    # XGBoost (excellent for 1M)
    if XGBOOST_AVAILABLE:
        estimators.append(
            (
                "xgb_balanced",
                XGBClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    eval_metric="logloss",
                    verbosity=0,
                ),
            )
        )

    # LightGBM (decent across all targets)
    if LIGHTGBM_AVAILABLE:
        estimators.append(
            (
                "lgb_optimized",
                LGBMClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight="balanced",
                    verbosity=-1,
                ),
            )
        )

    # Use SOFT VOTING instead of stacking to avoid CV issues
    voting_classifier = VotingClassifier(
        estimators=estimators, voting="soft", n_jobs=-1  # Use probability averages
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "feature_selector",
                SelectFromModel(
                    RandomForestClassifier(
                        random_state=42, n_estimators=100, class_weight="balanced"
                    ),
                    threshold="0.5*median",
                ),
            ),
            ("classifier", voting_classifier),
        ]
    )


def safe_cross_val_score_with_custom_splits(estimator, X, y, custom_splits, scoring):
    """
    FIXED VERSION: Improved error handling and individual model tracking
    """
    scores = []
    individual_predictions = {}
    individual_scores = {}
    all_confusion_matrices = []
    individual_confusion_matrices = {}

    # Initialize containers for individual models
    estimator_names = [
        name for name, _ in estimator.named_steps["classifier"].estimators
    ]
    for name in estimator_names:
        individual_predictions[name] = []
        individual_scores[name] = []
        individual_confusion_matrices[name] = []

    successful_folds = 0

    for fold_idx, (train_idx, val_idx) in enumerate(custom_splits):
        try:
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Double-check class distribution
            train_classes = y_train.unique()
            val_classes = y_val.unique()

            if len(train_classes) < 2:
                print(
                    f"Fold {fold_idx} skipped: training set has classes {train_classes}"
                )
                continue

            if len(val_classes) < 2:
                print(
                    f"Fold {fold_idx} skipped: validation set has classes {val_classes}"
                )
                continue

            # Handle class imbalance for XGBoost
            class_counts = y_train.value_counts()
            if XGBOOST_AVAILABLE:
                scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
                # Update XGBoost models
                classifier = estimator.named_steps["classifier"]
                for name, est in classifier.estimators:
                    if "xgb" in name and hasattr(est, "set_params"):
                        est.set_params(scale_pos_weight=scale_pos_weight)

            # Fit the pipeline
            estimator.fit(X.iloc[train_idx], y_train)

            # Get ensemble predictions
            y_pred_prob = estimator.predict_proba(X.iloc[val_idx])[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int)

            # Calculate ensemble score
            score = roc_auc_score(y_val, y_pred_prob)
            scores.append(score)

            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_val, y_pred)
            all_confusion_matrices.append(cm)

            # Get individual model predictions
            classifier = estimator.named_steps["classifier"]
            X_transformed = estimator[:-1].transform(X.iloc[val_idx])

            # For VotingClassifier, access individual estimators
            for (name, _), fitted_estimator in zip(
                classifier.estimators, classifier.estimators_
            ):
                try:
                    ind_pred_prob = fitted_estimator.predict_proba(X_transformed)[:, 1]
                    ind_pred = (ind_pred_prob > 0.5).astype(int)

                    individual_predictions[name].extend(ind_pred_prob)

                    ind_score = roc_auc_score(y_val, ind_pred_prob)
                    individual_scores[name].append(ind_score)

                    ind_cm = confusion_matrix(y_val, ind_pred)
                    individual_confusion_matrices[name].append(ind_cm)

                except Exception as e:
                    print(f"Error processing {name} in fold {fold_idx}: {e}")
                    continue

            successful_folds += 1
            print(f"✅ Fold {fold_idx} completed - AUC: {score:.3f}")

        except Exception as e:
            print(f"❌ Error in fold {fold_idx}: {e}")
            continue

    print(f"📊 Completed {successful_folds}/{len(custom_splits)} folds successfully")

    return (
        np.array(scores),
        all_confusion_matrices,
        individual_scores,
        individual_confusion_matrices,
    )


def train_models_and_predict(df, split_date="2024-01-01"):
    """
    COMPLETE improved training function with optimized ensembles and logical constraints
    """
    excluded = ["PriceUSD", "Target_1M", "Target_6M", "Target_1Y", "time"]
    features = [col for col in df.columns if col not in excluded]
    X = df[features].copy()

    # Clean features
    for col in X.select_dtypes(include=["int64", "float64"]).columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        q1, q99 = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(q1, q99)

    split_date = pd.to_datetime(split_date)
    train_mask = df["time"] < split_date

    models = {}
    performance = {}
    feature_importances = {}
    individual_model_predictions = {}

    for target in ["Target_1M", "Target_6M", "Target_1Y"]:
        try:
            print(f"\n🎯 Training {target}...")

            y_train = df[target][train_mask]
            unique_classes = y_train.unique()

            if len(unique_classes) < 2:
                print(f"⚠️ Skipping {target}: Only one class present")
                continue

            class_counts = y_train.value_counts()
            pos_ratio = class_counts.get(1, 0) / len(y_train)
            print(f"   Class distribution: {dict(class_counts)} (pos: {pos_ratio:.1%})")

            # Create pipeline
            pipeline = create_robust_pipeline_with_class_weights()

            # Create improved time splits
            custom_splits = create_stratified_time_splits(
                df[train_mask].reset_index(drop=True), target, n_splits=3
            )

            if len(custom_splits) == 0:
                print(f"⚠️ No valid splits for {target}, using simple training")
                # Fallback: fit on full training data
                pipeline.fit(X[train_mask], y_train)
                models[target] = pipeline

                # Generate predictions with ORIGINAL NAMES
                pred_proba = pipeline.predict_proba(X)[:, 1]
                df[f"pred_proba_{target}"] = pred_proba

                spans = {"Target_1M": 7, "Target_6M": 21, "Target_1Y": 30}
                df[f"pred_proba_{target}_smoothed"] = (
                    df[f"pred_proba_{target}"].ewm(span=spans[target]).mean()
                )

                print(f"✅ {target} - Training completed")
                continue

            # Cross-validation
            cv_scores, cv_confusion_matrices, individual_scores, individual_cms = (
                safe_cross_val_score_with_custom_splits(
                    pipeline,
                    X[train_mask].reset_index(drop=True),
                    y_train.reset_index(drop=True),
                    custom_splits,
                    scoring="roc_auc",
                )
            )

            # Fit final model on full training data
            pipeline.fit(X[train_mask], y_train)
            models[target] = pipeline

            # Generate predictions with ORIGINAL NAMES
            pred_proba = pipeline.predict_proba(X)[:, 1]
            df[f"pred_proba_{target}"] = pred_proba

            # Apply smoothing with ORIGINAL NAMES
            spans = {"Target_1M": 7, "Target_6M": 21, "Target_1Y": 30}
            df[f"pred_proba_{target}_smoothed"] = (
                df[f"pred_proba_{target}"].ewm(span=spans[target]).mean()
            )

            # Extract individual model predictions with ORIGINAL NAMES
            try:
                classifier = pipeline.named_steps["classifier"]
                X_transformed = pipeline[:-1].transform(X)

                individual_model_predictions[target] = {}

                for (name, _), fitted_estimator in zip(
                    classifier.estimators, classifier.estimators_
                ):
                    try:
                        ind_predictions = fitted_estimator.predict_proba(X_transformed)[
                            :, 1
                        ]
                        individual_model_predictions[target][name] = ind_predictions
                        df[f"pred_proba_{target}_{name}"] = ind_predictions
                        df[f"pred_proba_{target}_{name}_smoothed"] = (
                            df[f"pred_proba_{target}_{name}"]
                            .ewm(span=spans[target])
                            .mean()
                        )
                    except Exception as e:
                        print(f"   ⚠️ Error with model {name}: {e}")
            except Exception as e:
                print(f"   ⚠️ Could not extract individual predictions: {e}")

            # Store performance metrics
            if len(cv_scores) > 0:
                performance[target] = {
                    "cv_auc_mean": cv_scores.mean(),
                    "cv_auc_std": cv_scores.std(),
                    "cv_confusion_matrices": cv_confusion_matrices,
                    "individual_scores": individual_scores,
                    "individual_confusion_matrices": individual_cms,
                    "n_cv_folds": len(cv_scores),
                    "positive_ratio": pos_ratio,
                }

                print(
                    f"✅ {target} - CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                )

                # Print individual model performance
                for name, scores in individual_scores.items():
                    if scores:
                        print(
                            f"   📊 {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
                        )
            else:
                print(f"✅ {target} - Training completed (no CV scores)")

        except Exception as e:
            print(f"❌ Error training {target}: {str(e)}")

    # Apply temporal logic constraints with ORIGINAL NAMES
    print(f"\n🔄 APPLYING TEMPORAL LOGIC CONSTRAINTS")

    pred_1y = "pred_proba_Target_1Y_smoothed"
    pred_6m = "pred_proba_Target_6M_smoothed"
    pred_1m = "pred_proba_Target_1M_smoothed"

    if all(col in df.columns for col in [pred_1y, pred_6m, pred_1m]):
        violations_before = 0
        violations_after = 0

        for i in range(len(df)):
            prob_1y = df.loc[i, pred_1y]
            prob_6m = df.loc[i, pred_6m]
            prob_1m = df.loc[i, pred_1m]

            if prob_1y < prob_6m or prob_6m < prob_1m:
                violations_before += 1

            if prob_6m > prob_1y:
                df.loc[i, pred_6m] = 0.8 * prob_1y + 0.2 * prob_6m

            corrected_6m = df.loc[i, pred_6m]
            if prob_1m > corrected_6m:
                df.loc[i, pred_1m] = 0.8 * corrected_6m + 0.2 * prob_1m

            new_1y = df.loc[i, pred_1y]
            new_6m = df.loc[i, pred_6m]
            new_1m = df.loc[i, pred_1m]

            if new_1y < new_6m or new_6m < new_1m:
                violations_after += 1

        improvement = (
            (violations_before - violations_after) / max(violations_before, 1) * 100
        )
        print(
            f"   📊 Logic violations: {violations_before} → {violations_after} ({improvement:.1f}% improvement)"
        )

    # Generate future predictions
    future_df = generate_future_predictions_simple(df, models, features)

    print("✅ Model training completed successfully!")
    return (
        df,
        future_df,
        models,
        performance,
        feature_importances,
        individual_model_predictions,
    )


def generate_future_predictions_simple(df, models, features):
    """Generate simple future predictions"""
    from datetime import datetime
    import pandas as pd

    last_date = df["time"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90)
    future_features = []

    for date in future_dates:
        entry = {"time": date}
        # Add basic cycle features
        entry["days_since_last_halving"], entry["days_until_next_halving"] = (
            _calc_halving_features(date)
        )
        entry["phase_in_halving_cycle"] = entry["days_since_last_halving"] / 1379
        entry["phase_sin"] = np.sin(2 * np.pi * entry["phase_in_halving_cycle"])
        entry["phase_cos"] = np.cos(2 * np.pi * entry["phase_in_halving_cycle"])

        # Fill other features with recent values
        for feature in features:
            if feature not in entry:
                recent_values = df[feature].iloc[-30:].dropna()
                if len(recent_values) > 0:
                    entry[feature] = recent_values.mean()
                else:
                    entry[feature] = 0
        future_features.append(entry)

    future_df = pd.DataFrame(future_features)

    # Generate predictions for each target
    for target, model in models.items():
        try:
            future_X = future_df[features].copy()
            # Fill missing columns
            missing_cols = set(features) - set(future_X.columns)
            for col in missing_cols:
                future_X[col] = df[col].iloc[-1] if col in df.columns else 0
            future_X = future_X[features]

            predictions = model.predict_proba(future_X)[:, 1]
            future_df[f"pred_proba_{target}"] = predictions

            spans = {"Target_1M": 7, "Target_6M": 21, "Target_1Y": 30}
            future_df[f"pred_proba_{target}_smoothed"] = (
                future_df[f"pred_proba_{target}"].ewm(span=spans[target]).mean()
            )

        except Exception as e:
            print(f"Future prediction failed for {target}: {str(e)}")

    return future_df


def plot_dual_axis(
    df,
    price_col="PriceUSD",
    cycle_col="cycle",
    cycle_transformed_col="cycle_transformed",
    figsize=(14, 7),
):
    """Plot Bitcoin price with cycle data on dual axis"""
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot price
    ln1 = ax1.plot(
        df["time"], df[price_col], color="tab:blue", label="PriceUSD", linewidth=1.5
    )
    ax1.set_ylabel("PriceUSD (log)", color="tab:blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot original cycle
    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        df["time"], df[cycle_col], color="tab:orange", label="Cycle Original", alpha=0.7
    )
    ax2.set_ylabel("Cycle Original", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Plot transformed cycle if available
    if cycle_transformed_col in df.columns:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ln3 = ax3.plot(
            df["time"],
            df[cycle_transformed_col],
            color="tab:green",
            label="Cycle Transformed (0-1)",
            linewidth=2,
        )
        ax3.set_ylabel("Cycle Transformed (0-1)", color="tab:green")
        ax3.tick_params(axis="y", labelcolor="tab:green")
        ax3.set_ylim(0, 1)
        lns = ln1 + ln2 + ln3
    else:
        lns = ln1 + ln2

    # Add event markers
    for date in HALVING_DATES:
        if date <= df["time"].max() and date >= df["time"].min():
            ax1.axvline(date, color="purple", linestyle="--", alpha=0.7, linewidth=1)

    for date in TOPS_DATES:
        if date <= df["time"].max() and date >= df["time"].min():
            ax1.axvline(date, color="red", linestyle=":", alpha=0.7, linewidth=1)

    for date in BOTTOMS_DATES:
        if date <= df["time"].max() and date >= df["time"].min():
            ax1.axvline(date, color="green", linestyle=":", alpha=0.7, linewidth=1)

    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc="upper left")
    ax1.set_title("Bitcoin: Price, Cycle Original and Cycle Transformed")
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_predictions(
    df,
    future_df,
    individual_model_predictions=None,
    split_date="2024-01-01",
    activation_threshold=0.9,
):
    """Plot ensemble and individual model predictions"""
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pandas as pd
    
    split_date = pd.to_datetime(split_date)
    today = pd.to_datetime(datetime.today().date())

    # Calculate forecast dates ONLY from test set (after split_date)
    def get_forecast_dates(df, split_date, threshold=0.9):
        forecast_dates = {}
        targets_info = {
            "Target_1Y": {"days": 365, "color": "#8B0000"},
            "Target_6M": {"days": 180, "color": "#CD5C5C"}, 
            "Target_1M": {"days": 30, "color": "#FFB6C1"}
        }
        
        # Only look at test set data
        test_data = df[df["time"] >= split_date].copy()
        
        for target, info in targets_info.items():
            col = f"pred_proba_{target}_smoothed"
            if col in test_data.columns:
                # Find first date in TEST SET where threshold is exceeded
                above_threshold = test_data[test_data[col] >= threshold]
                if not above_threshold.empty:
                    trigger_date = above_threshold["time"].iloc[0]
                    forecast_date = trigger_date + pd.Timedelta(days=info["days"])
                    forecast_dates[target] = {
                        "trigger": trigger_date,
                        "forecast": forecast_date,
                        "color": info["color"]
                    }
                    print(f"📅 {target} forecast: Test set threshold exceeded {trigger_date.strftime('%Y-%m-%d')} → Top predicted {forecast_date.strftime('%Y-%m-%d')}")
        return forecast_dates
    
    forecast_dates = get_forecast_dates(df, split_date, activation_threshold)

    # Main ensemble plot
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1, 1, 1], "hspace": 0.06},
    )
    ax0 = axes[0]

    # Background shading
    ax0.axvspan(
        split_date,
        df["time"].max(),
        alpha=0.15,
        color="gray",
        zorder=0,
        label="Test Period",
    )
    if len(future_df) > 0:
        ax0.axvspan(
            df["time"].max(),
            future_df["time"].max(),
            alpha=0.08,
            color="lightblue",
            zorder=0,
            label="Future Forecast",
        )

    # Price plot
    ax0.plot(df["time"], df["PriceUSD"], "b-", linewidth=1.5, label="Bitcoin Price")
    ax0.set_yscale("log")
    ax0.set_ylabel("Price (USD)", fontsize=10)
    ax0.set_title(
        "Bitcoin Cycle Top Predictions - Ensemble", fontsize=16, fontweight="bold"
    )

    # Price formatter
    from matplotlib.ticker import FuncFormatter

    def price_formatter(x, pos):
        if x >= 1_000_000:
            return f"${x/1_000_000:.1f}M"
        elif x >= 1000:
            return f"${x/1000:.0f}K"
        else:
            return f"${x:.0f}"

    ax0.yaxis.set_major_formatter(FuncFormatter(price_formatter))

    # Event markers
    for date in HALVING_DATES:
        if date <= df["time"].max():
            price_idx = (df["time"] - date).abs().idxmin()
            ax0.scatter(
                date,
                df.loc[price_idx, "PriceUSD"],
                color="purple",
                s=100,
                alpha=0.9,
                marker="o",
                edgecolors="white",
                linewidth=2,
                zorder=10,
            )

    for date in TOPS_DATES:
        if date <= df["time"].max():
            price_idx = (df["time"] - date).abs().idxmin()
            ax0.scatter(
                date,
                df.loc[price_idx, "PriceUSD"],
                color="red",
                s=100,
                alpha=0.9,
                marker="o",
                edgecolors="white",
                linewidth=2,
                zorder=10,
            )

    for date in BOTTOMS_DATES:
        if date <= df["time"].max():
            price_idx = (df["time"] - date).abs().idxmin()
            ax0.scatter(
                date,
                df.loc[price_idx, "PriceUSD"],
                color="green",
                s=100,
                alpha=0.9,
                marker="o",
                edgecolors="white",
                linewidth=2,
                zorder=10,
            )

    # Add ONLY forecast lines from test set to price plot
    for target, info in forecast_dates.items():
        if info["forecast"] > df["time"].max():  # Only future forecasts
            ax0.axvline(
                info["forecast"],
                color=info["color"],
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"{target} Forecast"
            )

    ax0.axvline(
        df["time"].max(),
        color="black",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Today",
    )
    ax0.axvline(
        split_date,
        color="gray",
        linestyle="-.",
        alpha=0.7,
        linewidth=1.5,
        label="Train/Test Split",
    )

    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8)
    ax0.tick_params(axis="x", labelbottom=False)

    # Plot ensemble predictions for each target
    target_names = ["Target_1Y", "Target_6M", "Target_1M"]
    colors = ["#8B0000", "#CD5C5C", "#FFB6C1"]
    labels = [
        "1-Year Top Probability",
        "6-Month Top Probability",
        "1-Month Top Probability",
    ]

    for i, (target, color, label) in enumerate(zip(target_names, colors, labels), 1):
        ax = axes[i]

        ax.axvspan(split_date, df["time"].max(), alpha=0.15, color="gray", zorder=0)
        if len(future_df) > 0:
            ax.axvspan(
                df["time"].max(),
                future_df["time"].max(),
                alpha=0.08,
                color="lightblue",
                zorder=0,
            )

        col = f"pred_proba_{target}_smoothed"
        if col in df.columns:
            current_prob = df[col].iloc[-1] * 100
            ax.plot(
                df["time"],
                df[col],
                color=color,
                linewidth=3,
                label=f"Ensemble (Current: {current_prob:.1f}%)",
            )
            ax.fill_between(df["time"], 0, df[col], alpha=0.3, color=color)

            if col in future_df.columns:
                ax.plot(
                    future_df["time"],
                    future_df[col],
                    color=color,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                )

        # Add ONLY forecast line for this target (future only)
        if target in forecast_dates and forecast_dates[target]["forecast"] > df["time"].max():
            ax.axvline(
                forecast_dates[target]["forecast"],
                color=color,
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"Predicted Top"
            )

        ax.axvline(
            df["time"].max(), color="black", linestyle="--", alpha=0.6, linewidth=2
        )
        ax.axvline(split_date, color="gray", linestyle="-.", alpha=0.5, linewidth=1.5)
        ax.set_ylabel("Probability", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 1)

        if i < 3:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Individual models plots (if available)
    if individual_model_predictions:
        for target in target_names:
            if target in individual_model_predictions:
                n_models = len(individual_model_predictions[target])
                if n_models > 0:
                    fig, axes = plt.subplots(
                        n_models + 1,
                        1,
                        figsize=(12, 2 * (n_models + 1)),
                        gridspec_kw={"hspace": 0.3},
                    )

                    if n_models == 0:
                        axes = [axes]

                    # Price plot at top
                    ax0 = axes[0]
                    ax0.axvspan(
                        split_date, df["time"].max(), alpha=0.15, color="gray", zorder=0
                    )
                    if len(future_df) > 0:
                        ax0.axvspan(
                            df["time"].max(),
                            future_df["time"].max(),
                            alpha=0.08,
                            color="lightblue",
                            zorder=0,
                        )

                    ax0.plot(
                        df["time"],
                        df["PriceUSD"],
                        "b-",
                        linewidth=1.5,
                        label="Bitcoin Price",
                    )
                    ax0.set_yscale("log")
                    ax0.set_ylabel("Price (USD)", fontsize=10)
                    ax0.set_title(
                        f"Individual Model Predictions - {target}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax0.yaxis.set_major_formatter(FuncFormatter(price_formatter))
                    
                    # Add ONLY future forecast line to individual price plots
                    if target in forecast_dates and forecast_dates[target]["forecast"] > df["time"].max():
                        ax0.axvline(
                            forecast_dates[target]["forecast"],
                            color=forecast_dates[target]["color"],
                            linestyle=":",
                            alpha=0.8,
                            linewidth=3,
                            label=f"{target} Forecast"
                        )
                    
                    ax0.grid(True, alpha=0.3)
                    ax0.legend()
                    ax0.tick_params(axis="x", labelbottom=False)

                    # Individual model plots
                    model_colors = [
                        "#FF6B35",
                        "#F7931E", 
                        "#FFD23F",
                        "#06FFA5",
                        "#1FB7EA",
                    ]
                    for idx, (model_name, predictions) in enumerate(
                        individual_model_predictions[target].items()
                    ):
                        ax = axes[idx + 1]
                        color = model_colors[idx % len(model_colors)]

                        ax.axvspan(
                            split_date,
                            df["time"].max(),
                            alpha=0.15,
                            color="gray",
                            zorder=0,
                        )
                        if len(future_df) > 0:
                            ax.axvspan(
                                df["time"].max(),
                                future_df["time"].max(),
                                alpha=0.08,
                                color="lightblue",
                                zorder=0,
                            )

                        # Historical predictions
                        col_smoothed = f"pred_proba_{target}_{model_name}_smoothed"
                        if col_smoothed in df.columns:
                            current_prob = df[col_smoothed].iloc[-1] * 100
                            ax.plot(
                                df["time"],
                                df[col_smoothed],
                                color=color,
                                linewidth=2,
                                label=f"{model_name} (Current: {current_prob:.1f}%)",
                            )
                            ax.fill_between(
                                df["time"], 0, df[col_smoothed], alpha=0.3, color=color
                            )

                        # Add ONLY future forecast line for individual models
                        if target in forecast_dates and forecast_dates[target]["forecast"] > df["time"].max():
                            ax.axvline(
                                forecast_dates[target]["forecast"],
                                color=forecast_dates[target]["color"],
                                linestyle=":",
                                alpha=0.6,
                                linewidth=2
                            )

                        # Future predictions
                        if col_smoothed in future_df.columns:
                            ax.plot(
                                future_df["time"],
                                future_df[col_smoothed],
                                color=color,
                                linestyle="--",
                                alpha=0.8,
                                linewidth=2,
                            )

                        # Add event markers
                        for date in TOPS_DATES:
                            if date <= df["time"].max():
                                prob_idx = (df["time"] - date).abs().idxmin()
                                if col_smoothed in df.columns:
                                    ax.scatter(
                                        date,
                                        df.loc[prob_idx, col_smoothed],
                                        color="red",
                                        s=60,
                                        alpha=0.8,
                                        zorder=10,
                                    )

                        ax.axvline(
                            df["time"].max(),
                            color="black",
                            linestyle="--",
                            alpha=0.6,
                            linewidth=1,
                        )
                        ax.axvline(
                            split_date,
                            color="gray",
                            linestyle="-.",
                            alpha=0.5,
                            linewidth=1,
                        )
                        ax.set_ylabel("Probability", fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc="upper left", fontsize=8)
                        ax.set_ylim(0, 1)

                        if idx < n_models - 1:
                            ax.tick_params(axis="x", labelbottom=False)
                        else:
                            ax.set_xlabel("Date", fontsize=10)

                    plt.tight_layout()
                    plt.show()


def print_forecast_summary(
    df, future_df, activation_threshold=0.9, use_thermomodel=False, use_cycles=False
):
    """Print comprehensive forecast summary"""
    from datetime import datetime
    import pandas as pd
    
    split_date = pd.to_datetime("2024-01-01")  # Fixed split date
    today = pd.to_datetime(datetime.today().date())
    
    print("\n" + "=" * 60)
    print("🚀 BITCOIN CYCLE TOP FORECAST - ENHANCED")
    print("=" * 60)

    current_price = df["PriceUSD"].iloc[-1]
    print(f"💰 Current Price: ${current_price:,.0f}")

    if "days_until_next_halving" in df.columns:
        days_to_halving = df["days_until_next_halving"].iloc[-1]
        print(f"⛏️ Next Halving: {days_to_halving:.0f} days")

    # Calculate and display forecast dates ONLY from test set
    targets_info = {
        "Target_1Y": {"days": 365, "name": "1-Year"},
        "Target_6M": {"days": 180, "name": "6-Month"}, 
        "Target_1M": {"days": 30, "name": "1-Month"}
    }
    
    # Only look at test set data
    test_data = df[df["time"] >= split_date].copy()
    
    print(f"\n📅 CYCLE TOP FORECASTS (Threshold: {activation_threshold:.0%}, Test Set Only):")
    for target, info in targets_info.items():
        col = f"pred_proba_{target}_smoothed"
        if col in test_data.columns:
            above_threshold = test_data[test_data[col] >= activation_threshold]
            if not above_threshold.empty:
                trigger_date = above_threshold["time"].iloc[0]
                forecast_date = trigger_date + pd.Timedelta(days=info["days"])
                print(f"   🎯 {info['name']}: {forecast_date.strftime('%Y-%m-%d')} (triggered: {trigger_date.strftime('%Y-%m-%d')})")
            else:
                print(f"   ⏳ {info['name']}: Threshold not yet reached in test set")

    # Model status
    model_status = []
    if use_thermomodel and any("thermo" in col for col in df.columns):
        model_status.append("📊 ThermoModel")
        if "thermo_position_in_band" in df.columns:
            thermo_pos = df["thermo_position_in_band"].iloc[-1]
            if thermo_pos > 0.8:
                thermo_status = "🔴 OVERBOUGHT"
            elif thermo_pos < 0.2:
                thermo_status = "🟢 OVERSOLD"
            else:
                thermo_status = f"🟡 NEUTRAL ({thermo_pos:.1%})"
            print(f"🌡️ ThermoModel Status: {thermo_status}")

    if use_cycles and any("cycle" in col for col in df.columns):
        model_status.append("🔄 CycleModel")
        if "cycle" in df.columns:
            cycle_val = df["cycle"].iloc[-1]
            if cycle_val > 0.8:
                cycle_status = "🚨 CYCLE TOP ZONE"
            elif cycle_val > 0.6:
                cycle_status = "⚠️ ELEVATED"
            elif cycle_val < 0.4:
                cycle_status = "📉 LOW"
            else:
                cycle_status = f"📊 MODERATE ({cycle_val:.1%})"
            print(f"🔄 Cycle Model Status: {cycle_status}")

    if model_status:
        print(f"🧠 Enhanced with: {' + '.join(model_status)}")

    # Current ML signals
    print(f"\n📊 CURRENT ML SIGNALS:")
    targets = ["Target_1Y", "Target_6M", "Target_1M"]
    icons = ["🟣", "🟠", "🔴"]
    names = ["1-Year", "6-Month", "1-Month"]

    for target, icon, name in zip(targets, icons, names):
        col = f"pred_proba_{target}_smoothed"
        if col in df.columns:
            prob = df[col].iloc[-1]
            if prob >= 0.8:
                status = "🚨 CRITICAL"
            elif prob >= 0.6:
                status = "⚠️ HIGH"
            elif prob >= 0.4:
                status = "📈 ELEVATED"
            else:
                status = "😴 LOW"
            print(f" {icon} {name}: {prob:.0%} {status}")

    # Risk assessment
    current_signals = {}
    for target in targets:
        col = f"pred_proba_{target}_smoothed"
        if col in df.columns:
            current_signals[target] = df[col].iloc[-1]

    high_signals = {k: v for k, v in current_signals.items() if v > 0.6}
    critical_signals = {k: v for k, v in current_signals.items() if v > 0.8}

    print(f"\n⚠️ Risk Assessment:")
    print(f"   High signals (>60%): {len(high_signals)}/3")
    print(f"   Critical signals (>80%): {len(critical_signals)}/3")

    if len(critical_signals) >= 2:
        print("   🚨 MULTIPLE CRITICAL SIGNALS - HIGH RISK")
    elif len(high_signals) >= 2:
        print("   ⚠️ MULTIPLE HIGH SIGNALS - ELEVATED RISK")
    elif len(high_signals) >= 1:
        print("   📈 SOME ELEVATED SIGNALS - MODERATE RISK")
    else:
        print("   😴 LOW RISK SIGNALS")

    print("=" * 60)


def show_dataframe_info(df, show_sample=True):
    """Display comprehensive dataframe information"""
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("📊 DATAFRAME ANALYSIS")
    print("=" * 80)

    print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📅 Date range: {df['time'].min()} → {df['time'].max()}")

    print(f"\n🔍 COLUMNS ({df.shape[1]} total):")
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(
            f"  {i+1:2d}. {col:<35} | {dtype:<12} | {null_count:>4} nulls ({null_pct:>5.1f}%)"
        )

    if show_sample:
        print(f"\n📋 SAMPLE DATA (first 3 and last 3 rows):")
        sample_df = pd.concat([df.head(3), df.tail(3)])
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(sample_df.to_string(index=False))
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

    # Target distribution if available
    targets = ["Target_1M", "Target_6M", "Target_1Y"]
    available_targets = [t for t in targets if t in df.columns]
    if available_targets:
        print(f"\n🎯 TARGET DISTRIBUTION:")
        for target in available_targets:
            dist = df[target].value_counts().sort_index()
            total = len(df)
            print(f"  {target}:")
            for val, count in dist.items():
                pct = (count / total) * 100
                print(f"    {val}: {count:>6} ({pct:>5.1f}%)")

    # Prediction columns if available
    pred_cols = [col for col in df.columns if 'pred_proba' in col and '_smoothed' in col]
    if pred_cols:
        print(f"\n🎯 CURRENT PREDICTIONS:")
        for col in sorted(pred_cols):
            if col in df.columns:
                current_value = df[col].iloc[-1]
                print(f"  {col}: {current_value:.1%}")

    print("=" * 80)


def plot_feature_importances(feature_importances, show_top_n=30):
    """Plot feature importance charts"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    n_models = len(feature_importances)
    if n_models == 0:
        print("❌ No feature importances available")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    for idx, (target, imp_data) in enumerate(feature_importances.items()):
        try:
            features = imp_data["features"]
            importances = imp_data["importances"]

            imp_df = (
                pd.DataFrame({"feature": features, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(show_top_n)
            )

            ax = axes[idx]
            bars = ax.barh(
                imp_df["feature"][::-1], imp_df["importance"][::-1], color="tab:blue"
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {show_top_n} Features - {target}")
            ax.grid(axis="x", alpha=0.3)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width + max(imp_df["importance"]) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

        except Exception as e:
            print(f"❌ Cannot plot feature importances for {target}: {e}")

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n📊 TOP FEATURES SUMMARY:")
    for target, imp_data in feature_importances.items():
        try:
            features = imp_data["features"]
            importances = imp_data["importances"]
            estimators = imp_data.get("estimators", ["ensemble"])

            imp_df = (
                pd.DataFrame({"feature": features, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(10)
            )

            print(f"\n🎯 {target} - Top 10 Features (from {', '.join(estimators)}):")
            for _, row in imp_df.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            print(f"❌ Error processing {target}: {e}")


def plot_cv_confusion_matrices(performance):
    """Plot confusion matrices from cross-validation"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    targets_with_cm = {
        k: v
        for k, v in performance.items()
        if "cv_confusion_matrices" in v and v["cv_confusion_matrices"]
    }

    if not targets_with_cm:
        print("⚠️ No confusion matrices available from cross-validation")
        return

    # Plot ensemble confusion matrices
    n_targets = len(targets_with_cm)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    for idx, (target, metrics) in enumerate(targets_with_cm.items()):
        cm_list = metrics["cv_confusion_matrices"]

        if cm_list:
            avg_cm = np.mean(cm_list, axis=0).astype(int)

            ax = axes[idx]
            im = ax.imshow(avg_cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            thresh = avg_cm.max() / 2.0
            for i in range(avg_cm.shape[0]):
                for j in range(avg_cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(avg_cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if avg_cm[i, j] > thresh else "black",
                    )

            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            ax.set_title(f"{target} - Ensemble\nAvg Confusion Matrix (CV)")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Top", "Top"])
            ax.set_yticklabels(["No Top", "Top"])

    plt.tight_layout()
    plt.show()

    # Print detailed summary
    print("\n📊 CONFUSION MATRIX SUMMARY (from Cross-Validation):")
    for target, metrics in targets_with_cm.items():
        cm_list = metrics["cv_confusion_matrices"]
        if cm_list:
            avg_cm = np.mean(cm_list, axis=0)
            print(f"\n🎯 {target} - Ensemble (averaged over {len(cm_list)} CV folds):")
            print(f"   True Negatives: {avg_cm[0,0]:.1f}")
            print(f"   False Positives: {avg_cm[0,1]:.1f}")
            print(f"   False Negatives: {avg_cm[1,0]:.1f}")
            print(f"   True Positives: {avg_cm[1,1]:.1f}")

            if avg_cm[1, 1] + avg_cm[0, 1] > 0:
                precision = avg_cm[1, 1] / (avg_cm[1, 1] + avg_cm[0, 1])
                print(f"   Precision: {precision:.3f}")
            if avg_cm[1, 1] + avg_cm[1, 0] > 0:
                recall = avg_cm[1, 1] / (avg_cm[1, 1] + avg_cm[1, 0])
                print(f"   Recall: {recall:.3f}")

        # Individual model confusion matrix summary
        individual_cms = metrics.get("individual_confusion_matrices", {})
        for model_name, cm_list in individual_cms.items():
            if cm_list:
                avg_cm = np.mean(cm_list, axis=0)
                print(f"\n   📊 {model_name}:")
                print(
                    f"      TN: {avg_cm[0,0]:.1f}, FP: {avg_cm[0,1]:.1f}, FN: {avg_cm[1,0]:.1f}, TP: {avg_cm[1,1]:.1f}"
                )


def run_bitcoin_forecast(
    load_markets=True,
    load_fred=True,
    use_thermomodel=True,
    use_cycles=True,
    data_folder="data",
    show_feature_importance=False,
    show_df=False,
    activation_threshold=0.9,
    split_date="2024-01-01",
    apply_cycle_transformation=True,
    cycle_window=100,
):
    """Complete Bitcoin forecasting pipeline"""

    print("🚀 Starting Bitcoin Cycle Top Forecasting...")

    # Load data
    df = load_data(
        start_date="2010-01-01",
        load_btc=True,
        load_markets=load_markets,
        load_fred=load_fred,
        load_thermomodel=use_thermomodel,
        load_cycles=use_cycles,
        data_folder=data_folder,
        apply_cycle_transformation=apply_cycle_transformation,
        cycle_window=cycle_window,
    )

    if df is None:
        print("❌ Failed to load data")
        return None, None, None, None, None, None

    # Generate targets
    print("\n🎯 Generating targets...")
    df = generate_targets(df)

    # Feature engineering
    print("\n⚙️ Engineering features...")
    df = feature_engineering(df)

    # Show dataframe info if requested
    if show_df:
        show_dataframe_info(df, show_sample=True)

    # Reduce features
    print("\n🔧 Reducing features...")
    df = reduce_features(df)

    # Train models
    print("\n🤖 Training models...")
    (
        df,
        future_df,
        models,
        performance,
        feature_importances,
        individual_model_predictions,
    ) = train_models_and_predict(df, split_date=split_date)

    # Show performance
    print(f"\n📈 Model Performance Summary:")
    for target, metrics in performance.items():
        print(f"\n{target}:")
        print(
            f"  Ensemble CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}"
        )

        # Individual model performance
        individual_scores = metrics.get("individual_scores", {})
        for model_name, scores in individual_scores.items():
            if scores:
                print(
                    f"  {model_name} CV AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
                )

    # Plot results
    print("\n📊 Creating visualizations...")
    plot_predictions(
        df,
        future_df,
        individual_model_predictions,
        split_date=split_date,
        activation_threshold=activation_threshold,
    )

    # Show feature importance if requested
    if show_feature_importance and feature_importances:
        print("\n🎨 Showing feature importances...")
        plot_feature_importances(feature_importances)

    # Show confusion matrices
    print("\n🧪 Showing confusion matrices from Cross-Validation...")
    plot_cv_confusion_matrices(performance)

    # Print forecast summary
    print_forecast_summary(
        df, future_df, activation_threshold, use_thermomodel, use_cycles
    )

    print("\n✅ Bitcoin forecasting analysis complete!")

    return (
        df,
        future_df,
        models,
        performance,
        feature_importances,
        individual_model_predictions,
    )
