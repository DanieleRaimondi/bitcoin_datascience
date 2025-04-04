import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import requests
from pytrends.request import TrendReq


def plot_google_trends_crypto(
    keywords: list,
    long_period: str = "today 5-y",
    short_period: str = "today 1-m",
    geo: str = "",
    retries: int = 10,
    pause: int = 5,
    type: str = "double",
):
    """
    Plot Google Trends data alongside cryptocurrency price data.

    Parameters:
    keywords (list): List of cryptocurrency names to analyze (e.g., ["bitcoin", "ethereum"])
    long_period (str): Timeframe for long-term analysis (default: 'today 5-y')
    short_period (str): Timeframe for short-term analysis (default: 'today 1-m')
    geo (str): Geographic location code (e.g., "US" for United States, empty for global)
    retries (int): Number of retry attempts for failed requests
    pause (int): Pause duration between requests in seconds
    type (str): Plot type ('single' for one timeframe, 'double' for both)

    Returns:
    dict: Contains trends data and price data for all analyzed cryptocurrencies
    """

    if not isinstance(keywords, list) or len(keywords) == 0:
        raise ValueError("Keywords must be a non-empty list")
    if len(keywords) > 10:
        raise ValueError("Maximum 10 keywords allowed")

    CRYPTO_MAPPING = {
        "bitcoin": "btc",
        "ethereum": "eth",
        "litecoin": "ltc",
        "solana": "sol",
        "dogecoin": "doge",
        "ripple": "xrp",
        "cardano": "ada",
    }

    # Mapping per le colonne dei prezzi per ogni criptovaluta
    PRICE_COLUMN_MAPPING = {
        "bitcoin": "PriceUSD",
        "ethereum": "PriceUSD",
        "litecoin": "PriceUSD",
        "solana": "principal_market_price_usd",  # Colonna specifica per SOL
        "dogecoin": "PriceUSD",
        "ripple": "PriceUSD",
        "cardano": "PriceUSD",
    }

    COLOR_MAPPING = {
        "bitcoin": "orange",
        "ethereum": "darkblue",
        "litecoin": "lightgray",
        "ripple": "dodgerblue",
        "cardano": "lightblue",
        "dogecoin": "darkkhaki",
        "solana": "purple",
    }

    class CustomTrendReq(TrendReq):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.sess = requests.Session()
            self.sess.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )

    def fetch_crypto_data(crypto: str, keyword: str) -> pd.DataFrame:
        url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{crypto}.csv"
        df = pd.read_csv(url, parse_dates=["time"], low_memory=False)

        # Usa il mapping specifico per determinare la colonna del prezzo
        price_col = PRICE_COLUMN_MAPPING.get(keyword.lower(), "PriceUSD")

        # Verifica se la colonna esiste nel dataframe
        if price_col not in df.columns:
            # Fallback per retrocompatibilità
            price_col = (
                "PriceUSD" if "PriceUSD" in df.columns else "principal_market_price_usd"
            )
            if price_col not in df.columns:
                raise ValueError(f"No valid price column found for {crypto}")

        result_df = df[["time", price_col]].set_index("time")
        result_df.columns = ["PriceUSD"]  # Rinomina per standardizzare
        return result_df

    def fetch_trends_data(kw: list, tf: str, initial_pause: int) -> dict:
        all_data = {}
        for keyword in kw:
            current_pause = initial_pause
            print(f"Fetching trends data for: {keyword} ({tf})...")

            for attempt in range(retries):
                try:
                    time.sleep(current_pause + random.uniform(1, 3))
                    pytrends = CustomTrendReq(hl="en-US", tz=360)
                    pytrends.build_payload(
                        kw_list=[keyword], cat=0, timeframe=tf, geo=geo
                    )
                    data = pytrends.interest_over_time()

                    if not data.empty:
                        print(f"Data successfully retrieved for {keyword}!")
                        all_data[keyword] = data[keyword]
                        break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {keyword}: {str(e)}")
                    current_pause *= 2
                    if attempt == retries - 1:
                        print(f"Failed to retrieve data for {keyword}")
                        all_data[keyword] = None
        return all_data

    def plot_data(trends_data: dict, ax, keyword: str, idx: int, title: str = ""):
        color = COLOR_MAPPING.get(keyword.lower(), f"C{random.randint(0, 9)}")

        if trends_data[keyword] is not None:
            # Plot trends data
            line_trends = ax.plot(
                trends_data[keyword].index,
                trends_data[keyword],
                linestyle="-",
                linewidth=2,
                markersize=4,
                color=color,
                label="Google Trends",
            )
            ax.set_ylabel("Interest (0-100)", color=color)
            ax.tick_params(axis="y", labelcolor=color)

            ax.set_ylim(0, 100)  # Set fixed y-limits for trends (0-100)

            if keyword.lower() in CRYPTO_MAPPING and price_data[keyword] is not None:
                ax2 = ax.twinx()

                # Ottimizzazione: Usa lo stesso intervallo di date per il plot di trend e prezzo
                # e assicurati che vengano visualizzati tutti i dati disponibili
                start_date = trends_data[keyword].index[0]
                end_date = trends_data[keyword].index[-1]

                # Assicurati che i dati di prezzo coprano l'intero intervallo
                # Se ci sono dati di prezzo più recenti, li mostriamo comunque
                price_subset = price_data[keyword]

                # Se ci sono dati di prezzo disponibili che iniziano dopo i dati di trends
                if price_subset.index[0] > start_date:
                    start_date = price_subset.index[0]

                # Se ci sono dati di prezzo che finiscono dopo i dati di trends
                if price_subset.index[-1] > end_date:
                    # Estendi il plot per mostrare anche i dati più recenti
                    price_subset = price_subset[price_subset.index >= start_date]
                else:
                    # Altrimenti filtra solo nel range dei dati di trends
                    price_subset = price_subset[
                        (price_subset.index >= start_date)
                        & (price_subset.index <= end_date)
                    ]

                line_price = ax2.plot(
                    price_subset.index,
                    price_subset["PriceUSD"],
                    color="black",
                    linestyle="--",
                    alpha=0.7,
                    label="Price (USD)",
                )

                ax2.set_ylabel("Price (USD)", color="black")
                ax2.tick_params(axis="y", labelcolor="black")

                # Imposta i limiti esatti min e max
                if not price_subset.empty:
                    ax2.set_ylim(
                        price_subset["PriceUSD"].min() * 0.9,
                        price_subset["PriceUSD"].max() * 1.1,
                    )

                lines = line_trends + line_price
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc="upper left")

            ax.set_title(f'{title} "{keyword}"', fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)

            # Aggiungi statistiche
            if keyword.lower() in CRYPTO_MAPPING and price_data[keyword] is not None:
                stats = f"Trends Max: {trends_data[keyword].max():.1f}\n"
                stats += f"Trends Avg: {trends_data[keyword].mean():.1f}\n"

                # Aggiungi informazioni sui prezzi
                if not price_subset.empty:
                    price_start = price_subset["PriceUSD"].iloc[0]
                    price_end = price_subset["PriceUSD"].iloc[-1]
                    price_change = ((price_end - price_start) / price_start) * 100
                    stats += f"Price Start: ${price_start:.2f}\n"
                    stats += f"Price End: ${price_end:.2f}\n"
                    stats += f"Change: {price_change:.1f}%"

                ax.text(
                    0.02,
                    0.98,
                    stats,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

    price_data = {}
    for keyword in keywords:
        if keyword.lower() in CRYPTO_MAPPING:
            try:
                # Passa il keyword alla funzione per selezionare la colonna corretta
                price_data[keyword] = fetch_crypto_data(
                    CRYPTO_MAPPING[keyword.lower()], keyword
                )
                print(f"Successfully fetched price data for {keyword}")
            except Exception as e:
                print(f"Failed to fetch price data for {keyword}: {str(e)}")
                price_data[keyword] = None

    if type == "single":
        trends_data = fetch_trends_data(keywords, long_period, pause)
        fig, axes = plt.subplots(len(keywords), 1, figsize=(12, 4 * len(keywords)))
        if len(keywords) == 1:
            axes = [axes]

        for idx, keyword in enumerate(keywords):
            plot_data(trends_data, axes[idx], keyword, idx)

        long_data, short_data = trends_data, None
    else:
        long_data = fetch_trends_data(keywords, long_period, pause)
        short_data = fetch_trends_data(keywords, short_period, pause)

        fig, axes = plt.subplots(len(keywords), 2, figsize=(24, 4 * len(keywords)))
        if len(keywords) == 1:
            axes = axes.reshape(1, 2)

        plt.suptitle(
            "Crypto Analysis: Google Trends vs Price", fontsize=22, fontweight="bold"
        )

        for idx, keyword in enumerate(keywords):
            plot_data(long_data, axes[idx, 0], keyword, idx, "Long Period Analysis for")
            plot_data(
                short_data, axes[idx, 1], keyword, idx, "Short Period Analysis for"
            )

    plt.tight_layout()
    plt.savefig("../output/6a.GoogleTrends.jpeg", bbox_inches="tight", dpi=350)

    plt.show()

    return {
        "long_period": long_data,
        "short_period": short_data,
        "price_data": price_data,
    }
