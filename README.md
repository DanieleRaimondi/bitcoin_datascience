# **Bitcoin DataScience ₿** 📈

A collection of quantitative models for Bitcoin: long-run price corridors, halving-cycle structure,
on-chain valuation indicators, and the supply and demand studies that sit around them. The charts
are published for non-commercial, educational viewing.

> **Data**: Bitcoin price and on-chain metrics from the [Coin Metrics Community API](https://docs.coinmetrics.io/packages/coin-metrics-community-data)
> ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)). See [Data & Attribution](#data--attribution).

[![ThermoClock: levels from the bands, timing from the cycle](output/1j.ThermoClock_Animation.gif)](output/1j.ThermoClock_Animation_Narrated.mp4)

*ThermoClock in two minutes, silent preview above. [Watch the narrated version](output/1j.ThermoClock_Animation_Narrated.mp4)
(MP4, 2:16, English voice-over).*

---

## 1. Price Models

Where price sits against its own long-run trend, and how wide the corridor around it should be.

### ThermoClock 🕰️
![](output/1j.ThermoClock.jpg)
The ThermoModel's bands (below) answer *how high and how low* but say nothing about *when*; the
cycle analysis answers the opposite. Combined, they give a dated price path. A ThermoClock - the
same clock the cycle analysis draws, with every cycle the same length (1,419 days, fitted by least
squares on the tops and bottoms together) and bottoms a quarter of a cycle after each top, since
the cycle is asymmetric, roughly a year down and three years back up - is mapped against where in
the corridor price historically sat at each point of the cycle, then projected onto the extended
bands. The per-cycle profiles correlate 0.55–0.77 pairwise and agree tightly near the extremes
(interquartile spread down to 3 points) but less so late in the climb (up to 39 points), so the
dated extremes are worth more than the path between them.

### ThermoModel 🌡️
![](output/1a.ThermoModel.jpg)
Predicts and analyses Bitcoin price movements by combining cubic and logarithmic regression
models. Dynamic price bands and an oscillator identify potential buy and sell zones, giving a
single view of where price stands relative to its thermodynamic corridor.

### ThermoModel v2 🌡️📉
![](output/1a.ThermoModel_v2.jpg)
A rebuild of the ThermoModel that accounts for diminishing returns. The original fits a cubic
polynomial in day index with fixed log-space offsets: the cubic extrapolates explosively upward
(the opposite of diminishing returns) and a fixed offset keeps a constant band width forever.
The trend is a log-log fit whose exponent eases off with time: refitted in real time, a plain
power law's fixed exponent kept falling (6.8, 5.9, 5.8, 5.6 at the last four bottoms) and
overshot every next bottom, which the concave trend corrects. Measured against it, cycle tops came
in at 18.4x, 9.7x, 5.7x, 3.0x, 1.5x and the deepest close of each cycle at 0.32x, 0.32x, 0.42x,
0.47x, 0.56x, so both offsets move with the cycle: the upper one decays, fitted on the five tops,
the lower one rises, fitted by quantile regression on the daily closes so that **99.5% of the
history closes above it** (the old flat band, the mean of the five bottoms, was cut through on 15%
of days, the deepest 35% below). Validated leave-future-out on out-of-sample top prediction, the
median absolute error drops to **13% from 83%** (22% on a fixed-exponent trend).

### LogTimeLogPrice 🪜
![](output/1b.LogTimeLogPrice.jpg)
Analyses Bitcoin's price growth over time on a log-log graph. Beyond plotting support and
resistance lines via Ordinary Least Squares regression, it adds an oscillator subplot that reads
market sentiment from deviations against those trend lines.

### AritmTimeLogPrice ⚙️
![](output/1c.AritmTimeLogPrice.jpg)
An exponential transformation of the time dimension in the LogTimeLogPrice model, applied so that
its results become directly comparable with the ThermoModel's.

### ThermoLogTimeLogPrice 💪🏻
![](output/1d.EnsembleCorridor.jpg)
A straightforward ensemble of the ThermoModel and LogTimeLogPrice: averaging the predicted upper
and lower bands of both produces a more robust corridor than either alone.

### Metcalfe's Law 🕸️
![](output/1h.Metcalfe.jpg)
Quantifies Bitcoin's fundamental value from network activity to identify overvaluation and
undervaluation periods, turning network-effects theory into a data-driven indicator.

### 209-Week SMA 🚂
![](output/1g.BTC_SMA.jpg)
Bitcoin's price against its 209-week Simple Moving Average. The 209-week window is the halving
cycle itself — four years, roughly 209 weeks — and this average has historically acted as a
durable support level.

---

## 2. Cycles

The four-year halving rhythm, and where in it we currently stand.

### Cycles 🧲
![](output/2a.Cycles.jpg)
Analyses and visualises the cyclical patterns in Bitcoin's price, with particular focus on the
four-year cyclicity dictated by halving events, highlighting how those milestones shape the
broader cyclical behaviour of the market.

### CyclesNorm 🔄
![](output/2b.CyclesNorm.jpg)
Normalises price data from the 2016, 2020 and 2024 halving cycles - plus the nascent, still-open
2028 cycle - so they can be compared directly, identifying key inflection points and projecting
the timing of potential tops and bottoms. Colour-coded background shading marks the market
phases, with a sinusoidal overlay for the underlying cyclical structure.

### Epochs Growth 📆
![](output/2c.EpochsGrowth.jpg)
Bitcoin's price growth measured from each halving event, which makes the diminishing-returns
effect legible across epochs.

### MVRV 🔋
![](output/2c.MVRV_Oscillator.jpg)
Explores the relationship between Bitcoin's market and realized values to pinpoint tops and
bottoms. An oscillator marks overbought and oversold zones, highlighting the two-peak structure of
each cycle, where the second peak typically marks the all-time high.

---

## 3. Valuation & Risk

On-chain and price-derived indicators for cycle position. The composite comes first; the
components that feed the same question follow.

### Cycle Risk Model 🎚️
![](output/11.RiskModel.jpg)
A single composite 0–100 score answering "where are we in the cycle?": 0 = cycle lows, 100 = cycle
highs. It blends three indicators, each ranked against its own trailing 4-year distribution: the
deviation of price from its expanding power-law trend (weight 0.6), the MVRV ratio (0.2) and the
2-Year MA Multiplier (0.2). Weights were selected by backtest rather than by taste, and are
renormalized over whatever components are available so the score survives the on-chain feed lagging.

Validated on forward returns by risk bucket: readings in the 80–100 bucket have historically been
followed by *negative* median returns at every horizon (−14.5% at 90d, −29.1% at 365d), while low
readings precede strong gains. The edge is concentrated at the extremes and grows with the horizon
— this is a cycle-position tool, not a short-term timing signal. A conditional, empirical
projection (every past episode that started at today's risk level *and* today's 30-day direction
of travel) extends it six months forward; with only a handful of overlapping episodes from three
completed cycles, read it as "what happened the last few times", not as a calibrated probability.

### MVRV Z-Score 🧮
![](output/10b.MVRVZScore.jpg)
Measures the gap between Bitcoin's market cap and its realized cap (the aggregate cost basis of
all coins), normalized by the historical standard deviation of market cap. Plots the Realized
Price alongside BTC price, and has historically flagged euphoric tops (Z-Score > 7) and
capitulation bottoms (Z-Score < 0).

### NUPL 😨🤑
![](output/10b.NUPL.jpg)
Net Unrealized Profit/Loss expresses the same market cap vs. realized cap gap as a fraction of
market cap, split into five classic sentiment zones from Capitulation to Euphoria/Greed — a quick
read on how much of the market is sitting on unrealized gains or losses.

### Puell Multiple ⛏️💵
![](output/10a.PuellMultiple.jpg)
Compares the daily USD value of newly issued Bitcoin against its own 365-day moving average.
Spikes above 4x have historically coincided with cycle tops (miners cashing out into euphoric
prices), while drops below 0.5x have marked cycle bottoms (depressed miner revenue).

### Pi Cycle Top Indicator 🥧
![](output/10c.PiCycleTop.jpg)
Watches for the 111-day moving average crossing above 2x the 350-day moving average. Every time
this crossover has occurred, Bitcoin's price has been within days of the cycle's eventual top.

### Golden Ratio Multiplier 🌀
![](output/10d.GoldenRatioMultiplier.jpg)
Tracks Bitcoin's 350-day moving average and its Fibonacci multiples (1.6x, 2x, 3x, 5x, 8x, 13x,
21x). Each band has historically lined up with a resistance level reached during past bull-market
advances.

### 2-Year MA Multiplier 📏
![](output/10f.2YearMAMultiplier.jpg)
Plots Bitcoin's price against its own 730-day (2-year) moving average and 5x that average. The
2-year MA has historically acted as a strong long-term accumulation zone, while 5x the 2-year MA
has capped the blow-off top of every cycle to date.

### Hash Ribbons ⛏️📶
![](output/10e.HashRibbons.jpg)
Compares the 30-day and 60-day moving averages of Bitcoin's hashrate to flag miner capitulation
(30-day average below the 60-day) and its recovery. Historically, the recovery point has marked
strong accumulation zones. Short whipsaws around the crossing point are filtered out so only
sustained capitulations are flagged.

---

## 4. Supply & Demand

### BTC vs Supply 💭
![](output/1f.BTCvsSupply.jpg)
Explores the growth correlation between Bitcoin's supply curve and its price history, to see how
programmed scarcity influences value over time.

### Available Supply 💰
![](output/4b.AvailableSupply.jpg)
Estimates the percentage of Bitcoin lost over time and forecasts the future available supply,
accounting for both the 21M cap and a decaying loss rate. The result is a view of the effective
supply actually in circulation, and of how diminishing float could affect price over the long term.

### Demand 🙋🏽‍♂️
![](output/4c.Demand.jpg)
Analyses on-chain activity — active addresses and total transaction count — alongside price, using
both as proxies for Bitcoin demand and for the growth of real user engagement on the network.

---

## Data & Attribution

| Source | Used for | Terms |
| --- | --- | --- |
| [Coin Metrics Community API](https://docs.coinmetrics.io/packages/coin-metrics-community-data) | Bitcoin daily price and on-chain metrics (market and realized value, supply, issuance, hashrate, active addresses, transaction count) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

Source: Coin Metrics Community Data, licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The charts are derived from it: the data is filtered, transformed and run through the models described
here. Coin Metrics does not endorse this work. Only Bitcoin data is used.

The ThermoClock video is drawn from the same data, so the credit above applies to it too. Its text is
set in [Inter](https://rsms.me/inter/) (SIL Open Font License 1.1), and its English voice-over is
synthetic, generated locally with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache 2.0): it is not a recording of a person.

Several indicators implemented here are established, publicly documented concepts created by
others; the implementations are mine, the ideas are theirs:

- **Puell Multiple** — David Puell
- **MVRV / MVRV Z-Score** — Murad Mahmudov & David Puell
- **NUPL** — Tuur Demeester & Adamant Research
- **Pi Cycle Top Indicator**, **Golden Ratio Multiplier**, **2-Year MA Multiplier** — Philip Swift
- **Hash Ribbons** — Charles Edwards, Capriole Investments
- **Metcalfe's Law** — Robert Metcalfe

Price and on-chain metrics come from the same daily feed, so every chart runs to the same last
completed day.

---

## Notes

⭐ If you find this useful, star the repository. Feedback and corrections are welcome via issues.

**Disclaimer**: the content of this repository is for informational and educational purposes only
and is not financial advice. Always do your own research and consult a professional before making
any investment decision. 🚫💰📚

**Copyright** © Daniele Raimondi. All rights reserved. The charts and write-ups are published
here for viewing; no licence to use, copy, modify or redistribute the underlying code is granted.
Third-party data remain subject to their providers' terms, as listed above.
