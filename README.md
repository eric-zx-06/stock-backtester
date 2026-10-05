# stock-backtester

Backtest simple long-only stock strategies on daily price data. See whether a
rule would actually have made money, and how bumpy the ride was.

## Strategies

- **SMA crossover** — hold the stock when the 20-day average is above the
  50-day average, otherwise hold cash.
- **RSI mean reversion** — buy when RSI(14) drops below 30 (oversold), sell
  when it rises above 70 (overbought).

## Quick start

```bash
pip install -r requirements.txt

# command line
python main.py --ticker NVDA --strategy sma
python main.py --ticker QQQM --strategy rsi --period 5y

# web UI
streamlit run app.py
```

Each run prints total return vs buy & hold, CAGR, Sharpe ratio, max drawdown,
trade count and win rate, and saves a chart under `results/`.

## Notes

- A signal on day *t* is traded at *t*'s close, so there is no lookahead bias.
- No fees or slippage are modeled. Real results would be worse.
- This is a learning tool, not financial advice.

## Files

- `backtest.py` — data loading, strategies, backtest engine, metrics, charts
- `main.py` — command-line runner
- `app.py` — minimal Streamlit UI
