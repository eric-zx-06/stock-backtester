"""Minimal web UI for the backtester. Run with: streamlit run app.py"""

import streamlit as st

from backtest import (
    get_prices,
    sma_crossover,
    rsi_reversion,
    backtest,
    summarize,
    plot_backtest,
)

st.set_page_config(page_title="backtester", layout="wide")

# flat look: square corners everywhere, no gradients
st.markdown("""
<style>
* { border-radius: 0 !important; }
div[data-testid="stMetric"] {
    border: 1px solid #ddd;
    padding: 12px;
}
.stButton > button {
    border: 1px solid #333;
    background: #111;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

st.title("backtester")
st.caption("long-only strategy backtests on daily bars. for learning, not advice.")

with st.sidebar:
    st.header("setup")
    ticker = st.text_input("ticker", value="NVDA").upper().strip()
    strategy = st.selectbox("strategy", ["sma crossover", "rsi reversion"])
    period = st.selectbox("history", ["1y", "2y", "5y"], index=1)
    if strategy == "sma crossover":
        fast = st.slider("fast average (days)", 5, 60, 20)
        slow = st.slider("slow average (days)", 20, 200, 50)
    else:
        window = st.slider("rsi window", 5, 30, 14)
        low = st.slider("oversold level", 10, 40, 30)
        high = st.slider("overbought level", 60, 90, 70)
    run = st.button("run backtest")

if run and ticker:
    try:
        prices = get_prices(ticker, period=period)
    except Exception as e:
        st.error(f"could not load {ticker}: {e}")
        st.stop()

    if strategy == "sma crossover":
        label = f"SMA {fast}/{slow} crossover"
        df = sma_crossover(prices, fast=fast, slow=slow)
    else:
        label = f"RSI({window}) {low}/{high} reversion"
        df = rsi_reversion(prices, window=window, low=low, high=high)

    bt, trades = backtest(df)
    s = summarize(bt, trades)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("total return", f"{s['total_return']:+.1%}",
              delta=f"{s['total_return'] - s['buy_hold_return']:+.1%} vs buy&hold")
    c2.metric("sharpe", f"{s['sharpe']:.2f}")
    c3.metric("max drawdown", f"{s['max_drawdown']:.1%}")
    c4.metric("trades", s["num_trades"], delta=f"{s['win_rate']:.0%} win rate")

    plot_backtest(bt, trades, ticker, label, "results/app_chart.png")
    st.image("results/app_chart.png")

    if len(trades):
        st.subheader("trades")
        show = trades.copy()
        show["entry"] = show["entry"].dt.date
        show["exit"] = show["exit"].dt.date
        show["ret"] = (show["ret"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True)
    else:
        st.info("no trades in this period.")
