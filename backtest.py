"""
backtest.py - small long-only strategy backtester on daily bars.

A strategy is just a function that adds a "signal" column to a price
dataframe: 1 means hold the stock, 0 means hold cash.
"""

import os

import matplotlib
matplotlib.use("Agg")  # no display needed, we only save pngs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# ---------------- data ----------------

def get_prices(ticker, period="2y", cache_dir="data"):
    """Daily closes for ticker. Results are cached to csv so reruns are fast."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}_{period}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"no price data for {ticker}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance nests columns sometimes
            close = close.iloc[:, 0]
        df = pd.DataFrame({"Close": close})
        df.index = pd.to_datetime(df.index)
        df.to_csv(path)
    return df.dropna()


# ---------------- strategies ----------------

def sma_crossover(df, fast=20, slow=50):
    """Hold when the fast average is above the slow average."""
    out = df.copy()
    out["fast"] = out["Close"].rolling(fast).mean()
    out["slow"] = out["Close"].rolling(slow).mean()
    out["signal"] = (out["fast"] > out["slow"]).astype(int)
    return out.dropna()


def _rsi(close, window=14):
    """RSI with Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def rsi_reversion(df, window=14, low=30, high=70):
    """Buy when RSI drops below `low` (oversold), sell when above `high`."""
    out = df.copy()
    out["rsi"] = _rsi(out["Close"], window)
    sig = pd.Series(np.nan, index=out.index)
    sig[out["rsi"] < low] = 1
    sig[out["rsi"] > high] = 0
    out["signal"] = sig.ffill().fillna(0).astype(int)  # hold last state in between
    return out.dropna()


# ---------------- engine ----------------

def backtest(df):
    """Walk the bars. A signal on day t is traded at t's close (no peeking)."""
    out = df.copy()
    daily = out["Close"].pct_change()
    pos = out["signal"].shift(1).fillna(0)
    out["strat_ret"] = pos * daily
    out["equity"] = (1 + out["strat_ret"].fillna(0)).cumprod()
    out["buy_hold"] = (1 + daily.fillna(0)).cumprod()

    # pair up entries and exits from signal flips
    change = out["signal"].diff().fillna(0)
    entries = list(out.index[change == 1])
    exits = list(out.index[change == -1])
    if len(entries) > len(exits):
        entries = entries[:len(exits)]  # still holding at the end, ignore it
    trades = pd.DataFrame({
        "entry": entries,
        "exit": exits,
        "entry_px": out.loc[entries, "Close"].values,
        "exit_px": out.loc[exits, "Close"].values,
    })
    trades["ret"] = trades["exit_px"] / trades["entry_px"] - 1
    return out, trades


# ---------------- metrics ----------------

def summarize(bt, trades):
    """One dict of headline numbers for the strategy."""
    eq = bt["equity"]
    n = len(bt)
    rets = bt["strat_ret"].dropna()
    vol = rets.std()
    wins = trades[trades["ret"] > 0]
    return {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (252 / n) - 1,
        "sharpe": rets.mean() / vol * np.sqrt(252) if vol > 0 else 0.0,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades) if len(trades) else 0.0,
        "buy_hold_return": bt["buy_hold"].iloc[-1] - 1,
    }


# ---------------- plot ----------------

def plot_backtest(bt, trades, ticker, strategy_label, save_to):
    """Price panel with buy/sell markers on top, equity curves below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(bt.index, bt["Close"], color="black", lw=1, label="close")
    if "fast" in bt.columns:
        ax1.plot(bt.index, bt["fast"], lw=1, label="fast avg")
        ax1.plot(bt.index, bt["slow"], lw=1, label="slow avg")
    if len(trades):
        ax1.scatter(trades["entry"], trades["entry_px"], marker="^",
                    color="green", s=60, zorder=5, label="buy")
        ax1.scatter(trades["exit"], trades["exit_px"], marker="v",
                    color="red", s=60, zorder=5, label="sell")
    ax1.set_title(f"{ticker} - {strategy_label}")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(bt.index, bt["equity"], color="black", lw=1.5, label="strategy")
    ax2.plot(bt.index, bt["buy_hold"], color="gray", lw=1, ls="--", label="buy & hold")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    ax2.set_ylabel("growth of $1")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
