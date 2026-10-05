"""Run a backtest from the command line.

Examples:
    python main.py --ticker NVDA --strategy sma
    python main.py --ticker QQQM --strategy rsi --period 5y
"""

import argparse
import os

from backtest import (
    get_prices,
    sma_crossover,
    rsi_reversion,
    backtest,
    summarize,
    plot_backtest,
)

STRATEGIES = {
    "sma": ("SMA 20/50 crossover", sma_crossover),
    "rsi": ("RSI(14) mean reversion", rsi_reversion),
}


def main():
    ap = argparse.ArgumentParser(description="backtest a simple stock strategy")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--strategy", choices=STRATEGIES, default="sma")
    ap.add_argument("--period", default="2y",
                    help="how far back to test, e.g. 1y 2y 5y max")
    args = ap.parse_args()

    label, fn = STRATEGIES[args.strategy]
    prices = get_prices(args.ticker, period=args.period)
    bt, trades = backtest(fn(prices))
    s = summarize(bt, trades)

    print(f"\n{ticker_line(args.ticker, label, args.period)}")
    print(f"  total return   {s['total_return']:+.1%}   (buy & hold {s['buy_hold_return']:+.1%})")
    print(f"  CAGR           {s['cagr']:+.1%}")
    print(f"  Sharpe         {s['sharpe']:.2f}")
    print(f"  max drawdown   {s['max_drawdown']:.1%}")
    print(f"  trades         {s['num_trades']}   (win rate {s['win_rate']:.0%})")

    out = os.path.join("results", f"{args.ticker}_{args.strategy}.png")
    plot_backtest(bt, trades, args.ticker, label, out)
    print(f"\nchart saved to {out}\n")


def ticker_line(ticker, label, period):
    return f"{ticker} | {label} | last {period}"


if __name__ == "__main__":
    main()
