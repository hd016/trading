"""Backtesting utilities and summary reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import c


def backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rr = c(cfg, "labeling.rr", float)
    sl_atr_mult = c(cfg, "labeling.sl_atr_mult", float)
    fee_bps = c(cfg, "execution.fee_bps", float)
    slippage_bps = c(cfg, "execution.slippage_bps", float)
    cooldown_bars = c(cfg, "execution.cooldown_bars", int)

    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    position = 0
    entry_price = np.nan
    sl_price = np.nan
    tp_price = np.nan
    cooldown = 0

    equity = 1.0
    eq_curve = []

    open_prices = df["open"].to_numpy()
    high_prices = df["high"].to_numpy()
    low_prices = df["low"].to_numpy()
    atr_values = df["atr"].to_numpy()
    signals = df["signal"].fillna(0).to_numpy()

    for t in range(len(df)):
        eq_curve.append(equity)

        if cooldown > 0:
            cooldown -= 1
            continue

        if position == 0:
            if t + 1 >= len(df):
                continue

            if signals[t] == 1:
                position = 1
                entry_price = open_prices[t + 1] * (1 + slip)
                sl_dist = sl_atr_mult * atr_values[t]
                tp_dist = rr * sl_dist
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
                equity *= (1 - fee)

            elif signals[t] == -1:
                position = -1
                entry_price = open_prices[t + 1] * (1 - slip)
                sl_dist = sl_atr_mult * atr_values[t]
                tp_dist = rr * sl_dist
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                equity *= (1 - fee)

        else:
            if position == 1:
                hit_tp = high_prices[t] >= tp_price
                hit_sl = low_prices[t] <= sl_price

                if hit_tp and hit_sl:
                    dist_tp = abs(tp_price - open_prices[t])
                    dist_sl = abs(open_prices[t] - sl_price)
                    exit_price = (tp_price if dist_tp < dist_sl else sl_price) * (1 - slip)
                elif hit_tp:
                    exit_price = tp_price * (1 - slip)
                elif hit_sl:
                    exit_price = sl_price * (1 - slip)
                else:
                    exit_price = None

                if exit_price is not None:
                    equity *= (exit_price / entry_price)
                    equity *= (1 - fee)
                    position = 0
                    cooldown = cooldown_bars

            elif position == -1:
                hit_tp = low_prices[t] <= tp_price
                hit_sl = high_prices[t] >= sl_price

                if hit_tp and hit_sl:
                    dist_tp = abs(open_prices[t] - tp_price)
                    dist_sl = abs(sl_price - open_prices[t])
                    exit_price = (tp_price if dist_tp < dist_sl else sl_price) * (1 + slip)
                elif hit_tp:
                    exit_price = tp_price * (1 + slip)
                elif hit_sl:
                    exit_price = sl_price * (1 + slip)
                else:
                    exit_price = None

                if exit_price is not None:
                    equity *= (entry_price / exit_price)
                    equity *= (1 - fee)
                    position = 0
                    cooldown = cooldown_bars

    out = df.copy()
    out["equity"] = eq_curve
    return out


def summarize(bt: pd.DataFrame) -> None:
    equity = bt["equity"].dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = (equity / equity.cummax() - 1).min()
    changes = equity.pct_change().fillna(0)
    exits = int((changes != 0).sum())

    trade_returns = changes[changes != 0]
    if len(trade_returns) > 0:
        winrate = float((trade_returns > 0).mean())
        avg_win = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        avg_loss = float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 0.0
        expectancy = winrate * avg_win + (1 - winrate) * avg_loss
    else:
        winrate = avg_win = avg_loss = expectancy = 0.0

    print("==== Summary ====")
    print(f"Bars: {len(bt):,}")
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Max drawdown: {drawdown * 100:.2f}%")
    print(f"Exit count (approx): {exits:,}")
    print(
        f"Winrate (approx): {winrate * 100:.2f}% | "
        f"Avg win: {avg_win * 100:.3f}% | Avg loss: {avg_loss * 100:.3f}% | "
        f"Exp: {expectancy * 100:.3f}%"
    )
