"""
RF + Rule-Based Hybrid Crypto Scalper (LONG + SHORT) — Binance USDT-M Futures (1m)
CONFIG-ONLY VERSION (NO PARAMETERS IN CODE)

This entrypoint now delegates to smaller modules under ``scalper`` to improve
readability and maintainability while keeping the original CLI:

    python rf_hybrid_scalper.py --config config.yaml
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from scalper.backtest import backtest, summarize
from scalper.config import c, load_yaml_config, validate_cfg
from scalper.data import load_or_fetch_ohlcv
from scalper.features import make_features
from scalper.filters import add_pine_like_filters
from scalper.model import walk_forward_rf
from scalper.signals import make_hybrid_signals


def run(cfg_path: str, refresh: bool) -> None:
    cfg = load_yaml_config(cfg_path)
    validate_cfg(cfg)

    symbol = c(cfg, "data.symbol", str)
    interval = c(cfg, "data.interval", str)
    days = c(cfg, "data.days", int)
    out_csv = c(cfg, "output.backtest_csv", str)

    print("Loaded config:", Path(cfg_path).resolve())
    print(f"Symbol/TF: {symbol} {interval} | Days: {days}")

    df = load_or_fetch_ohlcv(cfg, refresh=refresh)
    print("Bars loaded:", len(df))

    df = make_features(df, cfg)
    df = add_pine_like_filters(df, cfg)

    wf = walk_forward_rf(df, cfg)
    wf = make_hybrid_signals(wf, cfg)

    print("Signals distribution:", wf["signal"].value_counts(dropna=False).to_dict())

    bt = backtest(wf, cfg)
    summarize(bt)

    bt.to_csv(out_csv)
    print("Saved:", out_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and refetch from Binance")
    args = parser.parse_args()
    run(args.config, args.refresh)


if __name__ == "__main__":
    main()
