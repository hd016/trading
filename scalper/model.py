"""Walk-forward RandomForest modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import c
from .labeling import tp_sl_labels


FEATURE_COLS = [
    "ret1",
    "ret3",
    "ret8",
    "vol10",
    "vol30",
    "ema_diff",
    "rsi14",
    "atr_n",
    "vol_ratio",
    "hl_range",
    "body",
    "upper_wick",
    "lower_wick",
    "preAlertScore_dump",
    "preAlertScore_pump",
    "entryScore_short",
    "entryScore_long",
]


def walk_forward_rf(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train_bars = c(cfg, "walkforward.train_bars", int)
    test_bars = c(cfg, "walkforward.test_bars", int)
    step_bars = c(cfg, "walkforward.step_bars", int)

    n_estimators = c(cfg, "model.n_estimators", int)
    max_depth = c(cfg, "model.max_depth")
    min_samples_leaf = c(cfg, "model.min_samples_leaf", int)
    random_state = c(cfg, "model.random_state", int)

    y_long, y_short, valid = tp_sl_labels(df, cfg)
    out = df.copy()
    out["y_long"] = y_long
    out["y_short"] = y_short
    out["valid"] = valid

    p_long = np.full(len(out), np.nan, dtype=float)
    p_short = np.full(len(out), np.nan, dtype=float)

    i = 0
    while True:
        train_start = i
        train_end = train_start + train_bars
        test_end = train_end + test_bars
        if test_end > len(out):
            break

        train = out.iloc[train_start:train_end]
        test = out.iloc[train_end:test_end]

        train = train[train["valid"] == 1]
        test_valid = (test["valid"] == 1).to_numpy()

        X_train = train[FEATURE_COLS].to_numpy()
        y_long_train = train["y_long"].to_numpy()
        y_short_train = train["y_short"].to_numpy()

        model_long = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        model_short = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state + 1,
            class_weight="balanced_subsample",
        )

        model_long.fit(X_train, y_long_train)
        model_short.fit(X_train, y_short_train)

        X_test = test[FEATURE_COLS].to_numpy()
        proba_long = model_long.predict_proba(X_test)[:, 1]
        proba_short = model_short.predict_proba(X_test)[:, 1]

        base = train_end
        for k in range(len(test)):
            if test_valid[k]:
                p_long[base + k] = float(proba_long[k])
                p_short[base + k] = float(proba_short[k])

        i += step_bars

    out["p_long"] = p_long
    out["p_short"] = p_short
    return out
