"""Unit tests for AI agent DB → VectorBT signal conversion (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.signal_builder import ai_recommendations_to_signals, normalize_reco_frame


def test_overweight_maps_to_buy_entry():
    idx = pd.bdate_range("2025-01-02", periods=5)
    price = pd.Series([100.0, 101, 102, 103, 104], index=idx)
    recs = pd.DataFrame(
        [
            {"trade_date": "2025-01-03", "decision": "OVERWEIGHT", "bucket": "unknown"},
            {"trade_date": "2025-01-06", "decision": "UNDERWEIGHT", "bucket": ""},
        ]
    )
    entries, exits = ai_recommendations_to_signals(recs, price)
    assert entries.any()
    assert exits.any()
    assert int(entries.sum()) == 1
    assert int(exits.sum()) == 1


def test_normalize_reco_frame_from_executions_shape():
    df = pd.DataFrame(
        [
            {
                "ticker": "tcs.ns",
                "trade_date": "2025-02-01",
                "action_taken": "BUY",
                "decision": "BUY",
                "price": 3500.0,
            }
        ]
    )
    out = normalize_reco_frame(df)
    assert out.iloc[0]["bucket"] == "buy"
    assert out.iloc[0]["ticker"] == "TCS.NS"
