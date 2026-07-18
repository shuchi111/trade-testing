"""Store VectorBT backtest results and trade logs in Supabase."""
import logging
import math

from supabase import create_client

from .config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

# Columns from migrations 008/009 — stripped on retry when not yet applied.
_OPTIONAL_RESULT_COLUMNS = frozenset({
    "ic",
    "rank_ic",
    "ml_horizon",
    "ml_train_rows",
    "ml_feature_count",
    "ml_retrain_date",
})


def _is_missing_column_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "pgrst204" in msg or ("could not find" in msg and "column" in msg)


def _insert_row(sb, row: dict):
    return sb.table("bt_strategy_results").insert(row).execute()


def _sanitize(val):
    """Replace NaN/Inf with None so JSON serialization works."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _sanitize_row(row: dict) -> dict:
    """Sanitize all values in a dict for JSON-safe Supabase insert."""
    return {k: _sanitize(v) for k, v in row.items()}


def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
            "in .env.local or environment variables."
        )
    logger.debug("Connecting to Supabase at %s", SUPABASE_URL)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def store_backtest_result(result: dict) -> str:
    """Insert one row into bt_strategy_results, return the id."""
    sb = get_supabase()
    row = _sanitize_row({
        "strategy_name":    result["strategy_name"],
        "ticker":           result["ticker"],
        "date_from":        result["date_from"],
        "date_to":          result["date_to"],
        "init_cash":        result.get("init_cash", 100000),
        "fees_pct":         result.get("fees_pct", 0.001),
        "slippage_pct":     result.get("slippage_pct", 0.002),
        "strategy_config":  result.get("strategy_config", {}),
        "total_return_pct": result.get("total_return_pct"),
        "cagr_pct":         result.get("cagr_pct"),
        "sharpe_ratio":     result.get("sharpe_ratio"),
        "sortino_ratio":    result.get("sortino_ratio"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "calmar_ratio":     result.get("calmar_ratio"),
        "win_rate_pct":     result.get("win_rate_pct"),
        "profit_factor":    result.get("profit_factor"),
        "total_trades":     result.get("total_trades"),
        "winning_trades":   result.get("winning_trades"),
        "losing_trades":    result.get("losing_trades"),
        "avg_win_pct":      result.get("avg_win_pct"),
        "avg_loss_pct":     result.get("avg_loss_pct"),
        "avg_holding_days": result.get("avg_holding_days"),
        "final_value":      result.get("final_value"),
        "buy_precision":    result.get("buy_precision"),
        "sell_precision":   result.get("sell_precision"),
        "directional_acc":  result.get("directional_acc"),
        "ic":               result.get("ic"),
        "rank_ic":          result.get("rank_ic"),
        "ml_horizon":       result.get("ml_horizon"),
        "ml_train_rows":    result.get("ml_train_rows"),
        "ml_feature_count": result.get("ml_feature_count"),
        "ml_retrain_date":  result.get("ml_retrain_date"),
        "expectancy_pct":   result.get("expectancy_pct"),
        "risk_reward":      result.get("risk_reward"),
        "diagnostic":       result.get("diagnostic"),
        "run_by":           result.get("run_by", "manual"),
    })
    try:
        resp = _insert_row(sb, row)
    except Exception as exc:
        if not _is_missing_column_error(exc):
            raise
        slim = {k: v for k, v in row.items() if k not in _OPTIONAL_RESULT_COLUMNS}
        logger.warning(
            "Optional bt_strategy_results columns missing (apply migrations 008/009). "
            "Retrying insert without ic/ml_* fields."
        )
        resp = _insert_row(sb, slim)
    logger.info(
        "Stored backtest result for %s/%s (id=%s)",
        result.get("ticker"), result.get("strategy_name"), resp.data[0]["id"],
    )
    return resp.data[0]["id"]


def store_trade_logs(result_id: str, trades: list[dict]):
    """Bulk insert trade logs for one backtest run."""
    if not trades:
        return
    sb = get_supabase()
    rows = []
    for t in trades:
        rows.append(_sanitize_row({
            "strategy_result_id": result_id,
            "ticker":             t["ticker"],
            "strategy_name":      t["strategy_name"],
            "entry_date":         t["entry_date"],
            "exit_date":          t["exit_date"],
            "entry_price":        t["entry_price"],
            "exit_price":         t["exit_price"],
            "direction":          t.get("direction", "long"),
            "size":               t.get("size"),
            "pnl":                t.get("pnl"),
            "return_pct":         t.get("return_pct"),
            "is_win":             t.get("is_win"),
            "entry_reason":       t.get("entry_reason"),
            "exit_reason":        t.get("exit_reason"),
            "entry_indicator_value": t.get("entry_indicator_value"),
            "exit_indicator_value":  t.get("exit_indicator_value"),
        }))
    for i in range(0, len(rows), 100):
        sb.table("bt_trade_log").insert(rows[i : i + 100]).execute()
    logger.info("Stored %d trade logs for result %s", len(rows), result_id)
