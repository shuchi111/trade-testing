"""Sync / load long-horizon daily OHLCV into ``market_daily_bars``."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    BACKTEST_END,
    BACKTEST_START,
    SUPABASE_KEY,
    SUPABASE_URL,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)

_CHUNK = 400
_PAGE = 1000

_AGENT = Path(__file__).resolve().parents[1]
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))


def _has_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _sb():
    if not _has_supabase():
        raise RuntimeError("NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY required")
    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _pg_conn():
    """Cron path: use DATABASE_URL / DIRECT_URL (no Supabase REST keys needed)."""
    from db_url import resolve_psycopg2_url
    import psycopg2  # type: ignore[reportMissingModuleSource]

    url = resolve_psycopg2_url()
    if not url:
        raise RuntimeError("DIRECT_URL / DATABASE_URL required")
    return psycopg2.connect(url)


def _backend() -> str:
    if _has_supabase():
        return "supabase"
    try:
        from db_url import resolve_psycopg2_url

        if resolve_psycopg2_url():
            return "psycopg2"
    except Exception:
        pass
    raise RuntimeError(
        "Set NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY, or DATABASE_URL / DIRECT_URL"
    )


def download_yf_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV via yfinance (preferred for 15y history).

    Parameters
    ----------
    ticker : str
        Yahoo-compatible symbol (e.g. ``TCS.NS``, ``BTC-USD``).
    start, end : str
        Inclusive ISO dates (``YYYY-MM-DD``).

    Returns
    -------
    pd.DataFrame
        Columns Open/High/Low/Close/Volume indexed by naive timestamps.
        Empty DataFrame on download failure.
    """
    try:
        import yfinance as yf  # type: ignore[reportMissingImports]
    except ImportError:
        logger.error("yfinance not installed")
        return pd.DataFrame()

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
    except Exception as exc:
        logger.error("yfinance history failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if hist is None or hist.empty:
        return pd.DataFrame()

    out = hist.copy()
    out.columns = [str(c).title() for c in out.columns]
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(out.columns)):
        return pd.DataFrame()
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["Close"])
    return out[["Open", "High", "Low", "Close", "Volume"]]


def upsert_ohlcv(ticker: str, ohlcv: pd.DataFrame, *, source: str = "yfinance") -> int:
    """Upsert OHLCV rows into ``market_daily_bars``.

    Parameters
    ----------
    ticker : str
        Instrument symbol (stored uppercased).
    ohlcv : pd.DataFrame
        Must include Close; Open/High/Low/Volume optional.
    source : str
        Provenance label written on each row (default ``yfinance``).

    Returns
    -------
    int
        Number of rows successfully upserted (chunk successes only).
    """
    if ohlcv is None or ohlcv.empty:
        return 0
    ticker_u = ticker.strip().upper()
    synced_at = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for ts, row in ohlcv.iterrows():
        try:
            d = pd.Timestamp(ts).date().isoformat()
            rows.append(
                {
                    "ticker": ticker_u,
                    "trade_date": d,
                    "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                    "high": float(row["High"]) if pd.notna(row["High"]) else None,
                    "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                    "source": source,
                    "synced_at": synced_at,
                }
            )
        except Exception:
            continue

    if not rows:
        return 0

    backend = _backend()
    if backend == "psycopg2":
        written = 0
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                for i in range(0, len(rows), _CHUNK):
                    chunk = rows[i : i + _CHUNK]
                    args = [
                        (
                            r["ticker"],
                            r["trade_date"],
                            r["open"],
                            r["high"],
                            r["low"],
                            r["close"],
                            r["volume"],
                            r["source"],
                            r["synced_at"],
                        )
                        for r in chunk
                    ]
                    cur.executemany(
                        """
                        INSERT INTO market_daily_bars
                          (ticker, trade_date, open, high, low, close, volume, source, synced_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (ticker, trade_date) DO UPDATE SET
                          open = EXCLUDED.open,
                          high = EXCLUDED.high,
                          low = EXCLUDED.low,
                          close = EXCLUDED.close,
                          volume = EXCLUDED.volume,
                          source = EXCLUDED.source,
                          synced_at = EXCLUDED.synced_at
                        """,
                        args,
                    )
                    written += len(chunk)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("psycopg2 upsert failed for %s: %s", ticker_u, exc)
            return 0
        finally:
            conn.close()
        return written

    sb = _sb()
    written = 0
    payload = [
        {**r, "synced_at": r["synced_at"].isoformat().replace("+00:00", "Z")}
        for r in rows
    ]
    for i in range(0, len(payload), _CHUNK):
        chunk = payload[i : i + _CHUNK]
        try:
            sb.table("market_daily_bars").upsert(chunk, on_conflict="ticker,trade_date").execute()
            written += len(chunk)
        except Exception as exc:
            logger.warning(
                "upsert chunk failed for %s (%s–%s): %s", ticker_u, i, i + len(chunk), exc
            )
    return written


def load_ohlcv_from_db(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load daily bars from ``market_daily_bars`` for ``[start, end]``.

    Parameters
    ----------
    ticker, start, end : str
        Symbol and inclusive ISO date bounds.

    Returns
    -------
    pd.DataFrame
        OHLCV with DatetimeIndex; empty if missing credentials or no rows.
        Paginated in ``_PAGE``-sized reads for large histories.
    """
    ticker_u = ticker.strip().upper()
    try:
        backend = _backend()
    except RuntimeError:
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []
    if backend == "psycopg2":
        try:
            conn = _pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT trade_date, open, high, low, close, volume
                        FROM market_daily_bars
                        WHERE ticker = %s AND trade_date >= %s AND trade_date <= %s
                        ORDER BY trade_date ASC
                        """,
                        (ticker_u, start, end),
                    )
                    for row in cur.fetchall():
                        all_rows.append(
                            {
                                "trade_date": row[0],
                                "open": row[1],
                                "high": row[2],
                                "low": row[3],
                                "close": row[4],
                                "volume": row[5],
                            }
                        )
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("market_daily_bars read failed for %s: %s", ticker_u, exc)
            return pd.DataFrame()
    else:
        sb = _sb()
        offset = 0
        try:
            while True:
                resp = (
                    sb.table("market_daily_bars")
                    .select("trade_date,open,high,low,close,volume")
                    .eq("ticker", ticker_u)
                    .gte("trade_date", start)
                    .lte("trade_date", end)
                    .order("trade_date", desc=False)
                    .range(offset, offset + _PAGE - 1)
                    .execute()
                )
                batch = resp.data or []
                all_rows.extend(batch)
                if len(batch) < _PAGE:
                    break
                offset += _PAGE
        except Exception as exc:
            logger.warning("market_daily_bars read failed for %s: %s", ticker_u, exc)
            return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Close"])


def coverage_summary(ticker: str) -> dict[str, Any]:
    """Return min/max date + bar count for UI (no full OHLCV scan).

    Parameters
    ----------
    ticker : str
        Instrument symbol.

    Returns
    -------
    dict
        Keys: ``ticker``, ``bars``, ``date_from``, ``date_to``, ``years``, ``source``.
        Zeroed defaults when the cache is empty or unreachable.
    """
    empty = {
        "ticker": ticker.strip().upper(),
        "bars": 0,
        "date_from": None,
        "date_to": None,
        "years": 0.0,
        "source": None,
    }
    ticker_u = ticker.strip().upper()
    try:
        backend = _backend()
    except RuntimeError:
        return empty

    if backend == "psycopg2":
        try:
            conn = _pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*), MIN(trade_date), MAX(trade_date),
                               (ARRAY_AGG(source ORDER BY trade_date DESC))[1]
                        FROM market_daily_bars WHERE ticker = %s
                        """,
                        (ticker_u,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("coverage_summary failed for %s: %s", ticker_u, exc)
            return empty
        bars = int(row[0] or 0) if row else 0
        if bars == 0:
            return empty
        d0, d1 = row[1], row[2]
        years = round((d1 - d0).days / 365.25, 2) if d0 and d1 else 0.0
        return {
            "ticker": ticker_u,
            "bars": bars,
            "date_from": d0.isoformat() if d0 else None,
            "date_to": d1.isoformat() if d1 else None,
            "years": years,
            "source": row[3],
        }

    sb = _sb()
    try:
        count_resp = (
            sb.table("market_daily_bars")
            .select("trade_date", count="exact")
            .eq("ticker", ticker_u)
            .limit(1)
            .execute()
        )
        bars = int(count_resp.count or 0)
        if bars == 0:
            return empty

        first = (
            sb.table("market_daily_bars")
            .select("trade_date,source")
            .eq("ticker", ticker_u)
            .order("trade_date", desc=False)
            .limit(1)
            .execute()
        )
        last = (
            sb.table("market_daily_bars")
            .select("trade_date,source")
            .eq("ticker", ticker_u)
            .order("trade_date", desc=True)
            .limit(1)
            .execute()
        )
        d0 = (first.data or [{}])[0].get("trade_date")
        d1 = (last.data or [{}])[0].get("trade_date")
        source = (last.data or [{}])[0].get("source") or "yfinance"
        if not d0 or not d1:
            return empty
        years = max(0.0, (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25)
        return {
            "ticker": ticker_u,
            "bars": bars,
            "date_from": str(d0)[:10],
            "date_to": str(d1)[:10],
            "years": round(years, 1),
            "source": source,
        }
    except Exception as exc:
        logger.warning("coverage query failed for %s: %s", ticker_u, exc)
        return empty


def sync_ticker_history(
    ticker: str,
    *,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
) -> dict[str, Any]:
    """Download from Yahoo and upsert into market_daily_bars."""
    ohlcv = download_yf_ohlcv(ticker, start, end)
    written = upsert_ohlcv(ticker, ohlcv)
    cov = coverage_summary(ticker)
    return {
        "ok": written > 0 or cov["bars"] > 0,
        "ticker": ticker.strip().upper(),
        "downloaded_rows": 0 if ohlcv is None else len(ohlcv),
        "upserted_rows": written,
        "coverage": cov,
        "requested_start": start,
        "requested_end": end,
        "timeframe": TIMEFRAME,
    }


def ensure_history(
    ticker: str,
    *,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    min_bars: int = 200,
) -> pd.DataFrame:
    """Prefer DB cache; sync from Yahoo when thin/missing, then reload.

    Parameters
    ----------
    ticker : str
        Instrument symbol.
    start, end : str
        Requested history window (ISO dates).
    min_bars : int
        Minimum cached rows before treating coverage as sufficient.

    Returns
    -------
    pd.DataFrame
        OHLCV indexed by date (from DB after sync, or live Yahoo as last resort).
    """
    cached = load_ohlcv_from_db(ticker, start, end)
    need_sync = cached.empty or len(cached) < min_bars
    if not cached.empty and not need_sync:
        first = cached.index.min()
        if pd.Timestamp(first) > pd.Timestamp(start) + timedelta(days=370):
            need_sync = True

    if need_sync:
        logger.info("Syncing %s daily bars from Yahoo (%s → %s)", ticker, start, end)
        sync_ticker_history(ticker, start=start, end=end)
        cached = load_ohlcv_from_db(ticker, start, end)

    if cached.empty:
        live = download_yf_ohlcv(ticker, start, end)
        if not live.empty:
            try:
                upsert_ohlcv(ticker, live)
            except Exception as exc:
                logger.warning("could not persist live bars for %s: %s", ticker, exc)
        return live

    return cached
