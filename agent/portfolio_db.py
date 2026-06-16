"""Paper wallet helpers for AI recommendation + execution pipelines."""
from __future__ import annotations

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"


def load_holding(conn, ticker: str) -> tuple[float, float]:
    """Return (quantity, avg_entry) for ticker in admin wallet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT quantity, avg_entry
            FROM portfolio_holdings
            WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
            """,
            (ADMIN_WALLET_ID, ticker),
        )
        row = cur.fetchone()
    if not row:
        return 0.0, 0.0
    return float(row[0] or 0), float(row[1] or 0)


def load_wallet_cash(conn) -> float:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_cash FROM wallet_accounts WHERE id = %s",
            (ADMIN_WALLET_ID,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else 0.0


def count_open_positions(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM portfolio_holdings WHERE wallet_id = %s AND quantity > 0",
            (ADMIN_WALLET_ID,),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def portfolio_value(conn, prices: dict[str, float]) -> float:
    """Cash + mark-to-market of holdings."""
    cash = load_wallet_cash(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, quantity, avg_entry FROM portfolio_holdings WHERE wallet_id = %s",
            (ADMIN_WALLET_ID,),
        )
        rows = cur.fetchall()
    holdings_val = 0.0
    for ticker, qty, avg_entry in rows:
        price = prices.get(str(ticker).upper()) or float(avg_entry or 0)
        holdings_val += float(qty) * price
    return cash + holdings_val


def execute_trade(conn, *, ticker: str, action: str, quantity: float, price: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT execute_wallet_trade(%s::uuid, %s, %s, %s, %s)",
            (ADMIN_WALLET_ID, ticker.upper(), action, quantity, price),
        )
    conn.commit()
