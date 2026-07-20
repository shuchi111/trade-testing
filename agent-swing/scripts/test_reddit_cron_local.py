"""Local cron smoke test for Reddit fetch (old.reddit session + JSON).

Run from agent-swing:
    python scripts/test_reddit_cron_local.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.dataflows.reddit import (  # noqa: E402
    _fetch_reddit_posts_uncached,
    subreddits_for_ticker,
)
from tradingagents.dataflows.symbol_utils import reddit_search_term  # noqa: E402

_SCORE_RE = re.compile(r"\d+\s*(?:↑|up)")


def _safe_print(text: str) -> None:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.buffer.write(text.encode(enc, errors="replace") + b"\n")


def _run_ticker(ticker: str) -> tuple[bool, int, bool]:
    subs = subreddits_for_ticker(ticker)
    term = reddit_search_term(ticker).upper()
    print(f"\n{'=' * 60}")
    print(f"fetch_reddit_posts path — {ticker} (search={term})")
    print(f"Subreddits: {', '.join(subs)}")
    print("=" * 60)
    out = _fetch_reddit_posts_uncached(ticker, subs, inter_request_delay=2.5)
    _safe_print(out[:2500])
    if len(out) > 2500:
        _safe_print("... [truncated]")
    has_posts = "<no Reddit posts found" not in out
    has_scores = bool(_SCORE_RE.search(out))
    via_rss = "via RSS feed" in out
    return has_posts, len(out), has_scores and not via_rss


def main() -> int:
    cases = [("AAPL", True), ("TCS.NS", True)]
    all_ok = True
    print("Reddit cron smoke test (production fetcher, no cache bypass on public API)")

    for ticker, expect_posts in cases:
        has_posts, length, has_scores = _run_ticker(ticker)
        ok = has_posts == expect_posts and (has_scores if expect_posts else True)
        status = "PASS" if ok else "FAIL"
        print(
            f"\n{ticker}: {status} | chars={length} | "
            f"has_posts={has_posts} | has_json_scores={has_scores}"
        )
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("VERDICT: production Reddit fetcher OK for cron.")
    else:
        print("VERDICT: one or more tickers failed — check rate limits or routing.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
