"""Reddit search fetcher for ticker-specific discussion posts.

Default path is ``old.reddit.com`` search JSON: prime a cookie session with the
HTML search page, then fetch ``/search.json`` for full post data (score, comment
counts, selftext). ``www.reddit.com/search.json`` is WAF-blocked for bare
clients; RSS (``search.rss``) is kept as fallback when the JSON path fails.
On a 429 we back off once (honouring ``Retry-After``).

No API key required. Returns formatted plaintext blocks ready for prompt
injection and degrades gracefully — returns a placeholder string rather than
raising, so callers never special-case missing data.
"""

from __future__ import annotations

import html
import http.client
import http.cookiejar
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from .fetch_cache import cached_fetch
from .symbol_utils import indian_equity_base, reddit_search_term

logger = logging.getLogger(__name__)

_OLD = "https://old.reddit.com"
_OLD_JSON = _OLD + "/r/{sub}/search.json?{qs}"
_OLD_HTML = _OLD + "/r/{sub}/search/?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# Browser-like UA for old.reddit session priming + JSON; RSS keeps the identified
# tradingagents token which Reddit still serves reliably.
_SESSION_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_RSS_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS_US = ("wallstreetbets", "stocks", "investing")
DEFAULT_SUBREDDITS_IN = ("IndiaInvestments", "IndianStreetBets", "dalalstreet")
DEFAULT_SUBREDDITS = DEFAULT_SUBREDDITS_US


def subreddits_for_ticker(ticker: str) -> tuple[str, ...]:
    """Pick finance subreddits that match the instrument's primary market."""
    return DEFAULT_SUBREDDITS_IN if indian_equity_base(ticker) else DEFAULT_SUBREDDITS_US


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 30s."""
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 30.0) if val else None
    except (ValueError, TypeError, AttributeError):
        return None


def _normalize_json_post(raw: dict) -> dict:
    return {
        "title": (raw.get("title") or "").strip(),
        "score": raw.get("score"),
        "num_comments": raw.get("num_comments"),
        "created_utc": raw.get("created_utc"),
        "selftext": (raw.get("selftext") or "").strip(),
        "source": "json",
    }


def _fetch_subreddit_old_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Fetch search results via old.reddit.com session + JSON.

    Visits the HTML search page first to obtain cookies, then requests
    ``search.json`` with a Referer — the same flow that works in a browser
    without OAuth. Carries score and comment counts.
    """
    qs = _search_qs(ticker, limit)
    html_url = _OLD_HTML.format(sub=sub, qs=qs)
    json_url = _OLD_JSON.format(sub=sub, qs=qs)
    base_headers = {
        "User-Agent": _SESSION_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    try:
        opener.open(Request(html_url, headers=base_headers), timeout=timeout)
        req = Request(
            json_url,
            headers={
                **base_headers,
                "Accept": "application/json",
                "Referer": html_url,
            },
        )
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        posts = [
            _normalize_json_post(c.get("data", {}))
            for c in children
            if isinstance(c, dict)
        ]
        return posts[:limit]
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit JSON 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_old_json(ticker, sub, limit, timeout, _retry=False)
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return []
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return []


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Fallback: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _RSS_UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fetch one subreddit: old.reddit JSON first, RSS on failure."""
    posts = _fetch_subreddit_old_json(ticker, sub, limit, timeout)
    if posts:
        return posts
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_reddit_posts_uncached(
    ticker: str,
    subreddits: Iterable[str],
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 2.5,
) -> str:
    """Fetch recent Reddit posts mentioning ``search_term`` across subreddits."""
    search_term = reddit_search_term(ticker)
    blocks = []
    total_posts = 0
    subs = tuple(subreddits)
    for i, sub in enumerate(subs):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(search_term, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(
                f"r/{sub}: <no posts found mentioning {search_term.upper()} in the past 7 days>"
            )
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {search_term.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {search_term.upper()} across "
            f"{', '.join(f'r/{s}' for s in subs)} in the past 7 days>"
        )
    return "\n\n".join(blocks)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] | None = None,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float | None = None,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    Uses old.reddit.com session + JSON by default (full score/comment data).
    ``inter_request_delay`` paces per-subreddit requests (2 HTTP calls each on
    the JSON path) to stay under Reddit's public per-IP rate limit.
    """
    subs = tuple(subreddits) if subreddits is not None else subreddits_for_ticker(ticker)
    delay = inter_request_delay
    if delay is None:
        delay = float(os.getenv("REDDIT_INTER_REQUEST_DELAY_SEC", "2.5"))
    return cached_fetch(
        ("reddit", reddit_search_term(ticker).upper(), subs, limit_per_sub),
        lambda: _fetch_reddit_posts_uncached(
            ticker,
            subs,
            limit_per_sub=limit_per_sub,
            timeout=timeout,
            inter_request_delay=delay,
        ),
        ttl_sec=float(os.getenv("SOCIAL_FETCH_CACHE_TTL_SEC", "300")),
    )
