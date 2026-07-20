"""Reddit search fetcher for ticker-specific discussion posts.

Default path is ``old.reddit.com`` search JSON: prime a shared cookie session,
then fetch ``/search.json`` for full post data (score, comment counts, selftext).
RSS (``search.rss``) is only used as fallback for transient failures — **not**
after WAF ``403`` blocks, which are common from datacenter/CI IPs and would
only double request volume and trigger ``429`` rate limits.

Set ``REDDIT_FETCH_MODE=rss`` in GitHub Actions when JSON is consistently blocked.
On a ``429`` we back off once (honouring ``Retry-After``).

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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
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

FetchMode = Literal["json", "rss"]
FetchError = Literal["403", "429", "other"]

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS_US = ("wallstreetbets", "stocks", "investing")
DEFAULT_SUBREDDITS_IN = ("IndiaInvestments", "IndianStreetBets", "dalalstreet")
DEFAULT_SUBREDDITS = DEFAULT_SUBREDDITS_US


def subreddits_for_ticker(ticker: str) -> tuple[str, ...]:
    """Pick finance subreddits that match the instrument's primary market."""
    return DEFAULT_SUBREDDITS_IN if indian_equity_base(ticker) else DEFAULT_SUBREDDITS_US


def _fetch_mode() -> FetchMode:
    mode = os.getenv("REDDIT_FETCH_MODE", "json").strip().lower()
    return "rss" if mode == "rss" else "json"


def _rss_fallback_enabled() -> bool:
    return os.getenv("REDDIT_RSS_FALLBACK", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


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


@dataclass
class _RedditSession:
    """Shared cookie session for one ticker fetch (reused across subreddits)."""

    timeout: float
    jar: http.cookiejar.CookieJar = field(default_factory=http.cookiejar.CookieJar)
    primed: bool = False
    waf_blocked: bool = False
    rate_limited: bool = False

    def __post_init__(self) -> None:
        self._opener = build_opener(HTTPCookieProcessor(self.jar))

    def _base_headers(self) -> dict[str, str]:
        return {
            "User-Agent": _SESSION_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "DNT": "1",
        }

    def _open(self, req: Request) -> http.client.HTTPResponse:
        return self._opener.open(req, timeout=self.timeout)

    def prime(self) -> None:
        if self.primed:
            return
        try:
            self._open(Request(f"{_OLD}/", headers=self._base_headers()))
            self.primed = True
        except (OSError, http.client.HTTPException) as exc:
            logger.debug("Reddit session prime failed (continuing): %s", exc)

    def fetch_json(
        self,
        ticker: str,
        sub: str,
        limit: int,
        *,
        prime_html: bool = True,
        _retry: bool = True,
    ) -> tuple[list[dict], FetchError | None]:
        if self.waf_blocked:
            return [], "403"

        self.prime()
        qs = _search_qs(ticker, limit)
        html_url = _OLD_HTML.format(sub=sub, qs=qs)
        json_url = _OLD_JSON.format(sub=sub, qs=qs)
        base_headers = self._base_headers()
        try:
            if prime_html:
                self._open(Request(html_url, headers=base_headers))
            req = Request(
                json_url,
                headers={
                    **base_headers,
                    "Accept": "application/json",
                    "Referer": html_url,
                },
            )
            with self._open(req) as resp:
                payload = json.loads(resp.read())
            children = (payload.get("data") or {}).get("children") or []
            posts = [
                _normalize_json_post(c.get("data", {}))
                for c in children
                if isinstance(c, dict)
            ]
            return posts[:limit], None
        except HTTPError as exc:
            if exc.code == 429:
                self.rate_limited = True
                if _retry:
                    wait = _retry_after_seconds(exc) or 8.0
                    logger.warning(
                        "Reddit JSON 429 for r/%s · %s — backing off %.1fs then retrying once",
                        sub, ticker, wait,
                    )
                    time.sleep(wait)
                    return self.fetch_json(
                        ticker, sub, limit, prime_html=prime_html, _retry=False,
                    )
                return [], "429"
            if exc.code == 403:
                self.waf_blocked = True
                logger.warning(
                    "Reddit JSON WAF-blocked (403) for r/%s · %s — skipping RSS fallback "
                    "to avoid rate-limit cascade. Set REDDIT_FETCH_MODE=rss in CI if persistent.",
                    sub, ticker,
                )
                return [], "403"
            logger.warning(
                "Reddit JSON fetch failed for r/%s · %s: HTTP %s",
                sub, ticker, exc.code,
            )
            return [], "other"
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            logger.warning("Reddit JSON fetch failed for r/%s · %s: %s", sub, ticker, exc)
            return [], "other"


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Fallback: parse the public Atom search feed for a subreddit."""
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _RSS_UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 8.0
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
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


def _fetch_subreddit_json(
    session: _RedditSession,
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    *,
    html_primed: bool = False,
) -> list[dict]:
    """JSON path with shared session; RSS only on transient errors."""
    posts, err = session.fetch_json(ticker, sub, limit, prime_html=not html_primed)
    if posts or err is None:
        return posts

    if err == "403" or not _rss_fallback_enabled():
        return []

    if session.waf_blocked:
        return []

    # Transient failure — one RSS attempt after a pause (avoid hammering).
    pause = 8.0 if err == "429" or session.rate_limited else 4.0
    logger.info(
        "Reddit JSON unavailable for r/%s · %s (%s) — trying RSS after %.0fs pause",
        sub, ticker, err, pause,
    )
    time.sleep(pause)
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    session: _RedditSession | None,
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    *,
    mode: FetchMode,
    html_primed: bool = False,
) -> list[dict]:
    if mode == "rss":
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)
    assert session is not None
    return _fetch_subreddit_json(
        session, ticker, sub, limit, timeout, html_primed=html_primed,
    )


def _fetch_reddit_posts_uncached(
    ticker: str,
    subreddits: Iterable[str],
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 5.0,
) -> str:
    """Fetch recent Reddit posts mentioning ``search_term`` across subreddits."""
    search_term = reddit_search_term(ticker)
    mode = _fetch_mode()
    blocks = []
    total_posts = 0
    subs = tuple(subreddits)
    session = None if mode == "rss" else _RedditSession(timeout=timeout)
    html_primed = False

    for i, sub in enumerate(subs):
        if i > 0:
            time.sleep(inter_request_delay)
        if session is not None and session.waf_blocked:
            blocks.append(
                f"r/{sub}: <Reddit blocked requests from this IP — data unavailable>"
            )
            continue

        posts = _fetch_subreddit(
            session,
            search_term,
            sub,
            limit_per_sub,
            timeout,
            mode=mode,
            html_primed=html_primed,
        )
        if session is not None and posts and not html_primed:
            html_primed = True

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

    if total_posts == 0 and session is not None and session.waf_blocked:
        return (
            f"<Reddit blocked requests from this IP for {search_term.upper()} — "
            f"try REDDIT_FETCH_MODE=rss or Reddit OAuth. "
            f"Checked {', '.join(f'r/{s}' for s in subs)}>"
        )

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
    ``REDDIT_FETCH_MODE=rss`` forces RSS-only (1 request/sub, better for CI IPs).
    ``inter_request_delay`` paces per-subreddit requests.
    """
    subs = tuple(subreddits) if subreddits is not None else subreddits_for_ticker(ticker)
    delay = inter_request_delay
    if delay is None:
        delay = float(os.getenv("REDDIT_INTER_REQUEST_DELAY_SEC", "5.0"))
    return cached_fetch(
        ("reddit", _fetch_mode(), reddit_search_term(ticker).upper(), subs, limit_per_sub),
        lambda: _fetch_reddit_posts_uncached(
            ticker,
            subs,
            limit_per_sub=limit_per_sub,
            timeout=timeout,
            inter_request_delay=delay,
        ),
        ttl_sec=float(os.getenv("SOCIAL_FETCH_CACHE_TTL_SEC", "300")),
    )
