"""Unit tests for Reddit data fetcher (proxy, JSON session, graceful degradation)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows import reddit as reddit_mod


def test_subreddits_for_ticker_us():
    assert reddit_mod.subreddits_for_ticker("AAPL") == reddit_mod.DEFAULT_SUBREDDITS_US


def test_subreddits_for_ticker_indian():
    assert reddit_mod.subreddits_for_ticker("TCS.NS") == reddit_mod.DEFAULT_SUBREDDITS_IN


def test_proxy_label_strips_credentials():
    url = "http://user:secret@proxy.example.com:8080"
    assert reddit_mod._proxy_label(url) == "proxy.example.com:8080"
    assert "user" not in reddit_mod._proxy_label(url)
    assert "secret" not in reddit_mod._proxy_label(url)


def test_proxy_label_direct_when_unset(monkeypatch):
    monkeypatch.delenv("REDDIT_HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    assert reddit_mod._proxy_label() == "direct"


def test_proxy_label_without_scheme():
    assert reddit_mod._proxy_label("proxy.example.com:8080") == "proxy.example.com:8080"


def test_fetch_mode_from_env(monkeypatch):
    monkeypatch.setenv("REDDIT_FETCH_MODE", "rss")
    assert reddit_mod._fetch_mode() == "rss"
    monkeypatch.setenv("REDDIT_FETCH_MODE", "json")
    assert reddit_mod._fetch_mode() == "json"


def test_build_opener_uses_proxy_handler(monkeypatch):
    monkeypatch.setenv("REDDIT_HTTP_PROXY", "http://user:pass@127.0.0.1:8888")
    opener = reddit_mod._build_opener()
    assert any(isinstance(h, ProxyHandler) for h in opener.handlers)


def test_build_opener_direct_without_proxy(monkeypatch):
    monkeypatch.delenv("REDDIT_HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    opener = reddit_mod._build_opener()
    assert not any(isinstance(h, ProxyHandler) for h in opener.handlers)


def test_fetch_json_403_sets_waf_blocked_no_rss(monkeypatch):
    monkeypatch.setenv("REDDIT_RSS_FALLBACK", "0")
    session = reddit_mod._RedditSession(timeout=5.0)

    def _raise_403(_req):
        raise HTTPError("https://old.reddit.com/x", 403, "Blocked", hdrs=None, fp=None)

    with patch.object(session, "_open", side_effect=_raise_403):
        with patch.object(reddit_mod, "_fetch_subreddit_rss") as rss_mock:
            posts, err = session.fetch_json("AAPL", "stocks", 3, prime_html=False)
            assert posts == []
            assert err == "403"
            assert session.waf_blocked is True
            rss_mock.assert_not_called()


def test_fetch_json_429_retries_once(monkeypatch):
    monkeypatch.setenv("REDDIT_RSS_FALLBACK", "0")
    session = reddit_mod._RedditSession(timeout=5.0)
    calls = {"n": 0}

    def _open_side_effect(_req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError("https://old.reddit.com/x", 429, "Too Many", hdrs=None, fp=None)
        payload = {"data": {"children": [{"data": {"title": "t", "score": 1, "num_comments": 2}}]}}
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch.object(session, "_open", side_effect=_open_side_effect):
        with patch.object(reddit_mod.time, "sleep"):
            posts, err = session.fetch_json("AAPL", "stocks", 3, prime_html=False)
            assert err is None
            assert len(posts) == 1
            assert posts[0]["score"] == 1
            assert calls["n"] == 2


def test_fetch_reddit_posts_uncached_waf_blocked_message(monkeypatch):
    monkeypatch.setenv("REDDIT_RSS_FALLBACK", "0")
    session = reddit_mod._RedditSession(timeout=5.0)
    session.waf_blocked = True

    with patch.object(reddit_mod, "_RedditSession", return_value=session):
        with patch.object(reddit_mod, "_fetch_subreddit", return_value=[]):
            out = reddit_mod._fetch_reddit_posts_uncached(
                "RELIANCE.NS",
                reddit_mod.DEFAULT_SUBREDDITS_IN,
                inter_request_delay=0,
            )
            assert "REDDIT_HTTP_PROXY" in out
            assert "blocked" in out.lower()


def test_fetch_reddit_posts_uncached_no_posts_placeholder():
    with patch.object(reddit_mod, "_fetch_subreddit", return_value=[]):
        out = reddit_mod._fetch_reddit_posts_uncached(
            "ZZZZ",
            ("wallstreetbets",),
            inter_request_delay=0,
        )
        assert "<no Reddit posts found" in out
        assert "ZZZZ" in out


def test_fetch_reddit_posts_formats_json_scores():
    posts = [{
        "title": "Test post",
        "score": 42,
        "num_comments": 7,
        "created_utc": 1_700_000_000.0,
        "selftext": "body",
        "source": "json",
    }]
    with patch.object(reddit_mod, "_fetch_subreddit", return_value=posts):
        out = reddit_mod._fetch_reddit_posts_uncached(
            "AAPL",
            ("stocks",),
            inter_request_delay=0,
        )
        assert "42" in out
        assert "Test post" in out
        assert "via RSS feed" not in out


def test_fetch_reddit_posts_uses_cache_wrapper():
    with patch.object(reddit_mod, "cached_fetch", return_value="cached-block") as cache_mock:
        result = reddit_mod.fetch_reddit_posts("AAPL")
        assert result == "cached-block"
        cache_mock.assert_called_once()
