"""Resolve a Postgres URL that psycopg2 can use (not the Supabase pooler DSN)."""
from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse


def resolve_psycopg2_url() -> str:
    """
    psycopg2 cannot connect with ``?pgbouncer=true`` (Supabase pooler URL).

    Prefer DIRECT_URL (port 5432). Fall back to DATABASE_URL with pooler
    query params stripped and port 6543 → 5432.
    """
    direct = os.getenv("DIRECT_URL", "").strip()
    if direct:
        return _sanitize(direct)

    pooled = os.getenv("DATABASE_URL", "").strip()
    if pooled:
        return _sanitize(pooled)

    return ""


def _sanitize(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc
    if ":6543" in netloc:
        netloc = netloc.replace(":6543", ":5432")
    # Drop query string — pgbouncer=true breaks psycopg2
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
