"""Polite, cached HTTP layer.

Every response is cached on disk keyed by URL, so a re-run against a frozen
chart makes zero network requests and is deterministic. Delete files under
data/cache/ to force a refresh.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from . import config

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})

_last_request_at: dict[str, float] = {}  # per-host politeness clock


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    return cache_dir / f"{key}.body", cache_dir / f"{key}.meta.json"


def _throttle(host: str, delay: float) -> None:
    last = _last_request_at.get(host, 0.0)
    wait = delay - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    _last_request_at[host] = time.monotonic()


def get(
    url: str,
    cache_dir: Path,
    delay: float,
    refresh: bool = False,
) -> tuple[bytes | None, dict]:
    """Fetch `url`, returning (body, meta). meta records status/error/cache hit.

    On network failure after retries, returns (None, meta) — callers decide
    whether that is an exclusion reason.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    body_path, meta_path = _cache_paths(cache_dir, url)
    if not refresh and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        body = body_path.read_bytes() if body_path.exists() else None
        meta["from_cache"] = True
        return body, meta

    host = requests.utils.urlparse(url).netloc
    meta = {"url": url, "from_cache": False}
    body = None
    for attempt in range(1, config.HTTP_RETRIES + 1):
        _throttle(host, delay)
        try:
            resp = _session.get(url, timeout=config.HTTP_TIMEOUT, allow_redirects=True)
            meta.update(
                status=resp.status_code,
                final_url=resp.url,
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                content_type=resp.headers.get("Content-Type", ""),
            )
            if resp.status_code == 200:
                body = resp.content
                break
            if resp.status_code in (403, 404, 410):
                break  # permanent-ish; don't hammer
        except requests.RequestException as exc:
            meta.update(status=None, error=f"{type(exc).__name__}: {exc}")
        if attempt < config.HTTP_RETRIES:
            time.sleep(2**attempt)

    if body is not None:
        body_path.write_bytes(body)
    meta_path.write_text(json.dumps(meta, indent=2))
    return body, meta
