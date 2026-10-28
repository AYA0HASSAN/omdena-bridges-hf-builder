"""All network access goes through here: retries with back-off, Overpass mirrors, and a swappable
`requests` (tests replace it with fakes)."""
import time

import requests as _requests

requests = _requests
UA = {"User-Agent": "fair-bridges-hf-builder/1.0 (HOT fAIr bridges dataset)"}
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
RETRY_STATUS = {429, 500, 502, 503, 504}


def _call(method, url, retries=3, backoff=5, **kw):
    kw.setdefault("headers", UA)
    kw.setdefault("timeout", 120)
    last = None
    for attempt in range(retries + 1):
        try:
            r = getattr(requests, method)(url, **kw)
            status = getattr(r, "status_code", 200)
            if status in RETRY_STATUS:
                raise IOError(f"HTTP {status}")
            r.raise_for_status()
            return r
        except Exception as e:          # network errors, timeouts, 5xx
            last = e
            if attempt < retries:
                wait = backoff * 2 ** attempt
                print(f"  [net] {method.upper()} {url[:70]} failed ({type(e).__name__}: {str(e)[:80]}), retry {attempt + 1}/{retries} in {wait}s")
                time.sleep(wait)
    raise last


def get(url, **kw):
    return _call("get", url, **kw)


def post(url, **kw):
    return _call("post", url, **kw)


def overpass(query, timeout=180, retries=2):
    """POST an Overpass query, trying each mirror in turn."""
    last = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            return post(ep, data={"data": query}, timeout=timeout + 60, retries=retries)
        except Exception as e:
            last = e
            print(f"  [net] Overpass mirror failed: {ep}")
    raise last
