"""
Local file cache for market data.

Two caches:
  1. Price downloads (yfinance) -> pickled DataFrames under cache/prices/.
     Windows whose end date is in the past are immutable and cached forever;
     windows that include today expire at the end of the day they were fetched.
  2. Ticker -> sector map -> cache/sectors.json (sectors effectively never
     change, so entries are permanent).

This makes repeated backtests near-instant, avoids hammering Yahoo, and lets
the app keep working from cache if the data source hiccups.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_ROOT = Path("cache")
PRICE_DIR = CACHE_ROOT / "prices"
SECTOR_FILE = CACHE_ROOT / "sectors.json"


def _price_key(tickers, start, end, kwargs) -> str:
    if isinstance(tickers, str):
        tickers = [tickers]
    payload = json.dumps(
        [sorted(str(t).upper() for t in tickers), str(start), str(end), sorted(kwargs.items())],
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def _window_is_closed(end) -> bool:
    """True if the requested window ends before today (data can't change)."""
    if end is None:
        return False
    try:
        end_d = pd.to_datetime(end).date()
    except Exception:
        return False
    return end_d < date.today()


def cached_download(tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
    """
    Drop-in wrapper around yf.download with on-disk caching.

    Only start/end windows are cached (period-based calls pass through, since
    'period="5d"' is relative to now).
    """
    if start is None or end is None:
        return yf.download(tickers, start=start, end=end, **kwargs)

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICE_DIR / f"{_price_key(tickers, start, end, kwargs)}.pkl"

    if path.exists():
        fresh = _window_is_closed(end) or (
            datetime.fromtimestamp(path.stat().st_mtime).date() == date.today()
        )
        if fresh:
            try:
                return pd.read_pickle(path)
            except Exception:
                pass  # corrupt cache entry; refetch

    df = yf.download(tickers, start=start, end=end, **kwargs)
    if df is not None and not df.empty:
        try:
            df.to_pickle(path)
        except Exception as e:
            warnings.warn(f"Could not write price cache: {e}")
    return df


def _load_sector_cache() -> dict:
    if SECTOR_FILE.exists():
        try:
            return json.loads(SECTOR_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_sector_cache(cache: dict) -> None:
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        SECTOR_FILE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
    except Exception as e:
        warnings.warn(f"Could not write sector cache: {e}")


SPLITS_FILE = CACHE_ROOT / "splits.json"


def cached_splits(tickers) -> dict:
    """
    Return {ticker: [[iso_date, ratio], ...]} of historical stock splits.
    Refreshed at most once per day per ticker (splits are rare events).
    """
    try:
        cache = json.loads(SPLITS_FILE.read_text(encoding="utf-8")) if SPLITS_FILE.exists() else {}
    except Exception:
        cache = {}

    today = date.today().isoformat()
    out, dirty = {}, False
    for t in tickers:
        t = str(t).upper()
        entry = cache.get(t)
        if entry and entry.get("as_of") == today:
            out[t] = entry["events"]
            continue
        events = []
        try:
            s = yf.Ticker(t).splits
            if s is not None and len(s):
                events = [[idx.strftime("%Y-%m-%d"), float(r)] for idx, r in s.items() if r and r > 0]
        except Exception:
            if entry:  # keep stale data over nothing
                out[t] = entry["events"]
                continue
        cache[t] = {"as_of": today, "events": events}
        out[t] = events
        dirty = True

    if dirty:
        try:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            SPLITS_FILE.write_text(json.dumps(cache), encoding="utf-8")
        except Exception as e:
            warnings.warn(f"Could not write splits cache: {e}")
    return out


def split_factor(events, since_iso: str) -> float:
    """
    Cumulative split multiple from `since_iso` (exclusive) to now.
    E.g. a 10-for-1 split after purchase returns 10.0: the buyer now holds
    10x the shares at 1/10th the per-share cost basis.
    """
    factor = 1.0
    for d, ratio in events or []:
        if d > since_iso and ratio and ratio > 0:
            factor *= float(ratio)
    return factor


def cached_sectors(tickers) -> dict:
    """
    Return {ticker: raw_sector_or_'Unknown'} using the on-disk cache; only
    tickers not seen before hit the network. 'Unknown' results are not cached,
    so transient failures get retried next time.
    """
    cache = _load_sector_cache()
    out, dirty = {}, False
    for t in tickers:
        t = str(t).upper()
        if t in cache:
            out[t] = cache[t]
            continue
        raw = "Unknown"
        try:
            info = yf.Ticker(t).get_info()
            raw = info.get("sector") or "Unknown"
        except Exception:
            raw = "Unknown"
        out[t] = raw
        if raw != "Unknown":
            cache[t] = raw
            dirty = True
    if dirty:
        _save_sector_cache(cache)
    return out
