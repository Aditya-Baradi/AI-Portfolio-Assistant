"""
Local file cache for market data.

All network access goes through `api.market_data`, so the provider (yfinance
for local development, or a commercially entitled Tiingo/Polygon account for
production) is a config choice and nothing downstream needs to know which one
is active.

Four caches:
  1. Price downloads -> non-executable numeric NPZ under cache/prices/. Windows whose
     end date is in the past refresh after 30 days; current windows refresh
     after 24 hours.
  2. Ticker -> sector map -> provider-namespaced cache/sectors.*.json.
  3. Splits -> cache/splits.json, refreshed at most daily per ticker.
  4. Trailing-12-month dividends -> cache/dividends.json, same cadence.

Every cache validates provider identity and price semantics before reuse.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

from api import market_data
from api.observability import report

logger = logging.getLogger("evergreen.cache")

from api.db import data_dir

CACHE_ROOT = data_dir() / "cache"
PRICE_DIR = CACHE_ROOT / "prices"
SECTOR_FILE = CACHE_ROOT / "sectors.json"
SPLITS_FILE = CACHE_ROOT / "splits.json"
DIVIDENDS_FILE = CACHE_ROOT / "dividends.json"


def _price_key(tickers, start, end, kwargs) -> str:
    if isinstance(tickers, str):
        tickers = [tickers]
    payload = json.dumps(
        [
            sorted(str(t).upper() for t in tickers),
            str(start),
            str(end),
            sorted(kwargs.items()),
            market_data.provider_name(),  # never serve one provider's data as another's
            market_data.configured_price_semantics(),
        ],
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def _window_is_closed(end) -> bool:
    """True if the requested window ends before today."""
    if end is None:
        return False
    try:
        end_d = pd.to_datetime(end).date()
    except Exception:
        return False
    return end_d < date.today()


def _cache_is_fresh(path: Path, end) -> bool:
    """
    Provider history can be corrected after publication, so even closed windows
    are refreshed periodically instead of being treated as immutable forever.
    """
    age_seconds = max(0.0, time.time() - path.stat().st_mtime)
    max_age = 30 * 24 * 3600 if _window_is_closed(end) else 24 * 3600
    return age_seconds <= max_age


def _cache_matches_provider(df, provider) -> bool:
    attrs = getattr(df, "attrs", {}) or {}
    return (
        attrs.get("source_provider") == provider.name
        and attrs.get("price_semantics") == provider.price_semantics
    )


def _requested_tickers(tickers) -> list[str]:
    if isinstance(tickers, str):
        tickers = [tickers]
    return list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))


def _mark_symbol_coverage(df, tickers) -> pd.DataFrame:
    """Attach explicit requested/present/missing symbol provenance."""
    if df is None:
        return df
    requested = _requested_tickers(tickers)
    present: set[str] = set()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                present.update(str(value).upper() for value in df.columns.get_level_values(level))
        else:
            flat = {str(value).upper() for value in df.columns}
            quote_fields = {"OPEN", "HIGH", "LOW", "CLOSE", "ADJ CLOSE", "VOLUME"}
            if len(requested) == 1 and flat.intersection(quote_fields):
                present.add(requested[0])
            else:
                present.update(flat)
    except Exception:
        present = set()
    covered = [ticker for ticker in requested if ticker in present]
    missing = [ticker for ticker in requested if ticker not in present]
    df.attrs["requested_tickers"] = requested
    df.attrs["present_tickers"] = covered
    df.attrs["missing_tickers"] = missing
    df.attrs["complete_coverage"] = not missing
    return df


def _read_price_cache(path: Path) -> pd.DataFrame:
    """
    Read the numeric NPZ cache with pickle explicitly disabled.

    The previous pandas-pickle cache could execute attacker-controlled opcodes
    if a writable data volume were tampered with.
    """
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Price cache entry exceeds the size limit.")
    with np.load(path, allow_pickle=False) as archive:
        required = {"values", "index", "metadata"}
        if not required.issubset(archive.files):
            raise ValueError("Price cache entry is incomplete.")
        values = archive["values"]
        index_raw = archive["index"]
        metadata_raw = archive["metadata"]
    metadata = json.loads(str(metadata_raw.item()))
    if values.ndim != 2 or index_raw.ndim != 1 or len(index_raw) != values.shape[0]:
        raise ValueError("Price cache dimensions are invalid.")
    if values.dtype.kind not in "fiu" or np.isinf(values).any():
        raise ValueError("Price cache contains invalid numeric data.")

    column_rows = metadata.get("columns")
    if not isinstance(column_rows, list) or len(column_rows) != values.shape[1]:
        raise ValueError("Price cache columns are invalid.")
    if metadata.get("multiindex"):
        columns = pd.MultiIndex.from_tuples(
            [tuple(str(value) for value in row) for row in column_rows],
            names=metadata.get("column_names"),
        )
    else:
        columns = pd.Index(
            [str(row[0]) for row in column_rows],
            name=(metadata.get("column_names") or [None])[0],
        )
    index = pd.to_datetime(index_raw.astype(str), utc=True).tz_localize(None)
    index.name = metadata.get("index_name")
    frame = pd.DataFrame(values.astype(float), index=index, columns=columns)
    attrs = metadata.get("attrs")
    if isinstance(attrs, dict):
        frame.attrs.update(attrs)
    return frame


def _write_price_cache(path: Path, frame: pd.DataFrame) -> None:
    """Atomically persist a numeric, non-executable NPZ cache entry."""
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Refusing to cache infinite price values.")
    multiindex = isinstance(frame.columns, pd.MultiIndex)
    if multiindex:
        column_rows = [[str(value) for value in column] for column in frame.columns]
        column_names = list(frame.columns.names)
    else:
        column_rows = [[str(value)] for value in frame.columns]
        column_names = [frame.columns.name]
    metadata = {
        "schema": 1,
        "multiindex": multiindex,
        "columns": column_rows,
        "column_names": column_names,
        "index_name": frame.index.name,
        "attrs": dict(frame.attrs),
    }
    encoded_metadata = json.dumps(metadata, default=str, sort_keys=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            np.savez_compressed(
                handle,
                values=values,
                index=np.asarray([str(value) for value in frame.index], dtype=str),
                metadata=np.asarray(encoded_metadata),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except Exception:
                pass


def _frame_is_degenerate(df, threshold: float = 0.4) -> bool:
    """
    True if a price frame is mostly empty — the signature of a flaky bulk
    download that returned data for only a couple of tickers and NaN for the
    rest. Price frames are normally dense (every row has a close for every
    column), so a non-NaN cell ratio below `threshold` means the download is
    broken and must not be cached or served. A single sparse column (e.g. a
    newly listed ticker in a large basket) barely moves the ratio, so
    legitimate data is not flagged.
    """
    try:
        if df is None or df.empty:
            return False  # emptiness is handled separately
        total = df.size
        if total == 0:
            return False
        non_na = int(df.notna().to_numpy().sum())
        return (non_na / total) < threshold
    except Exception:
        return False


def cached_download(tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
    """
    Cached price download against the configured provider.

    Only start/end windows are cached (period-based calls pass through, since
    'period="5d"' is relative to now).
    """
    provider = market_data.get_provider()

    if start is None or end is None:
        df = provider.download(tickers, start=start, end=end, **kwargs)
        df = _mark_symbol_coverage(df, tickers)
        if (
            df is None
            or df.empty
            or _frame_is_degenerate(df)
            or not _cache_matches_provider(df, provider)
        ):
            return pd.DataFrame()
        return df

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICE_DIR / f"{_price_key(tickers, start, end, kwargs)}.npz"

    if path.exists():
        if _cache_is_fresh(path, end):
            try:
                cached = _read_price_cache(path)
                # Ignore a poisoned (mostly-NaN) cache entry — a partial bulk
                # download that got persisted. Falling through refetches it.
                if (
                    not _frame_is_degenerate(cached)
                    and _cache_matches_provider(cached, provider)
                ):
                    return _mark_symbol_coverage(cached, tickers)
            except Exception as e:
                report(logger, "Corrupt price cache entry; refetching", e, path=str(path))

    primary_succeeded = False
    try:
        df = provider.download(tickers, start=start, end=end, **kwargs)
        df = _mark_symbol_coverage(df, tickers)
        primary_succeeded = (
            df is not None
            and not df.empty
            and not _frame_is_degenerate(df)
            and _cache_matches_provider(df, provider)
        )
        if df is not None and not df.empty and not primary_succeeded:
            logger.warning(
                "Rejecting malformed primary market-data response.",
                extra={"provider": provider.name, "tickers": str(tickers)[:200]},
            )
            df = None
    except Exception as e:
        report(logger, "Primary market-data provider failed", e,
               provider=provider.name, tickers=str(tickers)[:200])
        df = None

    if df is None or df.empty:
        # Providers hiccup. Two fallbacks, in order: a stale cache entry beats
        # nothing, then the secondary provider.
        if path.exists() and _cache_is_fresh(path, end):
            try:
                cached = _read_price_cache(path)
                if (
                    not _frame_is_degenerate(cached)
                    and _cache_matches_provider(cached, provider)
                ):
                    logger.warning(
                        "Provider returned no data; serving bounded-age cached prices.",
                        extra={"provider": provider.name},
                    )
                    return _mark_symbol_coverage(cached, tickers)
            except Exception as e:
                report(logger, "Stale cache read failed", e, path=str(path))
        fallback = market_data.get_fallback()
        if fallback is not None:
            try:
                df = fallback.download(tickers, start=start, end=end)
                df = _mark_symbol_coverage(df, tickers)
                if df is not None and not df.empty and not _frame_is_degenerate(df):
                    logger.warning("Prices served by the %s fallback provider.",
                                   fallback.name, extra={"fallback": fallback.name})
                else:
                    df = pd.DataFrame()
            except Exception as e:
                report(logger, "Fallback market-data provider failed", e,
                       provider=fallback.name)
                df = pd.DataFrame()

    # Only persist complete data. A partial/degenerate bulk download must never
    # poison the cache, or it silently starves every downstream consumer
    # (projections, metrics, backtests, recommendations).
    if (
        primary_succeeded
        and df is not None
        and not df.empty
        and not _frame_is_degenerate(df)
        and _cache_matches_provider(df, provider)
        and df.attrs.get("complete_coverage") is True
    ):
        try:
            _write_price_cache(path, df)
        except Exception as e:
            report(logger, "Could not write price cache", e, path=str(path))
    return df if df is not None else pd.DataFrame()


# --- provider-bound JSON caches ----------------------------------------------

def _provider_cache_path(path: Path, provider) -> Path:
    identity = f"{provider.name}|{provider.price_semantics}"
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return path.with_name(f"{path.stem}.{token}{path.suffix}")


def _load_provider_cache(path: Path, provider) -> dict:
    """Load entries only when the envelope matches this exact provider contract."""
    namespaced = _provider_cache_path(path, provider)
    try:
        raw = json.loads(namespaced.read_text(encoding="utf-8")) if namespaced.exists() else {}
    except Exception:
        return {}
    if (
        raw.get("schema") != 1
        or raw.get("provider") != provider.name
        or raw.get("price_semantics") != provider.price_semantics
        or not isinstance(raw.get("entries"), dict)
    ):
        return {}
    return raw["entries"]


def _save_provider_cache(path: Path, provider, entries: dict) -> None:
    namespaced = _provider_cache_path(path, provider)
    envelope = {
        "schema": 1,
        "provider": provider.name,
        "price_semantics": provider.price_semantics,
        "entries": entries,
    }
    temp_name = None
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=namespaced.parent,
            prefix=f".{namespaced.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(envelope, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, namespaced)
        temp_name = None
    except Exception as e:
        report(logger, f"Could not write {namespaced.name}", e)
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except Exception:
                pass


def cached_sectors(tickers) -> dict:
    """
    Return {ticker: raw_sector_or_'Unknown'} using the on-disk cache; only
    tickers not seen before hit the network. 'Unknown' results are not cached,
    so transient failures get retried next time.
    """
    provider = market_data.get_provider()
    cache = _load_provider_cache(SECTOR_FILE, provider)
    out, dirty = {}, False
    for t in tickers:
        t = str(t).upper()
        if t in cache:
            out[t] = cache[t]
            continue
        raw = "Unknown"
        if provider.supports_fundamentals:
            try:
                raw = provider.sector(t) or "Unknown"
            except Exception as e:
                report(logger, "Sector lookup failed", e, ticker=t, provider=provider.name)
                raw = "Unknown"
        out[t] = raw
        if raw != "Unknown":
            cache[t] = raw
            dirty = True
    if dirty:
        _save_provider_cache(SECTOR_FILE, provider, cache)
    return out


# --- corporate actions -------------------------------------------------------


def cached_splits(tickers) -> dict:
    """
    Return {ticker: [[iso_date, ratio], ...]} of historical stock splits.
    Refreshed at most once per day per ticker (splits are rare events).
    """
    provider = market_data.get_provider()
    cache = _load_provider_cache(SPLITS_FILE, provider)
    today = date.today().isoformat()
    out, dirty = {}, False

    for t in tickers:
        t = str(t).upper()
        entry = cache.get(t)
        if entry and entry.get("as_of") == today:
            out[t] = entry["events"]
            continue
        events = []
        if provider.supports_fundamentals:
            try:
                events = provider.splits(t) or []
            except Exception as e:
                report(logger, "Splits lookup failed", e, ticker=t, provider=provider.name)
                if entry:  # keep stale data over nothing
                    out[t] = entry["events"]
                    continue
        cache[t] = {"as_of": today, "events": events}
        out[t] = events
        dirty = True

    if dirty:
        _save_provider_cache(SPLITS_FILE, provider, cache)
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


def cached_dividends_ttm(tickers) -> dict:
    """
    Return {ticker: trailing-12-month dividends per share}. Refreshed at most
    once per day per ticker; stale values are kept when the fetch fails.
    """
    provider = market_data.get_provider()
    cache = _load_provider_cache(DIVIDENDS_FILE, provider)
    today = date.today().isoformat()
    out, dirty = {}, False

    for t in tickers:
        t = str(t).upper()
        entry = cache.get(t)
        if entry and entry.get("as_of") == today:
            out[t] = entry["ttm"]
            continue
        ttm = 0.0
        if provider.supports_fundamentals:
            try:
                ttm = float(provider.dividends_ttm(t) or 0.0)
            except Exception as e:
                report(logger, "Dividend lookup failed", e, ticker=t, provider=provider.name)
                if entry:
                    out[t] = entry["ttm"]
                    continue
        cache[t] = {"as_of": today, "ttm": round(ttm, 4)}
        out[t] = ttm
        dirty = True

    if dirty:
        _save_provider_cache(DIVIDENDS_FILE, provider, cache)
    return out
