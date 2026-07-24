"""
Market-data provider abstraction.

Why this exists
---------------
yfinance uses Yahoo endpoints without giving this project a data-distribution
contract. Shipping it as the data layer of a public app would therefore leave
both a rights gap and an operational one (rate limits, IP blocks, silent schema
changes). It remains available for local development, where the operator is
still responsible for complying with the applicable provider terms.

So the data source is an interface with several implementations, chosen by the
MARKET_DATA_PROVIDER environment variable. Moving to a production-capable
provider integration is a config change, not a rewrite. `data_cache.py` is the
only caller; every price,
split, dividend and sector lookup in the app funnels through it.

Providers
---------
yfinance  (default)  development-only here; refused in production.
stooq                keyless fallback, EOD only, no splits/dividends/sectors.
tiingo                commercial-plan integration. TIINGO_API_KEY.
polygon               commercial-plan integration. POLYGON_API_KEY.

Every provider returns prices in the same shape as `yf.download(...)`:
a DataFrame with a MultiIndex column level 0 containing "Close", indexed by
date, so the rest of the app is provider-agnostic.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime

import pandas as pd

logger = logging.getLogger("evergreen.market_data")

# Providers for which commercial/redistribution plans exist.  This is NOT proof
# that the operator has bought the required entitlement; production readiness
# separately requires an explicit operator acknowledgement.
PRODUCTION_CAPABLE_PROVIDERS = {"tiingo", "polygon"}
REDISTRIBUTION_ACK_ENV = "MARKET_DATA_REDISTRIBUTION_ENTITLED"

# Downstream analytics consume a total-return close series whose latest point is
# the current split-adjusted close.  Historical points are adjusted for both
# splits and cash dividends.  Keeping this contract explicit prevents a provider
# switch from silently changing every CAGR, Sharpe ratio and backtest.
TOTAL_RETURN_SEMANTICS = "split_and_dividend_adjusted_total_return"
SPLIT_ONLY_SEMANTICS = "split_adjusted_price_only"
UNKNOWN_PRICE_SEMANTICS = "provider_defined_close"


class MarketDataError(Exception):
    """Raised when a provider is misconfigured or a fetch fails unrecoverably."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def _as_close_frame(
    closes: dict[str, pd.Series],
    *,
    source_provider: str | None = None,
    price_semantics: str | None = None,
) -> pd.DataFrame:
    """Assemble {ticker: series} into the ('Close', ticker) MultiIndex layout."""
    if not closes:
        return _empty()
    frame = pd.DataFrame(closes).sort_index()
    frame.columns = pd.MultiIndex.from_product([["Close"], frame.columns])
    if source_provider:
        frame.attrs["source_provider"] = source_provider
    if price_semantics:
        frame.attrs["price_semantics"] = price_semantics
    return frame


def _total_return_close(
    split_adjusted_close: pd.Series,
    dividends: dict[pd.Timestamp, float],
) -> pd.Series:
    """
    Convert a split-adjusted price series to a dividend-adjusted total-return
    series, anchored so the latest value remains the latest quoted close.

    `dividends` must already be expressed on the same split-adjusted per-share
    basis as the close series.  A dividend is applied on its ex-date (or the
    first following trading date if the provider reports a non-trading date).
    """
    close = split_adjusted_close.dropna().astype(float).sort_index()
    if close.empty:
        return close
    if not dividends:
        return close

    gross = close.pct_change().add(1.0)
    gross.iloc[0] = 1.0
    for raw_date, cash in dividends.items():
        if not cash:
            continue
        ex_date = pd.Timestamp(raw_date).tz_localize(None)
        pos = int(close.index.searchsorted(ex_date, side="left"))
        if pos <= 0 or pos >= len(close):
            continue
        previous = float(close.iloc[pos - 1])
        current = float(close.iloc[pos])
        if previous > 0:
            gross.iloc[pos] = (current + float(cash)) / previous

    index = gross.cumprod()
    if not len(index) or float(index.iloc[-1]) <= 0:
        raise MarketDataError("Could not construct a total-return close series.")
    return index * (float(close.iloc[-1]) / float(index.iloc[-1]))


class PriceProvider:
    """Interface every data source implements."""

    name = "base"
    #: True when this adapter supports a provider that offers commercial plans.
    #: This says nothing about the operator's actual contract or entitlements.
    production_capable = False
    #: True when the provider can serve corporate actions and sectors.
    supports_fundamentals = False
    #: Meaning of the Close series returned by download().
    price_semantics = UNKNOWN_PRICE_SEMANTICS

    def download(self, tickers, start, end, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def splits(self, ticker: str) -> list[list]:
        """[[iso_date, ratio], ...] — empty when unsupported."""
        return []

    def dividends_ttm(self, ticker: str) -> float:
        """Trailing-12-month dividends per share — 0.0 when unsupported."""
        return 0.0

    def sector(self, ticker: str) -> str:
        return "Unknown"

    def news(self, ticker: str, limit: int = 10) -> list:
        """
        Recent news items for a ticker, in yfinance's shape:
        [{"content": {"title", "summary", "provider": {"displayName"},
                      "canonicalUrl": {"url"}}}, ...]

        Returns [] when the provider has no news entitlement.
        """
        return []


class YFinanceProvider(PriceProvider):
    """Yahoo via yfinance. This project has no public-use contract for it."""

    name = "yfinance"
    production_capable = False
    supports_fundamentals = True
    price_semantics = TOTAL_RETURN_SEMANTICS

    def download(self, tickers, start, end, **kwargs) -> pd.DataFrame:
        import yfinance as yf

        # The entire analytics stack expects total-return adjusted Close data.
        # Do not let a caller accidentally switch this off.
        kwargs["auto_adjust"] = True
        frame = yf.download(tickers, start=start, end=end, **kwargs)
        if frame is not None:
            frame.attrs["source_provider"] = self.name
            frame.attrs["price_semantics"] = self.price_semantics
        return frame

    def splits(self, ticker: str) -> list[list]:
        import yfinance as yf

        s = yf.Ticker(ticker).splits
        if s is None or not len(s):
            return []
        return [[idx.strftime("%Y-%m-%d"), float(r)] for idx, r in s.items() if r and r > 0]

    def dividends_ttm(self, ticker: str) -> float:
        import yfinance as yf

        s = yf.Ticker(ticker).dividends
        if s is None or not len(s):
            return 0.0
        cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=365)
        idx = s.index.tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
        return float(s[idx >= cutoff].sum())

    def sector(self, ticker: str) -> str:
        import yfinance as yf

        info = yf.Ticker(ticker).get_info()
        return info.get("sector") or "Unknown"

    def news(self, ticker: str, limit: int = 10) -> list:
        import yfinance as yf

        return list(yf.Ticker(ticker).news or [])[:limit]


class StooqProvider(PriceProvider):
    """Keyless EOD fallback via pandas-datareader. Prices only."""

    name = "stooq"
    production_capable = False
    supports_fundamentals = False
    price_semantics = UNKNOWN_PRICE_SEMANTICS

    def download(self, tickers, start, end, **kwargs) -> pd.DataFrame:
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            return _empty()
        if isinstance(tickers, str):
            tickers = [tickers]
        closes = {}
        for t in tickers:
            try:
                d = pdr.DataReader(str(t).upper(), "stooq", start=start, end=end)
                if d is not None and not d.empty and "Close" in d.columns:
                    closes[str(t).upper()] = d["Close"].sort_index()
            except Exception:
                continue
        return _as_close_frame(
            closes,
            source_provider=self.name,
            price_semantics=self.price_semantics,
        )


class _HttpProvider(PriceProvider):
    """Shared plumbing for production-capable HTTP/JSON integrations."""

    env_key = ""
    base_url = ""

    def __init__(self) -> None:
        self.api_key = os.getenv(self.env_key, "").strip()
        if not self.api_key:
            raise MarketDataError(
                f"MARKET_DATA_PROVIDER={self.name} requires {self.env_key} to be set."
            )

    def _client(self):
        import httpx

        return httpx.Client(timeout=30.0)

    @staticmethod
    def _iso(d) -> str:
        if isinstance(d, (datetime, date)):
            return d.strftime("%Y-%m-%d")
        return pd.to_datetime(d).strftime("%Y-%m-%d")


class TiingoProvider(_HttpProvider):
    """
    Tiingo EOD prices. Split/dividend adjusted closes come back directly, which
    matches the app's auto_adjust=True expectation from yfinance.
    """

    name = "tiingo"
    production_capable = True
    supports_fundamentals = True
    price_semantics = TOTAL_RETURN_SEMANTICS
    env_key = "TIINGO_API_KEY"
    base_url = "https://api.tiingo.com/tiingo/daily"

    def download(self, tickers, start, end, **kwargs) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        headers = {"Authorization": f"Token {self.api_key}"}
        params_base = {"startDate": self._iso(start), "endDate": self._iso(end), "format": "json"}
        closes = {}
        with self._client() as client:
            for t in tickers:
                sym = str(t).upper()
                try:
                    r = client.get(f"{self.base_url}/{sym}/prices",
                                   params=params_base, headers=headers)
                    if r.status_code != 200:
                        logger.warning("Tiingo %s -> HTTP %s", sym, r.status_code)
                        continue
                    rows = r.json()
                    if not rows:
                        continue
                    idx = pd.to_datetime([row["date"] for row in rows]).tz_localize(None)
                    # adjClose is split+dividend adjusted, like yfinance auto_adjust.
                    vals = [float(row.get("adjClose") or row.get("close")) for row in rows]
                    closes[sym] = pd.Series(vals, index=idx).sort_index()
                except Exception as e:
                    logger.warning(
                        "Tiingo fetch failed for %s (%s).", sym, type(e).__name__
                    )
                    continue
        return _as_close_frame(
            closes,
            source_provider=self.name,
            price_semantics=self.price_semantics,
        )

    def splits(self, ticker: str) -> list[list]:
        headers = {"Authorization": f"Token {self.api_key}"}
        out = []
        with self._client() as client:
            r = client.get(f"{self.base_url}/{str(ticker).upper()}/prices",
                           params={"startDate": "1990-01-01", "format": "json"},
                           headers=headers)
            if r.status_code != 200:
                return []
            for row in r.json():
                factor = float(row.get("splitFactor") or 1.0)
                if factor and factor != 1.0:
                    out.append([str(row["date"])[:10], factor])
        return out

    def dividends_ttm(self, ticker: str) -> float:
        headers = {"Authorization": f"Token {self.api_key}"}
        start = (pd.Timestamp(date.today()) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        with self._client() as client:
            r = client.get(f"{self.base_url}/{str(ticker).upper()}/prices",
                           params={"startDate": start, "format": "json"}, headers=headers)
            if r.status_code != 200:
                return 0.0
            return float(sum(float(row.get("divCash") or 0.0) for row in r.json()))

    def sector(self, ticker: str) -> str:
        # Tiingo's fundamentals endpoint is a separate (paid) entitlement; the
        # daily meta endpoint has no sector. Return Unknown rather than guess.
        return "Unknown"


class PolygonProvider(_HttpProvider):
    """
    Polygon.io daily aggregates plus corporate actions.

    Polygon's ``adjusted=true`` aggregates are split-adjusted but not
    dividend-adjusted.  We combine the close series with cash-dividend events
    below so this provider honours the same total-return contract as Tiingo.
    """

    name = "polygon"
    production_capable = True
    supports_fundamentals = True
    price_semantics = TOTAL_RETURN_SEMANTICS
    env_key = "POLYGON_API_KEY"
    base_url = "https://api.polygon.io"

    def _split_events(self, client, ticker: str, start=None) -> list[tuple[pd.Timestamp, float]]:
        params = {"ticker": ticker, "limit": 1000, "apiKey": self.api_key}
        if start is not None:
            params["execution_date.gte"] = self._iso(start)
        r = client.get(f"{self.base_url}/v3/reference/splits", params=params)
        if r.status_code != 200:
            raise MarketDataError(
                f"Polygon split lookup for {ticker} returned HTTP {r.status_code}."
            )
        events = []
        for row in (r.json() or {}).get("results") or []:
            try:
                ratio = float(row["split_to"]) / float(row["split_from"])
                if ratio > 0:
                    events.append(
                        (pd.Timestamp(row["execution_date"]).tz_localize(None), ratio)
                    )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
        return sorted(events, key=lambda item: item[0])

    def _adjusted_dividends(
        self,
        client,
        ticker: str,
        start,
        end,
        split_events: list[tuple[pd.Timestamp, float]],
    ) -> dict[pd.Timestamp, float]:
        r = client.get(
            f"{self.base_url}/v3/reference/dividends",
            params={
                "ticker": ticker,
                "limit": 1000,
                "ex_dividend_date.gte": self._iso(start),
                "ex_dividend_date.lte": self._iso(end),
                "apiKey": self.api_key,
            },
        )
        if r.status_code != 200:
            raise MarketDataError(
                f"Polygon dividend lookup for {ticker} returned HTTP {r.status_code}."
            )
        out: dict[pd.Timestamp, float] = {}
        for row in (r.json() or {}).get("results") or []:
            try:
                ex_date = pd.Timestamp(row["ex_dividend_date"]).tz_localize(None)
                cash = float(row.get("cash_amount") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            # Polygon's historic dividend cash amounts are not split-adjusted,
            # while adjusted aggregates are.  Express the cash amount on the
            # same current-share basis before applying it to the close series.
            future_split_factor = 1.0
            for split_date, ratio in split_events:
                if split_date > ex_date:
                    future_split_factor *= ratio
            if future_split_factor > 0 and cash:
                out[ex_date] = out.get(ex_date, 0.0) + cash / future_split_factor
        return out

    def download(self, tickers, start, end, **kwargs) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        closes = {}
        with self._client() as client:
            for t in tickers:
                sym = str(t).upper()
                try:
                    url = (f"{self.base_url}/v2/aggs/ticker/{sym}/range/1/day/"
                           f"{self._iso(start)}/{self._iso(end)}")
                    r = client.get(url, params={"adjusted": "true", "limit": 50000,
                                                "apiKey": self.api_key})
                    if r.status_code != 200:
                        logger.warning("Polygon %s -> HTTP %s", sym, r.status_code)
                        continue
                    results = (r.json() or {}).get("results") or []
                    if not results:
                        continue
                    idx = pd.to_datetime([row["t"] for row in results], unit="ms").tz_localize(None)
                    split_close = pd.Series(
                        [float(row["c"]) for row in results],
                        index=idx,
                    ).sort_index()
                    split_events = self._split_events(client, sym, start=start)
                    dividends = self._adjusted_dividends(
                        client, sym, start, end, split_events
                    )
                    closes[sym] = _total_return_close(split_close, dividends)
                except Exception as e:
                    logger.warning(
                        "Polygon fetch failed for %s (%s).", sym, type(e).__name__
                    )
                    continue
        return _as_close_frame(
            closes,
            source_provider=self.name,
            price_semantics=self.price_semantics,
        )

    def splits(self, ticker: str) -> list[list]:
        with self._client() as client:
            try:
                events = self._split_events(client, str(ticker).upper())
            except MarketDataError:
                return []
        return [[d.strftime("%Y-%m-%d"), ratio] for d, ratio in events]

    def dividends_ttm(self, ticker: str) -> float:
        cutoff = (pd.Timestamp(date.today()) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        with self._client() as client:
            r = client.get(f"{self.base_url}/v3/reference/dividends",
                           params={"ticker": str(ticker).upper(), "limit": 1000,
                                   "ex_dividend_date.gte": cutoff, "apiKey": self.api_key})
            if r.status_code != 200:
                return 0.0
            return float(sum(float(row.get("cash_amount") or 0.0)
                             for row in (r.json() or {}).get("results") or []))

    def sector(self, ticker: str) -> str:
        with self._client() as client:
            r = client.get(f"{self.base_url}/v3/reference/tickers/{str(ticker).upper()}",
                           params={"apiKey": self.api_key})
            if r.status_code != 200:
                return "Unknown"
            res = (r.json() or {}).get("results") or {}
            # Polygon exposes SIC descriptions rather than GICS sectors.
            return res.get("sic_description") or "Unknown"

    def news(self, ticker: str, limit: int = 10) -> list:
        """Reshaped into the yfinance news layout so `sentiment.py` is provider-agnostic."""
        with self._client() as client:
            r = client.get(f"{self.base_url}/v2/reference/news",
                           params={"ticker": str(ticker).upper(), "limit": int(limit),
                                   "apiKey": self.api_key})
            if r.status_code != 200:
                return []
            out = []
            for row in (r.json() or {}).get("results") or []:
                out.append({"content": {
                    "title": row.get("title") or "",
                    "summary": row.get("description") or "",
                    "provider": {"displayName": (row.get("publisher") or {}).get("name") or ""},
                    "canonicalUrl": {"url": row.get("article_url") or ""},
                }})
            return out


_PROVIDERS = {
    "yfinance": YFinanceProvider,
    "yahoo": YFinanceProvider,
    "stooq": StooqProvider,
    "tiingo": TiingoProvider,
    "polygon": PolygonProvider,
}

_active: PriceProvider | None = None
_fallback: PriceProvider | None = None


def provider_name() -> str:
    return os.getenv("MARKET_DATA_PROVIDER", "yfinance").strip().lower() or "yfinance"


def configured_price_semantics() -> str:
    """Declared Close-series meaning for the configured provider."""
    cls = _PROVIDERS.get(provider_name())
    return cls.price_semantics if cls is not None else UNKNOWN_PRICE_SEMANTICS


def is_production() -> bool:
    return os.getenv("EVERGREEN_ENV", "").strip().lower() in ("production", "prod")


def redistribution_entitled() -> bool:
    """
    Operator acknowledgement that its provider contract permits displaying or
    redistributing data to this application's users.

    A commercial-looking API key is not evidence of redistribution rights.
    Requiring an explicit acknowledgement makes that legal decision visible in
    deployment configuration instead of inferring it from a provider brand.
    """
    return os.getenv(REDISTRIBUTION_ACK_ENV, "").strip().lower() in {
        "1", "true", "yes"
    }


def get_provider() -> PriceProvider:
    """The configured primary provider (cached)."""
    global _active
    if _active is None:
        name = provider_name()
        cls = _PROVIDERS.get(name)
        if cls is None:
            raise MarketDataError(
                f"Unknown MARKET_DATA_PROVIDER={name!r}. "
                f"Valid options: {', '.join(sorted(set(_PROVIDERS)))}"
            )
        _active = cls()
        logger.info(
            "Market data provider: %s (production_capable=%s)",
            _active.name,
            _active.production_capable,
        )
    return _active


def get_fallback() -> PriceProvider | None:
    """
    Development-only secondary provider used when the primary returns nothing.

    A production service must never replace its configured commercial primary
    feed with an unentitled development source. It therefore fails closed.
    """
    global _fallback
    if is_production():
        return None
    if _fallback is None:
        if provider_name() in ("stooq",):
            return None
        _fallback = StooqProvider()
    return _fallback


def reset_provider() -> None:
    """Drop cached providers (tests, config reload)."""
    global _active, _fallback
    _active = None
    _fallback = None


def check_production_ready() -> list[str]:
    """
    Startup validation. Returns a list of human-readable problems; empty when
    the data layer is appropriate for a public deployment.
    """
    problems = []
    name = provider_name()
    if name not in _PROVIDERS:
        problems.append(
            f"Unknown MARKET_DATA_PROVIDER={name!r}. "
            f"Valid options: {', '.join(sorted(set(_PROVIDERS)))}."
        )
        return problems
    if name not in PRODUCTION_CAPABLE_PROVIDERS:
        problems.append(
            f"MARKET_DATA_PROVIDER={name!r} is not a production-capable integration. "
            f"Set MARKET_DATA_PROVIDER to one of "
            f"{sorted(PRODUCTION_CAPABLE_PROVIDERS)} with the matching API key and "
            "contractual entitlements before going live."
        )
        return problems

    # Instantiate the provider so missing/invalid local configuration (most
    # importantly the matching API key) blocks startup rather than the first
    # user request.
    try:
        provider = get_provider()
    except Exception as exc:
        problems.append(f"Market-data provider configuration is invalid: {exc}")
        return problems

    if not provider.production_capable:
        problems.append(
            f"MARKET_DATA_PROVIDER={name!r} is not marked production-capable."
        )
    if provider.price_semantics != TOTAL_RETURN_SEMANTICS:
        problems.append(
            f"MARKET_DATA_PROVIDER={name!r} returns {provider.price_semantics!r}; "
            f"analytics require {TOTAL_RETURN_SEMANTICS!r}."
        )
    if not redistribution_entitled():
        problems.append(
            f"{REDISTRIBUTION_ACK_ENV}=true is required in production. Set it only "
            "after confirming that the operator's market-data contract permits "
            "display/redistribution to application users."
        )
    return problems
