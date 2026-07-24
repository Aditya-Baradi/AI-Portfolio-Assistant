"""
News + sentiment analysis for the portfolio agent.

Pulls recent financial-news headlines for a ticker from the configured market
data provider (see api/market_data.py) and scores their sentiment. Sentiment
uses VADER when available, augmented with a finance-specific lexicon; if VADER
is not installed it falls back to a small built-in lexicon scorer so the module
always works.

IMPORTANT: headline sentiment is a measure of news TONE, not a price forecast
and not a recommendation. `signal_from_score` deliberately returns descriptive
tone bands ("Positive tone" / "Mixed" / "Negative tone") rather than
Buy/Hold/Sell — a lexicon score over a handful of headlines is nowhere near
strong enough evidence to justify a trade instruction, and presenting it as one
would be both bad statistics and a regulatory problem.

NOTE on X/Twitter: scraping X is intentionally NOT used here. It requires a paid
API, violates the ToS for scraping, and is noisy/unreliable. Reputable financial
news gives cleaner, higher-signal input.
"""
from __future__ import annotations

import math
import threading
import warnings
from collections import OrderedDict
from datetime import datetime, timezone

# --- sentiment backend -----------------------------------------------------
# Finance-tuned lexicon boosters (VADER is general-purpose and under-weights
# market jargon). Scores are on VADER's roughly [-4, 4] per-token scale.
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "beating": 2.0, "surge": 2.6, "surges": 2.6,
    "soar": 3.0, "soars": 3.0, "rally": 2.0, "rallies": 2.0, "jumps": 2.2,
    "upgrade": 2.2, "upgraded": 2.2, "upgrades": 2.2, "outperform": 2.2,
    "bullish": 3.2, "record": 1.6, "growth": 1.2, "profit": 1.6, "profits": 1.6,
    "gains": 1.6, "rebound": 1.8, "strong": 1.4, "raises": 1.4, "boost": 1.6,
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "plunge": -3.0, "plunges": -3.0,
    "slump": -2.6, "slumps": -2.6, "tumble": -2.6, "tumbles": -2.6, "sinks": -2.4,
    "downgrade": -2.2, "downgraded": -2.2, "downgrades": -2.2, "underperform": -2.2,
    "bearish": -3.2, "lawsuit": -2.0, "probe": -1.6, "recall": -2.0, "cuts": -1.2,
    "warns": -1.8, "warning": -1.6, "loss": -1.6, "losses": -1.6, "fraud": -3.4,
    "bankruptcy": -3.6, "layoffs": -2.0, "slashes": -2.0, "weak": -1.6, "halts": -1.8,
}

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _ANALYZER = SentimentIntensityAnalyzer()
    _ANALYZER.lexicon.update(FINANCE_LEXICON)
    _BACKEND = "vader+finance"
except Exception:  # pragma: no cover - fallback path
    _ANALYZER = None
    _BACKEND = "lexicon-fallback"

# Optional FinBERT backend (research: ~85-91% accuracy on financial headlines
# vs ~56% for VADER). Opt in with SENTIMENT_BACKEND=finbert; requires
# `pip install transformers` (torch is already a FinRL dependency). Model
# (~440MB, ProsusAI/finbert) downloads on first use. Falls back to VADER
# silently if unavailable.
_FINBERT = None
if __import__("os").getenv("SENTIMENT_BACKEND", "").lower() == "finbert":
    try:
        from transformers import pipeline as _hf_pipeline

        # FINBERT_MODEL can point at a local fine-tuned folder
        # (see scripts/train_sentiment.py) or any HF Hub model id.
        _model_id = __import__("os").getenv("FINBERT_MODEL", "ProsusAI/finbert")
        _FINBERT = _hf_pipeline(
            "sentiment-analysis", model=_model_id, top_k=None, truncation=True
        )
        _BACKEND = "finbert"
    except Exception as e:  # pragma: no cover - optional path
        warnings.warn(f"FinBERT requested but unavailable ({e}); using {_BACKEND}.")


def _finbert_score(text: str) -> float:
    """Map FinBERT's positive/negative/neutral probabilities to [-1, 1]."""
    out = _FINBERT(text[:512])[0]
    probs = {d["label"].lower(): float(d["score"]) for d in out}
    return round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4)


def sentiment_backend() -> str:
    return _BACKEND


def _fallback_score(text: str) -> float:
    """Tiny lexicon scorer used only if VADER is unavailable. Returns [-1, 1]."""
    tokens = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text).split()
    if not tokens:
        return 0.0
    raw = sum(FINANCE_LEXICON.get(tok, 0.0) for tok in tokens)
    # Squash to [-1, 1] similarly to VADER's compound normalization.
    return round(raw / (raw * raw + 15.0) ** 0.5, 4) if raw else 0.0


def score_text(text: str) -> float:
    """Return a compound sentiment score in [-1, 1] for a piece of text."""
    if not text:
        return 0.0
    if _FINBERT is not None:
        try:
            return _finbert_score(text)
        except Exception:
            pass  # fall through to VADER
    if _ANALYZER is not None:
        return float(_ANALYZER.polarity_scores(text)["compound"])
    return _fallback_score(text)


def label_from_score(score: float, pos_th: float = 0.15, neg_th: float = -0.15) -> str:
    if score >= pos_th:
        return "Positive language"
    if score <= neg_th:
        return "Negative language"
    return "Neutral language"


# Minimum headlines before we'll characterise tone at all. Below this the
# average is dominated by one or two articles and means nothing.
MIN_HEADLINES_FOR_TONE = 3


def signal_from_score(score: float, n_headlines: int) -> str:
    """
    Describe the TONE of recent coverage. Deliberately not a trade signal.

    This used to return Buy / Hold / Sell. It no longer does, for two reasons:

    1. Statistically, an average VADER score over a handful of headlines is a
       weak, high-variance measure with no established predictive relationship
       to forward returns. Rendering it as "Buy" implied evidence we don't have.
    2. Legally, publishing per-security buy/sell instructions to the public is
       the kind of thing that attracts securities regulators. Describing the
       news tone is reporting; telling someone to buy is advice.

    Returns one of: "Positive tone", "Mixed", "Negative tone", "No data".
    """
    if not n_headlines or n_headlines < MIN_HEADLINES_FOR_TONE:
        return "No data"
    if score >= 0.15:
        return "Positive tone"
    if score <= -0.15:
        return "Negative tone"
    return "Mixed"


# Headline sentiment barely moves minute to minute, but fetching news is a
# network round-trip per ticker — cache the aggregate for a while.
_SENT_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_SENT_INFLIGHT: dict[str, threading.Event] = {}
_SENT_LOCK = threading.RLock()
SENTIMENT_CACHE_MAX_ENTRIES = 512
SENTIMENT_TTL_SECONDS = 900  # 15 minutes


def cached_ticker_sentiment(ticker: str, limit: int = 8, ttl: float = SENTIMENT_TTL_SECONDS) -> dict:
    """Aggregate sentiment for one ticker (no headlines), cached for `ttl` seconds."""
    import time

    key = ticker.strip().upper()
    owner = False
    while True:
        now = time.time()
        with _SENT_LOCK:
            expired = [
                cached_key
                for cached_key, (saved_at, _value) in list(_SENT_CACHE.items())
                if now - saved_at >= ttl
            ]
            for cached_key in expired:
                _SENT_CACHE.pop(cached_key, None)
            hit = _SENT_CACHE.get(key)
            if hit:
                if hasattr(_SENT_CACHE, "move_to_end"):
                    _SENT_CACHE.move_to_end(key)
                return hit[1]
            event = _SENT_INFLIGHT.get(key)
            if event is None:
                event = threading.Event()
                _SENT_INFLIGHT[key] = event
                owner = True
                break
        # Another request is already paying for this ticker. Wait outside the
        # lock so different tickers can still be fetched concurrently.
        if event.wait(timeout=30):
            continue
        with _SENT_LOCK:
            if _SENT_INFLIGHT.get(key) is event:
                _SENT_INFLIGHT.pop(key, None)

    try:
        res = analyze_ticker_sentiment(key, limit=limit)
        slim = {
            "ticker": key,
            "avg_score": res["avg_score"],
            "label": res["label"],
            "n_headlines": res["n_headlines"],
            # Kept so the headline drill-down shows exactly what was scored.
            "headlines": res.get("headlines", []),
        }
        with _SENT_LOCK:
            _SENT_CACHE[key] = (time.time(), slim)
            if hasattr(_SENT_CACHE, "move_to_end"):
                _SENT_CACHE.move_to_end(key)
            while len(_SENT_CACHE) > SENTIMENT_CACHE_MAX_ENTRIES:
                if hasattr(_SENT_CACHE, "popitem"):
                    try:
                        _SENT_CACHE.popitem(last=False)
                        continue
                    except TypeError:
                        pass
                _SENT_CACHE.pop(next(iter(_SENT_CACHE)), None)
        return slim
    finally:
        if owner:
            with _SENT_LOCK:
                completed = _SENT_INFLIGHT.pop(key, None)
                if completed is not None:
                    completed.set()


def _extract_headlines(raw_news, limit: int) -> list[dict]:
    """Normalize yfinance news items across old and new (nested 'content') shapes."""
    out: list[dict] = []
    for item in (raw_news or [])[: limit * 2]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item

        title = content.get("title") or item.get("title")
        if not title:
            continue
        summary = content.get("summary") or content.get("description") or item.get("summary") or ""

        # publisher
        provider = content.get("provider") or {}
        publisher = (
            provider.get("displayName")
            if isinstance(provider, dict)
            else None
        ) or item.get("publisher") or "Unknown"

        # url
        cu = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        url = cu.get("url") if isinstance(cu, dict) else None
        url = url or item.get("link") or ""

        # published time
        published = content.get("pubDate") or content.get("displayTime")
        if not published and item.get("providerPublishTime"):
            try:
                published = datetime.fromtimestamp(
                    int(item["providerPublishTime"]), tz=timezone.utc
                ).isoformat()
            except Exception:
                published = ""

        out.append(
            {
                "title": str(title).strip(),
                "summary": _strip_html(str(summary))[:300],
                "publisher": publisher,
                "url": url,
                "published": published or "",
            }
        )
        if len(out) >= limit:
            break
    return out


def _strip_html(text: str) -> str:
    """Cheap HTML tag remover for summaries (avoids a bs4 dependency)."""
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


def fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    """Fetch recent news headlines for a ticker from the configured provider."""
    from api import market_data

    try:
        raw = market_data.get_provider().news(ticker, limit=limit)
    except Exception as e:
        raise RuntimeError(f"Headline data is unavailable for {ticker}.") from e
    return _extract_headlines(raw, limit)


def analyze_ticker_sentiment(ticker: str, limit: int = 10) -> dict:
    """
    Fetch and score recent headlines for a single ticker.

    Returns aggregate sentiment (avg score + label), pos/neu/neg counts, and the
    scored headlines (most negative and most positive are the most informative).
    """
    ticker = ticker.strip().upper()
    headlines = fetch_news(ticker, limit=limit)

    scored = []
    for h in headlines:
        text = f"{h['title']}. {h['summary']}".strip()
        s = score_text(text)
        scored.append({**h, "score": round(s, 4), "label": label_from_score(s)})

    if not scored:
        return {
            "ticker": ticker,
            "n_headlines": 0,
            "avg_score": 0.0,
            "label": "No data",
            "counts": {
                "Positive language": 0,
                "Neutral language": 0,
                "Negative language": 0,
            },
            "headlines": [],
            "backend": _BACKEND,
        }

    avg = sum(x["score"] for x in scored) / len(scored)
    counts = {
        "Positive language": 0,
        "Neutral language": 0,
        "Negative language": 0,
    }
    for x in scored:
        counts[x["label"]] += 1

    scored.sort(key=lambda x: x["score"])  # most negative language first
    return {
        "ticker": ticker,
        "n_headlines": len(scored),
        "avg_score": round(avg, 4),
        "label": label_from_score(avg),
        "counts": counts,
        "headlines": scored,
        "backend": _BACKEND,
    }


def analyze_portfolio_sentiment(weights: dict, limit_per_ticker: int = 6) -> dict:
    """
    Score sentiment for every holding and produce a weight-averaged portfolio read.

    `weights` maps ticker -> weight (dollar or fractional; renormalized here).
    """
    if not weights:
        raise ValueError("weights is empty.")

    total_w = sum(float(v) for v in weights.values() if v and float(v) > 0)
    per_ticker = {}
    weighted_sum = 0.0
    used_w = 0.0

    for t, w in weights.items():
        w = float(w)
        if w <= 0:
            continue
        try:
            res = analyze_ticker_sentiment(t, limit=limit_per_ticker)
        except Exception as e:
            per_ticker[t] = {"ticker": t.upper(), "error": str(e), "avg_score": 0.0, "label": "No data"}
            continue
        per_ticker[t] = {
            "avg_score": res["avg_score"],
            "label": res["label"],
            "n_headlines": res["n_headlines"],
        }
        if res["n_headlines"] > 0:
            weighted_sum += res["avg_score"] * w
            used_w += w

    overall = round(weighted_sum / used_w, 4) if used_w > 0 else 0.0

    ranked = sorted(
        [(t, d.get("avg_score", 0.0)) for t, d in per_ticker.items()],
        key=lambda kv: kv[1],
    )
    return {
        "portfolio_sentiment_score": overall,
        "portfolio_label": label_from_score(overall),
        "coverage": f"{round(100 * used_w / total_w, 1)}% of weight" if total_w else "0%",
        "most_negative_language": ranked[:3],
        "most_positive_language": list(reversed(ranked[-3:])),
        "per_ticker": per_ticker,
        "backend": _BACKEND,
    }


def apply_sentiment_tilt(weights: dict, strength: float = 0.2, limit_per_ticker: int = 5) -> dict:
    """
    Return normalized input weights plus descriptive headline tone.

    Headline sentiment is too noisy and weakly validated to alter a user's
    allocation. ``strength`` is retained for backwards-compatible callers but
    is never applied; the effective strength is always zero.
    """
    if not weights:
        raise ValueError("weights is empty.")

    scores = {}
    neutral = {}
    for t, w in weights.items():
        w = float(w)
        if not math.isfinite(w):
            raise ValueError("weights must be finite.")
        if w <= 0:
            continue
        neutral[str(t).upper()] = neutral.get(str(t).upper(), 0.0) + w
        try:
            res = analyze_ticker_sentiment(t, limit=limit_per_ticker)
            score = (
                res["avg_score"]
                if res["n_headlines"] >= MIN_HEADLINES_FOR_TONE
                else 0.0
            )
        except Exception:
            score = 0.0
        scores[str(t).upper()] = score

    total = sum(neutral.values())
    if total <= 0:
        raise ValueError("weights has no positive entries.")
    neutral = {ticker: round(value / total, 6) for ticker, value in neutral.items()}

    return {
        "weights": neutral,
        "sentiment": scores,
        "strength": 0.0,
        "requested_strength": max(0.0, min(float(strength), 1.0)),
        "disabled": True,
        "backend": _BACKEND,
        "disclaimer": (
            "Headline tone is descriptive only and does not change allocation weights."
        ),
    }


if __name__ == "__main__":
    import json

    print("backend:", sentiment_backend())
    print(json.dumps(analyze_ticker_sentiment("NVDA", limit=6), indent=2))
