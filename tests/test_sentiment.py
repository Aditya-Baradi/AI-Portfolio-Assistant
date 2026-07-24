"""Tests for the sentiment module (offline: scoring, labels, tilt, extraction)."""
from collections import OrderedDict

import pytest

from api.sentiment import (
    score_text,
    label_from_score,
    signal_from_score,
    cached_ticker_sentiment,
    _extract_headlines,
    apply_sentiment_tilt,
)


class TestScoring:
    def test_bullish_text_positive(self):
        assert score_text("Company beats estimates, shares surge on strong profit growth") > 0.2

    def test_bearish_text_negative(self):
        assert score_text("Shares plunge after fraud probe and bankruptcy warning") < -0.2

    def test_empty_text_zero(self):
        assert score_text("") == 0.0

    def test_labels(self):
        assert label_from_score(0.5) == "Positive language"
        assert label_from_score(-0.5) == "Negative language"
        assert label_from_score(0.0) == "Neutral language"
        assert label_from_score(0.15) == "Positive language"
        assert label_from_score(-0.15) == "Negative language"


class TestSignals:
    def test_tone_band_thresholds(self):
        assert signal_from_score(0.5, 3) == "Positive tone"
        assert signal_from_score(0.15, 3) == "Positive tone"   # boundary inclusive
        assert signal_from_score(0.0, 3) == "Mixed"
        assert signal_from_score(-0.15, 3) == "Negative tone"
        assert signal_from_score(-0.5, 3) == "Negative tone"

    def test_no_news_is_no_data(self):
        assert signal_from_score(0.9, 0) == "No data"

    def test_thin_coverage_is_not_characterised(self):
        """One or two headlines is noise, not a tone. It must not be labelled."""
        assert signal_from_score(0.9, 1) == "No data"
        assert signal_from_score(0.9, 2) == "No data"
        assert signal_from_score(0.9, 3) == "Positive tone"

    def test_never_emits_trade_instructions(self):
        """
        Guards the deliberate product decision that this function describes news
        tone and never instructs a trade. If someone reintroduces Buy/Sell here,
        this fails.
        """
        forbidden = {"buy", "sell", "hold", "strong buy", "outperform"}
        for score in (-1.0, -0.2, -0.15, 0.0, 0.15, 0.2, 1.0):
            for n in (0, 1, 3, 25):
                assert signal_from_score(score, n).strip().lower() not in forbidden


class TestSentimentCache:
    def test_second_call_hits_cache(self, monkeypatch):
        import api.sentiment as s

        calls = {"n": 0}

        def fake_analyze(t, limit=8):
            calls["n"] += 1
            return {"avg_score": 0.3, "label": "Bullish", "n_headlines": 4}

        monkeypatch.setattr(s, "analyze_ticker_sentiment", fake_analyze)
        monkeypatch.setattr(s, "_SENT_CACHE", OrderedDict())
        first = cached_ticker_sentiment("aaa")
        second = cached_ticker_sentiment("AAA")
        assert calls["n"] == 1  # second call served from cache
        assert first == second
        assert first["ticker"] == "AAA"
        assert first["avg_score"] == pytest.approx(0.3)

    def test_expired_entry_refetches(self, monkeypatch):
        import api.sentiment as s

        calls = {"n": 0}

        def fake_analyze(t, limit=8):
            calls["n"] += 1
            return {"avg_score": 0.1, "label": "Neutral", "n_headlines": 2}

        monkeypatch.setattr(s, "analyze_ticker_sentiment", fake_analyze)
        monkeypatch.setattr(s, "_SENT_CACHE", OrderedDict())
        cached_ticker_sentiment("BBB", ttl=0)
        cached_ticker_sentiment("BBB", ttl=0)
        assert calls["n"] == 2


class TestHeadlineExtraction:
    def test_new_nested_format(self):
        raw = [{
            "id": "1",
            "content": {
                "title": "Stock soars",
                "summary": "<p>Great news</p>",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://x/y"},
                "pubDate": "2026-07-09T10:00:00Z",
            },
        }]
        out = _extract_headlines(raw, limit=5)
        assert out[0]["title"] == "Stock soars"
        assert out[0]["summary"] == "Great news"  # HTML stripped
        assert out[0]["publisher"] == "Reuters"

    def test_old_flat_format(self):
        raw = [{"title": "Old style", "publisher": "MW", "link": "https://a", "providerPublishTime": 1700000000}]
        out = _extract_headlines(raw, limit=5)
        assert out[0]["title"] == "Old style"
        assert out[0]["publisher"] == "MW"

    def test_limit_and_junk_tolerance(self):
        raw = [{"content": {"title": f"T{i}"}} for i in range(10)] + ["junk", {"content": {}}]
        out = _extract_headlines(raw, limit=3)
        assert len(out) == 3


class TestSentimentTilt:
    def test_tone_never_changes_weights(self, monkeypatch):
        import api.sentiment as s

        fake = {"AAA": 1.0, "BBB": -1.0}

        def fake_analyze(t, limit=5):
            return {"avg_score": fake[t], "n_headlines": 3}

        monkeypatch.setattr(s, "analyze_ticker_sentiment", fake_analyze)
        out = apply_sentiment_tilt({"AAA": 0.5, "BBB": 0.5}, strength=0.2)
        w = out["weights"]
        assert w == {"AAA": 0.5, "BBB": 0.5}
        assert sum(w.values()) == pytest.approx(1.0)
        assert out["disabled"] is True
        assert out["strength"] == 0.0

    def test_zero_strength_is_identity(self, monkeypatch):
        import api.sentiment as s

        monkeypatch.setattr(s, "analyze_ticker_sentiment",
                            lambda t, limit=5: {"avg_score": 0.9, "n_headlines": 3})
        out = apply_sentiment_tilt({"AAA": 0.7, "BBB": 0.3}, strength=0.0)
        assert out["weights"]["AAA"] == pytest.approx(0.7)

    def test_failed_fetch_treated_as_neutral(self, monkeypatch):
        import api.sentiment as s

        def boom(t, limit=5):
            raise RuntimeError("network down")

        monkeypatch.setattr(s, "analyze_ticker_sentiment", boom)
        out = apply_sentiment_tilt({"AAA": 0.6, "BBB": 0.4}, strength=0.2)
        assert out["weights"]["AAA"] == pytest.approx(0.6)
        assert out["sentiment"] == {"AAA": 0.0, "BBB": 0.0}

    def test_empty_weights_raise(self):
        with pytest.raises(ValueError):
            apply_sentiment_tilt({})
