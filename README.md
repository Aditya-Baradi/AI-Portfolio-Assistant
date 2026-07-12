# Evergreen — AI Portfolio Assistant

A web app that helps you understand and improve your stock portfolio. Import your holdings, chat with an AI advisor about them, and see live analytics: a 12-month value projection, news sentiment with Buy / Hold / Sell reads per stock, and honest backtests against the S&P 500.

> ⚠️ **Disclaimer:** Educational purposes only. Not investment advice.

## Features

* **Chat advisor** — a LangChain agent (GPT-4o-mini) with tools for portfolio metrics, optimization, backtesting, charting, and news sentiment. Keeps your last 10 chats; 50 messages/day per user.
* **Accounts & security** — bcrypt-hashed passwords, SHA-256-hashed session tokens, optional two-factor sign-in (TOTP), login rate limiting, security headers, an activity log, data export, and account deletion.
* **Portfolio import** — holdings snapshots (CSV/JSON) or brokerage transaction histories, which are replayed to recover your true cost basis. Add/remove holdings in the UI.
* **Dashboard** — live value, 12-month outlook, what-if comparison vs an optimized mix (with a concrete trade list), diversification breakdown with concentration warnings, per-stock news sentiment with Buy/Hold/Sell tags and clickable headlines, and dividend income.
* **Stocks page** — 12-month estimated price ranges for every holding, a watchlist (star any ticker), lookup of any symbol, and your real performance history vs the S&P 500.
* **Plan page** — years to retirement, risk comfort, volatility limit, goal: a Monte Carlo retirement outlook, the probability of hitting your goal amount, the monthly contribution that would make it likely, and stocks screened to fit your risk profile.
* **Alerts** — a bell that warns when your portfolio drops sharply or a holding's news turns negative.
* **Backtesting** — portfolio and sector performance vs the S&P 500 with transaction costs, alpha/beta, and equity-curve charts.
* **Optimization** — conservative min-volatility blend (validated out-of-sample with a walk-forward harness), max-Sharpe via PyPortfolioOpt, and an optional FinRL (PPO) reinforcement-learning blend.

## Getting started

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Create a `.env` in the project root:

```text
OPENAI_API_KEY=sk-...
```

Run the app (serves both the API and the web page):

```bash
uvicorn api.backend:app --port 8000
```

Then open http://127.0.0.1:8000, create an account, and import your portfolio file.

## Project structure

```
api/
├─ backend.py            # FastAPI app: auth, chat, upload, projection, sentiment
├─ index.html            # Web UI (served by the backend at /)
├─ db.py                 # SQLite: users, tokens, portfolios, chats
├─ langchain_agent.py    # Chat agent setup and system prompt
├─ langchain_tools.py    # Agent tools (metrics, backtest, optimize, sentiment...)
├─ portfolio_core.py     # Parsing, weights, metrics, backtest engine
├─ predict_agent.py      # Optimizers: min-vol blend, max-Sharpe, FinRL PPO
├─ sentiment.py          # Headline fetching + scoring, Buy/Hold/Sell signals
├─ data_cache.py         # Disk caches for prices, sectors, splits
├─ charting.py           # Equity-curve PNGs for backtests
├─ backtest_vs_sp500.py  # Walk-forward research harness
└─ mcp_portfolio_server.py
```

## Deployment (Docker + HTTPS)

```bash
DOMAIN=yourdomain.com docker compose up -d
```

The compose file runs the app plus a Caddy reverse proxy that fetches and
renews HTTPS certificates automatically once `DOMAIN` points at the machine.
Data (accounts, caches) lives in the `appdata` volume. For local-only use,
skip Docker and run uvicorn directly.

## Tech stack

Python 3.11 · FastAPI · LangChain + OpenAI · yfinance · PyPortfolioOpt · FinRL + Stable-Baselines3 · vaderSentiment · SQLite · matplotlib
