# AI Portfolio Assistant

A web app that helps you understand and improve your stock portfolio. Import your holdings, chat with an AI advisor about them, and see live analytics: a 12-month value projection, news sentiment with Buy / Hold / Sell reads per stock, and honest backtests against the S&P 500.

> ⚠️ **Disclaimer:** Educational purposes only. Not investment advice.

## Features

* **Chat advisor** — a LangChain agent (GPT-4o-mini) with tools for portfolio metrics, optimization, backtesting, charting, and news sentiment. Keeps your last 10 chats.
* **Accounts** — register/login with bcrypt-hashed passwords and token auth; your portfolio and chats persist per user in SQLite.
* **Portfolio import** — upload a CSV or JSON of your holdings; the app derives weights and live values (yfinance).
* **Dashboard** — current value, estimated yearly gain, volatility, Sharpe ratio, a 12-month projection chart with a likely range, and a per-stock news check.
* **News sentiment** — recent headlines per ticker scored with a finance-tuned VADER (optional FinBERT via `SENTIMENT_BACKEND=finbert`), rolled up into Buy / Hold / Sell tags.
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

## Tech stack

Python 3.11 · FastAPI · LangChain + OpenAI · yfinance · PyPortfolioOpt · FinRL + Stable-Baselines3 · vaderSentiment · SQLite · matplotlib
