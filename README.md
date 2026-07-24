# Evergreen — AI Portfolio Assistant

A web app that helps you understand a stock portfolio. Import holdings, review descriptive historical analytics and scenarios, and ask an AI assistant questions grounded in the authenticated account's data.

> ⚠️ **Educational tool — not investment advice.** Evergreen is not a registered investment adviser. Nothing it produces is a recommendation to buy, sell or hold any security. Model output and backtests are hypothetical, use historical data, and do not predict future results. See [risk disclosures](#what-this-tool-does-not-do), [TERMS.md](TERMS.md) and [PRIVACY.md](PRIVACY.md).

## Features

* **Chat assistant** — direct OpenAI tool calling for portfolio summaries, historical metrics, benchmark comparisons, prices, and descriptive headline tone. The server binds tools to the authenticated account; user and chat identifiers are never model arguments. The assistant cannot generate target allocations, ranked picks, or trade lists.
* **Accounts & security** — bcrypt-hashed passwords, hashed session tokens with sliding expiry, **encryption at rest** for holdings and 2FA secrets, optional TOTP two-factor, per-IP rate limiting and per-account lockout, enforced email verification, a strict CSP, an activity log, data export, and account deletion.
* **Portfolio import** — holdings snapshots (CSV/JSON) or brokerage transaction histories, which are replayed to recover your true cost basis. Add/remove holdings in the UI.
* **Dashboard** — estimated current value, a 12-month scenario range, diversification breakdown with concentration warnings, per-stock headline tone with sources, and dividend income.
* **Stocks page** — 12-month total-return-equivalent scenario bands (not price targets), a watchlist, and descriptive lookup/news context. The app does not claim to reconstruct brokerage account performance.
* **Plan page** — a Monte Carlo retirement scenario driven by explicit assumptions, with the simulated frequency of reaching a goal and the monthly contribution used by the scenario.
* **Alerts** — periodically recomputed heuristics for a sharp decline in the current holding mix and currently negative headline tone. They do not reconstruct actual account history, detect a change from a prior tone state, or replace brokerage alerts.
* **Historical comparison** — retrospective portfolio and sector results vs the S&P 500, with transaction-cost assumptions and alpha/beta. These are hypothetical model comparisons, not brokerage account performance.

## What this tool does *not* do

Being straight about the limits, because the outputs look more authoritative than they are:

* **News tone is not a rating.** Headline sentiment is a lexicon score over a handful of articles with no established predictive relationship to returns. It is reported as "Positive tone / Mixed / Negative tone", never as Buy or Sell.
* **The retirement outlook does not model your holdings.** It models the risk level *you select*, using an assumed relationship between volatility and expected return. Only the starting balance comes from your portfolio. It's for comparing scenarios, not predicting your balance.
* **Projections assume the future resembles the past.** They extrapolate trailing return and volatility under a geometric-Brownian-motion model. Real markets have fat tails and regime changes.
* **It does not provide an allocation or trade plan.** Public routes and the assistant do not return target weights, ranked securities, personalized buy/sell/hold instructions, or generated trade lists.
* **The AI assistant can be confidently wrong.** Verify anything that matters.

## Getting started

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

```bash
cp .env.example .env            # then add your OPENAI_API_KEY
uvicorn api.backend:app --port 8000
```

Then open http://127.0.0.1:8000, create an account, and import your portfolio file.

On first run the app generates a development encryption key in `.evergreen_key` and prints a startup audit listing what would block a public deployment. Both are expected locally.

> **Research only:** `requirements-rl.txt` describes an isolated FinRL environment for offline experiments. It conflicts with the production market-data dependency graph, is not installed in the public image, and is not a supported in-process web feature.

## Configuration

Everything is environment-driven; see [.env.example](.env.example) for the annotated list. The ones that matter:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat assistant |
| `EVERGREEN_MASTER_KEY` | **Encryption at rest.** Generate with `python -m api.crypto keygen`. Losing it means losing every portfolio. |
| `BACKUP_ENCRYPTION_KEY` | Separate authenticated whole-backup encryption key. Generate with `python scripts/backup_db.py --generate-key`; never reuse the master key. |
| `MARKET_DATA_PROVIDER` | `yfinance` (dev only) · `tiingo` · `polygon` |
| `MARKET_DATA_REDISTRIBUTION_ENTITLED` | Set true only after the provider contract explicitly permits this public use. |
| `REDIS_URL` | Shared rate-limit/lockout/2FA state. It does not make SQLite horizontally scalable. |
| `RESEND_API_KEY` | Verification + password-reset email |
| `APP_BASE_URL`, `APP_ORIGINS`, `EMAIL_FROM` | Public URL, CORS allow-list, and verified sender identity. |
| `SENTRY_DSN` | Production error-monitoring destination; request bodies and credentials are scrubbed. |
| `LEGAL_*`, `HOSTING_REGION`, retention fields | Values rendered into public policy; legal-review confirmation is mandatory. |
| `EVERGREEN_ENV` | `production` makes the startup audit fatal |

## Project structure

```
api/
├─ backend.py            # App assembly: middleware, startup audit, router wiring
├─ index.html            # Web UI (served by the backend at /)
├─ deps.py               # Auth, rate limiting, lockout, shared validators
├─ routers/              # One module per slice of the API
│  ├─ auth.py            #   register, login, 2FA, verification, reset
│  ├─ account.py         #   profile, password, export, deletion
│  ├─ portfolio.py       #   upload, holdings, analytics, projection
│  ├─ chat.py            #   the assistant
│  ├─ stocks.py          #   watchlist, per-ticker projections, news
│  ├─ alerts.py          #   portfolio alerts
│  ├─ plan.py            #   investor profile, retirement scenario
│  └─ legal.py           #   terms, privacy, disclosures, healthz
├─ crypto.py             # Envelope encryption for data at rest
├─ state.py              # Shared rate-limit / lockout / 2FA state (memory|redis)
├─ market_data.py        # Pluggable price providers (yfinance|stooq|tiingo|polygon)
├─ observability.py      # Structured JSON logging, request ids, Sentry
├─ db.py                 # SQLite: users, tokens, portfolios, chats
├─ langchain_agent.py    # Direct OpenAI tool loop (legacy filename; no LangChain)
├─ agent_tools.py        # Allow-listed tools with server-owned auth context
├─ langchain_tools.py    # Compatibility shim for old imports
├─ portfolio_core.py     # Parsing, weights, metrics, backtest engine
├─ predict_agent.py      # Optimizers: min-vol blend, max-Sharpe, FinRL PPO
├─ sentiment.py          # Headline fetching + tone scoring
├─ data_cache.py         # Disk caches for prices, sectors, splits, dividends
├─ charting.py           # Dormant research helper; not on the public request path
├─ backtest_vs_sp500.py  # Walk-forward research harness
└─ mcp_portfolio_server.py

scripts/
├─ backup_db.py          # Consistent SQLite backups (online backup API)
├─ healthcheck.py        # Container health probe
└─ train_sentiment.py
```

## Testing

```bash
pytest -q          # 250+ tests, no network access required
```

CI additionally resolves and audits the production manifest, exercises a real Redis service, validates the isolated RL manifest, checks correctness-level lint rules, scans Git history for secrets, and builds/health-checks the Compose app.

## Deployment (Docker + HTTPS)

```bash
DOMAIN=yourdomain.com docker compose up -d
```

The compose file runs the app plus a Caddy reverse proxy that fetches and renews HTTPS certificates automatically once `DOMAIN` points at the machine. Data lives in the `appdata` volume.

The image sets `EVERGREEN_ENV=production`. It refuses to start until every production prerequisite in `.env.example` is configured: encryption and backup keys, contractually entitled market-data redistribution, Redis, public URL/CORS, verified email delivery, OpenAI, Sentry, completed operator/retention fields, and confirmed legal review. Placeholder values are not production configuration.

The app remains one process/replica while it uses SQLite. Redis makes short-lived security state consistent; it does not provide distributed database writes or job coordination. Uvicorn trusts forwarding headers only from Caddy's fixed internal IP (`TRUSTED_PROXY_IPS` may override it with an explicit IP/CIDR; never use `*`).

Schedule `python scripts/backup_db.py` daily. It takes an online SQLite snapshot, gzip-compresses it, and applies chunked authenticated encryption with `BACKUP_ENCRYPTION_KEY`; production backup creation fails without a valid, distinct key. Copy backups to encrypted off-host/object storage, alert on missed jobs and Redis evictions, keep both encryption keys in disaster-recovery escrow, and rehearse `--restore` at least quarterly. The script header contains exact Docker backup and stopped-app restore commands.

### Before going public

Read the operator checklists at the bottom of [TERMS.md](TERMS.md) and [PRIVACY.md](PRIVACY.md). The short version: complete the operator fields, obtain explicit contractual redistribution rights for this use, and have qualified counsel assess investment-adviser and data-protection obligations in every served jurisdiction.

## Licence

[Apache-2.0](LICENSE) — see [NOTICE](NOTICE) for why, and for the third-party data-licensing caveats.

## Tech stack

Python 3.11 · FastAPI · direct OpenAI SDK tool calling · PyPortfolioOpt · vaderSentiment · SQLite · cryptography · Redis · Sentry · isolated optional FinRL research manifest
