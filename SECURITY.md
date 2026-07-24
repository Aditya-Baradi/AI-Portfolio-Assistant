# Security notes

Running record of the app's security posture: what's hardened, what's a known
risk, and why. Audit the shipped manifest with
`pip-audit -r requirements.txt --strict` (CI fails on a finding).

## Reporting a vulnerability

Email **[SECURITY CONTACT — complete before publishing]**. Please don't open a
public issue for anything exploitable. We'll acknowledge within 3 working days.

## Hardened

- **Auth:** bcrypt (cost 12), session tokens stored SHA-256-hashed, constant-time
  login, per-IP rate limiting, **per-account lockout**, optional TOTP 2FA,
  parameterized SQL.
- **Sessions:** sliding idle expiry (2 days), an absolute cap (7 days), and at
  most 10 concurrent sessions per user. The client is prompted to rotate after
  12 hours, but rotation is not automatic revocation: an unrevoked stolen token
  can remain valid until idle/absolute expiry (at most 7 days).
- **Encryption at rest:** portfolios, investor profiles and TOTP secrets are
  encrypted with **envelope encryption** (`api/crypto.py`). A master key from
  the environment wraps a data key held in the database; the data key encrypts
  the fields. A stolen `app.db` yields neither holdings nor working 2FA secrets.
  Rotating the master key re-wraps one row — no data migration. Legacy
  plaintext rows are detected and migrated transparently on startup.
- **Email verification is enforced for protected product features.** Unverified
  accounts retain authentication, account/legal self-service, `/me`, and resend,
  but cannot upload financial data, call analytics, or spend model credit.
- **XSS:** all untrusted data (news headlines, tickers, holdings, alerts, audit
  IPs) is HTML-escaped before rendering; link URLs are scheme-validated
  (`http(s)` only); assistant markdown is escaped *before* it is parsed.
- **CSP:** `script-src` uses a fresh per-request nonce with **no `unsafe-inline`**,
  plus `frame-ancestors 'none'`, `object-src 'none'`, `base-uri 'self'` and
  `form-action 'self'`. `style-src` keeps `unsafe-inline` (inline styles can't
  exfiltrate; there are hundreds of them). HSTS is set when served over TLS.
- **Reverse proxy:** application code uses only `request.client`. Uvicorn
  rewrites it from forwarding headers only when the socket peer is in the
  explicit trusted-proxy allow-list (Caddy's fixed Compose IP by default);
  wildcard proxy trust is forbidden.
- **Request bodies:** a global ASGI cap rejects oversized declared and chunked
  bodies before endpoint parsing; portfolio upload also performs its own
  bounded read.
- **Agent:** direct OpenAI SDK tool calling (no LangChain runtime), capped at six
  tool rounds with bounded tool output/history/session cache. Tool schemas never
  contain a user, chat, session, path, or database identifier; authenticated
  context is bound by the server-side dispatcher. A deterministic output filter
  blocks obvious personalized trade instructions if model prompting regresses.
- **Observability:** structured JSON logs with request correlation, credential
  scrubbing across messages/tracebacks/context, and production-required Sentry
  with request bodies/PII disabled and outbound event scrubbing.
- **Container:** deny-by-default build context, multi-stage image, exact base
  tag, unprivileged/read-only runtime, bounded ephemeral Redis, and `/healthz`.
- **CI:** production dependency resolution and blocking audit (no blanket
  ignores), tests, real Redis integration, isolated RL-manifest resolution,
  correctness lint, Docker/Compose smoke, and full-history secret scanning.

## Startup audit

`api/backend.py` runs `_startup_audit()` on boot. With
`EVERGREEN_ENV=production` **the app refuses to start** unless:

- `EVERGREEN_MASTER_KEY` is set (otherwise secrets rest under a dev key file),
- `MARKET_DATA_PROVIDER` is a provider for which the operator has verified the
  actual contract permits this public display/redistribution,
- `MARKET_DATA_REDISTRIBUTION_ENTITLED` explicitly confirms the operator's
  contract permits this public use,
- `BACKUP_ENCRYPTION_KEY` is valid and distinct from the master key,
- Redis, public URL/CORS, verified email delivery, OpenAI, Sentry, operator
  identity/retention fields, hosting region, and confirmed legal review are set.

`GET /healthz` reports the same checks as `production_ready`.

## Market data licensing (a legal risk, not just a technical one)

The default provider, **yfinance, uses Yahoo endpoints for which this project
has not established public display, redistribution, or commercial-use
rights.** It is blocked in production. Operationally it is also prone to rate
limits, IP blocks, and upstream schema changes.

`api/market_data.py` makes the provider swappable. Selecting a provider or
buying an API plan is not proof of public redistribution rights:

| Provider | Commercial plans exist | Notes |
|---|---|---|
| `yfinance` (default) | N/A | No public-use rights established by this project; blocked in production. |
| `stooq` | N/A | Keyless development fallback; no public-use rights claimed; blocked in production. |
| `tiingo` | Adapter available | `TIINGO_API_KEY`; the operator must verify its executed contract permits this exact use. |
| `polygon` | Adapter available | `POLYGON_API_KEY`; the operator must verify its executed contract permits this exact use. |

## Known dependency risks (assessed, deferred with rationale)

The RL stack is now **optional and excluded from the shipped image**
(`requirements-rl.txt`), which removes the large majority of the project's
advisory surface from a default deployment.

| Package | Status | Rationale / plan |
|---|---|---|
| **torch 2.2.2** | Not installed by default | ~40 advisories, almost all `torch.load`/deserialization of **untrusted checkpoints**. Now confined to the opt-in RL extra; the default image and CI never install it. |
| **scikit-learn** | **Current** | Core pin moved off the previously deferred vulnerable 1.3 line. |
| OpenAI, requests, python-multipart, python-dotenv, cryptography, Redis, Sentry | **Current** | Exact direct pins; the blocking core audit has no blanket vulnerability ignores. |
| pip, setuptools | **Current** | Upgraded in the Docker build. |

## Deployment constraint: one process/replica while using SQLite

Without `REDIS_URL`, this security state is process-local (`api/state.py`):

- auth rate-limit windows and per-account lockout counters,
- pending-2FA and 2FA-setup handles.

Redis is mandatory for production so those controls are global. That does not
make the application horizontally scalable: portfolios, chats, quotas, and
other writes still share SQLite, and long-running job capacity is not
distributed. Keep one app process/replica until both are moved to managed,
distributed services.

If Redis is unreachable, the backend downgrades to memory and the production
startup audit refuses to serve. Compose bounds Redis at 128 MiB with LFU
eviction because the data is short-lived. An eviction can discard a cold
lockout/rate-limit/2FA handle, so alert on Redis memory and eviction metrics.

## Backups and key custody

`scripts/backup_db.py` takes consistent snapshots using SQLite's online backup
API, compresses them, then applies streaming authenticated encryption with a
separate `BACKUP_ENCRYPTION_KEY`. Production backup creation refuses a missing,
invalid, or reused key. Verification decrypts, decompresses, and runs SQLite
integrity checks; restore publishes a verified database atomically while the
app is stopped.

**Neither key is in the backup.** Keep the master and backup-encryption keys in
separate secret-manager/DR custody, copy encrypted backups off-host, alert on
missed schedules, and rehearse a restore at least quarterly.

> **Losing the master key means losing every portfolio in the database.**
> There is no recovery path by design: if the key could be recovered from the
> database, encrypting the database would be pointless. Treat key custody with
> the same seriousness as the backups themselves, and never start the app with
> a throwaway key against a database you care about — it will encrypt data under
> a key you don't have.

## Still outstanding

- Browser sessions use `HttpOnly; SameSite=Strict` cookies, and API Bearer
  tokens remain supported. There is no separate synchronizer CSRF token; the
  current cross-site request defense relies on the strict SameSite cookie,
  same-origin browser requests, and the CORS allow-list.
- Chat messages and the portfolio context are sent to a third-party model
  provider. Disclosed in the Privacy Policy; there is no way around it short of
  self-hosting a model.
- No formal penetration test has been performed.
