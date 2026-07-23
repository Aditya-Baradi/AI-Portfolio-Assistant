# Security notes

Running record of the app's security posture: what's hardened, what's a known
risk, and why. Audit dependencies with `python-dotenv`-style pins in
`requirements.txt`; run `pip-audit` against the **installed environment**
(`python -m pip_audit`), not `-r requirements.txt` — see the CI note below.

## Hardened

- **Auth:** bcrypt (cost 12), session tokens stored SHA-256-hashed, constant-time
  login, per-IP rate limiting, optional TOTP 2FA, parameterized SQL.
- **Email verification + password reset:** single-use, hashed, expiring tokens;
  `/auth/forgot` never reveals whether an account exists; reset revokes all sessions.
- **XSS:** all untrusted data (news headlines, tickers, holdings, alerts, audit
  IPs) is HTML-escaped before rendering; link URLs are scheme-validated
  (`http(s)` only).
- **CSP:** `script-src` uses a fresh per-request nonce with **no `unsafe-inline`**,
  so injected script cannot execute even if escaping is bypassed. `style-src`
  keeps `unsafe-inline` (inline styles can't exfiltrate; there are hundreds of them).
- **Reverse proxy:** client IP is taken from the rightmost `X-Forwarded-For`
  entry (the value Caddy appends), so rate limiting and audit logs see real IPs
  and can't be spoofed by a client prepending their own header.
- **Uploads:** rejected via `Content-Length` before reading, and the read is
  capped, so a large upload can't exhaust memory.
- **Container:** runs as an unprivileged user; build tooling (`pip`,
  `setuptools`) is upgraded during the image build.

## Known dependency risks (assessed, deferred with rationale)

The ML stack (`finrl`, `stable-baselines3`, `torch`) constrains upgrades, and
several CVEs are not reachable given how the app uses these libraries.

| Package | Status | Rationale / plan |
|---|---|---|
| **torch 2.2.2** | Deferred | ~40 advisories, almost all are `torch.load`/deserialization of **untrusted checkpoints** or unsafe operators. This app only trains and loads **its own** models — no user-supplied model files reach torch, so the practical exposure is low. Upgrading is pinned by `stable-baselines3`/`finrl`; plan is to test `torch>=2.6` on a branch against the FinRL path before bumping. |
| **langchain 0.1.20 / -core 0.1.52** | Deferred | Serious advisories are in `langchain-experimental` and specific document loaders, none of which this app imports. The 0.2/0.3 migration is a scoped effort, not a drop-in bump. |
| **urllib3 1.26.20** | Deferred | Latest 1.26.x; 2.x fixes several advisories but may conflict with libs pinning `<2`. Bump when the tree allows. |
| **scikit-learn 1.3.2** | Deferred | PYSEC-2024-110 (TfidfVectorizer data leak) — not on a user-facing path here. Bumping risks the pinned numeric stack; revisit with the torch branch. |
| requests, python-multipart, python-dotenv | **Fixed** | Bumped to patched versions. |
| pip, setuptools | **Fixed** | Upgraded in the Docker build. |

## Deployment constraint: run a single worker (until state moves to Redis)

Some state is held in process memory and is **not shared across workers**:

- the auth rate-limiter map and the pending-2FA / 2FA-setup maps (`backend.py`),
- the FinRL concurrency semaphore (`langchain_tools.py`).

The Docker image runs one uvicorn worker (the default), which is correct. **Do
not add `--workers N` (>1) or run multiple app replicas** without first moving
this state to a shared store (Redis): with >1 worker, rate limiting fragments,
in-progress 2FA logins fail intermittently, and the FinRL cap stops being global.
Portfolios and chat history are already in SQLite, so those are safe across
workers — it's only the in-memory maps above that pin us to one process.

Bounded/evicted state (no longer a leak): portfolios are read per-request from
the database (no growing in-memory cache), `chat_memory/` is age-pruned to
30 days, and concurrent FinRL runs are capped at 2.

## CI prerequisite: `requirements.txt` is not cleanly installable

`pip install -r requirements.txt` currently fails to resolve: `yfinance` needs
`websockets>=12` (we pin `13.1`), but `alpaca-trade-api==3.2.0` (an eager FinRL
import dep) requires `websockets<11`. The local venv works only because it was
installed in a way that let `websockets 13.1` win. This is why `pip-audit -r`
fails and why CI can't do a clean install.

**To unblock reproducible installs + CI, pick one:**
1. Drop `alpaca-trade-api`/`wrds` and confirm FinRL still imports (its import is
   already guarded; if it fails, `FINRL_AVAILABLE=False` and the app falls back
   to PyPortfolioOpt — a product tradeoff to decide deliberately).
2. Move the RL stack (`finrl`, `torch`, `stable-baselines3`, `alpaca`, ...) into
   an optional `requirements-rl.txt`, so the core app + tests install cleanly and
   CI runs fast, with RL as an opt-in extra.

Option 2 is recommended: it makes the core app cleanly installable, lets CI run
the full test suite, and isolates the heavy/conflicting ML dependencies.
