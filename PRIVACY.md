# Privacy Policy

**Effective date: 23 July 2026**

{{LEGAL_REVIEW_NOTICE}}

## 1. Who we are

Evergreen ("the Service") is operated by **{{OPERATOR_NAME}}**, contactable at
**{{LEGAL_CONTACT_EMAIL}}**. For GDPR purposes we
are the **data controller** for the personal data described below.

## 2. What we collect

We collect only what the Service needs to work. There is no advertising, no
tracking pixels, and no third-party analytics.

### You give us directly

| Data | Why | Where it lives |
|---|---|---|
| Email address | Account identity, sign-in, verification, password reset | `users` table |
| Password | Authentication — stored only as a **bcrypt hash** (cost 12), never in plaintext | `users` table |
| Display name (optional) | Shown in the UI | `users` table |
| Two-factor secret (optional) | TOTP sign-in — **encrypted at rest** | `users` table |
| **Portfolio holdings** — tickers, share counts, cost basis, purchase dates | The core function of the Service | `portfolios` table, **encrypted at rest** |
| **Investor profile** — retirement horizon, risk tolerance, contributions, goal amount | Planning tools | `profiles` table, **encrypted at rest** |
| Watchlist tickers | Stocks page | `watchlist` table |
| Chat messages | Conversation history with the assistant | `chats` / `messages` tables, encrypted at rest |

### Collected automatically

| Data | Why | Retention |
|---|---|---|
| IP address | Security: rate limiting, account-lockout, and the account activity log | Activity log keeps the most recent 100 events per user |
| Session tokens | Keeping you signed in — stored **SHA-256-hashed** | Expires after 2 days idle / 7 days absolute |
| Server logs | Debugging and abuse detection | **{{LOG_RETENTION_DAYS}} days**, unless a security incident or law requires longer |

### What we do **not** collect

- No brokerage credentials, and no connection to any brokerage account.
- No bank details, card numbers, or government identifiers.
- No location beyond what an IP address implies.
- No advertising identifiers, and no cross-site tracking.

## 3. How we use it

We process your data only to:

1. **Operate the Service** — authenticate you, store your portfolio, compute
   analytics, and answer your chat messages. *(Legal basis: performance of a
   contract.)*
2. **Keep it secure** — rate limiting, lockouts, the activity log, and abuse
   investigation. *(Legal basis: legitimate interests.)*
3. **Send transactional email** — verification and password reset only.
   *(Legal basis: performance of a contract.)*
4. **Comply with law** where we are required to. *(Legal basis: legal
   obligation.)*

We do **not** sell your data, share it with advertisers, use it for behavioural
profiling, or use it to train machine-learning models operated by us. The
language-model processor handles submitted content under its own terms and our
processor configuration.

## 4. Who we share it with

Only these processors, only for the purposes above:

| Processor | What it receives | Why |
|---|---|---|
| **Language-model provider** (default: OpenAI) | Your chat messages, and portfolio data the assistant needs to answer them | Powers the chat assistant |
| **Market data provider** (e.g. Tiingo, Polygon) | Ticker symbols only — never your holdings, share counts, or identity | Prices, splits, dividends, news |
| **Resend** | Your email address and the message | Verification and reset email |
| **Hosting provider** | Everything stored, as the infrastructure host | Runs the Service |
| **Sentry** (if enabled) | Error reports and diagnostic context, with PII sending disabled | Error monitoring |

> **Note on the chat assistant.** To answer questions about your portfolio, your
> messages and relevant holdings are sent to the language-model provider. Do not
> type anything into chat you are not comfortable sending to a third party.
> Review that provider's own data-handling terms; retention and training
> policies are theirs, not ours.

We may also disclose data if legally compelled, or to protect our rights or the
safety of others.

## 5. Where data is processed

The primary hosting region is **{{HOSTING_REGION}}**. Our processors may operate
internationally; where personal data leaves the EEA or UK, transfers rely on
Standard Contractual Clauses or an adequacy decision.

## 6. How long we keep it

- **Account data** — while your account exists.
- **Portfolio, profile, watchlist** — while your account exists.
- **Chat history** — the 10 most recent conversations per user and up to 500
  complete user/assistant turns in each conversation; older content is pruned
  automatically.
- **Activity log** — the 100 most recent events per user.
- **Session tokens** — deleted on expiry (2 days idle, 7 days absolute).
- **Backups** — the operator must schedule encrypted off-host backups and
  configure deletion after no more than **{{BACKUP_RETENTION_DAYS}} days**.
  The repository supplies backup, verification, restore, and pruning commands;
  it does not operate the schedule or object-store lifecycle itself.
- **Security and server logs** — the operator must configure the log platform
  to delete them after the period stated above. They may outlive account
  deletion where needed for security or legal obligations.
- **Active account data** — deleted when you delete your account (see §8);
  processor and backup copies age out under their applicable retention periods.

## 7. Security

What is actually implemented:

- Passwords hashed with **bcrypt** (cost 12); never stored or logged in plaintext.
- Session tokens stored **SHA-256-hashed**, with sliding idle expiry, an
  absolute cap, rotation, and a limit on concurrent sessions.
- **Encryption at rest** for the sensitive columns — portfolio holdings,
  investor profile, chat content/titles, token-rotation successors, and
  two-factor secrets — using envelope encryption with a master key held
  outside the database.
- Optional **two-factor authentication** (TOTP).
- Per-IP rate limiting and per-account lockout on repeated failed sign-ins.
- Strict **Content-Security-Policy** with per-request nonces, output escaping,
  and URL scheme validation to prevent script injection.
- HTTPS in transit; parameterised SQL throughout.
- Transport, storage and application dependencies audited in CI.

No system is perfectly secure. We cannot guarantee absolute security, and you
are responsible for your own password hygiene.

## 8. Your rights

Regardless of where you live, the Service gives you these directly in the app:

- **Access / portability** — *Account → Export my data* downloads the
  user-facing account, portfolio, profile, watchlist, alert, and chat data in
  the active database, including recent account activity. Server logs, secrets,
  token hashes, and backup copies are excluded from the self-service download.
- **Deletion** — *Account → Delete account* removes your active account,
  portfolio, profile, watchlist, chats, alerts, and account activity. Limited
  security logs and encrypted backups expire under the retention periods above.
- **Rectification** — edit your name, portfolio and profile at any time.

If you are in the EEA/UK you additionally have the right to restrict or object
to processing, and to lodge a complaint with your supervisory authority. In
California you have the rights to know, delete, correct, and to opt out of sale
or sharing — we do not sell or share personal information as those terms are
defined by the CPRA.

To exercise anything not available in-app, contact **{{LEGAL_CONTACT_EMAIL}}**. We aim
to respond within 30 days.

## 9. Children

The Service is not directed at anyone under 18 and we do not knowingly collect
their data. If you believe a minor has given us data, contact us and we will
delete it.

## 10. Cookies

In the public HTTPS deployment we use a strictly necessary Secure, HttpOnly,
SameSite=Strict session cookie to keep you signed in. It is not used for
advertising or cross-site tracking. State-changing cookie requests are also
checked against browser origin/fetch metadata; there is currently no separate
synchronizer token. Browser preferences such as theme and the last open chat
may be kept in local or session storage; authentication tokens are not.

## 11. Changes

We may update this policy. Material changes use a new policy version and require
acknowledgement before continued use of gated features.

## 12. Contact

**{{LEGAL_CONTACT_EMAIL}}**
