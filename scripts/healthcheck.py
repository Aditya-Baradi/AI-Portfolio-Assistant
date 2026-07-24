#!/usr/bin/env python
"""
Container health probe.

Exits 0 when the app is serving and non-zero otherwise. Docker records that
health state; an orchestrator or operator must act on it because plain Docker
Compose does not restart a container merely because it becomes unhealthy.
Kept as a file rather than an inline `python -c` in the Dockerfile so the
HEALTHCHECK line stays readable and the logic can grow without quoting pain.

Deliberately checks only liveness, not the readiness flags in /healthz: a
correctly running instance that is (say) still on the development market-data
provider is misconfigured, but restarting it would not help.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

URL = os.getenv("HEALTHCHECK_URL", "http://127.0.0.1:8000/healthz")
TIMEOUT = float(os.getenv("HEALTHCHECK_TIMEOUT", "4"))


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as resp:
            if resp.status != 200:
                print(f"unhealthy: HTTP {resp.status}", file=sys.stderr)
                return 1
            body = json.loads(resp.read() or b"{}")
    except urllib.error.URLError as e:
        print(f"unhealthy: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:  # malformed body, timeout, anything else
        print(f"unhealthy: {e}", file=sys.stderr)
        return 1

    if body.get("status") != "ok":
        print(f"unhealthy: status={body.get('status')!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
