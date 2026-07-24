"""The global request cap also covers chunked bodies without Content-Length."""

from __future__ import annotations

import asyncio

from api.backend import RequestBodyLimitMiddleware


def test_streamed_body_over_limit_never_completes_application_body_read():
    reached_application = False
    completed_body_read = False

    async def application(_scope, receive, send):
        nonlocal reached_application, completed_body_read
        reached_application = True
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        completed_body_read = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/auth/login",
        "headers": [(b"content-type", b"application/json")],
    }
    asyncio.run(RequestBodyLimitMiddleware(application, max_bytes=6)(
        scope, receive, send
    ))

    assert reached_application is True
    assert completed_body_read is False
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
