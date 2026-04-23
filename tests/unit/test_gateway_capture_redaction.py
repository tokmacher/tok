from __future__ import annotations

import json

from tok.gateway import BridgeSession


def test_capture_request_redacts_inline_and_key_based_secrets(tmp_path) -> None:
    session = BridgeSession(capture=True, memory_dir=tmp_path / ".tok")

    session.capture_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use Bearer super-secret-token and sk-live-secret while debugging.",
                }
            ],
            "system": "Authorization: Bearer top-secret",
            "x-api-key": "sk-top-level-secret",
        }
    )

    lines = session._capture_file.read_text().splitlines()  # type: ignore[union-attr]
    payload = json.loads(lines[-1])
    rendered = json.dumps(payload)

    assert "super-secret-token" not in rendered
    assert "top-secret" not in rendered
    assert "sk-live-secret" not in rendered
    assert "sk-top-level-secret" not in rendered
    assert "Bearer <redacted>" in rendered
    assert "sk-<redacted>" in rendered


def test_capture_event_redacts_nested_header_and_token_values(tmp_path) -> None:
    session = BridgeSession(capture=True, memory_dir=tmp_path / ".tok")

    session.capture_event(
        {
            "event": "fail_open_retry_provider_safe_invalid",
            "headers": {
                "authorization": "Bearer nested-secret",
                "x-api-key": "sk-nested-secret",
            },
            "nested": [
                {"openai_api_key": "sk-live-nested"},
                {"note": "upstream payload mentioned sk-inline-secret"},
            ],
        }
    )

    lines = session._capture_file.read_text().splitlines()  # type: ignore[union-attr]
    payload = json.loads(lines[-1])
    rendered = json.dumps(payload)

    assert "nested-secret" not in rendered
    assert "sk-nested-secret" not in rendered
    assert "sk-live-nested" not in rendered
    assert "sk-inline-secret" not in rendered
    assert payload["headers"]["authorization"] == "<redacted>"
    assert payload["headers"]["x-api-key"] == "<redacted>"
    assert payload["nested"][0]["openai_api_key"] == "<redacted>"
    assert "sk-<redacted>" in rendered
