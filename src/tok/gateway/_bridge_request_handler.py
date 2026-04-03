from __future__ import annotations

"""Fail-open request helpers for the Tok gateway."""

import json
from typing import Any

import httpx

from ..runtime.pipeline.request_validation import (
    has_provider_sensitive_failures,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)
from . import BridgeSession, _log_bridge_body_structure, logger
from ._bridge_comparison import _payloads_materially_differ
from ._bridge_preflight import (
    _count_user_messages_with_mixed_tool_result_content,
    _count_user_tool_result_split_boundaries,
)

__all__ = ["send_with_tok_fail_open_retry"]


def _decode_bridge_body(raw_content: bytes | None) -> dict[str, Any] | None:
    if raw_content is None:
        return None
    try:
        decoded = json.loads(raw_content)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


async def send_with_tok_fail_open_retry(
    session: BridgeSession,
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes,
    original_content: bytes | None,
    retry_content: bytes | None = None,
    allow_original_retry: bool = True,
    stream: bool = False,
    compressed_request: bool = False,
) -> tuple[httpx.Response, bool, dict[str, int]]:
    request_obj = client.build_request(
        method, url, headers=headers, content=content
    )
    response = await client.send(request_obj, stream=stream)
    retried_without_tok = False
    retry_signals: dict[str, int] = {}

    logger.warning(
        "Fail-open check: status=%d, compressed=%s, has_orig=%s, fail_open=%s",
        response.status_code,
        compressed_request,
        original_content is not None,
        session.fail_open,
    )

    if response.status_code == 429:
        # Pass through to client - let Claude Code handle quota exhaustion
        retry_after = response.headers.get("retry-after", "unknown")
        logger.warning(
            "rate_limit_429: upstream returned 429, passing through to client (retry_after=%s)",
            retry_after,
        )
        retry_signals["rate_limit_429_passed_through"] = 1

    if (
        response.status_code == 400
        and compressed_request
        and session.fail_open
    ):
        fallback_content = retry_content
        if fallback_content is None and allow_original_retry:
            fallback_content = original_content
        if stream:
            await response.aread()
        error_text = response.text
        _log_bridge_body_structure(
            "upstream_400_after_compressed_request",
            content=content,
            headers=headers,
            original_content=original_content,
            compressed_request=compressed_request,
        )
        prepared_summary: dict[str, Any] | str = {}
        prepared_pairing: list[dict[str, Any]] = []
        prepared_failures: list[str] = []
        provider_safe_summary: dict[str, Any] | str = {}
        provider_safe_pairing: list[dict[str, Any]] = []
        provider_safe_failures: list[str] = []
        prepared_mixed_user_tool_result_messages = 0
        prepared_split_boundaries = 0
        provider_safe_mixed_user_tool_result_messages = 0
        provider_safe_split_boundaries = 0
        prepared_body = _decode_bridge_body(content)
        if isinstance(prepared_body, dict):
            prepared_summary = summarize_message_structure(
                prepared_body.get("messages", [])
            )
            prepared_pairing = summarize_bridge_pairing(
                prepared_body.get("messages", [])
            )
            prepared_failures = validate_anthropic_bridge_body(prepared_body)
            prepared_mixed_user_tool_result_messages = (
                _count_user_messages_with_mixed_tool_result_content(
                    prepared_body.get("messages", [])
                )
            )
            prepared_split_boundaries = (
                _count_user_tool_result_split_boundaries(
                    prepared_body.get("messages", [])
                )
            )
        fallback_body = _decode_bridge_body(fallback_content)
        if isinstance(fallback_body, dict):
            provider_safe_summary = summarize_message_structure(
                fallback_body.get("messages", [])
            )
            provider_safe_pairing = summarize_bridge_pairing(
                fallback_body.get("messages", [])
            )
            provider_safe_failures = validate_anthropic_bridge_body(
                fallback_body
            )
            provider_safe_mixed_user_tool_result_messages = (
                _count_user_messages_with_mixed_tool_result_content(
                    fallback_body.get("messages", [])
                )
            )
            provider_safe_split_boundaries = (
                _count_user_tool_result_split_boundaries(
                    fallback_body.get("messages", [])
                )
            )
        retry_signals["fail_open_retry_prepared_forensics_logged"] = 1
        logger.warning(
            "bridge_pairing_forensics prepared_failures=%s prepared_pairing=%s prepared_summary=%s prepared_mixed_user_tool_result_messages=%s prepared_split_boundaries=%s provider_safe_failures=%s provider_safe_pairing=%s provider_safe_summary=%s provider_safe_mixed_user_tool_result_messages=%s provider_safe_split_boundaries=%s",
            prepared_failures,
            prepared_pairing,
            prepared_summary,
            prepared_mixed_user_tool_result_messages,
            prepared_split_boundaries,
            provider_safe_failures,
            provider_safe_pairing,
            provider_safe_summary,
            provider_safe_mixed_user_tool_result_messages,
            provider_safe_split_boundaries,
        )
        if (
            "`tool_use` ids were found without `tool_result` blocks immediately after"
            in error_text
            and not prepared_failures
        ):
            retry_signals["fail_open_retry_upstream_pairing_disagreement"] = 1
            if prepared_mixed_user_tool_result_messages > 0:
                retry_signals[
                    "fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present"
                ] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_mixed_user_message_present: prepared payload still contained mixed user tool_result+non-tool blocks"
                )
            elif prepared_split_boundaries > 0:
                retry_signals[
                    "fail_open_retry_upstream_pairing_disagreement_after_user_message_split"
                ] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_after_user_message_split: prepared payload had user tool_result/text split boundaries"
                )
            else:
                retry_signals[
                    "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message"
                ] = 1
                logger.warning(
                    "fail_open_retry_upstream_pairing_disagreement_without_mixed_user_message: no mixed user message or split boundary detected in prepared payload"
                )
            logger.warning(
                "fail_open_retry_upstream_pairing_disagreement: upstream reported pairing failure while local strict validation passed (prepared_mixed_user_tool_result_messages=%s prepared_split_boundaries=%s)",
                prepared_mixed_user_tool_result_messages,
                prepared_split_boundaries,
            )
            session.capture_event(
                {
                    "event": "fail_open_retry_upstream_pairing_disagreement",
                    "error_text": error_text[:500],
                    "prepared_summary": prepared_summary,
                    "prepared_pairing": prepared_pairing,
                    "provider_safe_summary": provider_safe_summary,
                    "provider_safe_pairing": provider_safe_pairing,
                    "behavior_signals": retry_signals,
                }
            )
        if fallback_content is not None and _payloads_materially_differ(
            content, fallback_content
        ):
            provider_safe_outgoing_failures = (
                validate_anthropic_outgoing_bridge_body(fallback_body)
                if isinstance(fallback_body, dict)
                else ["body_not_dict"]
            )
            if provider_safe_failures or provider_safe_outgoing_failures:
                logger.warning(
                    "fail_open_retry_provider_safe_invalid: refusing retry because provider-safe payload failed final local validation: strict=%s outgoing=%s",
                    provider_safe_failures,
                    provider_safe_outgoing_failures,
                )
                retry_signals["fail_open_retry_provider_safe_invalid"] = 1
                if has_provider_sensitive_failures(
                    provider_safe_outgoing_failures
                ):
                    retry_signals[
                        "fail_open_retry_provider_safe_blocked_local"
                    ] = 1
                    retry_signals[
                        "tok_bridge_provider_pairing_risk_detected"
                    ] = 1
                session.capture_event(
                    {
                        "event": "fail_open_retry_provider_safe_invalid",
                        "strict_failures": provider_safe_failures,
                        "outgoing_failures": provider_safe_outgoing_failures,
                        "prepared_summary": prepared_summary,
                        "prepared_pairing": prepared_pairing,
                        "provider_safe_summary": provider_safe_summary,
                        "provider_safe_pairing": provider_safe_pairing,
                        "behavior_signals": retry_signals,
                    }
                )
                return response, retried_without_tok, retry_signals
            retry_signals["fail_open_retry_provider_safe_validated"] = 1
            logger.info(
                "fail_open_retry_provider_safe_validated: provider-safe fallback passed local strict validation"
            )
            retry_kind = (
                "provider-safe" if retry_content is not None else "original"
            )
            logger.warning(
                "Upstream 400 after Tok request preparation: %s; retrying with %s payload",
                error_text[:500],
                retry_kind,
            )
            await response.aclose()
            request_obj = client.build_request(
                method, url, headers=headers, content=fallback_content
            )
            response = await client.send(request_obj, stream=stream)
            retried_without_tok = True
            if retry_content is not None:
                retry_signals["fail_open_retry_provider_safe"] = 1
            if not allow_original_retry:
                retry_signals["fail_open_raw_retry_blocked"] = 1
        elif not allow_original_retry and original_content is not None:
            logger.warning(
                "fail_open_raw_retry_blocked: refusing to resend raw original payload after tool-history repair"
            )
            retry_signals["fail_open_raw_retry_blocked"] = 1
    return response, retried_without_tok, retry_signals
