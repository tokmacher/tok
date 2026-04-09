"""
Tok agent runner — live proof of concept.

Inverted design: Tok tool calls live IN the text stream, not in a special
API field. Any model that can generate text can call tools — no JSON
function-calling support needed. The parser detects @UpperCase blocks and
dispatches them immediately as they stream.

Usage:
    Add OPENROUTER_API_KEY=... to a .env file in this directory, then:
    python agent.py
    python agent.py "What's the weather in Tokyo and London?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from tok.analysis.prompt import TOK_SYSTEM_PROMPT
from tok.protocol.models import TokNode
from tok.protocol.parser import TokParser, serialize
from tok.protocol.schema import DEFAULT_SCHEMA
from tok.utils.token_utils import count_tokens

from .adapters import TextLoopAdapter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tok.runtime.types import PreparedRuntimeRequest

# Threshold for short response detection - suppresses syntax error messages for brief responses
SHORT_RESPONSE_LIMIT = 100

logger = logging.getLogger(__name__)


class Agent:
    """Base class for Tok agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    def respond(self, message: str) -> str:
        """Generate a response to the given message."""
        raise NotImplementedError


load_dotenv()

client: OpenAI | None = None

MODEL = "openai/gpt-4.1-mini"

TEXT_LOOP_ADAPTER = TextLoopAdapter()


def get_weather(location: str, days: int = 3, **_kwargs: object) -> list[TokNode]:
    """Mock weather tool."""
    mock = {
        "San Francisco": [
            (65, 52, "partly cloudy"),
            (68, 55, "sunny"),
            (60, 50, "foggy"),
        ],
        "Tokyo": [(72, 61, "clear"), (75, 63, "sunny"), (69, 58, "cloudy")],
        "London": [
            (55, 48, "overcast"),
            (52, 46, "rainy"),
            (57, 50, "cloudy"),
        ],
        "New York": [
            (70, 58, "sunny"),
            (66, 55, "windy"),
            (63, 52, "overcast"),
        ],
    }
    city = next((k for k in mock if k.lower() in location.lower()), None)
    if city is None:
        city = "San Francisco"
    rows_data = mock.get(city, [(68, 54, "clear")] * days)[:days]

    result = TokNode(
        type="result",
        headers=["city", "high", "low", "condition"],
        rows=[[city or location, high, low, cond] for high, low, cond in rows_data],
    )
    return [result]


TOOLS: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
}


def dispatch(node: TokNode) -> TokNode | list[TokNode] | None:
    """Execute an @UpperCase (execution) block. Returns a result node, list of nodes, or None."""
    if not node.type or not node.type[0].isupper():
        return None
    fn = TOOLS.get(node.label)
    if not fn:
        return None
    try:
        result = fn(**node.attrs)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.debug("Tool dispatch failed: %s", e, exc_info=True)
        return None
    return cast("TokNode | list[TokNode] | None", result)


def run_legacy_turn(prompt: str, format_name: str) -> int:
    """Runs a single turn in a legacy format (JSON/XML) to measure tokens."""
    system_prompts = {
        "JSON": (
            "You are a helpful assistant. You MUST respond in valid JSON format. "
            "If you want to use a tool, use 'thought' and 'tool_call' fields."
        ),
        "XML": (
            "You are a helpful assistant. You MUST respond in valid XML. "
            "Wrap thoughts in <thought> and tool calls in <tool> tags."
        ),
    }

    if client is None:
        return 0
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompts[format_name]},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        text = completion.choices[0].message.content
        if text is None:
            return 0
        return count_tokens(text)
    except Exception:
        logger.debug("Legacy turn failed", exc_info=True)
        return 0


def run(user_input: str, *, compare: bool = False) -> None:
    """Multi-turn agent loop with Tok's 'Active Interruption' logic."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        return
    local_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )
    legacy_toks = {}
    if compare:
        for fmt in ["JSON", "XML"]:
            legacy_toks[fmt] = run_legacy_turn(user_input, fmt)

    # Wrap user input in a sandboxed @msg block
    user_tok = f"@msg role:user trust:untrusted\n  |> {user_input}"

    conversation_messages = [{"role": "user", "content": user_tok}]

    while True:
        parser = TokParser()
        messages, prepared = TEXT_LOOP_ADAPTER.prepare_messages(
            model=MODEL,
            messages=conversation_messages,
            system_prompt=TOK_SYSTEM_PROMPT,
        )

        start_time = time.time()
        try:
            stream = local_client.chat.completions.create(
                model=MODEL,
                messages=cast("Any", messages),
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://github.com/tok-lang/tok",
                    "X-Title": "Tok Agent",
                },
                extra_body={"provider": {"allow_fallbacks": True}},
            )
            (
                response_text,
                tool_results,
                interrupted,
                tta,
            ) = _collect_tok_stream(stream, parser, start_time)
        except Exception:
            logger.debug("Agent run failed", exc_info=True)
            break

        tok_count = _finalize_tok_turn(response_text, prepared, start_time, tta)
        _print_comparison(compare, legacy_toks, tok_count)
        if _should_exit_turn(tool_results, interrupted, response_text):
            break
        _append_turn_messages(conversation_messages, response_text, tool_results)


def _finalize_tok_turn(
    response_text: str,
    prepared: PreparedRuntimeRequest,
    _start_time: float,
    _tta: float | None,
) -> int:
    typed_prepared = prepared
    if response_text:
        TEXT_LOOP_ADAPTER.finalize(
            text=response_text,
            model=MODEL,
            behavior_signals=typed_prepared.behavior_signals,
        )
    tok_count = count_tokens(response_text)
    return tok_count


def _print_comparison(
    compare: bool,
    legacy_toks: dict[str, int],
    tok_count: int,
) -> None:
    if not compare:
        return
    legacy_total = legacy_toks.get("JSON", 0) + legacy_toks.get("XML", 0)
    diff = legacy_total - tok_count
    pct = (diff / legacy_total * 100) if legacy_total > 0 else 0.0
    print(f"[comparison] legacy={legacy_total} tok={tok_count} saved={diff} ({pct:.1f}%)")


def _should_exit_turn(tool_results: list[TokNode], interrupted: bool, response_text: str) -> bool:
    return not tool_results and (not interrupted or "@msg" in response_text)


def _append_turn_messages(
    conversation_messages: list[dict[str, Any]],
    response_text: str,
    tool_results: list[TokNode],
) -> None:
    conversation_messages.append({"role": "assistant", "content": response_text})
    if tool_results:
        error_nodes = [n for n in tool_results if n.type == "error"]
        actual_results = [n for n in tool_results if n.type != "error"]
        if error_nodes:
            errs = json.dumps([n.text for n in error_nodes])
            msg = f"SYNTAX ERROR(S) detected in your last response:\n{errs}\n\nPlease fix the attributes and try again."
            conversation_messages.append({"role": "user", "content": msg})
        elif actual_results:
            results_tok = serialize(actual_results)
            conversation_messages.append(
                {
                    "role": "user",
                    "content": f"Tool results:\n\n{results_tok}",
                }
            )
    else:
        conversation_messages.append({"role": "user", "content": "Please continue."})


def _extract_chunk_delta(chunk: object) -> str:
    if not hasattr(chunk, "choices") or not cast("Any", chunk).choices:
        return ""
    return cast("Any", chunk).choices[0].delta.content or ""


def _record_tta(tta: float | None, start_time: float) -> float:
    if tta is None:
        return time.time() - start_time
    return tta


def _handle_stream_node(
    node: TokNode,
    response_text: str,
    tool_results: list[TokNode],
    start_time: float,
    tta: float | None,
) -> tuple[str, list[TokNode], float | None, bool]:
    is_valid, err_msg = DEFAULT_SCHEMA.validate(node)
    if not is_valid:
        if (
            err_msg is not None
            and "Missing required attribute" in err_msg
            and len(response_text) < SHORT_RESPONSE_LIMIT
        ):
            return response_text, tool_results, tta, False

        tta = _record_tta(tta, start_time)
        response_text += f"\n[SYNTAX ERROR] {err_msg}"
        tool_results.append(TokNode(type="error", text=err_msg or "unknown error"))
        return response_text, tool_results, tta, True

    if node.type.lower() == "result":
        return response_text, tool_results, tta, False

    result = dispatch(node)
    if result:
        tta = _record_tta(tta, start_time)
        if isinstance(result, list):
            tool_results.extend(result)
        else:
            tool_results.append(result)

    if node.type and node.type[0].isupper():
        tta = _record_tta(tta, start_time)
        return response_text, tool_results, tta, True

    return response_text, tool_results, tta, False


def _handle_active_node_validation(
    parser: TokParser,
    response_text: str,
    tool_results: list[TokNode],
    start_time: float,
    tta: float | None,
) -> tuple[str, list[TokNode], float | None]:
    active_node = parser.current_node
    if not active_node:
        return response_text, tool_results, tta

    is_valid, err_msg = DEFAULT_SCHEMA.validate(active_node)
    if not is_valid and err_msg and "Unknown attribute" in err_msg:
        tta = _record_tta(tta, start_time)
        response_text += f"\n[SYNTAX ERROR] {err_msg}"
        tool_results.append(TokNode(type="error", text=err_msg))

    return response_text, tool_results, tta


def _process_flushed_nodes(
    parser: TokParser,
    _response_text: str,
    tool_results: list[TokNode],
    start_time: float,
    tta: float | None,
) -> tuple[list[TokNode], float | None]:
    for node in parser.flush():
        is_valid, err_msg = DEFAULT_SCHEMA.validate(node)
        if not is_valid:
            tool_results.append(
                TokNode(
                    type="error",
                    text=err_msg or "unknown error",
                )
            )
            continue

        if node.type.lower() == "result":
            continue
        result = dispatch(node)
        if result:
            tta = _record_tta(tta, start_time)
            if isinstance(result, list):
                tool_results.extend(result)
            else:
                tool_results.append(result)

    return tool_results, tta


def _collect_tok_stream(
    stream: object, parser: TokParser, start_time: float
) -> tuple[str, list[TokNode], bool, float | None]:
    response_text = ""
    tool_results: list[TokNode] = []
    interrupted = False
    tta: float | None = None

    for chunk in cast("Iterator[object]", stream):
        delta = _extract_chunk_delta(chunk)
        if not delta:
            continue
        response_text += delta

        for node in parser.feed(delta):
            response_text, tool_results, tta, node_interrupted = _handle_stream_node(
                node, response_text, tool_results, start_time, tta
            )
            if node_interrupted:
                interrupted = True
                break

        if interrupted:
            break

        response_text, tool_results, tta = _handle_active_node_validation(
            parser, response_text, tool_results, start_time, tta
        )

    return response_text, tool_results, interrupted, tta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What's the weather forecast for Tokyo?")
    parser.add_argument("--compare", action="store_true", help="Compare with JSON/XML")
    args = parser.parse_args()

    run(args.prompt, compare=args.compare)
