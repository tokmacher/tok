"""
Tok agent runner — live proof of concept.

Inverted design: Tok tool calls live IN the text stream, not in a special API
field. Any model that can generate text can call tools — no JSON function-calling
support needed. The parser detects @UpperCase blocks and dispatches them
immediately as they stream.

Usage:
    Add OPENROUTER_API_KEY=... to a .env file in this directory, then:
    python agent.py
    python agent.py "What's the weather in Tokyo and London?"
"""

import argparse
import json
import os
import sys
import time
from typing import Any, cast
from collections.abc import Callable

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .adapters import TextLoopAdapter
from ..protocol.models import TokNode
from ..protocol.parser import TokParser, serialize
from ..prompt import TOK_SYSTEM_PROMPT
from ..protocol.schema import DEFAULT_SCHEMA


class Agent:
    """Base class for Tok agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    def respond(self, message: str) -> str:
        raise NotImplementedError


load_dotenv()

# ── Client ────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=60.0,
    max_retries=0,
)

MODEL = "arcee-ai/trinity-large-preview:free"
ENCODER = tiktoken.get_encoding("cl100k_base")
TEXT_LOOP_ADAPTER = TextLoopAdapter()


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


# ── Mock tools ────────────────────────────────────────────────────────────────


def get_weather(location: str, days: int = 3, **kwargs: Any) -> list[TokNode]:
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
        rows=[
            [city or location, high, low, cond]
            for high, low, cond in rows_data
        ],
    )
    return [result]


TOOLS: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
}


def dispatch(node: TokNode) -> TokNode | None:
    """Execute an @UpperCase (execution) block. Returns a result node or None."""
    if not node.type or not node.type[0].isupper():
        return None
    fn = TOOLS.get(node.label)
    if not fn:
        return None
    try:
        result = fn(**node.attrs)
    except Exception:
        return None
    return cast("TokNode | None", result)


# ── Comparison Functions ──────────────────────────────────────────────────────


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

    print(f"\n[Comparison] Fetching {format_name} response...")
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
        toks = count_tokens(text)
        print(f"[{format_name}] Tokens: {toks}")
        return toks
    except Exception as e:
        print(f"[{format_name}] Failed: {e}")
        return 0


# ── Agent loop ────────────────────────────────────────────────────────────────


def run(user_input: str, compare: bool = False) -> None:
    """Multi-turn agent loop with Tok's 'Active Interruption' logic."""

    print(f"\n{'═' * 60}")
    print(f"  User: {user_input}")
    print(f"{'═' * 60}\n")

    legacy_toks = {}
    if compare:
        for fmt in ["JSON", "XML"]:
            legacy_toks[fmt] = run_legacy_turn(user_input, fmt)

    # Wrap user input in a sandboxed @msg block
    user_tok = f"@msg role:user trust:untrusted\n  |> {user_input}"

    conversation_messages = [{"role": "user", "content": user_tok}]

    while True:
        print("\n── Tok response (streaming) ──")
        parser = TokParser()
        messages, prepared = TEXT_LOOP_ADAPTER.prepare_messages(
            model=MODEL,
            messages=conversation_messages,
            system_prompt=TOK_SYSTEM_PROMPT,
        )

        start_time = time.time()
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=cast(Any, messages),
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
        except Exception as e:
            print(f"\n[error] LLM call failed: {e}")
            break

        tok_count = _finalize_tok_turn(
            response_text, prepared, start_time, tta
        )
        _print_comparison(compare, legacy_toks, tok_count)
        if _should_exit_turn(tool_results, interrupted, response_text):
            break
        _append_turn_messages(
            conversation_messages, response_text, tool_results
        )

    print("\n── Task Complete ──\n")


def _finalize_tok_turn(
    response_text: str,
    prepared: Any,
    start_time: float,
    tta: float | None,
) -> int:
    end_time = time.time()
    print()
    if response_text:
        TEXT_LOOP_ADAPTER.finalize(
            text=response_text,
            model=MODEL,
            behavior_signals=prepared.behavior_signals,
        )
    tok_count = count_tokens(response_text)
    duration = end_time - start_time
    tta_str = f"{tta:.2f}s" if tta else "N/A"
    print(
        f"\n[Tok] Tokens: {tok_count} | Time: {duration:.2f}s | TTA: {tta_str}"
    )
    return tok_count


def _print_comparison(
    compare: bool,
    legacy_toks: dict[str, int],
    tok_count: int,
) -> None:
    if not compare:
        return
    print("\nLIVE COMPARISON (This Turn):")
    for fmt, t in legacy_toks.items():
        savings = (t - tok_count) / t * 100 if t > 0 else 0
        print(
            f" {fmt}: {t} tokens vs Tok: {tok_count} ({savings:.1f}% savings)"
        )


def _should_exit_turn(
    tool_results: list[TokNode], interrupted: bool, response_text: str
) -> bool:
    return not tool_results and (not interrupted or "@msg" in response_text)


def _append_turn_messages(
    conversation_messages: list[dict[str, Any]],
    response_text: str,
    tool_results: list[TokNode],
) -> None:
    conversation_messages.append(
        {"role": "assistant", "content": response_text}
    )
    if tool_results:
        error_nodes = [n for n in tool_results if n.type == "error"]
        actual_results = [n for n in tool_results if n.type != "error"]
        if error_nodes:
            errs = json.dumps([n.text for n in error_nodes])
            msg = (
                f"SYNTAX ERROR(S) detected in your last response:\n"
                f"{errs}\n\nPlease fix the attributes and try again."
            )
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
        conversation_messages.append(
            {"role": "user", "content": "Please continue."}
        )


def _extract_chunk_delta(chunk: Any) -> str:
    if not hasattr(chunk, "choices") or not cast(Any, chunk).choices:
        return ""
    return cast(Any, chunk).choices[0].delta.content or ""


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
) -> tuple[list[TokNode], float | None, bool]:
    is_valid, err_msg = DEFAULT_SCHEMA.validate(node)
    if not is_valid:
        if (
            err_msg is not None
            and "Missing required attribute" in err_msg
            and len(response_text) < 100
        ):
            return tool_results, tta, False

        tta = _record_tta(tta, start_time)
        msg = f"\n  [agent] [tok-interrupt] syntax error: {err_msg}"
        print(msg)
        response_text += f"\n[SYNTAX ERROR] {err_msg}"
        tool_results.append(
            TokNode(type="error", text=err_msg or "unknown error")
        )
        return tool_results, tta, True

    if node.type.lower() == "result":
        return tool_results, tta, False

    result = dispatch(node)
    if result:
        tta = _record_tta(tta, start_time)
        tool_results.append(result)

    if node.type[0].isupper():
        tta = _record_tta(tta, start_time)
        print("\n  [agent] [tok-interrupt] interrupting stream...")
        return tool_results, tta, True

    return tool_results, tta, False


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
        msg = f"\n  [agent] [tok-interrupt] instant error: {err_msg}"
        print(msg)
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
            print(f"\n  [agent] syntax error: {err_msg}")
            response_text += f"\n[SYNTAX ERROR] {err_msg}"
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
            tool_results.append(result)

    return tool_results, tta


def _collect_tok_stream(
    stream: Any, parser: TokParser, start_time: float
) -> tuple[str, list[TokNode], bool, float | None]:
    response_text = ""
    tool_results: list[TokNode] = []
    interrupted = False
    tta: float | None = None

    for chunk in stream:
        delta = _extract_chunk_delta(chunk)
        if not delta:
            continue
        print(delta, end="", flush=True)
        response_text += delta

        for node in parser.feed(delta):
            tool_results, tta, node_interrupted = _handle_stream_node(
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
        if interrupted:
            break

        tool_results, tta = _process_flushed_nodes(
            parser, response_text, tool_results, start_time, tta
        )

    return response_text, tool_results, interrupted, tta


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="?", default="What's the weather forecast for Tokyo?"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with JSON/XML"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY missing.")
        sys.exit(1)

    run(args.prompt, compare=args.compare)
