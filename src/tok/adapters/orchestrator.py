from __future__ import annotations
import atexit
import concurrent.futures
import os
import re
import subprocess
from typing import Any, cast, TYPE_CHECKING
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .adapters import OrchestratorAdapter
from ..monitoring.profiler import TokProfiler
from ..prompt import MINIMAL_PULSE_PROMPT
from ..protocol import Bridge, TokParser  # noqa: F401
from ..stats import SavingsTracker

if TYPE_CHECKING:
    from ..utils.delta import (
        TokDeltaTracker,
        delta_to_tok,
        diff_tok,
        format_compact_delta,
    )
else:
    try:
        from ..utils.delta import (
            TokDeltaTracker,
            delta_to_tok,
            diff_tok,
            format_compact_delta,
        )
    except ImportError:
        TokDeltaTracker = None
        diff_tok = None
        delta_to_tok = None
        format_compact_delta = None

# TokRegistry is optional - gracefully handle if missing
TokRegistry: type[Any] | None = None
try:
    from ..utils.tok_registry import TokRegistry  # noqa: F401
except ImportError:
    pass


# Fallback pricing (Gemini 2.0 Flash Lite actuals)
FALLBACK_PRICING = {
    "prompt": 0.000075,  # $0.075/M
    "completion": 0.00030,  # $0.30/M
}


def fetch_model_pricing(client: OpenAI, model: str) -> tuple[float, float]:
    """Fetch pricing from OpenRouter API for a given model."""
    try:
        # Get model info from OpenRouter
        response = client.models.retrieve(model)
        # Access pricing from response attributes
        pricing_data = getattr(response, "pricing", None) or {}
        if pricing_data:
            prompt_price = float(pricing_data.get("prompt", 0) or 0)
            completion_price = float(pricing_data.get("completion", 0) or 0)
            return prompt_price, completion_price
    except Exception as e:
        print(f"[!] Could not fetch pricing for {model}: {e}")

    return FALLBACK_PRICING["prompt"], FALLBACK_PRICING["completion"]


TOK_HEARTBEAT = """
@protocol-pulse
  @Tool write path:"src/foo.py"
    |> # ALWAYS INVERT: Use the node body for code.
       def example():
           pass
  @Tool edit path:"..."
    @search |> ...
    @replace |> ...
  >>> t:N|usr:intent|agt:action|state:change
  INVERSION IS STABILITY.
"""


class TokOrchestrator:
    """
    The core orchestrator for the TokMemory protocol.
    Implements O(0) memory through cumulative Delta merging and autonomous tool-use.
    """

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-lite-001",
        max_retries: int = 3,
        strict_mode: bool = True,
        app_name: str | None = None,
        entropy_budget: int | None = None,
        compute_budget: int | None = None,
    ) -> None:
        """
        Initialize the orchestrator, loading environment variables and the OpenAI client.
        """
        load_dotenv()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.adapter = OrchestratorAdapter()
        self.tracker = SavingsTracker()
        self.model = model
        self.profiler = TokProfiler(model_name=model)
        self.max_retries = max_retries
        self.strict_mode = strict_mode
        self.app_name = app_name
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.todo_path = "todo.tok"

        # Delta tracking moved to runtime_tools.py
        self._pre_state: dict[str, str] = {}
        self._pending_deltas: list[Any] = []
        self.track_file_changes = True

        # Fetch pricing dynamically from OpenRouter (with timeout to avoid blocking init)
        (
            self.prompt_price,
            self.completion_price,
        ) = self._fetch_pricing_with_timeout(model)
        print(
            f"[*] Pricing for {model}: ${self.prompt_price}/1K prompt, ${self.completion_price}/1K completion"
        )

        self.system_prompt = """
        You are an autonomous coding agent using TOK-PROTOCOL v5.8 (Territory-Aware).

        SEMANTIC DELTA MODE:
        - Instead of full file reads, you receive @delta blocks describing changes
        - @delta: Structural change (add/remove/update of functions/classes)
        - @impact_context: Ripple effects (e.g., broken callers requiring updates)
        - Use @delta to update permanent_state surgically
        - Example:
          @Tool delta path:"src/utils.py"
            |> add @func|new_feature
                 +params: x, y

        REFERENCE:
        - Read agent.tok for task instructions
        - Read technical.tok for tech standards and library recommendations
        - Read TDD_GUIDE.tok for Red-Green-Refactor workflow

        PRINCIPLE (Dual-Speed Memory):
        1. WORLD STATE (`memory.tok`): Long-term discoveries. Managed via `>>> DELTA`.
        2. TASK QUEUE (`todo.tok`): Short-term scratchpad. Managed surgically.

        MISSION: Audit and refactor the Tok codebase.

        PRIMARY DIRECTIVE:
        1. Initialize: `@Tool write path:todo.tok text:"[ ] audit_codebase; [ ] fix_imports; [ ] verify_tests"`
        2. Audit: Use `@Tool run cmd:"ls src/tok/"` to understand structure.
        3. Persist: Use `>>>` to record findings in memory.tok immediately.
        4. TDD (Red-Green-Refactor):
           - RED: Create a failing test file (e.g., `tests/test_feature.py`) first.
           - GREEN: Implement minimal code to pass the test.
           - REFACTOR: Optimize while maintaining green status.
           - MISSION: No change without a test.

        GO.

        TOOLCHAIN:
        - Use `uv run` instead of `python` for scripts
        - Use `uv add <package>` to add dependencies
        - Use `uv pip install` for packages
        - NEVER use pip directly

        TERRITORY MAP (Codebase Navigation):
        - `todo.tok`: Current task queue and progress tracking.
        - `corpus.tok`: Detailed code bodies (use @Tool read to dive deep)

        DOCUMENTATION:
        - Prefer `.tok` versions of docs when available (e.g., README.tok, TOK_GRAMMAR_GUIDE.tok)
        - Use @Tool read to access documentation
        - Read TOK_GRAMMAR_GUIDE.tok for correct tool syntax

        STRICT TOOL SYNTAX (@Tool ONLY):
        - @Tool read path:"file.txt"
        - @Tool write path:"file.txt" text:"content"
        - @Tool edit path:"file.txt" search:"old" replace:"new"
        - @Tool run cmd:"command"
        - UNLOCK: Truncation limits removed.
        - WARNING: Legacy $WRITE/$READ syntax is DEPRECATED and will cause a SyntaxError.

        PROTOCOL:
        1. Read <<< WORLD STATE & TODO & TERRITORY.
        2. Consult `skeleton.tok` or `todo.tok` BEFORE reading large files like `corpus.tok`.
        3. Respond to user.
        4. MEMORY INVERSION:
           - Facts not recorded in `memory.tok` via `>>> DELTA` are purged from context.
           - SIGNAL is sovereign; history is noise.
           - Context window is strictly 4 messages. Keep your responses O(0).
        5. ACTIONS (Tools): Always use @Tool syntax.

            IMPORTANT: Output tools as PLAIN TEXT, NOT Markdown. Do NOT wrap in **bold**, ```code blocks```, or `backticks`.
            - Tool results are provided in the NEXT turn. Do NOT repeat the same call in one turn.
        6. MEMORY IS SOVEREIGN - MUST PERSIST (SENTINEL FILTER):
            - CRITICAL: After EVERY response, you MUST end with a SINGLE LINE starting with `>>>`.
            - NEVER write task output to stdout - it is LOST. Only memory.tok persists.
            - NOISE: Output `>>> [SAME]` if no state change.
            - SIGNAL: Output `>>> DELTA: key:value;` to append discoveries.
            - FORMAT: `>>> t:N|usr:intent|agt:action|state:change`
            - Example: `>>> t:6|usr:audit|agt:write_memory|state:protocol_enforced`
            - STRICT ADHERENCE: All summary text must be indented with `|>`.

        PROTOCOL DISCIPLINE:
        - Audit -> Delta -> Verify.
        - Use @Tool delta for structural changes (adding/removing functions).
        - Use @Tool edit for local line-level fixes.
        - NEVER wrap code in docstrings.
        """

        # Minimal prompt for after handshake - rely on memory
        self.minimal_prompt = self._load_system_prompt()

        self.turn_count = 0
        self.handshake_done = False
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.OPENROUTER_DISCOUNT = 0.01  # 1% flat discount
        self.MAX_MESSAGES = 10
        self.MAX_CONTEXT_TOKENS = (
            4096  # Extreme Inversion Test (forced memory reliance)
        )
        # 32k hard limit for sovereign context
        self.entropy_budget = (
            entropy_budget if entropy_budget is not None else 40
        )  # Max tool calls per session
        self.compute_budget = (
            compute_budget if compute_budget is not None else 20
        )  # Max compute-intensive operations
        self.MAX_PROMPT_TOKENS = 8192  # Total budget for all content
        self.workspace_root = os.getcwd()
        self.log_path = "execution.log"
        self.max_tokens = 4096  # Production default
        self.last_tool_sig: str | None = None
        self.consecutive_repeats = 0
        self.pulse_count = 0
        self.heartbeat_interval = 50
        atexit.register(self.tracker.merge_session_to_ledger)

    def _fetch_pricing_with_timeout(
        self, model: str, timeout: float = 2.0
    ) -> tuple[float, float]:
        """Fetch model pricing with a timeout to avoid blocking __init__ on slow networks."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fetch_model_pricing, self.client, model)
            try:
                return future.result(timeout=timeout)
            except (concurrent.futures.TimeoutError, Exception):
                return (
                    FALLBACK_PRICING["prompt"],
                    FALLBACK_PRICING["completion"],
                )

    def _get_pulse_prompt(self) -> str:
        """Return the appropriate instruction set (Full Heartbeat or Minimal Pulse)."""
        if self.pulse_count == 0:
            return self.system_prompt

        is_heartbeat = self.pulse_count % self.heartbeat_interval == 0
        if is_heartbeat:
            return self.minimal_prompt

        return MINIMAL_PULSE_PROMPT

    def _is_safe_path(self, path: str | Path) -> bool:
        """Delegate to runtime tools - kept for backward compatibility."""
        from ..runtime.tools import get_default_executor

        return get_default_executor()._is_safe_path(str(path))

    def _is_safe_rm(self, cmd: str) -> bool:
        """Delegate to runtime tools - kept for backward compatibility."""
        from ..runtime.tools import get_default_executor

        return get_default_executor()._is_safe_rm(cmd)

    def _load_system_prompt(self) -> str:
        """Load system prompt from file or use default."""
        if os.path.exists("system_prompt.tok"):
            with open("system_prompt.tok") as f:
                return f.read().strip()
        return """@agent mode:autonomous
@memory memory.tok todo.tok
@tools @Tool
@protocol >>> t:N|usr:intent|agt:action|state:change
"""

    def _get_active_memory_context(self) -> str:
        """Use adapter's memory system instead of separate orchestrator memory."""
        return self.adapter.session.load_memory()

    def _load_grammar(self) -> str:
        """Load immutable grammar from grammar.tok - always prepended to context."""
        if os.path.exists("grammar.tok"):
            with open("grammar.tok") as f:
                return f.read().strip()
        return ""

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a given text."""
        return len(self.enc.encode(text))

    def _regenerate_territory(self) -> None:
        """Regenerate the territory.tok map to stay in sync with filesystem."""
        try:
            # Shadowing fix: ensure we don't use root tok.py if it exists
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
            )

            # Temporary rename if tok.py exists to avoid shadowing
            shadowed = os.path.exists("tok.py")
            if shadowed:
                os.rename("tok.py", "tok.py.tmp")

            cmd = "uv run python -c \"from tok.sifter import Sifter; s = Sifter.from_dir('src/tok', naked=False, minify=True); open('territory.tok', 'w').write(s)\""
            proc = subprocess.run(
                cmd, shell=True, env=env, capture_output=True, text=True
            )
            if proc.returncode != 0:
                print(f"[!] Territory Sync Failed: {proc.stderr}")
            else:
                print("[!] Territory regenerated due to file changes.")
        except Exception as e:
            print(f"[!] Warning: Territory regeneration failed: {e}")
        finally:
            if shadowed:
                os.rename("tok.py.tmp", "tok.py")

    def _log_execution(
        self, cmd: str, stdout: str, stderr: str, returncode: int
    ) -> None:
        """Log command execution details to a file."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{'=' * 20}\n[{timestamp}] COMMAND: {cmd}\nEXIT CODE: {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n{'=' * 20}\n"

        # Simple rotation: keep the log file under a reasonable size (e.g., 500KB)
        try:
            if (
                os.path.exists(self.log_path)
                and os.path.getsize(self.log_path) > 500 * 1024
            ):
                with open(self.log_path) as f:
                    lines = f.readlines()
                # Keep last 1000 lines
                with open(self.log_path, "w") as f:
                    f.writelines(lines[-1000:])
        except Exception:
            pass

        with open(self.log_path, "a") as f:
            f.write(log_entry)

    def _extract_compression(self, response_text: str) -> str | None:
        """Extract memory delta from response - checks >>> lines and @state blocks."""

        # First, try to find >>> line (protocol format)
        try:
            for line in response_text.split("\n"):
                if line.strip().startswith(">>>"):
                    return line.strip()
        except Exception:
            pass

        # Fallback: Extract @state block content
        state_match = re.search(
            r"@state\s*\n(.*?)(?=\n@|\n$|$)", response_text, re.DOTALL
        )
        if state_match:
            state_content = state_match.group(1).strip()
            if state_content:
                return f">>> {state_content}"

        return None

    def _extract_state_from_response(self, response_text: str) -> str | None:
        """Extract @state block content from response for memory update."""

        state_match = re.search(
            r"@state\s*\n(.*?)(?=\n@|\n>>>|\n\n|$)", response_text, re.DOTALL
        )
        if state_match:
            return state_match.group(1).strip()
        return None

    def _is_valid_tok(self, text: str) -> bool:
        """Return True if text contains at least one valid Tok construct (lax for Lite models)."""
        # Allow leading text as long as a valid tag or memory marker exists
        return bool(
            re.search(r"(@[A-Za-z_][A-Za-z0-9_]*|>>>|\s+\|>)", text, re.DOTALL)
        )

    def _calc_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the USD cost based on fetched API pricing."""
        raw_cost = (
            prompt_tokens * self.prompt_price
            + completion_tokens * self.completion_price
        ) / 1_000_000
        return raw_cost * (1 - self.OPENROUTER_DISCOUNT)

    def chat(
        self,
        user_input: str,
        use_system_prompt: bool = True,
        verbose: bool = True,
    ) -> str:
        """
        Execute one or more turns of the Tok protocol:
        Load memory -> Prompt LLM -> Handle Tools -> (Recurse if results exist) -> Persist.
        """
        self.entropy_budget = 20
        self.consecutive_repeats = 0
        self.last_tool_sig = None

        grammar = self._load_grammar()
        todo_list = ""
        if os.path.exists(self.todo_path):
            with open(self.todo_path) as f:
                todo_list = f.read().strip()

        messages = [{"role": "user", "content": user_input}]
        final_agent_reply = ""

        while True:
            delta_block = self._build_delta_block()
            chat_messages, prepared = self._prepare_turn(
                messages,
                grammar,
                todo_list,
                delta_block,
            )

            response, error = self._call_llm_once(chat_messages)
            if response is None:
                return f"[ERROR] API failed: {error}"

            response_text = response.choices[0].message.content or ""
            final_agent_reply = response_text
            self.pulse_count += 1
            print(f"[*] Raw Response length: {len(response_text)} chars")
            if verbose:
                preview = response_text[:200].replace("\n", " ")
                print(f"    > {preview}...")

            processed = self.adapter.finalize(
                text=response_text,
                model=self.model,
                behavior_signals=prepared.behavior_signals,
            )
            response_signals = dict(processed.behavior_signals or {})

            usage = getattr(response, "usage", None)
            prompt_tokens, completion_tokens = self._get_usage_tokens(
                usage, chat_messages, response_text
            )
            self._record_turn_stats(
                prompt_tokens,
                completion_tokens,
                prepared,
                processed,
                response_signals,
            )

            tool_results = self._execute_tool_blocks(processed)
            tool_feedback = "\n".join(tool_results).strip()

            if (
                hasattr(processed, "updated_memory")
                and processed.updated_memory
            ):
                print(
                    f"[*] Memory updated: {len(processed.updated_memory)} chars"
                )

            messages.append({"role": "assistant", "content": response_text})

            if self._should_chain(tool_feedback):
                self._chain_tool_feedback(
                    messages, response_text, tool_feedback
                )
                continue
            return final_agent_reply

    def _build_delta_block(self) -> str:
        if not self._pending_deltas:
            return ""
        from ..utils.delta import delta_to_tok

        return delta_to_tok(self._pending_deltas[:10])

    def _prepare_turn(
        self,
        messages: list[dict[str, Any]],
        grammar: str,
        todo_list: str,
        delta_block: str,
    ) -> tuple[list[dict[str, Any]], Any]:
        return self.adapter.prepare_turn(
            model=self.model,
            system_prompt=self._get_pulse_prompt(),
            dynamic_messages=messages,
            grammar=grammar,
            todo=todo_list,
            deltas=delta_block,
        )

    def _call_llm_once(
        self, chat_messages: list[dict[str, Any]]
    ) -> tuple[Any | None, str | None]:
        print(f"[*] Calling {self.model} (Turn {self.turn_count})...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, chat_messages),
                temperature=0.0,
                max_tokens=2048,
            )
            return response, None
        except Exception as e:
            print(f"[!] API Error: {e}")
            return None, str(e)

    def _get_usage_tokens(
        self,
        usage: Any | None,
        chat_messages: list[dict[str, Any]],
        response_text: str,
    ) -> tuple[int, int]:
        prompt_tokens = (
            int(getattr(usage, "prompt_tokens", 0))
            if usage is not None
            else self.count_tokens(str(chat_messages))
        )
        completion_tokens = (
            int(getattr(usage, "completion_tokens", 0))
            if usage is not None
            else self.count_tokens(response_text)
        )
        return prompt_tokens, completion_tokens

    def _record_turn_stats(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        prepared: Any,
        processed: Any,
        response_signals: dict[str, int],
    ) -> None:
        turn_cost = self._calc_cost(prompt_tokens, completion_tokens)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += turn_cost
        self.tracker.record_call(
            model=self.model,
            actual_input=prompt_tokens,
            actual_output=completion_tokens,
            cache_read=0,
            cache_write=0,
            input_saved=prepared.input_saved_tokens,
            output_saved=processed.output_saved_tokens,
            type_breakdown=prepared.type_breakdown,
            behavior_signals=response_signals or None,
        )

    def _execute_tool_blocks(self, processed: Any) -> list[str]:
        from ..runtime.tools import get_default_executor
        from ..runtime.types import NormalizedToolEvent

        executor = get_default_executor()
        tool_results: list[str] = []

        for block in processed.content_blocks:
            if block.get("type") != "tool_use":
                continue
            name = cast(str, block.get("name"))
            args = cast(dict[str, Any], block.get("input", {}))

            sig = f"{name}:{sorted(args.items())}"
            if sig == self.last_tool_sig:
                self.consecutive_repeats += 1
            else:
                self.last_tool_sig = sig
                self.consecutive_repeats = 1

            if self.consecutive_repeats >= 3:
                tool_results.append(
                    f"@error type:gluttony\n"
                    f"  msg: Infinite_loop_detected_on_{name}.\n"
                    "  fix: Move_to_next_TODO_item."
                )
                self.entropy_budget -= 1
                continue

            if self.entropy_budget <= 0:
                tool_results.append("[ERROR] Entropy budget exhausted")
                continue
            if name in ("run", "edit") and self.compute_budget <= 0:
                tool_results.append("[ERROR] Compute budget exhausted")
                continue

            self.entropy_budget -= 1
            if name in ("run", "edit"):
                self.compute_budget -= 1

            path = args.get("path") or args.get("file")
            command = args.get("cmd") or args.get("command")

            event = NormalizedToolEvent(
                id=cast(str, block.get("id", f"tool_{len(tool_results)}")),
                name=name,
                args=args,
                path=cast(str, path) if path else None,
                command=cast(str, command) if command else None,
            )

            result = executor.execute_normalized_tool(event)
            if result["status"] == "SUCCESS":
                tool_results.append(f"[SUCCESS] {result['message']}")
            else:
                tool_results.append(f"[ERROR] {result['message']}")

        self._pending_deltas.extend(executor.get_pending_deltas())
        executor.clear_pending_deltas()
        return tool_results

    def _should_chain(self, tool_feedback: str) -> bool:
        return bool(tool_feedback and self.entropy_budget > 0)

    def _chain_tool_feedback(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        tool_feedback: str,
    ) -> None:
        print(f"[*] Tool Results ({len(tool_feedback)} chars), chaining...")
        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": f"@msg role:tool\n  |> Tool results:\n{tool_feedback}",
            }
        )
        self.turn_count += 1
        self.entropy_budget -= 1

    def _handle_forget_command(self, key_to_forget: str) -> None:
        """Remove a key from the adapter session's bridge memory."""
        mem = self.adapter.session.bridge_memory
        removed = False
        if key_to_forget in mem.hot:
            del mem.hot[key_to_forget]
            removed = True
        if key_to_forget in mem.durable:
            del mem.durable[key_to_forget]
            removed = True
        if removed:
            self.adapter.session._save_bridge_memory()
            print(f"[*] REMOVED: {key_to_forget} from session memory.")
        else:
            print(f"[!] Key not found in session memory: {key_to_forget}")

    def handshake(self) -> str:
        """Establish the Tok protocol with a cold call."""
        messages, _prepared = self.adapter.prepare_turn(
            model=self.model,
            system_prompt=self.system_prompt,
            dynamic_messages=[],
        )
        prompt_tokens = self.count_tokens(self.system_prompt)
        print(f">>> Handshake: Sending grammar ({prompt_tokens} tokens)")

        extra_headers = {}
        if self.app_name:
            extra_headers = {
                "HTTP-Referer": "https://tok.protocol",
                "X-Title": self.app_name,
            }

        # Stream handshake response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            stream=True,
            max_tokens=self.max_tokens,
            extra_headers=extra_headers,
        )

        print("<<< ", end="", flush=True)
        agent_reply = ""
        for chunk in stream:
            if (
                not isinstance(chunk, tuple)
                and chunk.choices
                and chunk.choices[0].delta.content
            ):
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                agent_reply += content
        print()

        if hasattr(stream, "usage") and stream.usage:
            prompt_tokens_actual = stream.usage.prompt_tokens
            completion_tokens = stream.usage.completion_tokens
        else:
            completion_tokens = len(self.enc.encode(agent_reply))
            prompt_tokens_actual = prompt_tokens

        turn_cost = self._calc_cost(prompt_tokens_actual, completion_tokens)
        self.total_prompt_tokens += prompt_tokens_actual
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += turn_cost

        print(
            f"<<< Handshake reply: {completion_tokens} tokens | Cost: ${turn_cost:.8f}"
        )
        processed_hs = self.adapter.finalize(
            text=agent_reply,
            model=self.model,
            behavior_signals=None,
        )
        del processed_hs  # telemetry recorded in session; return value not needed here

        self.handshake_done = True
        return agent_reply

    def get_stats(self) -> dict[str, Any]:
        """Return the current cumulative statistics of the orchestrator session."""
        return {
            "turns": self.turn_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens
            + self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "compressed_history_size": self.count_tokens(
                self.adapter.session.load_memory()
            ),
        }
