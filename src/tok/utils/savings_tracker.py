"""SavingsTracker extracted from stats.py for modularity."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import statistics
import threading
import time
from typing import TYPE_CHECKING, Any

from tok.runtime.policy.semantic_validation import (
    calculate_invisible_pressure,
    calculate_memory_lift,
    calculate_semantic_regression_score,
)

from ._savings_persistence import (
    GLOBAL_LEDGER_FILENAME,
    SESSION_STATS_FILENAME,
    STATS_KEY_MAP_INV,
    default_ledger_path,
    default_savings_file,
    empty_stats,
    legacy_ledger_path,
    parse_model_line,
)
from ._savings_quality import BASELINE_ONLY_SIGNAL, FALLBACK_SIGNAL
from ._savings_quality import PROMPT_METRIC_KEYS as _PROMPT_METRIC_KEYS
from ._savings_quality import degradation_reason as _degradation_reason
from ._savings_quality import session_quality as _session_quality
from .pricing import get_pricing
from .telemetry import emit_event_sync

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("tok.savings_tracker")

_default_savings_file = default_savings_file
_default_ledger_path = default_ledger_path
_legacy_ledger_path = legacy_ledger_path

__all__ = [
    "BASELINE_ONLY_SIGNAL",
    "FALLBACK_SIGNAL",
    "GLOBAL_LEDGER_FILENAME",
    "SESSION_STATS_FILENAME",
    "SavingsTracker",
    "_default_ledger_path",
    "_default_savings_file",
    "_degradation_reason",
    "_legacy_ledger_path",
    "_session_quality",
]


class SavingsTracker:
    """Thread-safe session savings accumulator."""

    def __init__(
        self,
        savings_file: str | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        self._savings_file = savings_file or default_savings_file()
        self._ledger_path = ledger_path or default_ledger_path()
        self._lock = threading.Lock()
        self._migrate_legacy_ledger()

    def _migrate_legacy_ledger(self) -> None:
        legacy = legacy_ledger_path()
        if legacy == self._ledger_path:
            return
        if not legacy.exists() or self._ledger_path.exists():
            return
        try:
            legacy.rename(self._ledger_path)
            logger.info("Migrated legacy savings ledger to %s", self._ledger_path)
        except Exception as exc:
            logger.warning("Failed to migrate legacy ledger: %s", exc)

    @property
    def savings_file(self) -> str:
        return self._savings_file

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def _parse_model_line(self, line: str) -> tuple[str, dict[str, Any]]:
        return parse_model_line(line)

    def _empty_stats(self) -> dict[str, Any]:
        return empty_stats()

    def load_stats(self) -> dict[str, Any]:
        try:
            if not os.path.exists(self._savings_file):
                return self._empty_stats()

            stats: dict[str, Any] = {"models": {}}
            with open(self._savings_file) as f:
                for line in f:
                    if line.startswith(">>> session:"):
                        stats["session_start"] = line.split(" session:", 1)[1].strip()
                    elif line.startswith(">>> m:"):
                        model_name, m = self._parse_model_line(line)
                        stats["models"][model_name] = m
            return stats
        except Exception as exc:
            logger.error("Stats load error (file may be corrupted): %s", exc)
            if os.path.exists(self._savings_file) and os.stat(self._savings_file).st_size > 0:
                raise
            return self._empty_stats()

    def save_stats(self, stats: dict[str, Any]) -> None:
        """Save stats atomically to prevent race conditions."""
        try:
            lines = [f">>> session:{stats.get('session_start', 'unknown')}"]
            for model, m in sorted(stats["models"].items()):
                parts = [model]
                for full_k, short_k in STATS_KEY_MAP_INV.items():
                    val = m.get(full_k, 0)
                    if isinstance(val, float):
                        parts.append(f"{short_k}:{val:.6f}")
                    else:
                        parts.append(f"{short_k}:{val}")
                bd = m.get("type_breakdown", {})
                if bd:
                    bd_str = ",".join(f"{k}={v}" for k, v in sorted(bd.items()))
                    parts.append(f"breakdown:{bd_str}")
                sigs = m.get("behavior_signals", {})
                if sigs:
                    sig_str = ",".join(f"{k}={v}" for k, v in sorted(sigs.items()))
                    parts.append(f"signals:{sig_str}")
                lines.append(">>> m:" + "|".join(parts))

            # Atomic write: write to temp file then rename
            temp_path = self._savings_file + ".tmp"
            with open(temp_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            # Atomic rename
            os.rename(temp_path, self._savings_file)
        except Exception as exc:
            logger.warning("Stats write error: %s", exc)

    def record_call(
        self,
        model: str,
        actual_input: int,
        actual_output: int,
        cache_read: int,
        cache_write: int,
        input_saved: int,
        output_saved: int,
        type_breakdown: dict[str, int] | None = None,
        behavior_signals: dict[str, int] | None = None,
        prompt_metrics: dict[str, int] | None = None,
    ) -> None:
        """Update per-model session stats."""
        inp_rate, out_rate, cr_rate, cw_rate = get_pricing(model)
        M = 1_000_000

        actual_cost = (
            actual_input * inp_rate / M
            + actual_output * out_rate / M
            + cache_read * cr_rate / M
            + cache_write * cw_rate / M
        )

        baseline_input = actual_input + input_saved
        baseline_output = actual_output + output_saved
        # Baseline represents cost without Tok's compression, but with caching intact.
        # Cache tokens keep their actual rates — caching is an API feature, not a Tok
        # contribution, so we don't credit it to savings.
        baseline_cost = (
            baseline_input * inp_rate / M
            + baseline_output * out_rate / M
            + cache_read * cr_rate / M
            + cache_write * cw_rate / M
        )

        with self._lock:
            stats = self.load_stats()
            m = stats["models"].setdefault(
                model,
                {
                    "calls": 0,
                    "actual_input_tokens": 0,
                    "actual_output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "input_saved_tokens": 0,
                    "output_saved_tokens": 0,
                    "baseline_cost_usd": 0.0,
                    "actual_cost_usd": 0.0,
                    "baseline_prompt_tokens": 0,
                    "prepared_prompt_tokens": 0,
                    "saved_prompt_tokens": 0,
                    "hot_hint_tokens_added": 0,
                    "reacquisition_tokens_avoided_estimate": 0,
                    "type_breakdown": {},
                    "behavior_signals": {},
                },
            )
            m["calls"] += 1
            m["actual_input_tokens"] += actual_input
            m["actual_output_tokens"] += actual_output
            m["cache_read_tokens"] += cache_read
            m["cache_write_tokens"] += cache_write
            m["input_saved_tokens"] += input_saved
            m["output_saved_tokens"] += output_saved
            m["baseline_cost_usd"] += baseline_cost
            m["actual_cost_usd"] += actual_cost
            if prompt_metrics:
                for key in _PROMPT_METRIC_KEYS:
                    m[key] = int(m.get(key, 0)) + int(prompt_metrics.get(key, 0))
            if type_breakdown:
                bd = m.setdefault("type_breakdown", {})
                for k, v in type_breakdown.items():
                    bd[k] = bd.get(k, 0) + v
            if behavior_signals:
                sigs = m.setdefault("behavior_signals", {})
                for k, v in behavior_signals.items():
                    sigs[k] = sigs.get(k, 0) + v

            total_turns = sum(m.get("calls", 0) for m in stats["models"].values())
            total_tokens = sum(
                m.get("actual_input_tokens", 0)
                + m.get("actual_output_tokens", 0)
                + m.get("cache_read_tokens", 0)
                + m.get("cache_write_tokens", 0)
                for m in stats["models"].values()
            )
            self.save_stats(stats)

        pct = (baseline_cost - actual_cost) / baseline_cost * 100 if baseline_cost > 0 else 0.0
        logger.info(
            "cost: baseline=$%.3f actual=$%.3f saved=$%.3f (%.1f%%) [%s]",
            baseline_cost,
            actual_cost,
            (baseline_cost - actual_cost),
            pct,
            model,
        )

        emit_event_sync(
            "token_savings",
            {
                "baseline_cost": baseline_cost,
                "actual_cost": actual_cost,
                "saved_usd": baseline_cost - actual_cost,
                "savings_pct": pct,
                "actual_input": actual_input,
                "actual_output": actual_output,
                "input_saved": input_saved,
                "output_saved": output_saved,
                "turn_count": total_turns,
                "total_recorded_tokens": total_tokens,
            },
            model=model,
        )

    def reset_session_stats(self) -> None:
        """Start a fresh session stats file (invoked on bridge startup)."""
        self.save_stats(
            {
                "session_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models": {},
            }
        )

    def session_summary(self) -> dict[str, int | float | bool | str] | None:
        """Return canonical user-facing savings fields for the current session."""
        stats = self.load_stats()
        models = stats.get("models", {})
        if not models or all(m.get("calls", 0) == 0 for m in models.values()):
            return None

        actual_prompt_tokens = sum(
            m.get("actual_input_tokens", 0) + m.get("cache_read_tokens", 0) + m.get("cache_write_tokens", 0)
            for m in models.values()
        )
        actual_completion_tokens = sum(m.get("actual_output_tokens", 0) for m in models.values())
        actual_tokens = actual_prompt_tokens + actual_completion_tokens
        saved_tokens = sum(m.get("input_saved_tokens", 0) + m.get("output_saved_tokens", 0) for m in models.values())
        baseline_prompt_tokens = sum(m.get("baseline_prompt_tokens", 0) for m in models.values())
        prepared_prompt_tokens = sum(m.get("prepared_prompt_tokens", 0) for m in models.values())
        saved_prompt_tokens = sum(m.get("saved_prompt_tokens", 0) for m in models.values())
        hot_hint_tokens_added = sum(m.get("hot_hint_tokens_added", 0) for m in models.values())
        reacquisition_tokens_avoided_estimate = sum(
            m.get("reacquisition_tokens_avoided_estimate", 0) for m in models.values()
        )
        baseline_tokens = actual_tokens + saved_tokens
        actual_cost = sum(m.get("actual_cost_usd", 0.0) for m in models.values())
        baseline_cost = sum(m.get("baseline_cost_usd", 0.0) for m in models.values())
        cost_saved = baseline_cost - actual_cost
        calls = sum(m.get("calls", 0) for m in models.values())
        signals = self.behavior_signals()
        reacquisition_cost = int(signals.get("reacquisition_cost_tokens", 0))
        net_saved_tokens = saved_tokens - reacquisition_cost
        fallback_count = int(signals.get(FALLBACK_SIGNAL, 0))
        baseline_only = bool(signals.get(BASELINE_ONLY_SIGNAL, 0))
        cost_savings_pct = cost_saved / baseline_cost * 100 if baseline_cost > 0 else 0.0
        savings_pct = saved_tokens / baseline_tokens * 100 if baseline_tokens > 0 else 0.0
        semantic_drift_count = int(signals.get("semantic_drift_detected", 0))
        fail_open_count = int(signals.get("fail_open_compat_response", 0))
        non_tok_count = int(signals.get("non_tok_response", 0))
        answer_anchor_miss_count = (
            1
            if signals.get("answer_anchor_present", 0) == 0
            and (
                signals.get("state_resend_suppressed_turn", 0)
                or signals.get("state_resend_delta_turn", 0)
                or signals.get("state_resend_full_turn", 0)
            )
            else 0
        )
        reacquisition_count = int(signals.get("repeat_file_read", 0)) + int(signals.get("repeat_search", 0))
        quality = _session_quality(
            signals,
            baseline_only=baseline_only,
            tokens_saved=saved_tokens,
        )
        degradation_reason = _degradation_reason(
            signals,
            baseline_only=baseline_only,
        )
        stream_recovery_attempt_count = int(signals.get("stream_recovery_started", 0)) + int(
            signals.get("stream_recovery_retry", 0)
        )
        stream_recovery_success_text_count = int(signals.get("stream_recovery_success_text", 0))
        stream_recovery_success_tool_use_count = int(signals.get("stream_recovery_success_tool_use", 0))
        stream_recovery_fallback_count = int(signals.get("stream_recovery_fallback", 0))
        stream_recovery_empty_success_count = int(signals.get("stream_recovery_empty_success", 0))
        stream_recovery_read_error_count = int(signals.get("stream_recovery_read_error", 0))
        tool_history_repaired_count = int(signals.get("tok_bridge_tool_history_repaired", 0))
        tool_history_pairing_repaired_count = int(signals.get("tok_bridge_tool_history_pairing_repaired", 0))
        tool_history_quarantined_count = int(signals.get("tok_bridge_invalid_tool_history_quarantined", 0))
        tool_history_blocked_count = int(signals.get("tok_bridge_invalid_tool_history_blocked", 0))
        invalid_tool_history_session_reset_count = int(signals.get("tok_bridge_invalid_tool_history_session_reset", 0))
        provider_pairing_disagreement_count = int(
            signals.get("fail_open_retry_upstream_pairing_disagreement", 0)
        ) + int(signals.get("tok_bridge_provider_pairing_risk_detected", 0))
        assistant_tool_use_text_interleaving_blocked_count = int(
            signals.get(
                "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                0,
            )
        )
        preflight_block_original_payload_count = int(signals.get("preflight_block_original_payload", 0))
        preflight_block_rewritten_payload_count = int(signals.get("preflight_block_rewritten_payload", 0))
        request_policy_natural_first_count = int(signals.get("request_policy_natural_first", 0))
        request_policy_tool_compatible_count = int(signals.get("request_policy_tool_compatible", 0))
        request_policy_escalations_count = int(signals.get("request_policy_escalations", 0))
        request_policy_deescalations_count = int(signals.get("request_policy_deescalations", 0))
        request_policy_interleaving_downgrades_count = int(signals.get("request_policy_interleaving_downgrades", 0))
        request_policy_reason_stream_recovery_count = int(signals.get("request_policy_reason_stream_recovery", 0))
        request_policy_reason_tool_recovery_count = int(signals.get("request_policy_reason_tool_recovery", 0))
        request_policy_reason_structured_tool_loop_count = int(
            signals.get("request_policy_reason_structured_tool_loop", 0)
        )
        request_policy_held_by_recovery_count = int(signals.get("request_policy_held_by_recovery", 0)) + int(
            signals.get("request_policy_recovery_sticky_continuations", 0)
        )

        return {
            "calls": calls,
            "actual_prompt_tokens": actual_prompt_tokens,
            "actual_completion_tokens": actual_completion_tokens,
            "actual_tokens": actual_tokens,
            "baseline_tokens": baseline_tokens,
            "tokens_saved": saved_tokens,
            "net_tokens_saved": net_saved_tokens,
            "reacquisition_cost_tokens": reacquisition_cost,
            "baseline_prompt_tokens": baseline_prompt_tokens,
            "prepared_prompt_tokens": prepared_prompt_tokens,
            "saved_prompt_tokens": saved_prompt_tokens,
            "hot_hint_tokens_added": hot_hint_tokens_added,
            "reacquisition_tokens_avoided_estimate": reacquisition_tokens_avoided_estimate,
            "savings_pct": round(savings_pct, 1),
            "cost_savings_pct": round(cost_savings_pct, 1),
            "actual_cost_usd": actual_cost,
            "baseline_cost_usd": baseline_cost,
            "cost_saved_usd": cost_saved,
            "fallback_count": fallback_count,
            "baseline_only": baseline_only,
            "semantic_drift_count": semantic_drift_count,
            "fail_open_count": fail_open_count,
            "non_tok_count": non_tok_count,
            "answer_anchor_miss_count": answer_anchor_miss_count,
            "reacquisition_count": reacquisition_count,
            "stream_recovery_attempt_count": stream_recovery_attempt_count,
            "stream_recovery_success_text_count": stream_recovery_success_text_count,
            "stream_recovery_success_tool_use_count": stream_recovery_success_tool_use_count,
            "stream_recovery_fallback_count": stream_recovery_fallback_count,
            "stream_recovery_empty_success_count": stream_recovery_empty_success_count,
            "stream_recovery_read_error_count": stream_recovery_read_error_count,
            "tool_history_repaired_count": tool_history_repaired_count,
            "tool_history_pairing_repaired_count": tool_history_pairing_repaired_count,
            "tool_history_quarantined_count": tool_history_quarantined_count,
            "tool_history_blocked_count": tool_history_blocked_count,
            "invalid_tool_history_session_reset_count": invalid_tool_history_session_reset_count,
            "provider_pairing_disagreement_count": provider_pairing_disagreement_count,
            "assistant_tool_use_text_interleaving_blocked_count": assistant_tool_use_text_interleaving_blocked_count,
            "preflight_block_original_payload_count": preflight_block_original_payload_count,
            "preflight_block_rewritten_payload_count": preflight_block_rewritten_payload_count,
            "request_policy_natural_first_count": request_policy_natural_first_count,
            "request_policy_tool_compatible_count": request_policy_tool_compatible_count,
            "request_policy_escalations_count": request_policy_escalations_count,
            "request_policy_deescalations_count": request_policy_deescalations_count,
            "request_policy_interleaving_downgrades_count": request_policy_interleaving_downgrades_count,
            "request_policy_reason_stream_recovery_count": request_policy_reason_stream_recovery_count,
            "request_policy_reason_tool_recovery_count": request_policy_reason_tool_recovery_count,
            "request_policy_reason_structured_tool_loop_count": request_policy_reason_structured_tool_loop_count,
            "request_policy_held_by_recovery_count": request_policy_held_by_recovery_count,
            "session_quality": quality,
            "last_degradation_reason": degradation_reason,
        }

    @staticmethod
    def _aggregate_session(models: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        int_fields = [
            "calls",
            "actual_input_tokens",
            "actual_output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "input_saved_tokens",
            "output_saved_tokens",
            "baseline_prompt_tokens",
            "prepared_prompt_tokens",
            "saved_prompt_tokens",
            "hot_hint_tokens_added",
            "reacquisition_tokens_avoided_estimate",
            "reacquisition_cost_tokens",
        ]
        float_fields = ["actual_cost_usd", "baseline_cost_usd"]
        for field in int_fields:
            result[field] = sum(m.get(field, 0) for m in models.values())
        for field in float_fields:
            result[field] = sum(m.get(field, 0.0) for m in models.values())
        result["prompt_tokens"] = (
            result["actual_input_tokens"] + result["cache_read_tokens"] + result["cache_write_tokens"]
        )
        result["total_tokens"] = result["prompt_tokens"] + result["actual_output_tokens"]
        result["saved_tokens"] = result["input_saved_tokens"] + result["output_saved_tokens"]
        result["net_tokens_saved"] = result["saved_tokens"] - result.get("reacquisition_cost_tokens", 0)
        result["saved_usd"] = result["baseline_cost_usd"] - result["actual_cost_usd"]
        return result

    @staticmethod
    def _default_ledger() -> dict[str, Any]:
        return {
            "sessions": 0,
            "total_turns": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "estimated_baseline_cost_usd": 0.0,
            "tokens_saved": 0,
            "net_tokens_saved": 0,
            "cost_saved_usd": 0.0,
            "baseline_prompt_tokens": 0,
            "prepared_prompt_tokens": 0,
            "saved_prompt_tokens": 0,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
            "reacquisition_cost_tokens": 0,
            "repeat_file_read": 0,
            "repeat_search": 0,
            "repeat_target_hot": 0,
            "repeat_target_stuck": 0,
            "hot_recent_hint_injected": 0,
            "repeat_tool_collapse_applied": 0,
            "predictive_cache_warm_applied": 0,
            "predictive_cache_candidates": 0,
            "predictive_cache_hits": 0,
            "python_c_workaround": 0,
            "stderr_workaround": 0,
            "non_tok_response": 0,
            "cold_start_structured_memory": 0,
            "cold_start_wire_fallback": 0,
            "durable_promotions": 0,
            "hot_promotions": 0,
            "hot_demotions": 0,
            "file_snapshot_recorded": 0,
            "search_snapshot_recorded": 0,
            "tok_native_response": 0,
            "fail_open_compat_response": 0,
            "tok_fallback_activated": 0,
            "baseline_only_session": 0,
            "malformed_tok_response": 0,
            "malformed_tok_hybrid_tool": 0,
            "malformed_tok_non_inverted_msg": 0,
            "malformed_tok_markdown_fallback": 0,
            "malformed_tok_bad_header": 0,
        }

    @staticmethod
    def _parse_ledger(path: Path, ledger: dict[str, Any]) -> list[str]:
        log_lines: list[str] = []
        if not path.exists():
            return log_lines
        in_log = False
        for line in path.read_text().splitlines():
            s = line.strip()
            if s.startswith("@lifetime_savings"):
                in_log = False
            elif s.startswith("@per_session_log"):
                in_log = True
            elif in_log:
                if s and not s.startswith("#"):
                    log_lines.append(s)
            elif ":" in s and not s.startswith("#") and not s.startswith("@"):
                k, _, v = s.partition(":")
                k = k.strip()
                v = v.strip()
                if k in ledger:
                    with contextlib.suppress(ValueError):
                        ledger[k] = float(v) if "." in v else int(v)
        return log_lines

    @staticmethod
    def _accumulate_ledger(
        ledger: dict[str, Any],
        sess: dict[str, Any],
        signals: dict[str, int],
    ) -> None:
        int_map = {
            "total_turns": "calls",
            "total_prompt_tokens": "prompt_tokens",
            "total_completion_tokens": "actual_output_tokens",
            "total_tokens": "total_tokens",
            "tokens_saved": "saved_tokens",
            "net_tokens_saved": "net_tokens_saved",
            "baseline_prompt_tokens": "baseline_prompt_tokens",
            "prepared_prompt_tokens": "prepared_prompt_tokens",
            "saved_prompt_tokens": "saved_prompt_tokens",
            "hot_hint_tokens_added": "hot_hint_tokens_added",
            "reacquisition_tokens_avoided_estimate": "reacquisition_tokens_avoided_estimate",
            "reacquisition_cost_tokens": "reacquisition_cost_tokens",
        }
        for ledger_key, sess_key in int_map.items():
            ledger[ledger_key] = int(ledger[ledger_key]) + int(sess[sess_key])
        ledger["total_cost_usd"] = float(ledger["total_cost_usd"]) + sess["actual_cost_usd"]
        ledger["estimated_baseline_cost_usd"] = float(ledger["estimated_baseline_cost_usd"]) + sess["baseline_cost_usd"]
        ledger["cost_saved_usd"] = float(ledger["cost_saved_usd"]) + sess["saved_usd"]
        for key, value in signals.items():
            if key in ledger:
                ledger[key] = int(ledger[key]) + value

    @staticmethod
    def _format_ledger_output(ledger: dict[str, Any], pct: float, log_lines: list[str]) -> list[str]:
        out = [
            "@lifetime_savings",
            f"  sessions: {ledger['sessions']}",
            f"  total_turns: {ledger['total_turns']}",
            f"  total_prompt_tokens: {ledger['total_prompt_tokens']}",
            f"  total_completion_tokens: {ledger['total_completion_tokens']}",
            f"  total_tokens: {ledger['total_tokens']}",
            f"  total_cost_usd: {ledger['total_cost_usd']:.6f}",
            f"  estimated_baseline_cost_usd: {ledger['estimated_baseline_cost_usd']:.6f}",
            f"  tokens_saved: {ledger['tokens_saved']}",
            f"  net_tokens_saved: {ledger['net_tokens_saved']}",
            f"  cost_saved_usd: {ledger['cost_saved_usd']:.6f}",
            f"  savings_pct: {pct:.1f}",
            f"  baseline_prompt_tokens: {ledger['baseline_prompt_tokens']}",
            f"  prepared_prompt_tokens: {ledger['prepared_prompt_tokens']}",
            f"  saved_prompt_tokens: {ledger['saved_prompt_tokens']}",
            f"  hot_hint_tokens_added: {ledger['hot_hint_tokens_added']}",
            f"  reacquisition_tokens_avoided_estimate: {ledger['reacquisition_tokens_avoided_estimate']}",
            f"  repeat_file_read: {ledger['repeat_file_read']}",
            f"  repeat_search: {ledger['repeat_search']}",
            f"  repeat_target_hot: {ledger['repeat_target_hot']}",
            f"  repeat_target_stuck: {ledger['repeat_target_stuck']}",
            f"  hot_recent_hint_injected: {ledger['hot_recent_hint_injected']}",
            f"  repeat_tool_collapse_applied: {ledger['repeat_tool_collapse_applied']}",
            f"  predictive_cache_warm_applied: {ledger['predictive_cache_warm_applied']}",
            f"  predictive_cache_candidates: {ledger['predictive_cache_candidates']}",
            f"  predictive_cache_hits: {ledger['predictive_cache_hits']}",
            f"  python_c_workaround: {ledger['python_c_workaround']}",
            f"  stderr_workaround: {ledger['stderr_workaround']}",
            f"  non_tok_response: {ledger['non_tok_response']}",
            f"  cold_start_structured_memory: {ledger['cold_start_structured_memory']}",
            f"  cold_start_wire_fallback: {ledger['cold_start_wire_fallback']}",
            f"  durable_promotions: {ledger['durable_promotions']}",
            f"  hot_promotions: {ledger['hot_promotions']}",
            f"  hot_demotions: {ledger['hot_demotions']}",
            f"  tok_native_response: {ledger['tok_native_response']}",
            f"  fail_open_compat_response: {ledger['fail_open_compat_response']}",
            f"  tok_fallback_activated: {ledger['tok_fallback_activated']}",
            f"  baseline_only_session: {ledger['baseline_only_session']}",
            f"  malformed_tok_response: {ledger['malformed_tok_response']}",
            f"  malformed_tok_hybrid_tool: {ledger['malformed_tok_hybrid_tool']}",
            f"  malformed_tok_non_inverted_msg: {ledger['malformed_tok_non_inverted_msg']}",
            f"  malformed_tok_markdown_fallback: {ledger['malformed_tok_markdown_fallback']}",
            f"  malformed_tok_bad_header: {ledger['malformed_tok_bad_header']}",
            "",
            "@per_session_log",
            "  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response;memory_lift;semantic_regression;reacquisition_count;answer_anchor_miss_count;degradation_reason;prompt_tokens;completion_tokens;fallback_activated;baseline_only;baseline_prompt_tokens;prepared_prompt_tokens;saved_prompt_tokens;hot_hint_tokens_added;reacquisition_avoided;reacquisition_cost_tokens",
        ]
        for ll in log_lines:
            out.append(f"  {ll}")
        out.append("")
        return out

    @staticmethod
    def _parse_session_log_core(line: str) -> dict[str, int | float | str] | None:
        parts = line.strip().split(";")
        if len(parts) < 10:
            return None
        try:
            return {
                "date": parts[0],
                "session_id": parts[1],
                "turns": int(parts[2]),
                "tokens": int(parts[3]),
                "actual_cost_usd": float(parts[4]),
                "baseline_cost_usd": float(parts[5]),
                "saved_usd": float(parts[6]),
                "tokens_saved": int(parts[7]),
                "prompt_tokens": int(parts[15]) if len(parts) > 15 and parts[15] else 0,
                "completion_tokens": int(parts[16]) if len(parts) > 16 and parts[16] else 0,
                FALLBACK_SIGNAL: int(parts[17]) if len(parts) > 17 and parts[17] else 0,
                BASELINE_ONLY_SIGNAL: int(parts[18]) if len(parts) > 18 and parts[18] else 0,
                "baseline_prompt_tokens": int(parts[19]) if len(parts) > 19 and parts[19] else 0,
                "prepared_prompt_tokens": int(parts[20]) if len(parts) > 20 and parts[20] else 0,
                "saved_prompt_tokens": int(parts[21]) if len(parts) > 21 and parts[21] else 0,
                "hot_hint_tokens_added": int(parts[22]) if len(parts) > 22 and parts[22] else 0,
                "reacquisition_tokens_avoided_estimate": int(parts[23]) if len(parts) > 23 and parts[23] else 0,
                "reacquisition_cost_tokens": int(parts[24]) if len(parts) > 24 and parts[24] else 0,
            }
        except ValueError:
            return None

    @staticmethod
    def _subtract_session_from_ledger(ledger: dict[str, Any], entry: dict[str, int | float | str]) -> None:
        ledger["sessions"] = max(0, int(ledger["sessions"]) - 1)
        ledger["total_turns"] = max(0, int(ledger["total_turns"]) - int(entry["turns"]))
        ledger["total_tokens"] = max(0, int(ledger["total_tokens"]) - int(entry["tokens"]))
        ledger["total_prompt_tokens"] = max(0, int(ledger["total_prompt_tokens"]) - int(entry["prompt_tokens"]))
        ledger["total_completion_tokens"] = max(
            0,
            int(ledger["total_completion_tokens"]) - int(entry["completion_tokens"]),
        )
        ledger["tokens_saved"] = max(0, int(ledger["tokens_saved"]) - int(entry["tokens_saved"]))
        ledger["net_tokens_saved"] = max(0, int(ledger["net_tokens_saved"]) - int(entry["tokens_saved"]))
        ledger["total_cost_usd"] = max(0.0, float(ledger["total_cost_usd"]) - float(entry["actual_cost_usd"]))
        ledger["estimated_baseline_cost_usd"] = max(
            0.0,
            float(ledger["estimated_baseline_cost_usd"]) - float(entry["baseline_cost_usd"]),
        )
        ledger["cost_saved_usd"] = max(0.0, float(ledger["cost_saved_usd"]) - float(entry["saved_usd"]))
        int_field_map = {
            "baseline_prompt_tokens": "baseline_prompt_tokens",
            "prepared_prompt_tokens": "prepared_prompt_tokens",
            "saved_prompt_tokens": "saved_prompt_tokens",
            "hot_hint_tokens_added": "hot_hint_tokens_added",
            "reacquisition_tokens_avoided_estimate": "reacquisition_tokens_avoided_estimate",
            "reacquisition_cost_tokens": "reacquisition_cost_tokens",
        }
        for ledger_key, entry_key in int_field_map.items():
            ledger[ledger_key] = max(0, int(ledger[ledger_key]) - int(entry.get(entry_key, 0)))
        signal_keys = [
            FALLBACK_SIGNAL,
            BASELINE_ONLY_SIGNAL,
        ]
        for sig_key in signal_keys:
            if sig_key in ledger and sig_key in entry:
                ledger[sig_key] = max(0, int(ledger[sig_key]) - int(entry.get(sig_key, 0)))

    def merge_session_to_ledger(self) -> None:
        """On shutdown, merge session stats into the persistent savings ledger."""
        try:
            with self._lock:
                stats = self.load_stats()

            models = stats.get("models", {})
            total_calls = sum(m.get("calls", 0) for m in models.values())
            if total_calls == 0:
                return

            sess = self._aggregate_session(models)
            sess_signals = self.behavior_signals()

            ledger = self._default_ledger()
            log_lines = self._parse_ledger(self._ledger_path, ledger)

            date_str = stats.get(
                "session_start",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            # Session IDs should be stable for a given session bucket (so reruns replace),
            # but not collide just because two sessions started in the same second.
            sess_id = hashlib.md5(f"{date_str}:{self._savings_file}".encode()).hexdigest()[:8]  # nosec B324
            invisible_pressure = calculate_invisible_pressure(sess_signals)
            memory_lift = calculate_memory_lift(sess_signals)
            semantic_regression = calculate_semantic_regression_score(sess_signals)
            answer_anchor_miss_count = (
                1
                if sess_signals.get("answer_anchor_present", 0) == 0
                and (
                    sess_signals.get("state_resend_suppressed_turn", 0)
                    or sess_signals.get("state_resend_delta_turn", 0)
                    or sess_signals.get("state_resend_full_turn", 0)
                )
                else 0
            )
            reacquisition_count = int(sess_signals.get("repeat_file_read", 0)) + int(
                sess_signals.get("repeat_search", 0)
            )
            baseline_only = bool(sess_signals.get(BASELINE_ONLY_SIGNAL, 0))
            degradation_reason = _degradation_reason(
                sess_signals,
                baseline_only=baseline_only,
            )

            new_entry = (
                f"{date_str};{sess_id};{sess['calls']};{sess['total_tokens']}"
                f";{sess['actual_cost_usd']:.6f};{sess['baseline_cost_usd']:.6f};{sess['saved_usd']:.6f}"
                f";{sess['saved_tokens']};{invisible_pressure};"
                f"{sess_signals.get('non_tok_response', 0)};{memory_lift}"
                f";{semantic_regression};{reacquisition_count}"
                f";{answer_anchor_miss_count};{degradation_reason}"
                f";{sess['prompt_tokens']};{sess['actual_output_tokens']}"
                f";{sess_signals.get(FALLBACK_SIGNAL, 0)};{int(sess_signals.get(BASELINE_ONLY_SIGNAL, 0))}"
                f";{sess['baseline_prompt_tokens']};{sess['prepared_prompt_tokens']}"
                f";{sess['saved_prompt_tokens']};{sess['hot_hint_tokens_added']}"
                f";{sess['reacquisition_tokens_avoided_estimate']};{sess['reacquisition_cost_tokens']}"
            )

            replacement_log_lines: list[str] = []
            replaced = 0
            for line in log_lines:
                prior = self._parse_session_log_core(line)
                if prior and prior["session_id"] == sess_id:
                    self._subtract_session_from_ledger(ledger, prior)
                    replaced += 1
                    continue
                replacement_log_lines.append(line)
            log_lines = replacement_log_lines

            ledger["sessions"] = int(ledger["sessions"]) + 1
            self._accumulate_ledger(ledger, sess, sess_signals)
            pct = (
                ledger["cost_saved_usd"] / ledger["estimated_baseline_cost_usd"] * 100
                if ledger["estimated_baseline_cost_usd"] > 0
                else 0.0
            )
            log_lines.append(new_entry)
            if replaced:
                logger.info("Replaced %d prior ledger row(s) for session %s", replaced, sess_id)

            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            out = self._format_ledger_output(ledger, pct, log_lines)
            self._ledger_path.write_text("\n".join(out))

            logger.info(
                "Ledger updated: %d turns | $%.3f saved (%.1f%% lifetime)",
                sess["calls"],
                sess["saved_usd"],
                pct,
            )
        except Exception as exc:
            logger.warning("Ledger merge error: %s", exc)

    def format_session(self) -> str | None:
        """Format current session stats for display."""
        summary = self.session_summary()
        if summary is None:
            return None

        return (
            "Current session:\n"
            f"- calls: {summary['calls']}\n"
            f"- actual tokens: {int(summary['actual_tokens']):,}\n"
            f"- baseline tokens: {int(summary['baseline_tokens']):,}\n"
            f"- saved: {int(summary['tokens_saved']):,} ({float(summary['savings_pct']):.1f}%)\n"
            f"- actual cost: ${float(summary['actual_cost_usd']):.4f}\n"
            f"- baseline cost: ${float(summary['baseline_cost_usd']):.4f}\n"
            f"- cost saved: ${float(summary['cost_saved_usd']):.4f}\n"
            f"- fallback count: {int(summary['fallback_count'])}\n"
            f"- baseline-only: {'yes' if bool(summary['baseline_only']) else 'no'}\n"
            f"- session quality: {summary['session_quality']}\n"
            f"- degradation reason: {summary['last_degradation_reason'] or 'none'}"
        )

    def format_compact_session_summary(self) -> str | None:
        """Backward-compatible compact session formatter used by legacy callers/tests."""
        summary = self.session_summary()
        if summary is None:
            return None

        status = "active and helping"
        if bool(summary.get("baseline_only")) or str(summary.get("session_quality", "")) == "degraded":
            status = "degraded"

        return (
            f"Saved ${float(summary['cost_saved_usd']):.4f} "
            f"({float(summary['savings_pct']):.1f}%) | "
            f"status={status} | "
            f"fallbacks={int(summary['fallback_count'])}"
        )

    def session_input_saved_tokens(self) -> int:
        """Return total input-side tokens saved for the current session."""
        stats = self.load_stats()
        total = 0
        for model_stats in stats.get("models", {}).values():
            total += int(model_stats.get("input_saved_tokens", 0))
        return total

    def format_ledger(self) -> str | None:
        """Format lifetime ledger stats for display."""
        summary = self.lifetime_summary()
        if summary is None:
            return None

        return (
            "Lifetime:\n"
            f"- sessions: {summary['sessions']}\n"
            f"- turns: {summary['total_turns']}\n"
            f"- actual tokens: {int(summary['actual_tokens']):,}\n"
            f"- baseline tokens: {int(summary['baseline_tokens']):,}\n"
            f"- saved: {int(summary['tokens_saved']):,} ({float(summary['savings_pct']):.1f}%)\n"
            f"- actual cost: ${float(summary['actual_cost_usd']):.4f}\n"
            f"- baseline cost: ${float(summary['baseline_cost_usd']):.4f}\n"
            f"- cost saved: ${float(summary['cost_saved_usd']):.4f}\n"
            f"- fallback count: {int(summary['fallback_count'])}\n"
            f"- baseline-only requests: {int(summary['baseline_only_requests'])}"
        )

    def lifetime_summary(self) -> dict[str, int | float] | None:
        """Return canonical user-facing savings fields for the lifetime ledger."""
        if not self._ledger_path.exists():
            return None

        ledger: dict[str, str] = {}
        for line in self._ledger_path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("@") and not s.startswith("#") and ":" in s:
                k, _, v = s.partition(":")
                ledger[k.strip()] = v.strip()

        entries = self._load_session_log_entries()
        if entries:
            actual_tokens = sum(int(entry["tokens"]) for entry in entries)
            saved_tokens = sum(int(entry["tokens_saved"]) for entry in entries)
            baseline_tokens = actual_tokens + saved_tokens
            actual_cost = sum(float(entry["actual_cost_usd"]) for entry in entries)
            baseline_cost = sum(float(entry["baseline_cost_usd"]) for entry in entries)
            cost_saved = sum(float(entry["saved_usd"]) for entry in entries)
            cost_savings_pct = cost_saved / baseline_cost * 100 if baseline_cost > 0 else 0.0
            savings_pct = saved_tokens / baseline_tokens * 100 if baseline_tokens > 0 else 0.0

            fallback_count = sum(int(entry.get(FALLBACK_SIGNAL, 0)) for entry in entries)
            baseline_only_count = sum(int(entry.get(BASELINE_ONLY_SIGNAL, 0)) for entry in entries)

            return {
                "sessions": len(entries),
                "total_turns": sum(int(entry["turns"]) for entry in entries),
                "actual_tokens": actual_tokens,
                "baseline_tokens": baseline_tokens,
                "tokens_saved": saved_tokens,
                "savings_pct": round(savings_pct, 1),
                "cost_savings_pct": round(cost_savings_pct, 1),
                "actual_cost_usd": actual_cost,
                "baseline_cost_usd": baseline_cost,
                "cost_saved_usd": cost_saved,
                "fallback_count": fallback_count,
                "baseline_only_requests": baseline_only_count,
            }

        if ledger.get("sessions", "0") == "0":
            return None

        actual_tokens = int(ledger.get("total_tokens", 0))
        saved_tokens = int(ledger.get("tokens_saved", 0))
        baseline_tokens = actual_tokens + saved_tokens
        actual_cost = float(ledger.get("total_cost_usd", 0.0))
        baseline_cost = float(ledger.get("estimated_baseline_cost_usd", 0.0))
        cost_savings_pct = (baseline_cost - actual_cost) / baseline_cost * 100 if baseline_cost > 0 else 0.0
        savings_pct = saved_tokens / baseline_tokens * 100 if baseline_tokens > 0 else 0.0

        return {
            "sessions": int(ledger.get("sessions", 0)),
            "total_turns": int(ledger.get("total_turns", 0)),
            "actual_tokens": actual_tokens,
            "baseline_tokens": baseline_tokens,
            "tokens_saved": saved_tokens,
            "savings_pct": round(savings_pct, 1),
            "cost_savings_pct": round(cost_savings_pct, 1),
            "actual_cost_usd": actual_cost,
            "baseline_cost_usd": baseline_cost,
            "cost_saved_usd": float(ledger.get("cost_saved_usd", 0.0)),
            "fallback_count": int(ledger.get(FALLBACK_SIGNAL, 0)),
            "baseline_only_requests": int(ledger.get(BASELINE_ONLY_SIGNAL, 0)),
        }

    def last_session_summary(self) -> dict[str, int | float | str] | None:
        """Return the most recent completed session from the lifetime log."""
        entries = self._load_session_log_entries()
        if not entries:
            return None
        entry = entries[-1]
        return {
            "date": str(entry["date"]),
            "turns": int(entry["turns"]),
            "actual_tokens": int(entry["tokens"]),
            "baseline_tokens": int(entry["tokens"]) + int(entry["tokens_saved"]),
            "tokens_saved": int(entry["tokens_saved"]),
            "savings_pct": float(entry["savings_pct"]),
            "cost_savings_pct": float(entry["cost_savings_pct"]),
            "actual_cost_usd": float(entry["actual_cost_usd"]),
            "baseline_cost_usd": float(entry["baseline_cost_usd"]),
            "cost_saved_usd": float(entry["saved_usd"]),
            "session_quality": (
                "degraded"
                if str(entry.get("degradation_reason", "")) == "baseline fallback"
                else ("watch" if str(entry.get("degradation_reason", "")) else "clean")
            ),
            "last_degradation_reason": str(entry.get("degradation_reason", "")),
        }

    def format_last_session(self) -> str | None:
        """Backward-compatible formatter for the most recent completed session."""
        summary = self.last_session_summary()
        if summary is None:
            return None

        return (
            "Last completed session:\n"
            f"- date: {summary['date']}\n"
            f"- turns: {int(summary['turns'])}\n"
            f"- actual tokens: {int(summary['actual_tokens']):,}\n"
            f"- baseline tokens: {int(summary['baseline_tokens']):,}\n"
            f"- saved: {int(summary['tokens_saved']):,} ({float(summary['savings_pct']):.1f}%)\n"
            f"- actual cost: ${float(summary['actual_cost_usd']):.4f}\n"
            f"- baseline cost: ${float(summary['baseline_cost_usd']):.4f}\n"
            f"- cost saved: ${float(summary['cost_saved_usd']):.4f}\n"
            f"- session quality: {summary['session_quality']}\n"
            f"- degradation reason: {summary['last_degradation_reason'] or 'none'}"
        )

    def recent_summary(self, recent_sessions: int) -> dict[str, int | float | str] | None:
        """Return an aggregate summary over the most recent completed sessions."""
        entries = self._load_session_log_entries()
        if not entries:
            return None

        recent_sessions = max(1, recent_sessions)
        recent = entries[-recent_sessions:]
        return self._session_window_summary(recent, label=f"Last {len(recent)} sessions")

    def since_summary(self, since: str) -> dict[str, int | float | str] | None:
        """Return an aggregate summary over completed sessions since an ISO date."""
        entries = self._load_session_log_entries()
        if not entries:
            return None

        window = [entry for entry in entries if str(entry["date"]) >= since]
        if not window:
            return None
        return self._session_window_summary(window, label=f"Since {since}")

    def trend_summary(self, recent_sessions: int = 5) -> dict[str, Any]:
        entries = self._load_session_log_entries()
        if not entries:
            return {
                "sessions_considered": 0,
                "direction": "none",
                "avg_savings_pct": 0.0,
                "avg_invisible_pressure": 0.0,
                "avg_tokens_saved": 0.0,
                "avg_semantic_regression": 0.0,
                "savings_velocity": 0.0,
                "pressure_velocity": 0.0,
                "memory_lift_velocity": 0.0,
                "semantic_regression_velocity": 0.0,
            }

        # Ensure we don't request more sessions than available
        recent_sessions = min(recent_sessions, len(entries))
        recent = entries[-recent_sessions:]
        savings = [
            float(entry["savings_pct"]) for entry in recent if isinstance(entry.get("savings_pct"), int | float | str)
        ]
        pressure = [
            float(entry["invisible_pressure"])
            for entry in recent
            if isinstance(entry.get("invisible_pressure"), int | float | str)
        ]
        saved_tokens = [
            float(entry["tokens_saved"]) for entry in recent if isinstance(entry.get("tokens_saved"), int | float | str)
        ]
        memory_lifts = [
            float(entry.get("memory_lift", 0))
            for entry in recent
            if isinstance(entry.get("memory_lift"), int | float | str)
        ]
        sem_regressions = [
            float(entry.get("semantic_regression", 0))
            for entry in recent
            if isinstance(entry.get("semantic_regression"), int | float | str)
        ]

        # Calculate trend direction
        direction = "flat"
        if len(recent) >= 2:
            if savings[-1] > savings[0] and pressure[-1] <= pressure[0]:
                direction = "improving"
            elif savings[-1] < savings[0] or pressure[-1] > pressure[0]:
                direction = "regressing"

        # Calculate trend velocity (rate of change per session)
        savings_velocity = 0.0
        pressure_velocity = 0.0
        memory_lift_velocity = 0.0
        semantic_regression_velocity = 0.0

        if len(recent) >= 2:
            # Simple linear regression slope calculation
            x_values = list(range(len(recent)))
            n = len(recent)
            sum_x = sum(x_values)
            sum_x2 = sum(x * x for x in x_values)
            denom = n * sum_x2 - sum_x * sum_x

            def _slope(y_values: list[float]) -> float:
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=True))
                return (n * sum_xy - sum_x * sum_y) / denom

            savings_velocity = _slope(savings)
            pressure_velocity = _slope(pressure)
            memory_lift_velocity = _slope(memory_lifts)
            semantic_regression_velocity = _slope(sem_regressions)

        return {
            "sessions_considered": len(recent),
            "direction": direction,
            "avg_savings_pct": round(statistics.mean(savings), 1),
            "avg_invisible_pressure": round(statistics.mean(pressure), 1),
            "avg_tokens_saved": round(statistics.mean(saved_tokens), 1),
            "avg_memory_lift": round(statistics.mean(memory_lifts), 1),
            "avg_semantic_regression": round(statistics.mean(sem_regressions), 1),
            "savings_velocity": round(savings_velocity, 2),
            "pressure_velocity": round(pressure_velocity, 2),
            "memory_lift_velocity": round(memory_lift_velocity, 2),
            "semantic_regression_velocity": round(semantic_regression_velocity, 2),
        }

    def _session_window_summary(
        self, entries: list[dict[str, float | int | str]], label: str
    ) -> dict[str, int | float | str]:
        actual_tokens = sum(int(entry["tokens"]) for entry in entries)
        tokens_saved = sum(int(entry["tokens_saved"]) for entry in entries)
        baseline_tokens = actual_tokens + tokens_saved
        actual_cost = sum(float(entry["actual_cost_usd"]) for entry in entries)
        baseline_cost = sum(float(entry["baseline_cost_usd"]) for entry in entries)
        cost_saved = baseline_cost - actual_cost
        cost_savings_pct = (cost_saved / baseline_cost * 100) if baseline_cost > 0 else 0.0
        savings_pct = tokens_saved / baseline_tokens * 100 if baseline_tokens > 0 else 0.0

        return {
            "label": label,
            "sessions": len(entries),
            "date_start": str(entries[0]["date"]),
            "date_end": str(entries[-1]["date"]),
            "turns": sum(int(entry["turns"]) for entry in entries),
            "actual_tokens": actual_tokens,
            "baseline_tokens": baseline_tokens,
            "tokens_saved": tokens_saved,
            "savings_pct": round(savings_pct, 1),
            "cost_savings_pct": round(cost_savings_pct, 1),
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "cost_saved_usd": round(cost_saved, 6),
            "session_quality": (
                "watch" if any(str(entry.get("degradation_reason", "")) for entry in entries) else "clean"
            ),
            "last_degradation_reason": str(entries[-1].get("degradation_reason", "")),
        }

    def behavior_signals(self) -> dict[str, int]:
        """Aggregate workaround/redundancy signals across current session stats."""
        stats = self.load_stats()
        merged: dict[str, int] = {}
        for model_stats in stats.get("models", {}).values():
            for key, value in model_stats.get("behavior_signals", {}).items():
                merged[key] = merged.get(key, 0) + value
        return merged

    def behavior_summary(self) -> dict[str, str | int]:
        """Interpret current session behavior signals into operator-friendly status."""
        signals = self.behavior_signals()
        invisible_pressure = calculate_invisible_pressure(signals)
        memory_lift = calculate_memory_lift(signals)
        semantic_regression = calculate_semantic_regression_score(signals)
        if invisible_pressure == 0:
            status = "clean"
        elif invisible_pressure <= 3:
            status = "watch"
        else:
            status = "noisy"
        return {
            "status": status,
            "invisible_pressure": invisible_pressure,
            "memory_lift": memory_lift,
            "semantic_regression_score": semantic_regression,
            "degradation_reason": _degradation_reason(
                signals,
                baseline_only=bool(signals.get(BASELINE_ONLY_SIGNAL, 0)),
            ),
        }

    def _load_session_log_entries(self) -> list[dict[str, Any]]:
        if not self._ledger_path.exists():
            return []

        entries: list[dict[str, float | int | str]] = []
        in_log = False
        for line in self._ledger_path.read_text().splitlines():
            s = line.strip()
            if s.startswith("@per_session_log"):
                in_log = True
                continue
            if not in_log or not s or s.startswith("#"):
                continue

            parts = s.split(";")
            if len(parts) < 10:
                continue

            try:
                baseline_cost = float(parts[5])
                actual_cost = float(parts[4])
                tokens_saved = int(parts[7])
                invisible_pressure = int(parts[8])
                memory_lift = int(parts[10]) if len(parts) > 10 else 0
                semantic_regression = int(parts[11]) if len(parts) > 11 else 0
                reacquisition_count = int(parts[12]) if len(parts) > 12 else 0
                answer_anchor_miss_count = int(parts[13]) if len(parts) > 13 else 0
                degradation_reason = parts[14] if len(parts) > 14 else ""
                fallback_activated = int(parts[17]) if len(parts) > 17 and parts[17] else 0
                baseline_only_session = int(parts[18]) if len(parts) > 18 and parts[18] else 0
                baseline_prompt_tokens = int(parts[19]) if len(parts) > 19 and parts[19] else 0
                prepared_prompt_tokens = int(parts[20]) if len(parts) > 20 and parts[20] else 0
                saved_prompt_tokens = int(parts[21]) if len(parts) > 21 and parts[21] else 0
                hot_hint_tokens_added = int(parts[22]) if len(parts) > 22 and parts[22] else 0
                reacquisition_avoided = int(parts[23]) if len(parts) > 23 and parts[23] else 0
                reacquisition_cost_tokens = int(parts[24]) if len(parts) > 24 and parts[24] else 0
                actual_tokens = int(parts[3])
                baseline_tokens = actual_tokens + tokens_saved
                cost_savings_pct = ((baseline_cost - actual_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0.0
                savings_pct = tokens_saved / baseline_tokens * 100 if baseline_tokens > 0 else 0.0
            except ValueError:
                continue

            entries.append(
                {
                    "date": parts[0],
                    "session_id": parts[1],
                    "turns": int(parts[2]),
                    "tokens": actual_tokens,
                    "actual_cost_usd": actual_cost,
                    "baseline_cost_usd": baseline_cost,
                    "saved_usd": float(parts[6]),
                    "tokens_saved": tokens_saved,
                    "invisible_pressure": invisible_pressure,
                    "non_tok_response": int(parts[9]),
                    "memory_lift": memory_lift,
                    "semantic_regression": semantic_regression,
                    "reacquisition_count": reacquisition_count,
                    "answer_anchor_miss_count": answer_anchor_miss_count,
                    "degradation_reason": degradation_reason,
                    FALLBACK_SIGNAL: fallback_activated,
                    BASELINE_ONLY_SIGNAL: baseline_only_session,
                    "baseline_prompt_tokens": baseline_prompt_tokens,
                    "prepared_prompt_tokens": prepared_prompt_tokens,
                    "saved_prompt_tokens": saved_prompt_tokens,
                    "hot_hint_tokens_added": hot_hint_tokens_added,
                    "reacquisition_tokens_avoided_estimate": reacquisition_avoided,
                    "reacquisition_cost_tokens": reacquisition_cost_tokens,
                    "savings_pct": savings_pct,
                    "cost_savings_pct": cost_savings_pct,
                }
            )
        seen: dict[str, dict[str, Any]] = {}
        for entry in entries:
            seen[str(entry["session_id"])] = entry
        result = list(seen.values())
        result.sort(key=lambda e: str(e["date"]))
        return result

    def reset_ledger(self) -> None:
        """Clear the persistent savings ledger."""
        with self._lock:
            if self._ledger_path.exists():
                self._ledger_path.unlink()
            default_ledger = self._default_ledger()
            default_ledger["sessions"] = 0
            out = self._format_ledger_output(default_ledger, 0.0, [])
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._ledger_path.write_text("\n".join(out))
            logger.info("Lifetime ledger reset")
