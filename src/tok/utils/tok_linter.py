import argparse
import sys
from typing import Any

from ..protocol.models import TokNode, Trust
from ..protocol.parser import TokParser
from ..utils.token_utils import count_tokens


class SemanticAuditor:
    RECURSION_LIMIT = 10

    def check_recursion(self, depth: int) -> bool:
        return depth < self.RECURSION_LIMIT

    """
    The Tok Semantic Auditor measures the efficiency and resilience of Tok payloads.
    It focuses on Reasoning-to-Noise Ratio (RNR) and Structural Debt.
    """

    def __init__(self, model: str = "gpt-4", threshold: float = 1.0) -> None:
        self.threshold = threshold

    count_tokens = staticmethod(count_tokens)

    def audit_v7(self, text: str) -> bool:
        # Check for strict typing (key: value without indent)
        import re

        lines = text.split("\n")
        strict_types = all(not re.search(r"^\s+\w+:", line) for line in lines)
        v7_headers = any(
            "@protocol version:7.0" in line or "@shard version:7.0" in line
            for line in lines
        )
        return strict_types and v7_headers

    def audit(self, text: str) -> dict[str, Any]:
        parser = TokParser()
        nodes = parser.parse(text)

        total_tokens = self.count_tokens(text)
        reasoning_tokens = self._count_reasoning(nodes)
        structural_tokens = self._count_structure(text, nodes)

        debt_report = self._analyze_debt(nodes)

        # RNR: (Reasoning) / (Total)
        rnr = (reasoning_tokens / total_tokens) if total_tokens > 0 else 0

        return {
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "structural_tokens": structural_tokens,
            "rnr": round(rnr, 4),
            "debt": debt_report,
            "suggestions": self._generate_suggestions(nodes, debt_report),
            "resilience": self._check_resilience(text),
            "efficiency_math": self._calculate_efficiency_math(nodes),
        }

    def _calculate_efficiency_math(
        self, _nodes: list[TokNode]
    ) -> dict[str, str | int | float]:
        """
        Compare 'Upfront Schema' (System Prompt) vs 'On-Demand Error Injection'.
        """
        # Assumptions based on typical LLM agent usage
        SCHEMA_AVG_TOKENS = (
            150  # Tokens for a detailed schema in system prompt
        )
        ERROR_MSG_AVG_TOKENS = 40  # Tokens for a surgical error message
        FAILURE_RATE = 0.05  # 5% chance of the model making a schema error

        # Upfront Cost: Paid every single turn
        upfront_turn_cost = SCHEMA_AVG_TOKENS

        # On-Demand Cost: Paid only on failure (Error Msg + Retry Tokens)
        # Assuming Retry takes ~50 tokens for thoughts + corrected block
        on_demand_turn_cost = (ERROR_MSG_AVG_TOKENS + 50) * FAILURE_RATE

        return {
            "upfront_cost_per_turn": upfront_turn_cost,
            "on_demand_expected_cost": round(on_demand_turn_cost, 2),
            "savings_per_turn": upfront_turn_cost
            - round(on_demand_turn_cost, 2),
            "verdict": (
                "On-Demand is significantly more efficient"
                if on_demand_turn_cost < upfront_turn_cost
                else "Upfront is better"
            ),
        }

    def _check_resilience(self, text: str) -> dict[str, float | str]:
        """
        Verify the 'Complete Partial Tok' philosophy.
        Tries parsing the text after truncating it at various points.
        """
        parser = TokParser()
        lines = text.split("\n")
        resilience_score = 1.0

        # Test truncation at 50% and 90%
        for pct in [0.5, 0.9]:
            cutoff = int(len(lines) * pct)
            partial_text = "\n".join(lines[:cutoff])
            try:
                # If it doesn't crash and returns at least some nodes, it's partially
                # valid
                nodes = parser.parse(partial_text)
                if not nodes and cutoff > 0:
                    resilience_score -= 0.2
            except Exception:
                resilience_score -= 0.5

        return {
            "score": max(0, resilience_score),
            "status": (
                "Resilient (Complete Partial Tok)"
                if resilience_score > 0.8
                else "Degraded"
            ),
        }

    def _count_reasoning(self, nodes: list[TokNode]) -> int:
        total = 0
        for node in nodes:
            # We treat text in trusted blocks (like @thought or lines with '>') as
            # reasoning
            if node.type.lower() in ("thought", "reasoning"):
                total += self.count_tokens(node.text)
            elif node.trust == Trust.SYSTEM:
                # In common blocks, we look for thoughts prefixed with '>'
                # However, the current parser merges text. We might need to check if
                # the source lines started with '>'.
                # For now, let's treat the entire text of a SYSTEM node as potential reasoning
                # if it's not a payload block.
                total += self.count_tokens(node.text)

            total += self._count_reasoning(node.children)
        return total

    def _count_structure(self, text: str, nodes: list[TokNode]) -> int:
        # Structure is roughly (Total - Content)
        # Content = text + attr values + row values
        content_tokens = 0

        def walk(ns: list[TokNode]) -> None:
            nonlocal content_tokens
            for n in ns:
                content_tokens += self.count_tokens(n.text)
                for v in n.attrs.values():
                    content_tokens += self.count_tokens(str(v))
                for row in n.rows:
                    for cell in row:
                        content_tokens += self.count_tokens(str(cell))
                walk(n.children)

        walk(nodes)
        total = self.count_tokens(text)
        return max(0, total - content_tokens)

    def _analyze_debt(self, nodes: list[TokNode]) -> list[dict[str, Any]]:
        debt: list[dict[str, Any]] = []

        def walk(ns: list[TokNode], path: str = "") -> None:
            for n in ns:
                curr_path = f"{path}/{n.type}"
                if n.label:
                    curr_path += f"[{n.label}]"

                # Check attributes for high waste
                for k, v in n.attrs.items():
                    key_tok = self.count_tokens(k)
                    val_tok = self.count_tokens(str(v))
                    if key_tok > (val_tok * self.threshold) and key_tok > 2:
                        debt.append(
                            {
                                "path": curr_path,
                                "type": "structural_waste",
                                "key": k,
                                "key_tokens": key_tok,
                                "val_tokens": val_tok,
                                "ratio": round(
                                    (
                                        key_tok / val_tok
                                        if val_tok > 0
                                        else key_tok
                                    ),
                                    2,
                                ),
                                "message": f"Attribute '{k}' ({key_tok} tokens) costs more than {self.threshold}x its value ({val_tok} tokens).",
                            }
                        )

                # Check for repetitive keys across children (Density opportunity)
                if len(n.children) > 2:
                    all_keys = [set(c.attrs.keys()) for c in n.children]
                    common_keys = (
                        set.intersection(*all_keys) if all_keys else set()
                    )
                    if common_keys:
                        for k in common_keys:
                            debt.append(
                                {
                                    "path": curr_path,
                                    "type": "repetition",
                                    "key": k,
                                    "message": f"Key '{k}' is repeated across {len(n.children)} siblings. Suggest moving to header.",
                                }
                            )

                walk(n.children, curr_path)

        walk(nodes)
        return debt

    def _generate_suggestions(
        self, _nodes: list[TokNode], debt: list[dict[str, str | int | float]]
    ) -> list[str]:
        suggestions = []
        processed_keys = set()

        for d in debt:
            if d["type"] == "structural_waste":
                suggestions.append(
                    f"Shorten key '{d['key']}' at {d['path']} to save {int(d['key_tokens']) - 1} tokens."
                )
            elif d["type"] == "repetition" and d["key"] not in processed_keys:
                suggestions.append(
                    f"Move repeated key '{d['key']}' in {d['path']} siblings to a parent-level header: [{d['key']}|...]."
                )
                processed_keys.add(d["key"])

        return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tok Semantic Auditor: Measure RNR and Structural Debt."
    )
    parser.add_argument("file", help="Path to Tok file to audit")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Structural debt threshold (default: 1.0)",
    )
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    auditor = SemanticAuditor(threshold=args.threshold)
    report = auditor.audit(content)

    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print("─" * 40)
        print(f"TOK SEMANTIC AUDIT: {args.file}")
        print("─" * 40)
        print(f"Total Tokens:      {report['total_tokens']}")
        print(f"Reasoning Tokens:   {report['reasoning_tokens']}")
        print(f"Structural Tokens: {report['structural_tokens']}")
        print(f"RNR (Reasoning-to-Noise): {report['rnr']}")
        print("─" * 40)

        if report["debt"]:
            print("\nDEBT DETECTED:")
            for d in report["debt"]:
                print(f" • [{d['type'].upper()}] {d['message']}")

        if report["suggestions"]:
            print("\nSUGGESTIONS:")
            for s in report["suggestions"]:
                print(f" → {s}")

        print(
            f"\nRESILIENCE REPORT: {report['resilience']['status']} ({report['resilience']['score']})"
        )

        em = report["efficiency_math"]
        print("\nEFFICIENCY MATH (Inverted Perspective):")
        print(
            f" • Upfront Schema Cost:   {em['upfront_cost_per_turn']} tokens/turn"
        )
        print(
            f" • On-Demand Expected:    {em['on_demand_expected_cost']} tokens/turn"
        )
        print(
            f" • Net Savings:           {em['savings_per_turn']} tokens/turn"
        )
        print(f" → Verdict: {em['verdict']}")
        print("─" * 40)


if __name__ == "__main__":
    main()
