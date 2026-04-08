"""Bridge module for converting between JSON/XML/MD and Tok format."""

import functools
import json
import logging
import random
import string
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any, cast

from .encoder import TokEncoder
from .models import TokNode
from .protocol import SerializationProtocol

logger = logging.getLogger(__name__)

MINIMAL_TOK_MANUAL = """
Tok: @Type (indent body). @UPPER: Tool. @lower: Msg.
Attrs: key: val (1/line). *label: Ref. !val: Literal.
Thought: > text. Sandbox: |> text. Table: col|row.
Stability: Repeat headers every 10 rows as heartbeats.
"""


class Bridge(SerializationProtocol):
    """The Invisible Bridge: JSON/XML/MD <-> Tok with JIT Activation."""

    @staticmethod
    def dispatch_async_v7(callback_id: str, _payload: object) -> str:
        """V7 asynchronous dispatch."""
        return f"v7_async_{callback_id}"

    verified_agents: set[str] = set()

    @staticmethod
    def should_upgrade(payload: str) -> bool:
        """JIT: Only use Tok if payload > 500 chars to offset instruction tax."""
        return len(payload) > 500

    def __init__(self) -> None:
        pass

    def __call__(self, arg: Callable[..., Any] | str | dict[str, Any] | list[Any]) -> Any:
        """Dual-mode entry: decorator OR direct conversion."""
        if callable(arg) and not isinstance(arg, str | dict | list):
            return self._decorator(arg)
        return self.detect_and_convert(arg)

    def _decorator(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            new_args = list(args)
            if new_args:
                input_val = new_args[0]
                if isinstance(input_val, dict):
                    new_args[0] = Bridge.json(json.dumps(input_val))
                elif isinstance(input_val, str) and (input_val.strip().startswith(("{", "[", "<", "|"))):
                    new_args[0] = self.detect_and_convert(input_val)

            tok_response = func(*tuple(new_args), **kwargs)

            if isinstance(tok_response, str) and (tok_response.startswith(("@", "|"))):
                return json.loads(Bridge.to_json(tok_response))
            return tok_response

        return wrapper

    def detect_and_convert(self, payload: str | dict[str, Any] | list[Any]) -> str:
        """Auto-detect format and convert to Tok."""
        if not isinstance(payload, str):
            if isinstance(payload, dict):
                return self.json(json.dumps(payload))
            return str(payload)

        s = payload.strip()
        if not s:
            return ""
        if s.startswith(("{", "[")):
            result = self.json(s)
            if result:
                return result
            return s
        if s.startswith("<"):
            return self.xml(s)
        if s.startswith("|"):
            return self.md(s)
        result = self.json(s)
        if result:
            return result
        return s

    @staticmethod
    def raw(content: str, role: str = "user") -> str:
        """Wrap any messy string in a Verbatim Tok block."""
        session_hash = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        return f"@msg role:{role}\n  |#{session_hash}>\n{content.strip()}\n  |#{session_hash}"

    @staticmethod
    def execute(
        llm_callable: Callable[[str], Any],
        payload: dict[str, Any],
        _schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The Invisible Bridge with JIT activation."""
        json_payload = json.dumps(payload)
        if Bridge.should_upgrade(json_payload):
            tok_input = MINIMAL_TOK_MANUAL + "\n" + Bridge.json(json_payload)
            logger.debug("Bridge upgrade: Tok + Manual (%d chars)", len(tok_input))
        else:
            tok_input = json_payload
            logger.debug("Bridge standard: JSON (%d chars)", len(tok_input))

        response_tok = llm_callable(tok_input)

        if isinstance(response_tok, str) and (
            response_tok.strip().startswith("@") or response_tok.strip().startswith("|")
        ):
            return cast("dict[str, Any]", json.loads(Bridge.to_json(response_tok)))
        try:
            return cast("dict[str, Any]", json.loads(response_tok))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"raw_response": response_tok}

    @staticmethod
    def is_valid_tok_response(response: str) -> bool:
        """Validate response contains valid Tok structure (lenient mode)."""
        if not isinstance(response, str) or not response.strip():
            return False

        from .parser import TokParser

        failure_phrases = [
            "don't understand",
            "cannot understand",
            "invalid format",
            "unknown token",
            "don't know how",
            "json",
            "syntax error",
        ]
        lower_res = response.lower()
        if any(phrase in lower_res for phrase in failure_phrases):
            return False

        parser = TokParser()
        nodes = parser.parse(response)
        if not nodes:
            return False

        valid_types = {
            "thought",
            "tool",
            "msg",
            "result",
            "delegate",
            "meta",
            "log",
        }
        return any(n.type.lower() in valid_types for n in nodes)

    @staticmethod
    def delegate(
        llm_callable: Callable[[str], Any],
        agent_id: str,
        task: str,
        bootstrap: str = "essentials",
        is_external: bool = True,
    ) -> dict[str, Any]:
        """Hand off task to agent with tiered grammar bootstrapping."""
        from tok.analysis.prompt import get_grammar_snippet

        if agent_id in Bridge.verified_agents:
            active_level = None
        else:
            active_level = "full" if is_external and bootstrap == "essentials" else bootstrap

        grammar = get_grammar_snippet(active_level) if active_level else ""
        if active_level:
            prompt = f"{grammar}\n\n@Delegate agent:{agent_id}\n  task: {task}"
        else:
            prompt = (
                f"@Delegate agent:{agent_id}\n  task: {task}\n\n"
                "Respond using Tok format:\n"
                "@msg role:assistant\n"
                "  |> your response"
            )

        try:
            result = llm_callable(prompt)

            response = result[0] if isinstance(result, tuple) else result

            if not Bridge.is_valid_tok_response(response):
                full_grammar = get_grammar_snippet("full")
                retry_prompt = (
                    f"[PROTOCOL ERROR] Use Tok format.\n{full_grammar}\n\n@Delegate agent:{agent_id}\n  task: {task}"
                )
                result = llm_callable(retry_prompt)

                response = result[0] if isinstance(result, tuple) else result

                if not Bridge.is_valid_tok_response(response):
                    return {"error": "protocol_mismatch", "raw": response}

            Bridge.verified_agents.add(agent_id)
            return cast("dict[str, Any]", json.loads(Bridge.to_json(response)))

        except Exception as e:
            return {"error": "transport_failure", "details": str(e)}

    @staticmethod
    def json(json_str: str) -> str:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ""

        def _to_nodes(name: str, value: str | dict[str, Any] | list[Any] | None) -> TokNode:
            if isinstance(value, dict):
                if not value:
                    return TokNode(type=name, attrs={"_empty_dict": True})
                node = TokNode(type=name)
                for k, v in value.items():
                    if isinstance(v, dict | list):
                        node.children.append(_to_nodes(k, v))
                    else:
                        node.attrs[k] = v
                return node
            if isinstance(value, list):
                if not value:
                    return TokNode(type=name, attrs={"_empty_list": True})

                if all(isinstance(i, dict) for i in value):
                    first_keys = set(value[0].keys())
                    if all(set(i.keys()) == first_keys for i in value):
                        headers = sorted(first_keys)
                        node = TokNode(type=name, headers=headers)
                        for i in value:
                            node.rows.append([i[h] for h in headers])
                        return node

                container = TokNode(type=name)
                for item in value:
                    if isinstance(item, dict | list):
                        container.children.append(_to_nodes("item", item))
                    else:
                        container.children.append(TokNode(type="item", text=str(item)))
                return container
            return TokNode(type=name, text=str(value))

        root = _to_nodes("data", data)
        return TokEncoder.encode([root])

    @staticmethod
    def xml(xml_str: str) -> str:
        try:
            root = ET.fromstring(xml_str)
            node = TokNode(type=root.tag, attrs=root.attrib, text=root.text or "")
            for child in root:
                node.children.append(
                    TokNode(
                        type=child.tag,
                        attrs=child.attrib,
                        text=child.text or "",
                    )
                )
            return TokEncoder.encode([node])
        except Exception:
            return ""

    @staticmethod
    def md(md_str: str, table_name: str = "result") -> str:
        lines = [line.strip() for line in md_str.strip().split("\n") if line.strip()]
        if not lines:
            return ""
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        rows = []
        for line in lines[2:]:
            rows.append([c.strip() for c in line.split("|") if c.strip()])
        node = TokNode(type=table_name, headers=headers, rows=rows)
        return TokEncoder.encode([node])

    @staticmethod
    def markdown(md_str: str, table_name: str = "result") -> str:
        """Alias for md() for legacy test scripts."""
        return Bridge.md(md_str, table_name)

    @staticmethod
    def to_json(tok_str: str) -> str:
        from .parser import TokParser

        parser = TokParser()
        nodes = parser.parse(tok_str)
        if not nodes:
            trimmed = tok_str.strip()
            if trimmed:
                nodes = parser.parse(f"@msg\n  |> {trimmed}")
            if not nodes:
                return "{}"

        def _node_value(node: TokNode) -> Any:
            attrs = dict(node.attrs)
            is_empty_list = attrs.pop("_empty_list", False)
            is_empty_dict = attrs.pop("_empty_dict", False)

            if is_empty_list:
                return []
            if is_empty_dict:
                return {}

            if node.headers and node.rows:
                return [dict(zip(node.headers, r, strict=True)) for r in node.rows]

            if node.children:
                if all(child.type == "item" for child in node.children):
                    return [_node_value(child) for child in node.children]

                merged: dict[str, Any] = dict(attrs)
                for child in node.children:
                    child_value = _node_value(child)
                    existing = merged.get(child.type)
                    if existing is None:
                        merged[child.type] = child_value
                    elif isinstance(existing, list):
                        existing.append(child_value)
                    else:
                        merged[child.type] = [existing, child_value]
                return merged

            if attrs:
                return dict(attrs)

            text = node.text.strip()
            if text:
                return parser._cast(text)

            return ""

        def _rehydrate(nodes_list: list[TokNode]) -> Any:
            if len(nodes_list) == 1 and nodes_list[0].type.lower() == "data":
                return _node_value(nodes_list[0])

            aggregated: dict[str, Any] = {}
            for node in nodes_list:
                value = _node_value(node)
                if node.type in aggregated:
                    existing = aggregated[node.type]
                    if isinstance(existing, list):
                        existing.append(value)
                    else:
                        aggregated[node.type] = [existing, value]
                else:
                    aggregated[node.type] = value
            return aggregated

        hydrated = _rehydrate(nodes)
        try:
            return json.dumps(hydrated, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"

    @staticmethod
    def to_xml(tok_str: str) -> str:
        from .parser import TokParser

        parser = TokParser()
        nodes = parser.parse(tok_str)
        if not nodes:
            return ""
        node = nodes[0]
        attrs = {k: str(v) for k, v in node.attrs.items()}
        root = ET.Element(node.type, attrs)
        root.text = node.text
        for child in node.children:
            c_attrs = {k: str(v) for k, v in child.attrs.items()}
            c_elem = ET.SubElement(root, child.type, c_attrs)
            c_elem.text = child.text
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def to_md(tok_str: str) -> str:
        from .parser import TokParser

        parser = TokParser()
        nodes = parser.parse(tok_str)
        if not nodes or not nodes[0].headers:
            return ""
        node = nodes[0]
        lines = ["| " + " | ".join(node.headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(node.headers)) + " |")
        for row in node.rows:
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)

    def encode(self, data: object) -> str:
        """
        Encode data to Tok text. Satisfies SerializationProtocol.

        Args:
            data: Any data to encode (dict, str, list, etc.)

        Returns:
            Serialized Tok text

        """
        # Type assertion to handle the generic object parameter
        if not isinstance(data, (str, dict, list)):
            raise TypeError(f"Expected str, dict, or list, got {type(data)}")
        return Bridge().detect_and_convert(data)

    @staticmethod
    def decode(text: str) -> Any:
        """
        Decode Tok text to data. Satisfies SerializationProtocol.

        Args:
            text: Tok text to decode

        Returns:
            Deserialized data (typically dict or list)

        """
        result = Bridge.to_json(text)
        if not result:
            return {}
        return json.loads(result)
