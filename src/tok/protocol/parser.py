"""Tok parser module - LL(1) line-driven state machine."""

import logging
import random
import re
import string
from typing import Any, cast

from .models import TokNode, Trust
from .protocol import SerializationProtocol

_logger = logging.getLogger("tok.protocol.parser")


def tok_to_dict(node: TokNode) -> dict[str, Any]:
    """Convert a TokNode tree to a plain dictionary for visualization."""
    d: dict[str, Any] = {"_type": node.type}
    if node.label:
        d["_label"] = node.label
    if node.trust != Trust.SYSTEM:
        d["_trust"] = node.trust.value
    if node.text.strip():
        d["_text"] = node.text.strip()
    if node.cardinality is not None:
        d["_cardinality"] = node.cardinality
    d.update(node.attrs)
    if node.headers and node.rows:
        d["_rows"] = [dict(zip(node.headers, row, strict=True)) for row in node.rows]
    if node.children:
        d["_children"] = [tok_to_dict(c) for c in node.children]
    return d


def serialize(nodes: list[TokNode], _depth: int = 0, compact: bool = False) -> str:
    """Serialize a list of TokNodes back to canonical Tok text."""
    lines = []
    pad = "" if compact else "  " * _depth
    child_pad = "" if compact else pad + "  "

    for node in nodes:
        # Header: @type label [headers] {attrs}
        parts = [f"@{node.type}"]
        if node.label:
            parts.append(node.label)
        if node.ref:
            parts.append(f"*{node.ref}")
        if node.trust != Trust.SYSTEM:
            parts.append(f"trust:{node.trust.value}")
        if node.headers:
            parts.append(f"[{'|'.join(node.headers)}]")
        elif node.cardinality is not None:
            parts.append(f"[{node.cardinality}]")

        # Attributes in header if they are simple (single-word values only)
        for k, v in node.attrs.items():
            if (
                not k.startswith("_")
                and isinstance(v, str | int | float | bool)
                and len(str(v)) < 50
                and " " not in str(v)
            ):
                parts.append(f"{k}:{v}")

        lines.append(pad + " ".join(parts))

        # Rows (for tables)
        for row in node.rows:
            row_str = " | ".join(str(v) if v is not None else "" for v in row)
            lines.append(child_pad + row_str)

        # Text content
        text = node.text.strip()
        if text:
            lines_raw = text.split("\n")
            needs_verbatim = False
            if any(line.strip().startswith(("@", ">", "|", "#")) for line in lines_raw) or (
                ":" in text and len(text) > 100
            ):
                needs_verbatim = True

            if needs_verbatim and not compact:
                h = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                lines.append(f"{child_pad}|#{h}>")
                lines.append(text)
                lines.append(f"{child_pad}|#{h}")
            else:
                for text_line in text.rstrip("\n").split("\n"):
                    if compact:
                        lines.append(text_line.strip())
                    else:
                        prefix = "> "
                        if node.trust == Trust.UNTRUSTED:
                            prefix = "|> "
                        # PRESERVE leading whitespace for verbatim blocks or just be careful here
                        # Actually, Tok usually indents text content.
                        lines.append(f"{child_pad}{prefix}{text_line}")

        # Complex attributes — anything not inlined in the header
        for k, v in node.attrs.items():
            if k.startswith("_") or not isinstance(v, str | int | float | bool) or len(str(v)) >= 50 or " " in str(v):
                lines.append(f"{child_pad}{k}: {v}")

        # Children
        if node.children:
            lines.append(serialize(node.children, _depth + 1, compact=compact))

        if not compact:
            lines.append("")

    if compact:
        return " ".join(line.strip() for line in lines if line.strip())
    return "\n".join(lines).rstrip()


def tok_to_tok(node: TokNode) -> str:
    """Serialize a single TokNode to canonical Tok text."""
    return serialize([node], compact=False)


class TokParser(SerializationProtocol):
    """LL(1) line-driven state machine for Tok. Supports streaming via feed()/flush()."""

    MAX_LINE_LENGTH = 65536

    def __init__(self) -> None:
        self._stack: list[tuple[int, TokNode]] = []
        self._buffer: str = ""
        self._active_boundary: str | None = None
        self._pending_header: str = ""
        self._in_inverted_body: bool = False

    @property
    def current_node(self) -> TokNode | None:
        if self._stack:
            return self._stack[-1][1]
        return None

    def parse(self, text: str) -> list[TokNode]:
        """Parse a complete Tok document."""
        self._stack = []
        self._buffer = ""
        self._active_boundary = None
        self._pending_header = ""
        self._in_inverted_body = False
        result = []

        # Strip UTF-8 BOM if present
        if text.startswith("\ufeff"):
            text = text[1:]
        elif text.startswith("\xef\xbb\xbf"):
            text = text[3:]

        lines = text.split("\n")

        # Robustness: Check if there's any content before the first @type
        # or if there are no @type declarations at all.
        first_at_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("@"):
                first_at_idx = i
                break

        if first_at_idx == -1 and text.strip():
            # No @type at all, wrap everything
            result.extend(self._ingest_line("@msg"))
            for line in lines:
                result.extend(self._ingest_line("  " + line))
        elif first_at_idx > 0:
            # Content before first @type, wrap that part
            pre_content = lines[:first_at_idx]
            if any(line.strip() for line in pre_content):
                result.extend(self._ingest_line("@msg"))
                for line in pre_content:
                    result.extend(self._ingest_line("  " + line))
                # Then process the rest normally
                for line in lines[first_at_idx:]:
                    result.extend(self._ingest_line(line))
            else:
                for line in lines:
                    result.extend(self._ingest_line(line))
        else:
            for line in lines:
                result.extend(self._ingest_line(line))

        result.extend(self._flush_stack())
        return result

    def feed(self, chunk: str) -> list[TokNode]:
        """Streaming: feed a partial chunk, get newly completed top-level nodes."""
        self._buffer += chunk
        lines = self._buffer.split("\n")
        self._buffer = lines[-1]
        result = []
        for line in lines[:-1]:
            result.extend(self._ingest_line(line))
        return result

    def flush(self) -> list[TokNode]:
        """End of stream: finalize pending blocks and return them."""
        try:
            result = []
            if self._buffer:
                result.extend(self._ingest_line(self._buffer))
                self._buffer = ""
            result.extend(self._flush_stack())
            return result
        except Exception:
            _logger.debug("Parser recovered from error during flush")
            return self._flush_stack()

    def _ingest_line(self, line: str) -> list[TokNode]:
        try:
            # Handle multi-line headers (quotes)
            if self._pending_header:
                self._pending_header += "\n" + line
                if self._is_balanced(self._pending_header):
                    full_header = self._pending_header
                    self._pending_header = ""
                    # Now re-ingest the combined header line
                    # But we need to keep the original indentation of the first line
                    return self._ingest_line_inner(full_header)
                return []

            return self._ingest_line_inner(line)
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            _logger.warning("Parser recovered from error in line ingestion: %r: %s", line[:80], exc)
            if self._stack:
                self._stack[-1][1].text += line + "\n"
            return []

    def _is_balanced(self, text: str) -> bool:
        """Check if quotes and brackets are balanced in a potential header."""
        in_q = False
        in_tq = False
        depth_b = 0
        depth_c = 0
        escaped = False
        i = 0
        while i < len(text):
            if escaped:
                escaped = False
                i += 1
                continue
            if text[i] == "\\":
                escaped = True
                i += 1
                continue

            if not in_q and text[i : i + 3] == '"""':
                in_tq = not in_tq
                i += 3
                continue

            if not in_tq and text[i] == '"':
                in_q = not in_q
                i += 1
                continue

            if not in_q and not in_tq:
                if text[i] == "[":
                    depth_b += 1
                elif text[i] == "]":
                    depth_b -= 1
                elif text[i] == "{":
                    depth_c += 1
                elif text[i] == "}":
                    depth_c -= 1
            i += 1
        return not in_q and not in_tq and depth_b == 0 and depth_c == 0

    def _ingest_line_inner(self, line: str) -> list[TokNode]:
        if len(line) > self.MAX_LINE_LENGTH:
            line = line[: self.MAX_LINE_LENGTH]

        stripped = line.strip()

        if self._active_boundary is not None:
            if stripped == f"|#{self._active_boundary}":
                if self._stack and self._stack[-1][1].text.endswith("\n"):
                    self._stack[-1][1].text = self._stack[-1][1].text[:-1]
                self._active_boundary = None
                return []
            if self._stack:
                self._stack[-1][1].text += line + "\n"
            return []

        if not stripped:
            return []

        if stripped.startswith("#"):
            # Robustness: Treat comments as @comment if they are at the top level
            # or if we want to preserve them. For now, let's keep them as ignored
            # unless they are inside a node.
            if self._stack:
                # If we are inside a node, maybe it's just a comment line in text?
                # Actually, TokMasterSpec says # is a comment.
                return []
            return []

        indent = self._indent(line)
        completed = []

        if stripped.startswith("@"):
            # Check if header is balanced
            if not self._is_balanced(line):
                self._pending_header = line
                return []

            # Ensure we pop off siblings to reach the correct parent depth
            while self._stack and self._stack[-1][0] >= indent:
                _, done = self._stack.pop()
                if self._stack:
                    self._stack[-1][1].children.append(done)
                else:
                    completed.append(done)

            header_text = stripped[1:]
            node = self._parse_header(header_text)
            self._in_inverted_body = False  # Reset on new node

            if not node.type:
                if self._stack:
                    self._process_content_line(line, self._stack[-1][1])
                return completed
            if node.type.lower() == "end":
                return completed
            self._stack.append((indent, node))
        elif self._stack:
            self._process_content_line(line, self._stack[-1][1])

        return completed

    def _process_content_line(self, raw_line: str, node: TokNode) -> None:
        stripped = raw_line.strip()

        if self._in_inverted_body:
            node.text += raw_line + "\n"
            return

        if (stripped.startswith(("|#", "#"))) and stripped.endswith(">"):
            label = stripped.lstrip("|#")[:-1].strip()
            if label in ("SEARCH", "REPLACE", "RAW") or self._active_boundary:
                self._active_boundary = label
                if node.trust == Trust.SYSTEM:
                    node.trust = Trust.EXTERNAL
                return

        if stripped.startswith("|#") and ">" in stripped:
            sep_idx = stripped.index(">")
            # FIX: Use empty string instead of None to represent empty label
            self._active_boundary = stripped[2:sep_idx].strip()

            if node.trust == Trust.SYSTEM:
                node.trust = Trust.EXTERNAL

            # Robustness: Capture inline content after >
            inline = stripped[sep_idx + 1 :].strip()
            if inline:
                node.text += inline + "\n"
            return

        if stripped.startswith("|>"):
            if node.trust == Trust.SYSTEM:
                node.trust = Trust.UNTRUSTED
            idx = raw_line.find("|>")
            content = raw_line[idx + 2 :]
            # If there's exactly one space after |>, trim it (common delimiter)
            content = content.removeprefix(" ")
            node.text += content + "\n"
            self._in_inverted_body = True  # STICKY MODE: following lines are text

        elif stripped.startswith("|") and ">" in stripped:
            marker_end = stripped.index(">")
            try:
                length = int(stripped[1:marker_end])
                start = raw_line.index(">") + 1
                data = raw_line[start : start + length]
                node.text += data + "\n"
                if node.trust == Trust.SYSTEM:
                    node.trust = Trust.EXTERNAL
            except (ValueError, IndexError):
                node.text += stripped + "\n"

        elif stripped.startswith(">"):
            node.text += stripped[1:].strip() + "\n"

        elif node.headers and "|" in stripped:
            raw_cells = [c.strip() for c in stripped.split("|")]
            # Robustness: Treat everything as data row. Header repeat detection is too
            # risky.

            row = [""] * len(node.headers)
            curr_idx = 0
            for cell in raw_cells:
                if not cell:
                    curr_idx += 1
                    continue

                if ":" in cell and not (cell.startswith(('"', "'"))):
                    parts = cell.split(":", 1)
                    key, val = parts[0].strip(), parts[1].strip()
                    if key in node.headers:
                        curr_idx = node.headers.index(key)
                        row[curr_idx] = str(self._cast(val))
                        curr_idx += 1
                        continue

                if curr_idx < len(node.headers):
                    cast_val = self._cast(cell)
                    if cast_val is not None:
                        row[curr_idx] = str(cast_val)
                    curr_idx += 1
                else:
                    # Robustness: keep extra cells by appending to a hidden attr or last cell
                    # For now, let's just append to the row list itself if we want to be
                    # truly lossless
                    row.append(str(self._cast(cell)))

            node.rows.append(row)

        elif (
            (":" in stripped or "=" in stripped)
            and not (stripped.startswith(('"', "'")))
            and len(stripped) < self.MAX_LINE_LENGTH
        ):
            sep = ":" if ":" in stripped else "="
            parts = stripped.split(sep)
            if len(parts) > 2:
                # Robustness: key must be preceded by space or start of line,
                # and followed by a space. This prevents URL/time colons/equals from being
                # keys.
                matches = list(re.finditer(r"(?:^|\s)([a-zA-Z0-9_-]+)[:=](?=\s|$)", stripped))
                if matches:
                    for i, m in enumerate(matches):
                        k = m.group(1).strip()
                        val_start = m.end()
                        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)
                        v = stripped[val_start:val_end].strip()
                        # Robustness: if value contains :, check if it's a URL or just
                        # more attrs
                        node.attrs[k] = self._cast(v)
                else:
                    node.attrs[parts[0].strip()] = self._cast(":".join(parts[1:]).strip())
            else:
                k, v = parts[0], parts[1]
                node.attrs[k.strip()] = self._cast(v.strip())

        elif stripped:
            node.text += stripped + "\n"

    def _flush_stack(self) -> list[TokNode]:
        while len(self._stack) > 1:
            _, child = self._stack.pop()
            self._stack[-1][1].children.append(child)
        if self._stack:
            return [self._stack.pop()[1]]
        return []

    def _parse_header(self, header: str) -> TokNode:
        node = TokNode(type="")
        tokens = self._tokenize(header)
        if not tokens:
            return node

        node.type = tokens[0]
        for tok in tokens[1:]:
            if tok.startswith("[") and "]" in tok:
                bracket_end = tok.index("]")
                content = tok[1:bracket_end]
                if "{" in tok:
                    try:
                        node.cardinality = int(content)
                        hdr_part = tok[tok.index("{") + 1 : tok.rindex("}")]
                        node.headers = [h.strip() for h in hdr_part.split(",")]
                    except (ValueError, IndexError):
                        node.headers = [tok]
                elif "|" in content:
                    node.headers = [h.strip() for h in content.split("|")]
                else:
                    try:
                        node.cardinality = int(content)
                    except ValueError:
                        node.headers = [content]
            elif tok.startswith("*"):
                node.ref = tok[1:]
            elif (":" in tok or "=" in tok) and not (tok.startswith((":", "="))):
                sep = ":" if ":" in tok else "="
                k, v = tok.split(sep, 1)
                if k == "trust":
                    try:
                        node.trust = Trust(v)
                    except ValueError:
                        if v:
                            node.attrs[k] = self._cast(v)
                elif v:
                    node.attrs[k] = self._cast(v)
            elif tok == "|>" or tok.startswith("|>"):
                # Phase 7d: In-lined text content
                content = tok[2:] if tok.startswith("|>") else ""
                # Collect remaining tokens
                remaining = tokens[tokens.index(tok) + 1 :]
                node.text = (content + " " + " ".join(remaining)).strip()
                break
            elif node.type[0].isupper() and not node.label:
                node.label = tok
            else:
                node.text += tok + " "

        if node.text:
            node.text = node.text.strip()
        return node

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        current = ""
        in_q = False
        in_tq = False
        depth_b = 0
        depth_c = 0
        escaped = False
        i = 0
        while i < len(text):
            c = text[i]
            if escaped:
                current += c
                escaped = False
                i += 1
                continue

            if c == "\\":
                current += c
                escaped = True
                i += 1
                continue

            if not in_q and text[i : i + 3] == '"""':
                current += '"""'
                in_tq = not in_tq
                i += 3
                continue

            if not in_tq and c == '"':
                current += '"'
                in_q = not in_q
                i += 1
                continue

            if not in_q and not in_tq:
                if c == "[":
                    depth_b += 1
                elif c == "]":
                    if depth_b > 0:
                        depth_b -= 1
                    else:
                        current += c
                        i += 1
                        continue
                elif c == "{":
                    depth_c += 1
                elif c == "}":
                    if depth_c > 0:
                        depth_c -= 1
                    else:
                        current += c
                        i += 1
                        continue
                elif c == " " and depth_b == 0 and depth_c == 0:
                    if current:
                        tokens.append(current)
                    current = ""
                    i += 1
                    continue

            current += c
            i += 1

        if current:
            tokens.append(current)
        return tokens

    def _indent(self, line: str) -> int:
        # Standardize: treat tabs as 2 spaces, but use raw count for depth
        expanded = line.replace("\t", "  ")
        return len(expanded) - len(expanded.lstrip())

    def _cast(self, v: str) -> str | list[Any] | int | float | bool | None:
        if not v:
            return v
        if v.startswith("\\") and len(v) > 1 and v[1] in ("!", "*"):
            return v[1:]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []

            # Robustness: handle nested lists by split only on top-level commas
            parts = []
            stack = 0
            current = ""
            for char in inner:
                if char == "[":
                    stack += 1
                elif char == "]":
                    stack -= 1

                if char == "," and stack == 0:
                    parts.append(current.strip())
                    current = ""
                else:
                    current += char
            if current:
                parts.append(current.strip())
            return [self._cast(x) for x in parts]
        if v.startswith("!"):
            return v[1:]
        if v.startswith("*"):
            return v[1:]
        v_low = v.lower()
        if v_low == "true":
            return True
        if v_low == "false":
            return False
        if v_low == "null":
            return None
        for conv in (int, float):
            try:
                val = conv(v)
                if str(val) == v or (isinstance(val, float) and v.replace(".", "").isdigit()):
                    return val
            except ValueError:
                continue
        m = re.match(
            r"^(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?=[@\s\|\[\{]|$)",
            v,
            re.IGNORECASE,
        )
        if m:
            val_str = m.group(1)
            try:
                if "." in val_str.lower() or "e" in val_str.lower():
                    return float(val_str)
                return int(val_str)
            except ValueError:
                pass

        # Final fallback: string cleanup and escape decoding
        clean_str = v.strip()
        if clean_str.startswith('"""') and clean_str.endswith('"""') and len(clean_str) >= 6:
            clean_str = clean_str[3:-3]
        elif clean_str.startswith('"') and clean_str.endswith('"') and len(clean_str) >= 2:
            clean_str = clean_str[1:-1]
        try:
            if "\\" in clean_str:
                # ONLY decode specific escapes that Tok needs for its own structure
                # Avoid aggressive unescaping (like \n) here. Favor verbatim blocks for
                # multi-line content.
                return clean_str.replace('\\"', '"').replace("\\\\", "\\")
        except Exception:
            pass
        return clean_str

    def encode(self, data: object) -> str:
        """
        Encode data to Tok text. Satisfies SerializationProtocol.

        Args:
            data: A TokNode or list of TokNodes to encode

        Returns:
            Serialized Tok text

        """
        # Type assertion to handle the generic object parameter
        if not isinstance(data, list):
            data = [data]
        return serialize(cast(list[TokNode], data))

    def decode(self, text: str) -> list[TokNode]:
        """
        Decode Tok text to nodes. Satisfies SerializationProtocol.

        Args:
            text: Tok text to parse

        Returns:
            List of parsed TokNode objects

        """
        return self.parse(text)
