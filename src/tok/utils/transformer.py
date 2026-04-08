import re
from collections.abc import Callable
from typing import Any

import tiktoken

from tok.protocol.models import TokNode

# Constants for pointer computation
MIN_OCCURRENCES = 2
MIN_TOKEN_LENGTH = 2
MIN_PROFIT = 2
MAX_ALPHA_LENGTH = 8


from tok.protocol.parser import TokParser


class DocumentTransformer:
    """
    Transforms structured documents (Markdown-like) into optimized Tok nodes.
    Supports hierarchy mapping, footnote pointers, node flattening, and TOC.
    """

    def __init__(self, flattening_threshold: int = 5) -> None:
        self.flattening_threshold = flattening_threshold
        self.citations: dict[str, TokNode] = {}
        self.nodes_by_id: dict[str, TokNode] = {}
        self._next_id = 0

    def _generate_id(self, prefix: str = "node") -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    @staticmethod
    def _extract_metadata(lines: list[str]) -> tuple[dict[str, str], int]:
        metadata: dict[str, str] = {}
        md_start = 0
        while md_start < len(lines) and not lines[md_start].strip():
            md_start += 1
        if md_start < len(lines) and lines[md_start].strip() == "---":
            md_start += 1
            while md_start < len(lines) and lines[md_start].strip() != "---":
                match = re.match(r"^(\w+):\s*(.*)", lines[md_start])
                if match:
                    metadata[match.group(1)] = match.group(2)
                md_start += 1
            if md_start < len(lines):
                md_start += 1
        return metadata, md_start

    @staticmethod
    def _extract_footnotes(
        lines: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        footnote_defs: dict[str, str] = {}
        content_lines: list[str] = []
        for line in lines:
            match = re.match(r"^\[\^(\w+)\]:\s*(.*)", line)
            if match:
                footnote_defs[match.group(1)] = match.group(2)
            else:
                content_lines.append(line)
        return footnote_defs, content_lines

    @staticmethod
    def _parse_table_rows(
        lines: list[str],
        i: int,
        headers: list[str],
        strip_fn: Callable[[str], str],
    ) -> tuple[TokNode, int]:
        node = TokNode(type="T", headers=headers)
        while i < len(lines) and "|" in lines[i]:
            row = [c.strip() for c in lines[i].split("|")]
            if len(row) > 1:
                if not row[0]:
                    row.pop(0)
                if row and not row[-1]:
                    row.pop()
                if len(row) >= len(headers):
                    node.rows.append([strip_fn(c) for c in row[: len(headers)]])
            i += 1
        return node, i

    def _attach_node(
        self,
        node: TokNode,
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
    ) -> None:
        if stack:
            stack[-1][1].children.append(node)
        else:
            root_nodes.append(node)

    @staticmethod
    def _parse_code_block(lines: list[str], i: int, lang: str) -> tuple[TokNode, int]:
        i += 1
        code: list[str] = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            code.append(lines[i])
            i += 1
        if i < len(lines):
            i += 1
        node = TokNode(
            type="CODE",
            text="\n".join(code),
            attrs={"lang": lang} if lang else {},
        )
        return node, i

    @staticmethod
    def _is_structural_break(line: str, i: int, lines: list[str], rich: bool) -> bool:
        if re.match(r"^(#{1,6})\s+", line):
            return True
        if re.match(r"^\s*[-*+]\s+", line):
            return True
        if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", line):
            return True
        if i + 1 < len(lines) and "|" in line and re.match(r"^\s*\|?[\s:|:-]+\|?\s*$", lines[i + 1]):
            return True
        if not rich and "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
            return True
        return bool(line.strip().startswith("```"))

    def _try_parse_heading(
        self,
        line: str,
        strip_decorators: Callable[[str], str],
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
    ) -> bool:
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if not match:
            return False
        lvl = len(match.group(1))
        node = TokNode(
            type=f"H{lvl}",
            text=strip_decorators(match.group(2).strip()),
        )
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        self._attach_node(node, stack, root_nodes)
        stack.append((lvl, node))
        return True

    def _try_parse_item(
        self,
        line: str,
        strip_decorators: Callable[[str], str],
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
    ) -> bool:
        match = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if not match:
            return False
        node = TokNode(type="I", text=strip_decorators(match.group(2).strip()))
        self._attach_node(node, stack, root_nodes)
        return True

    def _try_parse_naked_table(
        self,
        lines: list[str],
        i: int,
        strip_decorators: Callable[[str], str],
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
    ) -> tuple[bool, int]:
        if not ("|" in lines[i] and i + 1 < len(lines) and "|" in lines[i + 1]):
            return False, i
        headers = [h.strip() for h in lines[i].split("|") if h.strip()]
        node, new_i = self._parse_table_rows(lines, i + 1, headers, strip_decorators)
        self._attach_node(node, stack, root_nodes)
        return True, new_i

    def _parse_paragraph(
        self,
        lines: list[str],
        i: int,
        strip_decorators: Callable[[str], str],
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
        rich: bool,
    ) -> int:
        para: list[str] = []
        while i < len(lines) and lines[i].strip():
            if self._is_structural_break(lines[i], i, lines, rich):
                break
            para.append(lines[i])
            i += 1
        if para:
            node = TokNode(type="P", text=strip_decorators("\n".join(para)))
            self._attach_node(node, stack, root_nodes)
        else:
            i += 1
        return i

    def _parse_content_lines(
        self,
        lines: list[str],
        strip_decorators: Callable[[str], str],
        stack: list[tuple[int, TokNode]],
        root_nodes: list[TokNode],
        rich: bool,
    ) -> None:
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            if i + 1 < len(lines) and "|" in line and re.match(r"^\s*\|?([\s:|-]+)+\s*$", lines[i + 1]):
                headers = [h.strip() for h in line.split("|") if h.strip()]
                i += 2
                node, i = self._parse_table_rows(lines, i, headers, strip_decorators)
                self._attach_node(node, stack, root_nodes)
                continue

            if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", line):
                self._attach_node(TokNode(type="HR"), stack, root_nodes)
                i += 1
                continue

            if line.strip().startswith("```"):
                lang = line.strip()[3:].strip()
                node, i = self._parse_code_block(lines, i, lang)
                self._attach_node(node, stack, root_nodes)
                continue

            if self._try_parse_heading(line, strip_decorators, stack, root_nodes):
                i += 1
                continue

            if self._try_parse_item(line, strip_decorators, stack, root_nodes):
                i += 1
                continue

            if not rich:
                parsed, new_i = self._try_parse_naked_table(lines, i, strip_decorators, stack, root_nodes)
                if parsed:
                    i = new_i
                    continue

            i = self._parse_paragraph(lines, i, strip_decorators, stack, root_nodes, rich)

    @staticmethod
    def _strip_md_decorators(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
        return re.sub(r"~~(.*?)~~", r"\1", text)

    def _post_process_rich(
        self,
        root_nodes: list[TokNode],
        footnote_defs: dict[str, str],
    ) -> list[TokNode]:
        self.citations = {}
        self._extract_citations(root_nodes, footnote_defs)
        ptr_nodes = self._pointerize(root_nodes)

        needed: set[int] = set()
        flattened_nodes = self._flatten(root_nodes, needed)
        toc_node, head_to_node = self._generate_toc(root_nodes, needed)
        self._assign_labels(root_nodes + flattened_nodes, needed)

        if toc_node:
            for row in toc_node.rows:
                toc_target: TokNode | None = head_to_node.get(str(row[1]))
                row[1] = f"*{toc_target.label}" if toc_target and toc_target.label else ""

        result: list[TokNode] = [toc_node] if toc_node else []
        if ptr_nodes:
            pool = TokNode(type="POOL")
            pool.children = ptr_nodes
            result.append(pool)

        result.extend(flattened_nodes)
        if self.citations:
            cites = TokNode(type="CITATIONS")
            cites.children = sorted(self.citations.values(), key=lambda x: x.label)
            result.append(cites)
        return result

    @staticmethod
    def _identity(text: str) -> str:
        return text

    def transform(self, markdown_text: str, rich: bool = True) -> list[TokNode]:
        """
        Convert markdown text to a list of TokNodes.

        rich=True: Preserves all symbols (v1.6).
        rich=False: Strips markdown syntax for raw semantic density (v1.7).
        """
        lines = markdown_text.split("\n")

        if not rich:
            lines = [line for line in lines if not re.match(r"^\s*\|?[\s:|:-]+\|?\s*$", line)]
            strip_decorators = self._strip_md_decorators
        else:
            strip_decorators = self._identity

        root_nodes: list[TokNode] = []
        stack: list[tuple[int, TokNode]] = []

        metadata, md_start = self._extract_metadata(lines)
        footnote_defs, content_lines = self._extract_footnotes(lines[md_start:])
        lines = content_lines

        if metadata:
            root_nodes.append(TokNode(type="META", attrs=metadata))

        self._parse_content_lines(lines, strip_decorators, stack, root_nodes, rich)

        if not rich:
            return root_nodes

        return self._post_process_rich(root_nodes, footnote_defs)

    def _extract_citations(self, nodes: list[TokNode], defs: dict[str, str]) -> None:
        """Extract footnote reflections and replace with pointers."""
        self.citations_by_text: dict[str, TokNode] = {}

        for node in nodes:
            self._process_node_citations(node, defs)

    def _process_node_citations(self, node: TokNode, defs: dict[str, str]) -> None:
        """Process citations for a single node and its children."""
        if not node.text:
            return

        self._replace_citation_markers(node, defs)

        for child in node.children:
            self._process_node_citations(child, defs)

    def _replace_citation_markers(self, node: TokNode, defs: dict[str, str]) -> None:
        """Replace citation markers in node text with pointers."""
        matches = list(re.finditer(r"\[\^(\w+)\]", node.text))
        for match in reversed(matches):
            ref_id = match.group(1)
            text = defs.get(ref_id, f"Citation {ref_id}")

            cite_node = self._get_or_create_citation_node(ref_id, text)
            ptr = cite_node.label

            node.text = node.text[: match.start()] + f"*{ptr}" + node.text[match.end() :]

    def _get_or_create_citation_node(self, ref_id: str, text: str) -> TokNode:
        """Get existing citation node or create a new one."""
        if text not in self.citations_by_text:
            cite_node = TokNode(
                type="CITE",
                label=f"ref_{len(self.citations)}",
                text=text,
            )
            self.citations[ref_id] = cite_node
            self.citations_by_text[text] = cite_node

        return self.citations_by_text[text]

    @staticmethod
    def _collect_text_counts(nodes: list[TokNode]) -> dict[str, int]:
        text_counts: dict[str, int] = {}

        def collect_text(node: TokNode) -> None:
            if node.text:
                urls = re.findall(r"https?://[^\s\)]+", node.text)
                for url in urls:
                    text_counts[url] = text_counts.get(url, 0) + 1
                words = re.findall(r"\b[A-Za-z0-9\-]{4,}\b", node.text)
                for word in words:
                    text_counts[word] = text_counts.get(word, 0) + 1
            for child in node.children:
                collect_text(child)

        for node in nodes:
            collect_text(node)
        return text_counts

    @staticmethod
    def _compute_profitable_pointers(text_counts: dict[str, int], max_pointers: int = 60) -> dict[str, str]:
        enc = tiktoken.get_encoding("cl100k_base")
        profitable = []
        for text, count in text_counts.items():
            if count < MIN_OCCURRENCES:
                continue
            t_toks = len(enc.encode(text))
            if t_toks < MIN_TOKEN_LENGTH:
                continue
            profit = (t_toks - MIN_TOKEN_LENGTH) * count - t_toks
            if profit > MIN_PROFIT:
                profitable.append((text, profit))

        profitable.sort(key=lambda x: x[1], reverse=True)
        return {text: f"p{i}" for i, (text, _) in enumerate(profitable[:max_pointers])}

    @staticmethod
    def _apply_pointers(nodes: list[TokNode], pointers: dict[str, str]) -> None:
        def apply_pointers(node: TokNode) -> None:
            if node.text:
                for text in sorted(pointers.keys(), key=len, reverse=True):
                    ptr = pointers[text]
                    if text.isalpha() and len(text) < MAX_ALPHA_LENGTH:
                        node.text = re.sub(rf"\b{re.escape(text)}\b", f"*{ptr}", node.text)
                    else:
                        node.text = node.text.replace(text, f"*{ptr}")
            for child in node.children:
                apply_pointers(child)

        for node in nodes:
            apply_pointers(node)

    def _pointerize(self, nodes: list[TokNode]) -> list[TokNode]:
        """Find repeated long strings and replace with pointers."""
        text_counts = self._collect_text_counts(nodes)
        pointers = self._compute_profitable_pointers(text_counts)

        if not pointers:
            return []

        self._apply_pointers(nodes, pointers)

        return [TokNode(type="PTR", label=ptr, text=text) for text, ptr in pointers.items()]

    def _assign_labels(self, nodes: list[TokNode], needed: set[int]) -> None:
        visited: set[int] = set()

        def assign(ns: list[TokNode]) -> None:
            for n in ns:
                if id(n) in visited:
                    continue
                visited.add(id(n))
                if id(n) in needed and not n.label:
                    n.label = self._generate_id(n.type.lower())
                assign(n.children)

        assign(nodes)

    @staticmethod
    def _patch_parent_pointers(
        all_nodes: list[TokNode],
        needed: set[int],
        generate_id: Callable[[str], str],
    ) -> None:
        id_to_label: dict[str, str] = {}
        for n in all_nodes:
            if id(n) in needed:
                if not n.label:
                    n.label = generate_id(n.type.lower())
                id_to_label[f"NESTED_{id(n)}"] = f"*{n.label}"

        def patch(ns: list[TokNode]) -> None:
            for n in ns:
                if "parent" in n.attrs and n.attrs.get("parent") in id_to_label:
                    n.attrs["parent"] = id_to_label[n.attrs["parent"]]
                patch(n.children)

        patch(all_nodes)

    def _flatten(
        self,
        nodes: list[TokNode],
        needed: set[int],
        current_depth: int = 0,
        root_list: list[TokNode] | None = None,
        parent_node: TokNode | None = None,
    ) -> list[TokNode]:
        """Promote deep structural nodes to root level with parent pointers."""
        if root_list is None:
            root_list = []
        retained: list[TokNode] = []
        for node in nodes:
            if current_depth >= self.flattening_threshold and current_depth > 0:
                if parent_node:
                    needed.add(id(parent_node))
                    node.attrs["parent"] = f"NESTED_{id(parent_node)}"
                root_list.append(node)
                needed.add(id(node))
            else:
                retained.append(node)

            if node.children:
                node.children = self._flatten(node.children, needed, current_depth + 1, root_list, node)

        if current_depth == 0:
            all_reachable = retained + root_list
            self._patch_parent_pointers(all_reachable, needed, self._generate_id)
            return all_reachable
        return retained

    def _generate_toc(self, nodes: list[TokNode], needed: set[int]) -> tuple[TokNode | None, dict[str, TokNode]]:
        toc_headers = ["h", "lbl", "title"]
        toc_rows = []
        visited: set[int] = set()
        head_to_node: dict[str, TokNode] = {}

        def collect_toc(n: list[TokNode]) -> None:
            for node in n:
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if node.type.upper().startswith("H") and node.type[1:].isdigit():
                    needed.add(id(node))
                    title = node.text.split("\n")[0][:30].strip()
                    placeholder = f"PENDING_{id(node)}"
                    toc_rows.append([node.type, placeholder, title])
                    head_to_node[placeholder] = node
                collect_toc(node.children)

        collect_toc(nodes)
        if not toc_rows:
            return None, {}

        toc_node = TokNode(type="TOC", headers=toc_headers, rows=toc_rows)
        return toc_node, head_to_node

    def to_markdown(self, nodes: list[TokNode]) -> str:
        """Deterministic Reconstruction Engine (Re-hydration)."""
        return self.detransform_nodes(nodes)

    def detransform(self, tok_text: str) -> str:
        """Convert Tok text back to markdown format."""
        parser = TokParser()
        nodes = parser.parse(tok_text)
        return self.detransform_nodes(nodes)

    @staticmethod
    def _build_pointer_map(
        nodes: list[TokNode],
    ) -> dict[str, str]:
        pointer_map: dict[str, str] = {}

        def collect_pointers(ns: list[TokNode]) -> None:
            for n in ns:
                if n.type == "PTR" and n.label:
                    pointer_map[f"*{n.label}"] = n.text
                collect_pointers(n.children)

        collect_pointers(nodes)

        nodes_by_label: dict[str, TokNode] = {}

        def index_nodes(ns: list[TokNode]) -> None:
            for n in ns:
                if n.label:
                    nodes_by_label[n.label] = n
                index_nodes(n.children)

        index_nodes(nodes)

        for lbl, node in nodes_by_label.items():
            if f"*{lbl}" not in pointer_map:
                pointer_map[f"*{lbl}"] = node.text.split("\n")[0] if node.text else f"#{lbl}"

        return pointer_map

    @staticmethod
    def _render_table_md(n: TokNode) -> list[str]:
        lines: list[str] = ["\n| " + " | ".join(n.headers) + " |"]
        aligns = n.attrs.get("aligns", ["left"] * len(n.headers))
        sep_cells: list[str] = []
        for a in aligns:
            if a == "center":
                sep_cells.append(":---:")
            elif a == "right":
                sep_cells.append("---:")
            else:
                sep_cells.append("---")
        lines.append("| " + " | ".join(sep_cells) + " |")
        for row in n.rows:
            lines.extend(["| " + " | ".join(str(v) if v is not None else "" for v in row) + " |"])
        lines.append("")
        return lines

    @staticmethod
    def _node_to_md(
        n: TokNode,
        depth: int,
        citation_defs: dict[str, str],
        rehydrate: Callable[[str], str],
    ) -> list[str]:
        lines: list[str] = []
        text = n.text.strip() if n.text else ""
        t = n.type.upper()

        for cite_lbl in citation_defs:
            text = text.replace(f"*{cite_lbl}", f"[^{cite_lbl}]")

        if t.startswith("H") and t[1:].isdigit():
            level = int(t[1:])
            lines.append(f"\n{'#' * level} {rehydrate(text)}\n")
        elif t in ("ITEM", "I"):
            indent = "  " * depth
            lines.append(f"{indent}- {rehydrate(text)}")
        elif t in ("TABLE", "T"):
            lines.extend(DocumentTransformer._render_table_md(n))
        elif t == "HR":
            lines.append("\n---\n")
        elif t == "CODE":
            lang = n.attrs.get("lang", "")
            lines.append(f"\n```{lang}\n{text}\n```\n")
        elif t in ("PARA", "P") and text:
            lines.append(rehydrate(text) + "\n")

        for child in n.children:
            lines.extend(DocumentTransformer._node_to_md(child, depth + 1, citation_defs, rehydrate))
        return lines

    @staticmethod
    def _classify_nodes(
        nodes: list[TokNode],
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, Any],
        list[TokNode],
        TokNode | None,
    ]:
        pool: dict[str, str] = {}
        citation_defs: dict[str, str] = {}
        metadata: dict[str, Any] = {}
        content_nodes: list[TokNode] = []
        toc_node: TokNode | None = None

        for node in nodes:
            ut = node.type.upper()
            if ut == "POOL":
                for child in node.children:
                    pool[child.label] = child.text
            elif ut == "CITATIONS":
                for child in node.children:
                    citation_defs[child.label] = child.text
            elif ut == "TOC":
                toc_node = node
            elif ut == "META":
                metadata = node.attrs
            else:
                content_nodes.append(node)

        return pool, citation_defs, metadata, content_nodes, toc_node

    @staticmethod
    def _resolve_parent_structure(
        content_nodes: list[TokNode],
        nodes_by_label: dict[str, TokNode],
    ) -> list[TokNode]:
        final_roots: list[TokNode] = []
        for n in content_nodes:
            parent_ptr = n.attrs.get("parent")
            if parent_ptr and str(parent_ptr).startswith("*"):
                parent_lbl = str(parent_ptr)[1:]
                if parent_lbl in nodes_by_label:
                    if n not in nodes_by_label[parent_lbl].children:
                        nodes_by_label[parent_lbl].children.append(n)
                else:
                    final_roots.append(n)
            else:
                final_roots.append(n)
        return final_roots

    @staticmethod
    def _render_toc(toc_node: TokNode) -> list[str]:
        md_lines: list[str] = ["\n# Table of Contents"]
        for row in toc_node.rows:
            row[1] = str(row[1])
            level_str = row[0].upper().replace("H", "")
            level = int(level_str) if level_str.isdigit() else 1
            indent = "  " * (level - 1)
            label = row[1]
            title = row[2]
            if label.startswith("*"):
                md_lines.append(f"{indent}- [{title}](#{label[1:]})")
            else:
                md_lines.append(f"{indent}- {title}")
        md_lines.append("")
        return md_lines

    @staticmethod
    def _render_metadata(metadata: dict[str, Any]) -> list[str]:
        md_lines: list[str] = ["---"]
        for k, v in sorted(metadata.items()):
            md_lines.append(f"{k}: {v}")
        md_lines.append("---\n")
        return md_lines

    @staticmethod
    def _render_footnotes(citation_defs: dict[str, str]) -> list[str]:
        md_lines: list[str] = ["\n# Footnotes"]
        for lbl, txt in sorted(citation_defs.items()):
            md_lines.append(f"[^{lbl}]: {txt}")
        return md_lines

    @staticmethod
    def _index_nodes_by_label(
        nodes: list[TokNode],
    ) -> dict[str, TokNode]:
        nodes_by_label: dict[str, TokNode] = {}

        def index(ns: list[TokNode]) -> None:
            for n in ns:
                if n.label:
                    nodes_by_label[n.label] = n
                index(n.children)

        index(nodes)
        return nodes_by_label

    @staticmethod
    def _make_rehydrator(pointer_map: dict[str, str]) -> Callable[[str], str]:
        sorted_ptrs = sorted(pointer_map.keys(), key=len, reverse=True)

        def rehydrate(text: str) -> str:
            if not text:
                return ""
            for ptr in sorted_ptrs:
                text = text.replace(ptr, pointer_map[ptr])
            return text

        return rehydrate

    def detransform_nodes(self, nodes: list[TokNode]) -> str:
        """Convert a list of TokNodes back to markdown string."""
        (
            _pool,
            citation_defs,
            metadata,
            content_nodes,
            toc_node,
        ) = self._classify_nodes(nodes)

        nodes_by_label = self._index_nodes_by_label(nodes)
        pointer_map = self._build_pointer_map(nodes)
        rehydrate = self._make_rehydrator(pointer_map)

        final_roots = self._resolve_parent_structure(content_nodes, nodes_by_label)

        md_lines: list[str] = []
        if metadata:
            md_lines.extend(self._render_metadata(metadata))
        if toc_node:
            md_lines.extend(self._render_toc(toc_node))

        for n in final_roots:
            md_lines.extend(self._node_to_md(n, 0, citation_defs, rehydrate))

        if citation_defs:
            md_lines.extend(self._render_footnotes(citation_defs))

        md = "\n".join(md_lines)
        return re.sub(r"\n{3,}", "\n\n", md).strip()
