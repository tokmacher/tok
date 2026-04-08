"""Tok breakdown analysis for token usage by component."""

import tiktoken

from tok.protocol import TokEncoder
from tok.utils.transformer import DocumentTransformer


def breakdown(file_path: str) -> None:
    """Analyze Tok file and breakdown token usage by component."""
    enc = tiktoken.get_encoding("cl100k_base")
    transformer = DocumentTransformer(flattening_threshold=5)

    with open(file_path) as f:
        md_text = f.read()

    len(enc.encode(md_text))
    tok_nodes = transformer.transform(md_text)
    tok_text = TokEncoder.encode(tok_nodes, compact=True)
    len(enc.encode(tok_text))

    # Analyze components
    import re

    # 1. Structural Tags (@H, @P, @I, @T, @HR)
    tags = re.findall(r"@[A-Z0-9]+", tok_text)
    sum(len(enc.encode(t)) for t in tags)

    # 2. Text Prefixes (> )
    prefixes = re.findall(r"> ", tok_text)
    sum(len(enc.encode(p)) for p in prefixes)

    # 3. Indentation & Newlines
    len(enc.encode(tok_text)) - len(enc.encode(re.sub(r"[\n\s]", "", tok_text)))
    # Note: sub above is too aggressive, let's just count newlines and leading spaces
    lines = tok_text.split("\n")
    len(lines)  # Approximate

    # 4. TOC
    toc_match = re.search(r"@TOC.*?(@|$)", tok_text, re.DOTALL)
    if toc_match:
        len(enc.encode(toc_match.group(0)))

    # 5. POOL
    pool_match = re.search(r"@POOL.*?(@|$)", tok_text, re.DOTALL)
    if pool_match:
        len(enc.encode(pool_match.group(0)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        breakdown(sys.argv[1])
    else:
        breakdown("research_prompts/ds_research.md")
