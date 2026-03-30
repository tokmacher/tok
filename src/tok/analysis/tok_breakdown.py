import tiktoken

from ..protocol import TokEncoder
from ..utils.transformer import DocumentTransformer


def breakdown(file_path: str) -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    transformer = DocumentTransformer(flattening_threshold=5)

    with open(file_path) as f:
        md_text = f.read()

    md_tokens = len(enc.encode(md_text))
    tok_nodes = transformer.transform(md_text)
    tok_text = TokEncoder.encode(tok_nodes, compact=True)
    tok_tokens = len(enc.encode(tok_text))

    print(f"File: {file_path}")
    print(f"MD Tokens: {md_tokens}")
    print(f"Tok Tokens: {tok_tokens}")
    print(
        f"Diff: {tok_tokens - md_tokens} ({(tok_tokens / md_tokens - 1) * 100:.2f}%)"
    )

    # Analyze components
    import re

    # 1. Structural Tags (@H, @P, @I, @T, @HR)
    tags = re.findall(r"@[A-Z0-9]+", tok_text)
    tag_tokens = sum(len(enc.encode(t)) for t in tags)
    print(f"Tag Tokens (@...): {tag_tokens}")

    # 2. Text Prefixes (> )
    prefixes = re.findall(r"> ", tok_text)
    prefix_tokens = sum(len(enc.encode(p)) for p in prefixes)
    print(f"Prefix Tokens (> ): {prefix_tokens}")

    # 3. Indentation & Newlines
    len(enc.encode(tok_text)) - len(
        enc.encode(re.sub(r"[\n\s]", "", tok_text))
    )
    # Note: sub above is too aggressive, let's just count newlines and leading spaces
    lines = tok_text.split("\n")
    newline_tokens = len(lines)  # Approximate
    print(f"Newline Tokens: {newline_tokens}")

    # 4. TOC
    toc_match = re.search(r"@TOC.*?(@|$)", tok_text, re.DOTALL)
    if toc_match:
        toc_tokens = len(enc.encode(toc_match.group(0)))
        print(f"TOC Tokens: {toc_tokens}")

    # 5. POOL
    pool_match = re.search(r"@POOL.*?(@|$)", tok_text, re.DOTALL)
    if pool_match:
        pool_tokens = len(enc.encode(pool_match.group(0)))
        print(f"POOL Tokens: {pool_tokens}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        breakdown(sys.argv[1])
    else:
        breakdown("research_prompts/ds_research.md")
