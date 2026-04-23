"""Batch transform markdown files to Tok format and back."""

from pathlib import Path

import tiktoken

from tok.protocol import TokEncoder

from .transformer import DocumentTransformer


def batch_transform() -> None:
    """Transform markdown files to Tok format and generate summary report."""
    input_dir = Path("research_prompts")
    output_dir = Path("research_tok_output")
    output_dir.mkdir(exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")
    transformer = DocumentTransformer(flattening_threshold=5)

    results = []

    files = list(input_dir.glob("*.md"))
    for md_file in files:
        with open(md_file) as f:
            content = f.read()

        md_tokens = len(enc.encode(content))

        # 1. Transform to Tok
        tok_nodes = transformer.transform(content)
        tok_text = TokEncoder.encode(tok_nodes, compact=False)
        tok_tokens = len(enc.encode(tok_text))

        # Save Tok
        tok_out = output_dir / md_file.with_suffix(".tok").name
        with open(tok_out, "w") as f:
            f.write(tok_text)

        # 2. Detransform back to MD
        detransformed_md = transformer.detransform(tok_text)
        md_out = output_dir / md_file.with_suffix(".round_trip.md").name
        with open(md_out, "w") as f:
            f.write(detransformed_md)

        savings = (1 - tok_tokens / md_tokens) * 100
        results.append(
            {
                "name": md_file.name,
                "md_tokens": md_tokens,
                "tok_tokens": tok_tokens,
                "savings": savings,
            }
        )

    # Generate Summary Report
    report_path = output_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("# Batch Transformation Summary Report\n\n")
        f.write("| Document | MD Tokens | Tok Tokens | Savings |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.writelines(
            f"| {res['name']} | {res['md_tokens']} | {res['tok_tokens']} | {res['savings']:.2f}% |\n" for res in results
        )


if __name__ == "__main__":
    batch_transform()
