"""Generate a live-bridge per-prompt bloat audit report."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tok.analysis.live_bridge_bloat import (
    generate_live_bridge_bloat_report,
    render_live_bridge_bloat_markdown,
)

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "tmp", "live_bridge_bloat_report.json")
FINDINGS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "docs",
    "live_bridge_prompt_bloat_findings.md",
)


def main() -> None:
    report = generate_live_bridge_bloat_report()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as handle:
        json.dump(report, handle, indent=2)

    with open(FINDINGS_PATH, "w") as handle:
        handle.write(render_live_bridge_bloat_markdown(report))


if __name__ == "__main__":
    main()
