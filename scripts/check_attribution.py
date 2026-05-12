"""Post-session attribution consistency checker.

Usage: python scripts/check_attribution.py <session-name>
Reads capture files from tmp/attribution_<session>_*.txt and checks consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _read_capture(session: str, kind: str) -> str:
    path = Path(f"tmp/attribution_{session}_{kind}.txt")
    if not path.exists():
        return ""
    return path.read_text()


def check_session(session: str) -> dict[str, bool]:
    status = _read_capture(session, "status")
    doctor = _read_capture(session, "doctor")
    stats = _read_capture(session, "stats")
    audit = _read_capture(session, "audit")

    return {
        "status_file_exists": bool(status),
        "doctor_file_exists": bool(doctor),
        "stats_file_exists": bool(stats),
        "audit_file_exists": bool(audit),
        "status_has_bridge_state": "running" in status.lower() or "not running" in status.lower(),
        "doctor_has_verdict": "verdict" in doctor.lower() or "recommendation" in doctor.lower(),
        "stats_has_session_data": "session" in stats.lower() or "calls" in stats.lower(),
        "audit_no_trace_error": "error" not in audit.lower()[:200] if audit else False,
        "capability_visible_when_running": ("bridge capability" in status.lower()) or ("not running" in status.lower()),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_attribution.py <session-name>")
        sys.exit(2)

    session = sys.argv[1]
    results = check_session(session)
    all_pass = all(results.values())
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    if not all_pass:
        print(f"\nSome checks failed for session '{session}'.")
        sys.exit(1)
    print(f"\nAll checks passed for session '{session}'.")


if __name__ == "__main__":
    main()
