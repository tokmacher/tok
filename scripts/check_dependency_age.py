#!/usr/bin/env python3
"""Check dependency artifact age for manual security review."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def check_dependencies(lock_path: Path = Path("uv.lock")) -> list[str]:
    violations: list[str] = []
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))

    for package in lock.get("package", []):
        if not isinstance(package, dict):
            continue
        source = package.get("source", {})
        if isinstance(source, dict) and "editable" in source:
            continue

        artifacts: list[dict[str, Any]] = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            artifacts.append(sdist)
        wheels = package.get("wheels", [])
        if isinstance(wheels, list):
            artifacts.extend(artifact for artifact in wheels if isinstance(artifact, dict))

        upload_times: list[datetime] = []
        for artifact in artifacts:
            upload_time = artifact.get("upload-time")
            if isinstance(upload_time, str):
                upload_times.append(datetime.fromisoformat(upload_time.replace("Z", "+00:00")))

        if not upload_times:
            continue

        package_age_days = (datetime.now(timezone.utc) - min(upload_times)).days
        if package_age_days < 5:
            violations.append(f"{package['name']}@{package['version']} is only {package_age_days} days old")

    return violations


def main() -> int:
    violations = check_dependencies()
    if violations:
        print("SECURITY REVIEW REQUIRED:")
        for violation in violations:
            print(f"  - {violation}")
        print("\nDependencies must be at least 5 days old for automatic approval")
        print("Manual security review required before merging")
        return 1

    print("All dependencies meet 5-day age requirement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
