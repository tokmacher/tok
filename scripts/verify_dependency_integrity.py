#!/usr/bin/env python3

"""Dependency integrity verification script for Tok security pipeline."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from _uv_lock import load_uv_lock, normalize_hash
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._uv_lock import load_uv_lock, normalize_hash

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    "min_package_age_days": 90,
    "allowed_sources": [
        "https://pypi.org/simple",
        "https://files.pythonhosted.org",
    ],
    "blocked_packages": [
        "python3-lib",
        "pycrypto",
    ],
}
BLOCKED_PACKAGES = {blocked.lower() for blocked in SECURITY_CONFIG["blocked_packages"]}

PACKAGE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|(?:[._-](?=[a-zA-Z0-9])))*$")
VERSION_REGEX = re.compile(r"^[a-zA-Z0-9._+-]+$")
HASH_REGEX = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_package_name(package_name: str) -> bool:
    """Validate package name for security."""
    if not package_name or len(package_name) > 100:
        return False
    return bool(PACKAGE_NAME_REGEX.fullmatch(package_name))


def validate_version(version: str) -> bool:
    """Validate version string for security."""
    if not version or len(version) > 50:
        return False
    return bool(VERSION_REGEX.fullmatch(version))


def validate_hash(hash_str: str) -> bool:
    """Validate SHA-256 hash format."""
    return bool(HASH_REGEX.fullmatch(normalize_hash(hash_str)))


def parse_uv_lock() -> list[dict[str, Any]]:
    """Parse uv.lock file to extract package information."""
    return load_uv_lock()


def check_blocked_packages(package_name: str) -> bool:
    """Check if package is in blocked list."""
    return package_name.lower() in BLOCKED_PACKAGES


def verify_source_integrity(source: str) -> bool:
    """Verify package comes from an allowed source."""
    return any(source.startswith(allowed_source) for allowed_source in SECURITY_CONFIG["allowed_sources"])


def check_package_age(upload_time: datetime | None) -> int | None:
    """Return package age in days if the lockfile captured upload metadata."""
    if upload_time is None:
        return None
    return (datetime.now(timezone.utc) - upload_time).days


def evaluate_package(package: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Evaluate a single package entry for warnings and violations."""
    warnings: list[str] = []
    violations: list[str] = []

    package_name = str(package["name"])
    version = str(package["version"])
    source = package["source"]
    artifact_url = str(package["artifact_url"])
    hashes = [str(hash_value) for hash_value in package["hashes"]]
    upload_time = package["upload_time"]

    if check_blocked_packages(package_name):
        violations.append(f"❌ Blocked package found: {package_name}")
        return warnings, violations

    if isinstance(source, dict) and "editable" in source:
        editable_path = Path(str(source["editable"]))
        if not editable_path.exists():
            violations.append(f"❌ Editable source missing for {package_name}: {editable_path}")
        return warnings, violations

    registry = ""
    if isinstance(source, dict):
        registry = str(source.get("registry", ""))
    if registry and not verify_source_integrity(registry):
        violations.append(f"❌ Untrusted registry source for {package_name}: {registry}")
        return warnings, violations
    if artifact_url and not verify_source_integrity(artifact_url):
        violations.append(f"❌ Untrusted artifact source for {package_name}: {artifact_url}")
        return warnings, violations

    age_days = check_package_age(upload_time)
    if age_days is None:
        warnings.append(f"⚠️  No upload timestamp available for {package_name}@{version}")
    elif age_days < SECURITY_CONFIG["min_package_age_days"]:
        warnings.append(f"⚠️  Recent package: {package_name}@{version} ({age_days} days old, review before release)")

    if not hashes:
        warnings.append(f"⚠️  No artifact hashes recorded for {package_name}@{version}")
        return warnings, violations
    if any(not validate_hash(hash_value) for hash_value in hashes):
        violations.append(f"❌ Invalid hash format recorded for {package_name}@{version}")

    return warnings, violations


def main() -> int:
    """Main verification function."""
    logger.info("🔒 Starting dependency integrity verification...")

    packages = parse_uv_lock()
    violations: list[str] = []
    warnings: list[str] = []

    for package in packages:
        package_warnings, package_violations = evaluate_package(package)
        warnings.extend(package_warnings)
        violations.extend(package_violations)

    if violations:
        for violation in violations:
            logger.error("%s", violation)

    if warnings:
        for warning in warnings:
            logger.warning("%s", warning)

    if violations:
        logger.error(
            "Security verification failed with %d violations",
            len(violations),
        )
        return 1

    if warnings:
        logger.warning(
            "Security verification passed with %d warnings",
            len(warnings),
        )
        return 0

    logger.info("All dependency integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
