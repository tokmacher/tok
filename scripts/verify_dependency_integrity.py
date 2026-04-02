#!/usr/bin/env python3

"""Dependency integrity verification script for Tok security pipeline."""

from __future__ import annotations

import logging
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
BLOCKED_PACKAGES = {
    blocked.lower() for blocked in SECURITY_CONFIG["blocked_packages"]
}

PACKAGE_NAME_REGEX = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|(?:[._-](?=[a-zA-Z0-9])))*$"
)
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


def normalize_hash(hash_str: str) -> str:
    """Strip the algorithm prefix from uv.lock hashes."""
    return hash_str.removeprefix("sha256:")


def validate_hash(hash_str: str) -> bool:
    """Validate SHA-256 hash format."""
    return bool(HASH_REGEX.fullmatch(normalize_hash(hash_str)))


def parse_upload_time(upload_time: str) -> datetime | None:
    """Parse uv.lock upload timestamps."""
    try:
        return datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_artifacts(package: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect wheel and sdist metadata from a uv.lock package entry."""
    artifacts: list[dict[str, Any]] = []

    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        artifacts.append(sdist)

    wheels = package.get("wheels", [])
    if isinstance(wheels, list):
        artifacts.extend(
            artifact for artifact in wheels if isinstance(artifact, dict)
        )

    return artifacts


def parse_uv_lock() -> list[dict[str, Any]]:
    """Parse uv.lock file to extract package information."""
    lock_file = Path("uv.lock")
    if not lock_file.exists():
        logger.error("uv.lock file not found")
        sys.exit(1)

    try:
        with open(lock_file, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.error(f"Failed to read uv.lock file: {e}")
        sys.exit(1)

    packages: list[dict[str, Any]] = []
    for raw_package in data.get("package", []):
        if not isinstance(raw_package, dict):
            continue

        package_name = str(raw_package.get("name", ""))
        version = str(raw_package.get("version", ""))
        if not validate_package_name(package_name):
            logger.warning("Invalid package name in uv.lock: %s", package_name)
            continue
        if not validate_version(version):
            logger.warning(
                "Invalid package version in uv.lock: %s@%s",
                package_name,
                version,
            )
            continue

        source = raw_package.get("source", {})
        artifacts = collect_artifacts(raw_package)
        upload_times = [
            parsed
            for artifact in artifacts
            for parsed in [
                parse_upload_time(str(artifact.get("upload-time", "")))
            ]
            if parsed is not None
        ]
        hashes = [
            str(artifact.get("hash", ""))
            for artifact in artifacts
            if artifact.get("hash")
        ]
        primary_artifact = next(
            (artifact for artifact in artifacts if artifact.get("url")), None
        )

        packages.append(
            {
                "name": package_name,
                "version": version,
                "source": source if isinstance(source, dict) else {},
                "artifact_url": (
                    str(primary_artifact.get("url", ""))
                    if isinstance(primary_artifact, dict)
                    else ""
                ),
                "hashes": hashes,
                "upload_time": min(upload_times) if upload_times else None,
            }
        )

    logger.info("Parsed %d packages from uv.lock", len(packages))
    return packages


def check_blocked_packages(package_name: str) -> bool:
    """Check if package is in blocked list."""
    return package_name.lower() in BLOCKED_PACKAGES


def verify_source_integrity(source: str) -> bool:
    """Verify package comes from an allowed source."""
    return any(
        source.startswith(allowed_source)
        for allowed_source in SECURITY_CONFIG["allowed_sources"]
    )


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
            violations.append(
                f"❌ Editable source missing for {package_name}: {editable_path}"
            )
        return warnings, violations

    registry = ""
    if isinstance(source, dict):
        registry = str(source.get("registry", ""))
    if registry and not verify_source_integrity(registry):
        violations.append(
            f"❌ Untrusted registry source for {package_name}: {registry}"
        )
        return warnings, violations
    if artifact_url and not verify_source_integrity(artifact_url):
        violations.append(
            f"❌ Untrusted artifact source for {package_name}: {artifact_url}"
        )
        return warnings, violations

    age_days = check_package_age(upload_time)
    if age_days is None:
        warnings.append(
            f"⚠️  No upload timestamp available for {package_name}@{version}"
        )
    elif age_days < SECURITY_CONFIG["min_package_age_days"]:
        warnings.append(
            f"⚠️  Recent package: {package_name}@{version} "
            f"({age_days} days old, review before release)"
        )

    if not hashes:
        warnings.append(
            f"⚠️  No artifact hashes recorded for {package_name}@{version}"
        )
        return warnings, violations
    if any(not validate_hash(hash_value) for hash_value in hashes):
        violations.append(
            f"❌ Invalid hash format recorded for {package_name}@{version}"
        )

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

    print("\n" + "=" * 60)
    print("DEPENDENCY INTEGRITY VERIFICATION RESULTS")
    print("=" * 60)

    if violations:
        print(f"\n❌ SECURITY VIOLATIONS ({len(violations)}):")
        for violation in violations:
            print(f"  - {violation}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if violations:
        print(
            f"\n❌ Security verification failed with {len(violations)} violations"
        )
        logger.error(
            "Security verification failed with %d violations",
            len(violations),
        )
        return 1

    if warnings:
        print(
            f"\n⚠️  Security verification passed with {len(warnings)} warnings"
        )
        logger.warning(
            "Security verification passed with %d warnings",
            len(warnings),
        )
        return 0

    print("\n✅ All dependency integrity checks passed!")
    logger.info("All dependency integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
