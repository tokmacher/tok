#!/usr/bin/env python3

"""Shared uv.lock parsing helpers for release and security scripts."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

logger = logging.getLogger(__name__)

PACKAGE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|(?:[._-](?=[a-zA-Z0-9])))*$")
VERSION_REGEX = re.compile(r"^[a-zA-Z0-9._+-]+$")


def validate_package_name(package_name: str) -> bool:
    """Validate package name for security-sensitive script use."""
    if not package_name or len(package_name) > 100:
        return False
    return bool(PACKAGE_NAME_REGEX.fullmatch(package_name))


def validate_version(version: str) -> bool:
    """Validate version string for security-sensitive script use."""
    if not version or len(version) > 50:
        return False
    return bool(VERSION_REGEX.fullmatch(version))


def normalize_hash(hash_str: str) -> str:
    """Strip the algorithm prefix from uv.lock hashes."""
    return hash_str.removeprefix("sha256:")


def parse_upload_time(upload_time: str) -> datetime | None:
    """Parse uv.lock upload timestamps."""
    try:
        parsed = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def collect_artifacts(package: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect wheel and sdist metadata from a uv.lock package entry."""
    artifacts: list[dict[str, Any]] = []

    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        artifacts.append(sdist)

    wheels = package.get("wheels", [])
    if isinstance(wheels, list):
        artifacts.extend(artifact for artifact in wheels if isinstance(artifact, dict))

    return artifacts


def extract_dependency_names(raw_dependencies: Any) -> list[str]:
    """Extract dependency names from uv.lock dependency entries."""
    if not isinstance(raw_dependencies, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for dependency in raw_dependencies:
        candidate = ""
        if isinstance(dependency, dict):
            candidate = str(dependency.get("name", ""))
        elif isinstance(dependency, str):
            candidate = dependency
        if not validate_package_name(candidate) or candidate in seen:
            continue
        seen.add(candidate)
        names.append(candidate)
    return names


def load_uv_lock(lock_path: str | Path = "uv.lock") -> list[dict[str, Any]]:
    """Parse uv.lock into normalized package records."""
    resolved_lock_path = Path(lock_path)
    if not resolved_lock_path.exists():
        logger.error("uv.lock file not found")
        sys.exit(1)

    try:
        with resolved_lock_path.open("rb") as lock_file:
            data = tomllib.load(lock_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.exception("Failed to read uv.lock file: %s", exc)
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
            logger.warning("Invalid package version in uv.lock: %s@%s", package_name, version)
            continue

        source = raw_package.get("source", {})
        normalized_source = source if isinstance(source, dict) else {}
        artifacts = collect_artifacts(raw_package)
        primary_artifact = artifacts[0] if artifacts else None
        hashes = [str(artifact.get("hash", "")) for artifact in artifacts if artifact.get("hash")]
        primary_upload_time = (
            parse_upload_time(str(primary_artifact.get("upload-time", "")))
            if isinstance(primary_artifact, dict)
            else None
        )

        packages.append(
            {
                "name": package_name,
                "version": version,
                "source": normalized_source,
                "editable": str(normalized_source["editable"]) if "editable" in normalized_source else None,
                "dependencies": extract_dependency_names(raw_package.get("dependencies", [])),
                "artifacts": artifacts,
                "primary_artifact": primary_artifact,
                "artifact_url": (str(primary_artifact.get("url", "")) if isinstance(primary_artifact, dict) else ""),
                "artifact_hash": (str(primary_artifact.get("hash", "")) if isinstance(primary_artifact, dict) else ""),
                "hashes": hashes,
                "upload_time": primary_upload_time,
            }
        )

    logger.info("Parsed %d packages from uv.lock", len(packages))
    return packages
