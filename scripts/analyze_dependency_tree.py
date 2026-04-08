#!/usr/bin/env python3

"""
Dependency tree analysis script for Tok security monitoring.

Analyzes dependency depth, transitive dependencies, and security metrics.
"""

import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    "allowed_sources": ["https://pypi.org", "https://files.pythonhosted.org"],
    "request_timeout": 30,
    "max_retries": 3,
    "rate_limit_delay": 0.1,  # 100ms between requests
    "user_agent": "tok-dependency-analyzer/0.1.0",
}

# Package name validation regex based on PyPI requirements
# PyPI allows: letters, numbers, hyphens, underscores, and dots
# Must start and end with letter or number, no consecutive special chars
PACKAGE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|(?:[._-](?=[a-zA-Z0-9])))*$")
VERSION_REGEX = re.compile(r"^[a-zA-Z0-9._+-]+$")


def _read_lock_file() -> list[str]:
    """Read the uv.lock file and return its lines."""
    lock_file = Path("uv.lock")
    if not lock_file.exists():
        logger.error("uv.lock file not found")
        sys.exit(1)

    try:
        with open(lock_file, encoding="utf-8") as f:
            return f.readlines()
    except OSError as e:
        logger.exception(f"Failed to read uv.lock file: {e}")
        sys.exit(1)


def _update_package_field(package: dict[str, Any], key: str, value: str, line_num: int) -> None:
    """Update a single field in the package based on the key."""
    if key == "version":
        _set_package_version(package, value, line_num)
    elif key == "hash":
        package["hash"] = value
    elif key == "url":
        _set_package_url(package, value, line_num)
    elif key == "dependencies":
        deps = _parse_dependencies(value, line_num)
        if deps:
            package["dependencies"] = deps


def _process_lock_line(
    line: str,
    line_num: int,
    current_package: dict[str, Any] | None,
    packages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Process a single line from the lock file."""
    stripped = line.strip()
    if not stripped or "=" not in stripped:
        return current_package, packages

    key, value = _split_lock_line(stripped)

    if key == "name":
        if current_package:
            packages.append(current_package)
        return _begin_package(value, line_num), packages

    if current_package:
        _update_package_field(current_package, key, value, line_num)

    return current_package, packages


def parse_uv_lock() -> list:
    """Parse uv.lock file to extract package information."""
    lines = _read_lock_file()
    packages: list[dict[str, Any]] = []
    current_package: dict[str, Any] | None = None

    for line_num, raw_line in enumerate(lines, 1):
        current_package, packages = _process_lock_line(raw_line, line_num, current_package, packages)

    if current_package:
        packages.append(current_package)

    logger.info(f"Parsed {len(packages)} packages from uv.lock")
    return packages


def _split_lock_line(line: str) -> tuple[str, str]:
    key, value = line.split("=", 1)
    return key.strip(), value.strip().strip('"')


def _begin_package(name: str, line_num: int) -> dict[str, Any] | None:
    if not validate_package_name(name):
        logger.warning(f"Invalid package name at line {line_num}: {name}")
        return None
    return {"name": name}


def _set_package_version(package: dict[str, Any], version: str, line_num: int) -> None:
    if not validate_version(version):
        logger.warning(f"Invalid version at line {line_num}: {version}")
        return
    package["version"] = version


def _set_package_url(package: dict[str, Any], url: str, line_num: int) -> None:
    if not any(url.startswith(source) for source in SECURITY_CONFIG["allowed_sources"]):
        logger.warning(f"Untrusted package source at line {line_num}: {url}")
        return
    package["url"] = url


def _parse_dependencies(value: str, line_num: int) -> list[str]:
    if not value.startswith("[") or not value.endswith("]"):
        return []
    deps_list = value[1:-1]
    dependencies = []
    for dep in deps_list.split(","):
        candidate = dep.strip().strip('"')
        if not candidate:
            continue
        if validate_package_name(candidate):
            dependencies.append(candidate)
        else:
            logger.warning(f"Invalid dependency at line {line_num}: {candidate}")
    return dependencies


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


def create_secure_session() -> requests.Session:
    """Create a secure HTTP session with proper configuration."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=SECURITY_CONFIG["max_retries"],
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set secure headers
    session.headers.update(
        {
            "User-Agent": SECURITY_CONFIG["user_agent"],
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "close",
        }
    )

    return session


def get_package_security_data(package_name: str, version: str) -> dict:
    """Fetch security data for a package with proper security measures."""
    # Input validation
    if not validate_package_name(package_name):
        logger.error(f"Invalid package name: {package_name}")
        return {}

    if not validate_version(version):
        logger.error(f"Invalid version: {version}")
        return {}

    # Rate limiting
    time.sleep(SECURITY_CONFIG["rate_limit_delay"])

    session = create_secure_session()

    try:
        url = f"https://pypi.org/pypi/{package_name}/{version}/json"
        logger.debug(f"Fetching package data: {package_name}@{version}")

        response = session.get(
            url,
            timeout=SECURITY_CONFIG["request_timeout"],
            verify=True,  # SSL verification enabled
        )

        response.raise_for_status()
        data = response.json()

        # Validate response structure
        if not isinstance(data, dict) or "releases" not in data:
            logger.warning(f"Invalid response structure for {package_name}@{version}")
            return {}

        # Extract relevant security information with validation
        uploads = data.get("uploads", [])
        releases = data.get("releases", {}).get(version, [])

        return {
            "upload_time": uploads[0].get("upload_time") if uploads else None,
            "package_size": sum(f.get("size", 0) for f in releases if isinstance(f, dict)),
            "has_wheel": any(isinstance(f, dict) and f.get("packagetype") == "bdist_wheel" for f in releases),
            "has_source": any(isinstance(f, dict) and f.get("packagetype") == "sdist" for f in releases),
        }

    except requests.exceptions.SSLError as e:
        logger.exception(f"SSL error fetching {package_name}@{version}: {e}")
        return {}
    except requests.exceptions.Timeout as e:
        logger.exception(f"Timeout fetching {package_name}@{version}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.exception(f"Network error fetching {package_name}@{version}: {e}")
        return {}
    except (json.JSONDecodeError, ValueError) as e:
        logger.exception(f"Invalid JSON response for {package_name}@{version}: {e}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error fetching {package_name}@{version}: {e}")
        return {}
    finally:
        session.close()


def analyze_dependency_tree(packages: list) -> dict:
    """Analyze the dependency tree for security metrics."""
    # Build dependency graph
    dependency_graph = defaultdict(set)
    package_info = {}

    for package in packages:
        name = package.get("name", "")
        version = package.get("version", "")

        if not name:
            continue

        package_info[name] = {
            "version": version,
            "hash": package.get("hash", ""),
            "url": package.get("url", ""),
            "dependencies": package.get("dependencies", []),
            **get_package_security_data(name, version),
        }

        for dep in package.get("dependencies", []):
            dependency_graph[name].add(dep)

    # Calculate metrics
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_packages": len(packages),
            "total_dependencies": sum(len(p.get("dependencies", [])) for p in packages),
            "packages_with_hashes": sum(1 for p in packages if p.get("hash")),
            "packages_with_wheels": sum(
                1 for p in packages if package_info.get(p.get("name", ""), {}).get("has_wheel")
            ),
            "total_size_mb": sum(package_info.get(p.get("name", ""), {}).get("package_size", 0) for p in packages)
            / (1024 * 1024),
        },
        "security_metrics": {
            "packages_with_integrity_checks": sum(1 for p in packages if p.get("hash")),
            "packages_from_trusted_sources": sum(1 for p in packages if "pypi.org" in p.get("url", "")),
            "recent_packages": 0,  # Will be calculated below
        },
        "dependency_analysis": {
            "max_dependency_depth": calculate_max_depth(dependency_graph),
            "packages_with_no_dependencies": sum(1 for p in packages if not p.get("dependencies")),
            "most_depended_upon": find_most_depended_upon(dependency_graph),
            "dependency_cycles": find_dependency_cycles(dependency_graph),
        },
        "package_details": package_info,
    }

    # Calculate recent packages (less than 90 days)
    ninety_days_ago = datetime.now(timezone.utc).timestamp() - (90 * 24 * 3600)
    for package in packages:
        name = package.get("name", "")
        upload_time_str = package_info.get(name, {}).get("upload_time")
        if upload_time_str:
            try:
                upload_time = datetime.fromisoformat(upload_time_str.replace("Z", "+00:00")).timestamp()
                if upload_time > ninety_days_ago:
                    analysis["security_metrics"]["recent_packages"] += 1
            except ValueError:
                pass

    return analysis


def calculate_max_depth(graph: dict) -> int:
    """Calculate maximum dependency depth."""

    def get_depth(package, visited=None):
        if visited is None:
            visited = set()

        if package in visited:
            return 0  # Cycle detected

        visited.add(package)
        max_depth = 0

        for dep in graph.get(package, set()):
            max_depth = max(max_depth, 1 + get_depth(dep, visited.copy()))

        return max_depth

    return max(get_depth(pkg) for pkg in graph)


def find_most_depended_upon(graph: dict) -> list:
    """Find packages that are depended upon by the most other packages."""
    dependency_count = Counter()

    for deps in graph.values():
        for dep in deps:
            dependency_count[dep] += 1

    return dependency_count.most_common(10)


def find_dependency_cycles(graph: dict) -> list:
    """Find dependency cycles in the graph."""
    cycles = []
    visited = set()
    rec_stack = set()

    def dfs(package, path) -> None:
        if package in rec_stack:
            # Found a cycle
            cycle_start = path.index(package)
            cycle = [*path[cycle_start:], package]
            cycles.append(cycle)
            return

        if package in visited:
            return

        visited.add(package)
        rec_stack.add(package)

        for dep in graph.get(package, set()):
            dfs(dep, [*path, package])

        rec_stack.remove(package)

    for package in graph:
        if package not in visited:
            dfs(package, [])

    return cycles


def main() -> None:
    """Main analysis function."""
    logger.info("🔍 Analyzing dependency tree...")

    try:
        packages = parse_uv_lock()
        logger.info(f"📦 Found {len(packages)} packages")

        analysis = analyze_dependency_tree(packages)

        # Save analysis
        output_file = Path("dependency-analysis.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"✅ Analysis saved to {output_file}")
        except OSError as e:
            logger.exception(f"Failed to save analysis: {e}")
            sys.exit(1)

        # Print summary

        if analysis["dependency_analysis"]["dependency_cycles"]:
            for _cycle in analysis["dependency_analysis"]["dependency_cycles"][:5]:
                pass

        # Security warnings
        if analysis["security_metrics"]["recent_packages"] > 0:
            pass

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
