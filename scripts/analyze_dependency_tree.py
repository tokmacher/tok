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
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
PACKAGE_NAME_REGEX = re.compile(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|(?:[._-](?=[a-zA-Z0-9])))*$')
VERSION_REGEX = re.compile(r'^[a-zA-Z0-9._+-]+$')


def parse_uv_lock() -> list:
    """Parse uv.lock file to extract package information."""
    lock_file = Path("uv.lock")
    if not lock_file.exists():
        logger.error("uv.lock file not found")
        sys.exit(1)

    packages = []
    current_package = {}

    try:
        with open(lock_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        logger.error(f"Failed to read uv.lock file: {e}")
        sys.exit(1)

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith("name = "):
            if current_package:
                packages.append(current_package)
            name = line.split("=")[1].strip().strip('"')
            # Validate package name
            if not validate_package_name(name):
                logger.warning(f"Invalid package name at line {line_num}: {name}")
                continue
            current_package = {"name": name}
        elif line.startswith("version = ") and current_package:
            version = line.split("=")[1].strip().strip('"')
            # Validate version
            if not validate_version(version):
                logger.warning(f"Invalid version at line {line_num}: {version}")
                continue
            current_package["version"] = version
        elif line.startswith("hash = ") and current_package:
            current_package["hash"] = line.split("=")[1].strip().strip('"')
        elif line.startswith("url = ") and current_package:
            url = line.split("=")[1].strip().strip('"')
            # Validate URL is from trusted source
            if not any(url.startswith(source) for source in SECURITY_CONFIG["allowed_sources"]):
                logger.warning(f"Untrusted package source at line {line_num}: {url}")
                continue
            current_package["url"] = url
        elif line.startswith("dependencies = ") and current_package:
            # Parse dependencies list
            deps_str = line.split("=")[1].strip()
            if deps_str.startswith("[") and deps_str.endswith("]"):
                deps_str = deps_str[1:-1]
                deps = []
                for dep in deps_str.split(","):
                    dep = dep.strip().strip('"')
                    if dep and validate_package_name(dep):
                        deps.append(dep)
                    elif dep:
                        logger.warning(f"Invalid dependency at line {line_num}: {dep}")
                current_package["dependencies"] = deps

    if current_package:
        packages.append(current_package)

    logger.info(f"Parsed {len(packages)} packages from uv.lock")
    return packages


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
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1,
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Set secure headers
    session.headers.update({
        'User-Agent': SECURITY_CONFIG["user_agent"],
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'close',
    })
    
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
            verify=True  # SSL verification enabled
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
        
        security_info = {
            "upload_time": uploads[0].get("upload_time") if uploads else None,
            "package_size": sum(
                f.get("size", 0) for f in releases if isinstance(f, dict)
            ),
            "has_wheel": any(
                isinstance(f, dict) and f.get("packagetype") == "bdist_wheel"
                for f in releases
            ),
            "has_source": any(
                isinstance(f, dict) and f.get("packagetype") == "sdist"
                for f in releases
            ),
        }
        
        return security_info
        
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error fetching {package_name}@{version}: {e}")
        return {}
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout fetching {package_name}@{version}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching {package_name}@{version}: {e}")
        return {}
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON response for {package_name}@{version}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching {package_name}@{version}: {e}")
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
            "total_dependencies": sum(
                len(p.get("dependencies", [])) for p in packages
            ),
            "packages_with_hashes": sum(1 for p in packages if p.get("hash")),
            "packages_with_wheels": sum(
                1
                for p in packages
                if package_info.get(p.get("name", ""), {}).get("has_wheel")
            ),
            "total_size_mb": sum(
                package_info.get(p.get("name", ""), {}).get("package_size", 0)
                for p in packages
            )
            / (1024 * 1024),
        },
        "security_metrics": {
            "packages_with_integrity_checks": sum(
                1 for p in packages if p.get("hash")
            ),
            "packages_from_trusted_sources": sum(
                1 for p in packages if "pypi.org" in p.get("url", "")
            ),
            "recent_packages": 0,  # Will be calculated below
        },
        "dependency_analysis": {
            "max_dependency_depth": calculate_max_depth(dependency_graph),
            "packages_with_no_dependencies": sum(
                1 for p in packages if not p.get("dependencies")
            ),
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
                upload_time = datetime.fromisoformat(
                    upload_time_str.replace("Z", "+00:00")
                ).timestamp()
                if upload_time > ninety_days_ago:
                    analysis["security_metrics"]["recent_packages"] += 1
            except:
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

    return max(get_depth(pkg) for pkg in graph.keys())


def find_most_depended_upon(graph: dict) -> list:
    """Find packages that are depended upon by the most other packages."""
    dependency_count = Counter()

    for package, deps in graph.items():
        for dep in deps:
            dependency_count[dep] += 1

    return dependency_count.most_common(10)


def find_dependency_cycles(graph: dict) -> list:
    """Find dependency cycles in the graph."""
    cycles = []
    visited = set()
    rec_stack = set()

    def dfs(package, path):
        if package in rec_stack:
            # Found a cycle
            cycle_start = path.index(package)
            cycle = path[cycle_start:] + [package]
            cycles.append(cycle)
            return

        if package in visited:
            return

        visited.add(package)
        rec_stack.add(package)

        for dep in graph.get(package, set()):
            dfs(dep, path + [package])

        rec_stack.remove(package)

    for package in graph.keys():
        if package not in visited:
            dfs(package, [])

    return cycles


def main():
    """Main analysis function."""
    logger.info("🔍 Analyzing dependency tree...")

    try:
        packages = parse_uv_lock()
        logger.info(f"📦 Found {len(packages)} packages")

        analysis = analyze_dependency_tree(packages)

        # Save analysis
        output_file = Path("dependency-analysis.json")
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"✅ Analysis saved to {output_file}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save analysis: {e}")
            sys.exit(1)

        # Print summary
        print("\n📊 Dependency Analysis Summary:")
        print(f"  Total packages: {analysis['summary']['total_packages']}")
        print(f"  Total dependencies: {analysis['summary']['total_dependencies']}")
        print(
            f"  Max dependency depth: {analysis['dependency_analysis']['max_dependency_depth']}"
        )
        print(
            f"  Packages with hashes: {analysis['summary']['packages_with_hashes']}"
        )
        print(
            f"  Recent packages (<90 days): {analysis['security_metrics']['recent_packages']}"
        )
        print(
            f"  Dependency cycles: {len(analysis['dependency_analysis']['dependency_cycles'])}"
        )

        if analysis["dependency_analysis"]["dependency_cycles"]:
            print("\n⚠️  Dependency cycles detected:")
            for cycle in analysis["dependency_analysis"]["dependency_cycles"][:5]:
                print(f"  - {' -> '.join(cycle)}")

        # Security warnings
        if analysis['security_metrics']['recent_packages'] > 0:
            print(f"\n⚠️  {analysis['security_metrics']['recent_packages']} packages are less than 90 days old")
            print("  Consider reviewing these packages for security")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
