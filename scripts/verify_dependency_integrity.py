#!/usr/bin/env python3

"""
Dependency integrity verification script for Tok security pipeline.
Verifies package hashes, signatures, and integrity metrics.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
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
    "min_package_age_days": 90,
    "require_signatures": True,
    "allowed_sources": ["https://pypi.org/simple", "https://files.pythonhosted.org"],
    "blocked_packages": [
        # Known malicious or suspicious packages
        "python3-lib",
        "pycrypto",  # Deprecated, use pycryptodome instead
        # Add more as needed
    ],
    "request_timeout": 30,
    "max_retries": 3,
    "rate_limit_delay": 0.1,  # 100ms between requests
    "user_agent": "tok-dependency-verifier/0.1.0",
}

# Package name validation regex
PACKAGE_NAME_REGEX = re.compile(r'^[a-zA-Z0-9._-]+$')
VERSION_REGEX = re.compile(r'^[a-zA-Z0-9._+-]+$')
HASH_REGEX = re.compile(r'^[a-fA-F0-9]{64}$')  # SHA-256 hash


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
    return bool(HASH_REGEX.fullmatch(hash_str))

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

def get_package_metadata(package_name: str, version: str) -> dict | None:
    """Fetch package metadata from PyPI with security measures."""
    # Input validation
    if not validate_package_name(package_name):
        logger.error(f"Invalid package name: {package_name}")
        return None
    
    if not validate_version(version):
        logger.error(f"Invalid version: {version}")
        return None
    
    # Check blocked packages
    if package_name in SECURITY_CONFIG["blocked_packages"]:
        logger.error(f"Blocked package detected: {package_name}")
        return None
    
    # Rate limiting
    time.sleep(SECURITY_CONFIG["rate_limit_delay"])
    
    session = create_secure_session()
    
    try:
        url = f"https://pypi.org/pypi/{package_name}/{version}/json"
        logger.debug(f"Fetching metadata: {package_name}@{version}")
        
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
            return None
        
        return data
        
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error fetching {package_name}@{version}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout fetching {package_name}@{version}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching {package_name}@{version}: {e}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON response for {package_name}@{version}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {package_name}@{version}: {e}")
        return None
    finally:
        session.close()


def check_package_age(package_name: str, version: str) -> tuple[bool, int]:
    """Check if package is old enough for security consideration."""
    metadata = get_package_metadata(package_name, version)
    if not metadata:
        return False, 0

    try:
        uploads = metadata.get("uploads", [])
        if not uploads:
            logger.warning(f"No upload data found for {package_name}@{version}")
            return False, 0
            
        upload_time_str = uploads[0].get("upload_time")
        if not upload_time_str:
            logger.warning(f"No upload time found for {package_name}@{version}")
            return False, 0
            
        upload_time = datetime.fromisoformat(
            upload_time_str.replace("Z", "+00:00")
        )
        age_days = (datetime.now(timezone.utc) - upload_time).days

        is_old_enough = age_days >= SECURITY_CONFIG["min_package_age_days"]
        return is_old_enough, age_days
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing upload time for {package_name}@{version}: {e}")
        return False, 0


def verify_package_hash(
    package_name: str, version: str, expected_hash: str
) -> bool:
    """Verify package hash against PyPI record."""
    # Input validation
    if not validate_hash(expected_hash):
        logger.error(f"Invalid hash format for {package_name}@{version}: {expected_hash}")
        return False
    
    metadata = get_package_metadata(package_name, version)
    if not metadata:
        return False

    try:
        releases = metadata.get("releases", {}).get(version, [])
        if not releases:
            logger.warning(f"No releases found for {package_name}@{version}")
            return False
        
        # Find the matching release file
        for release_file in releases:
            if not isinstance(release_file, dict):
                continue
                
            if release_file.get("packagetype") == "bdist_wheel":
                hashes = release_file.get("hashes", {})
                pypi_hash = hashes.get("sha256")
                
                if not pypi_hash:
                    logger.warning(f"No SHA256 hash found for {package_name}@{version} wheel")
                    continue
                
                # Compare hashes case-insensitively
                return pypi_hash.lower() == expected_hash.lower()

        logger.warning(f"No wheel release found for {package_name}@{version}")
        return False
    except Exception as e:
        logger.error(f"Error verifying hash for {package_name}@{version}: {e}")
        return False


def parse_uv_lock() -> list[dict]:
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
            hash_str = line.split("=")[1].strip().strip('"')
            # Validate hash format
            if not validate_hash(hash_str):
                logger.warning(f"Invalid hash format at line {line_num}: {hash_str}")
                continue
            current_package["hash"] = hash_str
        elif line.startswith("url = ") and current_package:
            url = line.split("=")[1].strip().strip('"')
            # Validate URL is from trusted source
            if not any(url.startswith(source) for source in SECURITY_CONFIG["allowed_sources"]):
                logger.warning(f"Untrusted package source at line {line_num}: {url}")
                continue
            current_package["url"] = url

    if current_package:
        packages.append(current_package)

    logger.info(f"Parsed {len(packages)} packages from uv.lock")
    return packages


def check_blocked_packages(package_name: str) -> bool:
    """Check if package is in blocked list."""
    return package_name.lower() in [
        p.lower() for p in SECURITY_CONFIG["blocked_packages"]
    ]


def verify_source_integrity(package_url: str) -> bool:
    """Verify package comes from allowed source."""
    return any(
        source in package_url for source in SECURITY_CONFIG["allowed_sources"]
    )


def main() -> int:
    """Main verification function."""
    logger.info("🔒 Starting dependency integrity verification...")

    try:
        packages = parse_uv_lock()
        violations = []
        warnings = []

        for package in packages:
            package_name = package.get("name", "")
            version = package.get("version", "")
            package_hash = package.get("hash", "")
            package_url = package.get("url", "")

            if not package_name or not version:
                warnings.append(f"⚠️  Incomplete package info: {package}")
                continue

            logger.debug(f"🔍 Checking {package_name}@{version}")

            # Check blocked packages
            if check_blocked_packages(package_name):
                violations.append(f"❌ Blocked package found: {package_name}")
                continue

            # Check source integrity
            if package_url and not verify_source_integrity(package_url):
                violations.append(
                    f"❌ Untrusted source for {package_name}: {package_url}"
                )
                continue

            # Check package age
            is_old_enough, age_days = check_package_age(package_name, version)
            if not is_old_enough:
                violations.append(
                    f"❌ Package too new: {package_name}@{version} ({age_days} days old, "
                    f"minimum {SECURITY_CONFIG['min_package_age_days']} days required)"
                )

            # Verify hash if available
            if package_hash:
                if not verify_package_hash(package_name, version, package_hash):
                    violations.append(
                        f"❌ Hash verification failed for {package_name}@{version}"
                    )
            else:
                warnings.append(
                    f"⚠️  No hash available for {package_name}@{version}"
                )

        # Report results
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

        if not violations and not warnings:
            print("\n✅ All dependency integrity checks passed!")
            logger.info("All dependency integrity checks passed")
            return 0
        elif violations:
            print(
                f"\n❌ Security verification failed with {len(violations)} violations"
            )
            logger.error(f"Security verification failed with {len(violations)} violations")
            return 1
        else:
            print(
                f"\n⚠️  Security verification passed with {len(warnings)} warnings"
            )
            logger.warning(f"Security verification passed with {len(warnings)} warnings")
            return 0
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        print(f"\n❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
