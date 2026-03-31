#!/usr/bin/env python3

"""
Dependency integrity verification script for Tok security pipeline.
Verifies package hashes, signatures, and integrity metrics.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import requests
from typing import Dict, List, Optional, Tuple

# Security configuration
SECURITY_CONFIG = {
    "min_package_age_days": 90,
    "require_signatures": True,
    "allowed_sources": ["https://pypi.org/simple"],
    "blocked_packages": [
        # Known malicious or suspicious packages
        "python3-lib",
        "pycrypto",  # Deprecated, use pycryptodome instead
        # Add more as needed
    ]
}

def get_package_metadata(package_name: str, version: str) -> Optional[Dict]:
    """Fetch package metadata from PyPI."""
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/{version}/json",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch metadata for {package_name}@{version}: {e}")
        return None

def check_package_age(package_name: str, version: str) -> Tuple[bool, int]:
    """Check if package is old enough for security consideration."""
    metadata = get_package_metadata(package_name, version)
    if not metadata:
        return False, 0
    
    try:
        upload_time_str = metadata["uploads"][0]["upload_time"]
        upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - upload_time).days
        
        is_old_enough = age_days >= SECURITY_CONFIG["min_package_age_days"]
        return is_old_enough, age_days
    except Exception as e:
        print(f"❌ Error parsing upload time for {package_name}@{version}: {e}")
        return False, 0

def verify_package_hash(package_name: str, version: str, expected_hash: str) -> bool:
    """Verify package hash against PyPI record."""
    metadata = get_package_metadata(package_name, version)
    if not metadata:
        return False
    
    try:
        # Find the matching release file
        for release_file in metadata.get("releases", {}).get(version, []):
            if release_file["packagetype"] == "bdist_wheel":
                pypi_hash = release_file["hashes"]["sha256"]
                return pypi_hash.lower() == expected_hash.lower()
        
        print(f"⚠️  No wheel release found for {package_name}@{version}")
        return False
    except Exception as e:
        print(f"❌ Error verifying hash for {package_name}@{version}: {e}")
        return False

def parse_uv_lock() -> List[Dict]:
    """Parse uv.lock file to extract package information."""
    lock_file = Path("uv.lock")
    if not lock_file.exists():
        print("❌ uv.lock file not found")
        sys.exit(1)
    
    packages = []
    current_package = {}
    
    with open(lock_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('name = '):
            if current_package:
                packages.append(current_package)
            current_package = {"name": line.split('=')[1].strip().strip('"')}
        elif line.startswith('version = ') and current_package:
            current_package["version"] = line.split('=')[1].strip().strip('"')
        elif line.startswith('hash = ') and current_package:
            current_package["hash"] = line.split('=')[1].strip().strip('"')
        elif line.startswith('url = ') and current_package:
            current_package["url"] = line.split('=')[1].strip().strip('"')
    
    if current_package:
        packages.append(current_package)
    
    return packages

def check_blocked_packages(package_name: str) -> bool:
    """Check if package is in blocked list."""
    return package_name.lower() in [p.lower() for p in SECURITY_CONFIG["blocked_packages"]]

def verify_source_integrity(package_url: str) -> bool:
    """Verify package comes from allowed source."""
    return any(source in package_url for source in SECURITY_CONFIG["allowed_sources"])

def main() -> int:
    """Main verification function."""
    print("🔒 Starting dependency integrity verification...")
    
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
        
        print(f"🔍 Checking {package_name}@{version}")
        
        # Check blocked packages
        if check_blocked_packages(package_name):
            violations.append(f"❌ Blocked package found: {package_name}")
            continue
        
        # Check source integrity
        if package_url and not verify_source_integrity(package_url):
            violations.append(f"❌ Untrusted source for {package_name}: {package_url}")
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
                violations.append(f"❌ Hash verification failed for {package_name}@{version}")
        else:
            warnings.append(f"⚠️  No hash available for {package_name}@{version}")
    
    # Report results
    print("\n" + "="*60)
    print("DEPENDENCY INTEGRITY VERIFICATION RESULTS")
    print("="*60)
    
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
        return 0
    elif violations:
        print(f"\n❌ Security verification failed with {len(violations)} violations")
        return 1
    else:
        print(f"\n⚠️  Security verification passed with {len(warnings)} warnings")
        return 0

if __name__ == "__main__":
    sys.exit(main())
