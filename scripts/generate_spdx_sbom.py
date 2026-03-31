#!/usr/bin/env python3

"""
SPDX SBOM generation script for Tok security compliance.
Generates SPDX 2.3 format Software Bill of Materials.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

def parse_uv_lock() -> list:
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

def generate_spdx_sbom(packages: list) -> dict:
    """Generate SPDX 2.3 format SBOM."""
    
    # Document creation info
    creation_info = {
        "created": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "creators": [
            "Tool: tok-security-pipeline-1.0.0",
            "Organization: Tok Team"
        ]
    }
    
    # Document descriptor
    document_descriptor = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "tok-protocol-dependencies",
        "documentNamespace": f"https://tok-protocol.dev/sbom/{uuid.uuid4()}",
        "creationInfo": creation_info
    }
    
    # Package information
    document_packages = []
    
    # Add the main package
    main_package = {
        "name": "tok-protocol",
        "SPDXID": "SPDXRef-tok-protocol",
        "versionInfo": "0.1.0",
        "downloadLocation": "https://github.com/tokmacher/tok",
        "filesAnalyzed": False,
        "licenseConcluded": "Apache-2.0",
        "licenseDeclared": "Apache-2.0",
        "copyrightText": "Copyright 2024 Tok Team",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": "pkg:pypi/tok-protocol@0.1.0"
            }
        ]
    }
    document_packages.append(main_package)
    
    # Add dependencies
    for package in packages:
        package_name = package.get("name", "")
        version = package.get("version", "")
        package_hash = package.get("hash", "")
        package_url = package.get("url", "")
        
        if not package_name or not version:
            continue
        
        spdx_package = {
            "name": package_name,
            "SPDXID": f"SPDXRef-{package_name.replace('-', '_').replace('.', '_')}",
            "versionInfo": version,
            "downloadLocation": package_url or "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{package_name}@{version}"
                }
            ]
        }
        
        # Add checksum if available
        if package_hash:
            spdx_package["checksums"] = [
                {
                    "algorithm": "SHA256",
                    "checksumValue": package_hash
                }
            ]
        
        document_packages.append(spdx_package)
    
    # Relationships
    relationships = []
    
    # Add relationships for dependencies
    for package in packages:
        package_name = package.get("name", "")
        if package_name:
            spdx_id = f"SPDXRef-{package_name.replace('-', '_').replace('.', '_')}"
            relationships.append({
                "spdxElementId": "SPDXRef-tok-protocol",
                "relatedSpdxElement": spdx_id,
                "relationshipType": "DEPENDS_ON"
            })
    
    # Build complete SBOM
    sbom = {
        **document_descriptor,
        "packages": document_packages,
        "relationships": relationships
    }
    
    return sbom

def main():
    """Main function to generate SBOM."""
    print("📋 Generating SPDX SBOM...")
    
    packages = parse_uv_lock()
    print(f"📦 Found {len(packages)} dependencies")
    
    sbom = generate_spdx_sbom(packages)
    
    output_file = Path("sbom.spdx")
    with open(output_file, 'w') as f:
        json.dump(sbom, f, indent=2)
    
    print(f"✅ SPDX SBOM generated: {output_file}")
    print(f"📊 Total packages: {len(sbom['packages'])}")
    print(f"🔗 Relationships: {len(sbom['relationships'])}")

if __name__ == "__main__":
    main()
