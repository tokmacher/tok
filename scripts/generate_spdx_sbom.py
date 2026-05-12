#!/usr/bin/env python3

"""
SPDX SBOM generation script for Tok security compliance.
Generates SPDX 2.3 format Software Bill of Materials.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from _project_metadata import read_project_metadata
    from _uv_lock import load_uv_lock, normalize_hash
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._project_metadata import read_project_metadata
    from scripts._uv_lock import load_uv_lock, normalize_hash


def parse_uv_lock() -> list[dict[str, Any]]:
    """Parse uv.lock file to extract package information."""
    return load_uv_lock()


def generate_spdx_sbom(
    packages: list[dict[str, Any]], project_metadata: dict[str, str] | None = None
) -> dict[str, Any]:
    """Generate SPDX 2.3 format SBOM."""
    project_metadata = project_metadata or read_project_metadata()
    project_name = project_metadata["name"]
    project_version = project_metadata["version"]

    # Document creation info
    creation_info = {
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "creators": [
            "Tool: tok-security-pipeline-1.0.0",
            "Organization: tokmacher",
        ],
    }

    # Document descriptor
    document_descriptor = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "tok-protocol-dependencies",
        "documentNamespace": f"https://tok-protocol.dev/sbom/{uuid.uuid4()}",
        "creationInfo": creation_info,
    }

    # Package information
    document_packages = []

    # Add the main package
    main_package = {
        "name": project_name,
        "SPDXID": "SPDXRef-tok-protocol",
        "versionInfo": project_version,
        "downloadLocation": "https://github.com/tokmacher/tok",
        "filesAnalyzed": False,
        "licenseConcluded": "Apache-2.0",
        "licenseDeclared": "Apache-2.0",
        "copyrightText": f"Copyright {datetime.now(timezone.utc).year} tokmacher",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:pypi/{project_name}@{project_version}",
            }
        ],
    }
    document_packages.append(main_package)

    # Add dependencies
    for package in packages:
        package_name = package.get("name", "")
        version = package.get("version", "")
        package_hash = str(package.get("artifact_hash", ""))
        package_url = str(package.get("artifact_url", ""))

        if not package_name or not version or package_name == project_name:
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
                    "referenceLocator": f"pkg:pypi/{package_name}@{version}",
                }
            ],
        }

        # Add checksum if available
        if package_hash:
            spdx_package["checksums"] = [
                {
                    "algorithm": "SHA256",
                    "checksumValue": normalize_hash(package_hash),
                }
            ]

        document_packages.append(spdx_package)

    # Relationships
    relationships = []

    # Add relationships for dependencies
    for package in packages:
        package_name = package.get("name", "")
        if package_name and package_name != project_name:
            spdx_id = f"SPDXRef-{package_name.replace('-', '_').replace('.', '_')}"
            relationships.append(
                {
                    "spdxElementId": "SPDXRef-tok-protocol",
                    "relatedSpdxElement": spdx_id,
                    "relationshipType": "DEPENDS_ON",
                }
            )

    # Build complete SBOM
    return {
        **document_descriptor,
        "packages": document_packages,
        "relationships": relationships,
    }


def main() -> None:
    """Main function to generate SBOM."""
    packages = parse_uv_lock()

    sbom = generate_spdx_sbom(packages)

    output_file = Path("sbom.spdx")
    output_file.write_text(json.dumps(sbom, indent=2) + "\n")


if __name__ == "__main__":
    main()
