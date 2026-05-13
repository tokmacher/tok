from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ResolverPrivacy:
    allow_remote_resolution: bool
    allow_referral_following: bool
    metadata_export_policy: Literal["local_only"]


@dataclass(frozen=True)
class ResolverStoragePolicy:
    max_total_bytes: int
    object_ttl_seconds: int
    eviction_policy: Literal["lru"]


@dataclass(frozen=True)
class ResolverManifest:
    manifest_version: Literal["tok-resolver/v0.1"]
    resolver_id: str
    supported_hash_algorithms: tuple[Literal["sha256"], ...]
    max_conformance_level: Literal["L3a"]
    routing_scope: Literal["local_only"]
    privacy: ResolverPrivacy
    storage_policy: ResolverStoragePolicy
    configured_peers: tuple[str, ...] = ()
    configured_gateways: tuple[str, ...] = ()

    @staticmethod
    def default() -> ResolverManifest:
        return ResolverManifest(
            manifest_version="tok-resolver/v0.1",
            resolver_id=str(uuid.uuid4()),
            supported_hash_algorithms=("sha256",),
            max_conformance_level="L3a",
            routing_scope="local_only",
            privacy=ResolverPrivacy(
                allow_remote_resolution=False,
                allow_referral_following=False,
                metadata_export_policy="local_only",
            ),
            storage_policy=ResolverStoragePolicy(
                max_total_bytes=512 * 1024 * 1024,
                object_ttl_seconds=30 * 24 * 60 * 60,
                eviction_policy="lru",
            ),
            configured_peers=(),
            configured_gateways=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "resolver_id": self.resolver_id,
            "supported_hash_algorithms": list(self.supported_hash_algorithms),
            "max_conformance_level": self.max_conformance_level,
            "routing_scope": self.routing_scope,
            "privacy": {
                "allow_remote_resolution": self.privacy.allow_remote_resolution,
                "allow_referral_following": self.privacy.allow_referral_following,
                "metadata_export_policy": self.privacy.metadata_export_policy,
            },
            "storage_policy": {
                "max_total_bytes": self.storage_policy.max_total_bytes,
                "object_ttl_seconds": self.storage_policy.object_ttl_seconds,
                "eviction_policy": self.storage_policy.eviction_policy,
            },
            "configured_peers": list(self.configured_peers),
            "configured_gateways": list(self.configured_gateways),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def load(path: Path) -> ResolverManifest:
        data = json.loads(path.read_text())
        return ResolverManifest.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ResolverManifest:
        manifest_version = data.get("manifest_version")
        if manifest_version != "tok-resolver/v0.1":
            raise ValueError(f"Unsupported manifest_version: {manifest_version!r}")

        privacy_data = data.get("privacy") or {}
        if privacy_data.get("allow_remote_resolution") is not False:
            raise ValueError("privacy.allow_remote_resolution must be false in 0.2.0")
        if privacy_data.get("allow_referral_following") is not False:
            raise ValueError("privacy.allow_referral_following must be false in 0.2.0")
        if privacy_data.get("metadata_export_policy") != "local_only":
            raise ValueError("privacy.metadata_export_policy must be 'local_only' in 0.2.0")

        configured_peers = tuple(data.get("configured_peers") or [])
        configured_gateways = tuple(data.get("configured_gateways") or [])
        if configured_peers:
            raise ValueError("configured_peers must be empty in 0.2.0")
        if configured_gateways:
            raise ValueError("configured_gateways must be empty in 0.2.0")

        supported = tuple(data.get("supported_hash_algorithms") or [])
        if supported != ("sha256",) and supported != ["sha256"]:
            raise ValueError("supported_hash_algorithms must be ['sha256'] in 0.2.0")

        storage = data.get("storage_policy") or {}
        return ResolverManifest(
            manifest_version="tok-resolver/v0.1",
            resolver_id=str(data.get("resolver_id") or ""),
            supported_hash_algorithms=("sha256",),
            max_conformance_level="L3a",
            routing_scope="local_only",
            privacy=ResolverPrivacy(
                allow_remote_resolution=False,
                allow_referral_following=False,
                metadata_export_policy="local_only",
            ),
            storage_policy=ResolverStoragePolicy(
                max_total_bytes=int(storage.get("max_total_bytes") or 0),
                object_ttl_seconds=int(storage.get("object_ttl_seconds") or 0),
                eviction_policy="lru",
            ),
            configured_peers=(),
            configured_gateways=(),
        )
