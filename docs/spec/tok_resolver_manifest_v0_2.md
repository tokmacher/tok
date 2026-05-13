# Tok Resolver Manifest v0.2 (Local Beta)

Tok 0.2.0 adds a **local-only resolver beta**. It is a content-addressed store used to
back trace content URIs and make audits repeatable **without any remote routing**.

This spec defines the manifest format and the on-disk layout for the local store.

## Non-goals (0.2.0)

- No global network routing.
- No automatic remote resolution.
- No referral following.
- No claims of stable protocol or protocol compliance.

## Manifest file

Path:

- User-wide manifest: `~/.tok/resolver/manifest.tok`
- Project manifest: `~/.tok/projects/<project-digest>/manifest.tok` (reserved, not
  implemented in 0.2.0)

Format: UTF-8 JSON.

### Required fields

- `manifest_version`: `"tok-resolver/v0.1"`
- `resolver_id`: UUID4 string (generated on first `tok resolver init`)
- `supported_hash_algorithms`: `["sha256"]` (0.2.0 only)
- `max_conformance_level`: `"L3a"`
- `routing_scope`: `"local_only"`
- `privacy`:
  - `allow_remote_resolution`: `false`
  - `allow_referral_following`: `false`
  - `metadata_export_policy`: `"local_only"`
- `storage_policy`:
  - `max_total_bytes`: integer (default `536870912` = 512MB)
  - `object_ttl_seconds`: integer (default `2592000` = 30 days)
  - `eviction_policy`: `"lru"`

Note: `storage_policy` fields are declared intent for 0.2.x enforcement. Tok 0.2.0 does
not enforce quotas or TTL eviction in the store yet.

### Optional fields

- `configured_peers`: `[]` (must be empty in 0.2.0)
- `configured_gateways`: `[]` (must be empty in 0.2.0)

## Invariants

- The manifest must not claim content exists unless the object exists in the local store
  at the expected path.
- If the manifest is malformed or corrupted, Tok must surface an error (and not silently
  ignore it).
- Digest strings must be validated strictly. Only `sha256:<64 lowercase hex>` is valid
  in 0.2.0.

## Storage layout

Resolver root: `~/.tok/resolver/`

Objects:

- `~/.tok/resolver/objects/<first2>/<rest>`

Where:

- Digest is `sha256:<64 hex>`.
- `<first2>` is the first two hex characters of the digest (after `sha256:`).
- `<rest>` is the remaining 62 hex characters.

Example:

- Digest: `sha256:0123...` (64 hex chars)
- Path: `~/.tok/resolver/objects/01/23...`
