from __future__ import annotations

import re
import uuid

from tok.protocol._common import _prefixed_id, _utc_now_z


def test_utc_now_z_ends_with_z() -> None:
    ts = _utc_now_z()
    assert ts.endswith("Z")


def test_utc_now_z_has_no_plus_offset() -> None:
    ts = _utc_now_z()
    assert "+" not in ts


def test_utc_now_z_matches_iso8601_pattern() -> None:
    ts = _utc_now_z()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts)


def test_prefixed_id_starts_with_prefix() -> None:
    id_ = _prefixed_id("test_")
    assert id_.startswith("test_")
    assert id_ == "test_" + uuid.UUID(hex=id_[len("test_") :]).hex


def test_prefixed_id_unique_each_call() -> None:
    a = _prefixed_id("pref_")
    b = _prefixed_id("pref_")
    assert a != b


def test_prefixed_id_valid_uuid_hex() -> None:
    id_ = _prefixed_id("foo_")
    hex_part = id_[len("foo_") :]
    uuid.UUID(hex=hex_part)
