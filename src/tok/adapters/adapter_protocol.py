"""Formal adapter boundary protocol for Tok runtimes."""

from __future__ import annotations

from typing import Protocol

from tok.runtime.core import ProcessedRuntimeResponse, RuntimeRequest


class AdapterProtocol(Protocol):
    """Transport boundary every runtime adapter should expose."""

    def identify_runtime(self) -> str:
        """Return the runtime identity, such as ``claude-code``."""

    def parse_inbound_request(self, raw_bytes: bytes) -> RuntimeRequest:
        """Parse transport bytes into a runtime request."""

    def build_outbound_response(self, processed: ProcessedRuntimeResponse) -> bytes:
        """Build transport response bytes from a processed runtime response."""

    def supported_capabilities(self) -> frozenset[str]:
        """Return capability labels supported by this adapter."""

    def transport_boundary(self) -> str:
        """Return the transport boundary kind, such as ``http-proxy``."""


__all__ = ["AdapterProtocol"]
