"""Exact-once cleanup and ownership proof tests for streaming implementations."""

import pytest

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import (
    buffer_strip_restream_impl,
    passthrough_stream_impl,
)


class _CloseSpyClient:
    """Spy client that records close calls and order."""

    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self._order.append("client")


class _CloseSpyResponse:
    """Spy response that records close calls and order with optional failure."""

    def __init__(
        self,
        chunks: list[bytes],
        order: list[str],
        *,
        fail_after_chunks: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._order = order
        self._fail_after_chunks = fail_after_chunks
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self._order.append("response")

    async def aiter_bytes(self):
        """Yield chunks with optional synthetic failure."""
        emitted = 0
        for chunk in self._chunks:
            if self._fail_after_chunks is not None and emitted >= self._fail_after_chunks:
                raise RuntimeError("synthetic stream failure")
            emitted += 1
            yield chunk


def _make_session() -> BridgeSession:
    """Create a real BridgeSession for testing."""
    return BridgeSession()


@pytest.mark.asyncio
async def test_passthrough_cleanup_occurs_exactly_once_on_normal_completion() -> None:
    """Response and client should close exactly once, in order, on successful completion."""
    order: list[str] = []
    response = _CloseSpyResponse([b"a", b"b"], order)
    client = _CloseSpyClient(order)
    session = _make_session()

    collected: list[bytes] = []
    async for chunk in passthrough_stream_impl(
        session=session,
        client=client,  # type: ignore[arg-type]
        response=response,  # type: ignore[arg-type]
        client_owned=True,
    ):
        collected.append(chunk)

    assert collected == [b"a", b"b"]
    assert response.close_calls == 1, "response closed more than once on normal completion"
    assert client.close_calls == 1, "client closed more than once on normal completion"
    assert order == ["response", "client"]


@pytest.mark.asyncio
async def test_passthrough_cleanup_occurs_exactly_once_on_early_consumer_close() -> None:
    """Response and client should close exactly once, in order, on early consumer close."""
    order: list[str] = []
    response = _CloseSpyResponse([b"a", b"b"], order)
    client = _CloseSpyClient(order)
    session = _make_session()

    stream = passthrough_stream_impl(
        session=session,
        client=client,  # type: ignore[arg-type]
        response=response,  # type: ignore[arg-type]
        client_owned=True,
    )

    first = await anext(stream)
    assert first == b"a"

    await stream.aclose()

    assert response.close_calls == 1, "response closed more than once on early close"
    assert client.close_calls == 1, "client closed more than once on early close"
    assert order == ["response", "client"]


@pytest.mark.asyncio
async def test_passthrough_cleanup_occurs_exactly_once_on_upstream_error() -> None:
    """Response and client should close exactly once, in order, on upstream stream error."""
    order: list[str] = []
    response = _CloseSpyResponse([b"a", b"b"], order, fail_after_chunks=1)
    client = _CloseSpyClient(order)
    session = _make_session()

    with pytest.raises(RuntimeError, match="synthetic stream failure"):
        async for _ in passthrough_stream_impl(
            session=session,
            client=client,  # type: ignore[arg-type]
            response=response,  # type: ignore[arg-type]
            client_owned=True,
        ):
            pass

    assert response.close_calls == 1, "response closed more than once on upstream error"
    assert client.close_calls == 1, "client closed more than once on upstream error"
    assert order == ["response", "client"]


@pytest.mark.asyncio
async def test_passthrough_does_not_close_unowned_client() -> None:
    """Unowned client should not be closed - only response should close."""
    order: list[str] = []
    response = _CloseSpyResponse([b"a"], order)
    client = _CloseSpyClient(order)
    session = _make_session()

    collected: list[bytes] = []
    async for chunk in passthrough_stream_impl(
        session=session,
        client=client,  # type: ignore[arg-type]
        response=response,  # type: ignore[arg-type]
        client_owned=False,
    ):
        collected.append(chunk)

    assert collected == [b"a"]
    assert response.close_calls == 1
    assert client.close_calls == 0, "unowned client was closed unexpectedly"
    assert order == ["response"]


@pytest.mark.asyncio
async def test_buffer_strip_cleanup_occurs_exactly_once_on_normal_completion() -> None:
    """Buffer-strip: response and client should close exactly once, in order."""
    order: list[str] = []
    response = _CloseSpyResponse([b'event: message_start\ndata: {"type": "message_start"}\n\n'], order)
    client = _CloseSpyClient(order)
    session = _make_session()

    collected: list[bytes] = []
    async for chunk in buffer_strip_restream_impl(
        session=session,
        client=client,  # type: ignore[arg-type]
        response=response,  # type: ignore[arg-type]
        client_owned=True,
    ):
        collected.append(chunk)

    assert response.close_calls == 1, "response closed more than once on normal completion"
    assert client.close_calls == 1, "client closed more than once on normal completion"
    assert order == ["response", "client"]
