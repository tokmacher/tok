import pytest

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import passthrough_stream_impl


class _ClosableClient:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self._order.append("client")


class _StreamingResponse:
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
        emitted = 0
        for chunk in self._chunks:
            if self._fail_after_chunks is not None and emitted >= self._fail_after_chunks:
                raise RuntimeError("synthetic stream failure")
            emitted += 1
            yield chunk


@pytest.mark.asyncio
async def test_passthrough_stream_impl_closes_response_then_client_when_owned() -> None:
    order: list[str] = []
    session = BridgeSession()
    response = _StreamingResponse([b"a", b"b"], order)
    client = _ClosableClient(order)

    collected: list[bytes] = []
    async for chunk in passthrough_stream_impl(
        session=session,
        response=response,
        client=client,
        client_owned=True,
    ):
        collected.append(chunk)

    assert collected == [b"a", b"b"]
    assert response.close_calls == 1
    assert client.close_calls == 1
    assert order == ["response", "client"]


@pytest.mark.asyncio
async def test_passthrough_stream_impl_does_not_close_client_when_not_owned() -> None:
    order: list[str] = []
    session = BridgeSession()
    response = _StreamingResponse([b"a"], order)
    client = _ClosableClient(order)

    collected: list[bytes] = []
    async for chunk in passthrough_stream_impl(
        session=session,
        response=response,
        client=client,
        client_owned=False,
    ):
        collected.append(chunk)

    assert collected == [b"a"]
    assert response.close_calls == 1
    assert client.close_calls == 0
    assert order == ["response"]


@pytest.mark.asyncio
async def test_passthrough_stream_impl_closes_owned_resources_on_iteration_failure() -> None:
    order: list[str] = []
    session = BridgeSession()
    response = _StreamingResponse([b"a", b"b"], order, fail_after_chunks=1)
    client = _ClosableClient(order)

    collected: list[bytes] = []

    with pytest.raises(RuntimeError, match="synthetic stream failure"):
        async for chunk in passthrough_stream_impl(
            session=session,
            response=response,
            client=client,
            client_owned=True,
        ):
            collected.append(chunk)

    assert collected == [b"a"]
    assert response.close_calls == 1
    assert client.close_calls == 1
    assert order == ["response", "client"]


@pytest.mark.asyncio
async def test_passthrough_stream_impl_closes_owned_resources_on_early_consumer_close() -> None:
    order: list[str] = []
    session = BridgeSession()
    response = _StreamingResponse([b"a", b"b"], order)
    client = _ClosableClient(order)

    stream = passthrough_stream_impl(
        session=session,
        response=response,
        client=client,
        client_owned=True,
    )

    first = await anext(stream)
    assert first == b"a"

    await stream.aclose()

    assert response.close_calls == 1
    assert client.close_calls == 1
    assert order == ["response", "client"]
