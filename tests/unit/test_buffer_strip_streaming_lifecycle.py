import pytest

from tok.gateway import BridgeSession
from tok.gateway._bridge_streaming import buffer_strip_restream_impl


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
async def test_buffer_strip_restream_impl_closes_response_then_client_when_owned() -> None:
    order: list[str] = []
    session = BridgeSession()
    # SSE formatted content for minimal parsing
    sse_content = b'event: message_start\ndata: {"type":"message_start","message":{"model":"test","usage":{}}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
    response = _StreamingResponse([sse_content], order)
    client = _ClosableClient(order)

    collected: list[bytes] = []
    async for chunk in buffer_strip_restream_impl(
        session=session,
        response=response,
        client=client,
        client_owned=True,
    ):
        collected.append(chunk)

    assert response.close_calls == 1
    assert client.close_calls == 1
    assert order == ["response", "client"]
