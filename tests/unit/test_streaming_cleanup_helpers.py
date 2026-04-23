import pytest

from tok.gateway._app_factory import (
    _aclose_if_possible,
    _close_streaming_setup_resources,
)


class _Closable:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_aclose_if_possible_ignores_none() -> None:
    await _aclose_if_possible(None)


@pytest.mark.asyncio
async def test_aclose_if_possible_ignores_nonclosable_object() -> None:
    await _aclose_if_possible(object())


@pytest.mark.asyncio
async def test_close_streaming_setup_resources_closes_response_then_client() -> None:
    order: list[str] = []

    class _OrderedClosable:
        def __init__(self, label: str) -> None:
            self.label = label
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            order.append(self.label)

    response = _OrderedClosable("response")
    client = _OrderedClosable("client")

    await _close_streaming_setup_resources(response, client)

    assert response.close_calls == 1
    assert client.close_calls == 1
    assert order == ["response", "client"]
