import asyncio

from geosgan_hive.llm import FakeLLM
from geosgan_hive.messages import Message


def test_fake_llm_acomplete_with_sync_handler():
    llm = FakeLLM(handler=lambda messages: f"echo:{len(messages)}")
    out = asyncio.run(llm.acomplete([Message("user", "hi")], run_id="r1"))
    assert out == "echo:1"
    assert len(llm.acomplete_calls) == 1
    assert llm.calls == []  # async path did not touch the sync log
    assert llm.run_ids == ["r1"]


def test_fake_llm_acomplete_supports_async_handlers():
    async def handler(messages):
        await asyncio.sleep(0)
        return "async answer"

    llm = FakeLLM(handler=handler)
    out = asyncio.run(llm.acomplete([Message("user", "hi")]))
    assert out == "async answer"


def test_fake_llm_acomplete_consumes_response_queue():
    llm = FakeLLM(responses=["first", "second"])
    assert asyncio.run(llm.acomplete([Message("user", "a")])) == "first"
    assert asyncio.run(llm.acomplete([Message("user", "b")])) == "second"


def test_fake_llm_implements_async_protocol():
    llm = FakeLLM(responses=["x"])
    assert hasattr(llm, "acomplete")  # structural AsyncLLMClient conformance
    out = asyncio.run(llm.acomplete([Message("user", "hi")]))
    assert out == "x"
