from __future__ import annotations

import asyncio
import importlib
import json

import httpx
import pytest

from utils import search_ops


SEARCH_DATA = {
    "results": [
        {"title": "Source", "url": "https://example.org/paper", "excerpts": ["First excerpt", "Second excerpt"]},
        {"title": None, "url": "https://example.org/other", "excerpts": []},
    ],
}


@pytest.fixture()
def parallel_server(monkeypatch):
    """Exercise the real SDK transport with recorded HTTP requests."""
    import mcp.client.streamable_http as transport

    requests = []
    options = {"data": SEARCH_DATA, "text_only": False, "error": False, "status": 200, "block": False}
    entered = asyncio.Event()
    closed = []

    async def handle(request):
        requests.append(request)
        assert str(request.url) == search_ops.PARALLEL_MCP_URL
        assert request.headers["user-agent"] == search_ops.PARALLEL_USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        message = json.loads(request.content)
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert method == "tools/call"
            entered.set()
            if options["block"]:
                await asyncio.Event().wait()
            if options["status"] != 200:
                return httpx.Response(options["status"], json={"error": "unavailable"})
            result = {"content": [{"type": "text", "text": json.dumps(options["data"])}],
                      "isError": options["error"]}
            if not options["text_only"]:
                result["structuredContent"] = options["data"]
        return httpx.Response(200, headers={"mcp-session-id": "fixture-session"},
                              json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    class Client(httpx.AsyncClient):
        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            closed.append(True)

    def factory(**kwargs):
        return Client(transport=httpx.MockTransport(handle), **kwargs)

    real_transport = transport.streamablehttp_client

    def configured_transport(*args, **kwargs):
        return real_transport(*args, httpx_client_factory=factory, **kwargs)

    monkeypatch.setattr(transport, "streamablehttp_client", configured_transport)
    monkeypatch.setattr(search_ops, "SEARCH_PROVIDER", "parallel")
    return requests, options, entered, closed


@pytest.mark.asyncio
@pytest.mark.parametrize("text_only", [False, True])
async def test_parallel_selection_mapping_and_limit(parallel_server, text_only):
    requests, options, _, closed = parallel_server
    options["text_only"] = text_only
    result = await search_ops.search_web("climate research sources", num_results=1, session_id="a" * 32)
    assert result == [{"title": "Source", "url": "https://example.org/paper",
                       "snippet": "First excerpt\n\nSecond excerpt"}]
    call = next(json.loads(r.content) for r in requests if r.method == "POST"
                and json.loads(r.content)["method"] == "tools/call")
    assert call["params"] == {"name": "web_search", "arguments": {
        "objective": "climate research sources", "search_queries": ["climate research sources"],
        "session_id": "a" * 32,
    }}
    assert any(r.method == "DELETE" for r in requests)
    assert closed


@pytest.mark.asyncio
async def test_nullable_title_and_empty_results(parallel_server):
    _, options, _, _ = parallel_server
    result = await search_ops.search_web("climate research sources")
    assert result[1] == {"title": "", "url": "https://example.org/other", "snippet": ""}
    options["data"] = {"results": []}
    assert await search_ops.search_web("climate research sources") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool", "http", "malformed"])
async def test_provider_failure_returns_empty_without_fallback(parallel_server, monkeypatch, failure):
    _, options, _, _ = parallel_server
    if failure == "tool":
        options["error"] = True
    elif failure == "http":
        options["status"] = 503
    else:
        options["data"] = {"unexpected": []}

    async def forbidden(*args, **kwargs):
        pytest.fail("An explicit Parallel selection must not contact SearXNG")

    monkeypatch.setattr(search_ops, "_search_web_searxng", forbidden)
    assert await search_ops.search_web("climate research sources") == []


@pytest.mark.asyncio
async def test_total_timeout_closes_transport(parallel_server, monkeypatch):
    _, options, _, closed = parallel_server
    options["block"] = True
    monkeypatch.setattr(search_ops, "PARALLEL_TIMEOUT_SECONDS", 0.05)
    assert await asyncio.wait_for(search_ops.search_web("climate research sources"), 2) == []
    assert closed


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_transport(parallel_server):
    _, options, entered, closed = parallel_server
    options["block"] = True
    task = asyncio.create_task(search_ops.search_web("climate research sources"))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert closed


@pytest.mark.asyncio
async def test_parallel_rejects_unsupported_categories_and_nonpositive_limits(parallel_server):
    requests, _, _, _ = parallel_server
    assert await search_ops.search_web("query", categories="science") == []
    assert await search_ops.search_web("query", num_results=0) == []
    assert await search_ops.search_web("query", num_results=-1) == []
    assert not requests


@pytest.mark.asyncio
async def test_default_searxng_preserves_url_categories_and_result_contract(monkeypatch):
    monkeypatch.delenv("CITEFLOW_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("SEARXNG_URL", "https://my-searxng.example")
    importlib.reload(search_ops)
    assert search_ops.SEARCH_PROVIDER == "searxng"
    real_client = httpx.AsyncClient
    seen = []

    def handle(request):
        seen.append(request)
        assert str(request.url.copy_with(query=None)) == "https://my-searxng.example/search"
        assert dict(request.url.params) == {"q": "research", "format": "json", "categories": "science", "pageno": "1"}
        return httpx.Response(200, json={"results": [{"title": "Paper", "url": "https://example.org", "content": "Summary"}]})

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(search_ops.httpx, "AsyncClient", client)
    assert await search_ops.search_web("research", categories="science") == [
        {"title": "Paper", "url": "https://example.org", "snippet": "Summary"},
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_unknown_provider_does_not_silently_search(monkeypatch):
    monkeypatch.setattr(search_ops, "SEARCH_PROVIDER", "typo")
    assert await search_ops.search_web("query") == []


@pytest.mark.asyncio
async def test_research_tool_dispatch_consumes_results_and_reuses_session(parallel_server):
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode
    from utils.agent import web_search

    requests, _, _, _ = parallel_server
    node = ToolNode([web_search])
    config = {"configurable": {"search_session_id": "b" * 32}}
    for tool_id in ("first", "second"):
        result = await node.ainvoke({"messages": [AIMessage(content="", tool_calls=[{
            "name": "web_search", "args": {"query": "climate research sources"}, "id": tool_id,
        }])]}, config=config)
        output = json.loads(result["messages"][0].content)
        assert output[0]["url"] == "https://example.org/paper"
        assert output[0]["snippet"] == "First excerpt\n\nSecond excerpt"
    calls = [json.loads(r.content) for r in requests if r.method == "POST"
             and json.loads(r.content)["method"] == "tools/call"]
    assert len(calls) == 2
    assert {c["params"]["arguments"]["session_id"] for c in calls} == {"b" * 32}
    assert list(web_search.args) == ["query"]
