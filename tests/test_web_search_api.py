from __future__ import annotations

from typing import Any

import pytest

from metaculus_bot.research import web_search_api
from metaculus_bot.research.providers import choose_provider_with_name


@pytest.mark.parametrize("key_name", ["EXA_API_KEY", "FIRECRAWL_API_KEY"])
def test_new_search_keys_select_shared_provider(monkeypatch: pytest.MonkeyPatch, key_name: str) -> None:
    for env_name in ("TAVILY_API_KEY", "EXA_API_KEY", "FIRECRAWL_API_KEY", "NIMBLE_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("RESEARCH_PROVIDER", "auto")
    monkeypatch.setenv(key_name, "test-key")

    _, provider_name = choose_provider_with_name()

    assert provider_name == "web_search"


@pytest.mark.asyncio
async def test_tavily_request_uses_body_key_and_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((url, headers, payload))
        return {"results": [{"title": "Source", "url": "https://example.com", "content": "Evidence"}]}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("Will it happen?", end_date="2026-09-01")

    assert provider == "tavily"
    assert "Evidence" in result
    assert calls == [
        (
            "https://api.tavily.com/search",
            {},
            {
                "api_key": "test-tavily-key",
                "query": "Will it happen?",
                "search_depth": "basic",
                "max_results": 8,
                "include_answer": False,
                "include_raw_content": False,
                "topic": "general",
                "end_date": "2026-09-01",
            },
        )
    ]


@pytest.mark.asyncio
async def test_explicit_nimble_provider_uses_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_PROVIDER", "nimble")
    monkeypatch.setenv("NIMBLE_API_KEY", "test-nimble-key")

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        assert url == "https://sdk.nimbleway.com/v2/search"
        assert headers == {"Authorization": "Bearer test-nimble-key"}
        return {"results": [{"title": "Backup", "url": "https://backup.example", "description": "Found"}]}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("Will it happen?", topic="news")

    assert provider == "nimble"
    assert "Nimbleway" in result
    assert "Found" in result


@pytest.mark.asyncio
async def test_nimble_is_not_an_automatic_extra_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("NIMBLE_API_KEY", "test-nimble-key")
    providers: list[str] = []

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        providers.append(url)
        if "tavily" in url:
            return {"results": []}
        return {"results": [{"title": "Backup", "url": "https://backup.example", "content": "Found"}]}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("Will it happen?", preferred="tavily")

    assert provider == "none"
    assert result == ""
    assert providers == ["https://api.tavily.com/search"]


@pytest.mark.asyncio
async def test_nimble_preference_falls_back_to_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_PROVIDER", "nimble")
    monkeypatch.setenv("NIMBLE_API_KEY", "nimble-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    providers: list[str] = []

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        providers.append(url)
        if "nimbleway" in url:
            raise RuntimeError("temporary error")
        return {"results": [{"title": "Backup", "url": "https://backup.example", "content": "Found"}]}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("Will it happen?")

    assert provider == "tavily"
    assert "Found" in result
    assert providers == ["https://sdk.nimbleway.com/v2/search", "https://api.tavily.com/search"]


@pytest.mark.asyncio
async def test_tavily_and_exa_primary_results_are_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    calls: list[str] = []

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "exa" in url:
            assert headers == {"x-api-key": "exa-test"}
            return {"results": [{"title": "Exa source", "url": "https://exa.test", "highlights": ["Exa evidence"]}]}
        return {"results": [{"title": "Tavily source", "url": "https://tavily.test", "content": "Tavily evidence"}]}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("forecast query")

    assert provider == "tavily+exa"
    assert "Tavily evidence" in result
    assert "Exa evidence" in result
    assert set(calls) == {"https://api.tavily.com/search", "https://api.exa.ai/search"}


@pytest.mark.asyncio
async def test_firecrawl_is_used_only_when_both_primaries_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-test")
    calls: list[str] = []

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "firecrawl" in url:
            assert headers == {"Authorization": "Bearer firecrawl-test"}
            return {"data": [{"title": "Backup", "url": "https://backup.test", "markdown": "Fallback evidence"}]}
        return {"results": []}

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("forecast query")

    assert provider == "firecrawl"
    assert "Fallback evidence" in result
    assert calls.count("https://api.firecrawl.dev/v1/search") == 1


@pytest.mark.asyncio
async def test_firecrawl_is_not_used_when_either_primary_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-test")

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        if "tavily" in url:
            return {"results": [{"title": "Source", "url": "https://source.test", "content": "Evidence"}]}
        if "exa" in url:
            return {"results": []}
        pytest.fail("Firecrawl should not be called while a primary has usable results")

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("forecast query")

    assert provider == "tavily"
    assert "Evidence" in result


@pytest.mark.asyncio
async def test_firecrawl_is_skipped_for_date_bounded_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-test")

    async def fake_post(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        pytest.fail(f"Date-bounded search must not call {url}")

    monkeypatch.setattr(web_search_api, "_post_json", fake_post)

    provider, result = await web_search_api.search_web_fallback("forecast query", end_date="2026-09-01")

    assert provider == "none"
    assert result == ""
