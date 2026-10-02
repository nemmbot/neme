"""Tavily and Exa primary web search with Firecrawl fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any

import aiohttp

from metaculus_bot.constants import (
    EXA_API_KEY_ENV,
    FIRECRAWL_API_KEY_ENV,
    NIMBLE_API_KEY_ENV,
    RESEARCH_PROVIDER_ENV,
    TAVILY_API_KEY_ENV,
    WEB_SEARCH_API_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_EXA_SEARCH_URL = "https://api.exa.ai/search"
_FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"
_NIMBLE_SEARCH_URL = "https://sdk.nimbleway.com/v2/search"


async def _post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_API_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(url, headers=headers, json=payload) as response,
    ):
        response.raise_for_status()
        body = await response.json()
    if not isinstance(body, dict):
        raise ValueError("Search API returned a non-object JSON response")
    return body


def _render_results(results: object, *, provider: str) -> str:
    if not isinstance(results, list):
        return ""

    sections: list[str] = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        title = str(result.get("title") or "Untitled result").strip()
        url = str(result.get("url") or "").strip()
        content_value = (
            result.get("content")
            or result.get("description")
            or result.get("markdown")
            or result.get("text")
            or result.get("highlights")
            or ""
        )
        if isinstance(content_value, list):
            content_value = "\n".join(str(item) for item in content_value if item)
        content = str(content_value).strip()
        lines = [f"### {title}"]
        if url:
            lines.append(f"URL: {url}")
        published = str(result.get("published_date") or result.get("publishedDate") or "").strip()
        if published:
            lines.append(f"Published: {published}")
        if content:
            lines.append(content)
        if url or content:
            sections.append("\n".join(lines))
    if not sections:
        return ""
    return f"Search source: {provider}\n\n" + "\n\n".join(sections)


async def _search_tavily(query: str, *, end_date: str | None, topic: str) -> str:
    api_key = os.getenv(TAVILY_API_KEY_ENV)
    if not api_key:
        return ""
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 8,
        "include_answer": False,
        "include_raw_content": False,
        "topic": topic,
    }
    if end_date:
        payload["end_date"] = end_date
    response = await _post_json(_TAVILY_SEARCH_URL, headers={}, payload=payload)
    return _render_results(response.get("results"), provider="Tavily")


async def _search_exa(query: str, *, end_date: str | None, topic: str) -> str:
    api_key = os.getenv(EXA_API_KEY_ENV)
    if not api_key:
        return ""
    payload: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "numResults": 8,
        "contents": {"highlights": {"numSentences": 3}, "text": {"maxCharacters": 2000}},
    }
    if end_date:
        payload["endPublishedDate"] = end_date
    if topic == "news":
        payload["category"] = "news"
    response = await _post_json(_EXA_SEARCH_URL, headers={"x-api-key": api_key}, payload=payload)
    return _render_results(response.get("results"), provider="Exa")


async def _search_firecrawl(query: str, *, end_date: str | None, topic: str) -> str:
    _ = topic
    if end_date is not None:
        return ""
    api_key = os.getenv(FIRECRAWL_API_KEY_ENV)
    if not api_key:
        return ""
    response = await _post_json(
        _FIRECRAWL_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={"query": query, "limit": 8, "sources": ["web"], "scrapeOptions": {"formats": ["markdown"]}},
    )
    return _render_results(response.get("data"), provider="Firecrawl")


async def _search_nimble(query: str, *, end_date: str | None, topic: str) -> str:
    api_key = os.getenv(NIMBLE_API_KEY_ENV)
    if not api_key:
        return ""
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": "lite",
        "max_results": 8,
        "focus": topic,
    }
    if end_date:
        payload["end_date"] = end_date
    response = await _post_json(
        _NIMBLE_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
    )
    return _render_results(response.get("results"), provider="Nimbleway")


async def _search_nimble_preferred(query: str, *, end_date: str | None, topic: str) -> tuple[str, str]:
    last_error: Exception | None = None
    candidates = (
        ("nimble", NIMBLE_API_KEY_ENV, _search_nimble),
        ("tavily", TAVILY_API_KEY_ENV, _search_tavily),
    )
    for provider, key_env, search in candidates:
        if not os.getenv(key_env):
            continue
        try:
            result = await asyncio.wait_for(
                search(query, end_date=end_date, topic=topic), timeout=WEB_SEARCH_API_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # explicit legacy failover boundary
            last_error = exc
            logger.warning("%s search failed (%s); trying the next configured provider", provider, type(exc).__name__)
            continue
        if result:
            return provider, result
    if last_error is not None:
        raise RuntimeError("All configured Nimbleway search providers failed") from last_error
    return "none", ""


async def _search_primary_sources(
    query: str, *, end_date: str | None, topic: str
) -> tuple[str, str, Exception | None]:
    primaries: list[tuple[str, Any]] = []
    if os.getenv(TAVILY_API_KEY_ENV):
        primaries.append(("tavily", _search_tavily))
    if os.getenv(EXA_API_KEY_ENV):
        primaries.append(("exa", _search_exa))
    if not primaries:
        return "", "", None

    outcomes = await asyncio.gather(
        *(
            asyncio.wait_for(search(query, end_date=end_date, topic=topic), timeout=WEB_SEARCH_API_TIMEOUT_S)
            for _, search in primaries
        ),
        return_exceptions=True,
    )
    sections: list[str] = []
    successful_sources: list[str] = []
    last_error: Exception | None = None
    for (provider, _), outcome in zip(primaries, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, Exception):
                last_error = outcome
            logger.warning("%s search failed (%s)", provider, type(outcome).__name__)
        elif outcome:
            sections.append(outcome)
            successful_sources.append(provider)
    return "\n\n---\n\n".join(sections), "+".join(successful_sources), last_error


async def search_web_fallback(
    query: str,
    *,
    end_date: str | None = None,
    topic: str = "general",
    preferred: str | None = None,
) -> tuple[str, str]:
    """Aggregate Tavily and Exa results, using Firecrawl if neither primary succeeds.

    Explicit Nimble preference remains supported for older deployments. When only one
    primary key is configured, that provider runs alone; Firecrawl is considered only
    if the primary set has no usable results. Firecrawl is skipped for date-bounded queries.
    """
    if preferred is None:
        preferred = os.getenv(RESEARCH_PROVIDER_ENV, "tavily").strip().lower()

    if preferred == "nimble":
        return await _search_nimble_preferred(query, end_date=end_date, topic=topic)

    research, provider_names, last_error = await _search_primary_sources(query, end_date=end_date, topic=topic)
    if research:
        return provider_names, research

    if os.getenv(FIRECRAWL_API_KEY_ENV) and end_date is None:
        try:
            result = await asyncio.wait_for(
                _search_firecrawl(query, end_date=end_date, topic=topic), timeout=WEB_SEARCH_API_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # provider failover boundary
            last_error = exc
            logger.warning("firecrawl search failed (%s)", type(exc).__name__)
        else:
            if result:
                return "firecrawl", result

    if last_error is not None:
        raise RuntimeError("All configured web search providers failed") from last_error
    return "none", ""
