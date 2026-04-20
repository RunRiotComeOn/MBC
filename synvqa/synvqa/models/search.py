"""Web search + URL fetch.

Search backends: tavily, serpapi, bing. Pick via stage2.search_backend.
URL fetch uses requests + BeautifulSoup and respects a short timeout.
robots.txt compliance is deferred to an external middleware (e.g. a
politeness-aware fetcher); here we check disallow on a best-effort basis.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from ..utils.io import load_env_file


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    rank: int
    raw_content: str = ""


class SearchClient:
    def __init__(self, backend: str = "tavily", api_key: str | None = None,
                 results_per_query: int = 10):
        load_env_file()
        self.backend = backend
        self.api_key = api_key or os.environ.get(f"{backend.upper()}_API_KEY")
        self.results_per_query = results_per_query
        self._client = None
        self.last_answer: str = ""
        self.last_response: dict[str, Any] = {}

    def search(self, query: str) -> list[SearchHit]:
        if self.backend == "tavily":
            return self._search_tavily(query)
        if self.backend == "serpapi":
            return self._search_serpapi(query)
        if self.backend == "bing":
            return self._search_bing(query)
        raise ValueError(f"Unknown search backend: {self.backend}")

    def _search_tavily(self, query: str) -> list[SearchHit]:
        import requests  # type: ignore

        if not self.api_key:
            raise RuntimeError("missing Tavily API key")
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": self.results_per_query,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": True,
                "include_images": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        resp = resp.json()
        self.last_response = resp
        self.last_answer = resp.get("answer", "") or ""
        return [
            SearchHit(url=r["url"], title=r.get("title", ""),
                      snippet=r.get("content", ""),
                      rank=i,
                      raw_content=r.get("raw_content", "") or "")
            for i, r in enumerate(resp.get("results", []))
        ]

    def _search_serpapi(self, query: str) -> list[SearchHit]:
        try:
            from serpapi import GoogleSearch  # type: ignore
        except ImportError as e:
            raise RuntimeError("google-search-results not installed") from e
        params = {"q": query, "api_key": self.api_key, "num": self.results_per_query}
        results = GoogleSearch(params).get_dict()
        hits = []
        for i, r in enumerate(results.get("organic_results", [])):
            hits.append(SearchHit(url=r["link"], title=r.get("title", ""),
                                  snippet=r.get("snippet", ""), rank=i))
        return hits

    def _search_bing(self, query: str) -> list[SearchHit]:
        import requests  # type: ignore
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        r = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": self.results_per_query},
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        hits = []
        for i, v in enumerate(data.get("webPages", {}).get("value", [])):
            hits.append(SearchHit(url=v["url"], title=v.get("name", ""),
                                  snippet=v.get("snippet", ""), rank=i))
        return hits


# ---------- URL fetching ----------

_ROBOTS_CACHE: dict[str, RobotFileParser] = {}


def _robots_allows(url: str, user_agent: str = "synvqa-bot") -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _ROBOTS_CACHE.get(base)
    if rp is None:
        rp = RobotFileParser()
        rp.set_url(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            # if robots.txt unreachable, default-allow (best effort)
            pass
        _ROBOTS_CACHE[base] = rp
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def fetch_url(url: str, *, timeout_s: int = 20, max_chars: int = 40000,
              respect_robots: bool = True) -> dict[str, Any]:
    """Fetch and extract readable text from a URL.

    Returns {url, ok, text, status, content_type, fetched_at, error}.
    """
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "text": "",
        "status": None,
        "content_type": None,
        "fetched_at": time.time(),
        "error": None,
    }
    if respect_robots and not _robots_allows(url):
        result["error"] = "blocked_by_robots"
        return result
    try:
        import requests  # type: ignore
    except ImportError as e:
        result["error"] = f"missing_dep:{e}"
        return result

    try:
        r = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "synvqa-bot/0.1 (+research)"},
        )
        result["status"] = r.status_code
        result["content_type"] = r.headers.get("Content-Type", "")
        r.raise_for_status()
        if "html" in result["content_type"].lower():
            text = _strip_html(r.text)
        else:
            text = r.text
        result["text"] = text[:max_chars]
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


def _strip_html(html: str) -> str:
    import html as _html
    import re

    html = re.sub(r"(?is)<(script|style|noscript|nav|footer|header|aside)[^>]*>.*?</\\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = _html.unescape(html)
    html = re.sub(r"\s+", " ", html).strip()
    return html
