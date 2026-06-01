from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from .text import html_to_text


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebResearcher:
    def __init__(self, *, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "JarvisLocalCodingAssistant/1.0 (+local-rag-research)"}
        )

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        html = response.text
        results: list[SearchResult] = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?>(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            result_url = self._clean_duck_url(unescape(match.group("url")))
            title = html_to_text(unescape(match.group("title")))
            snippet = html_to_text(unescape(match.group("snippet")))
            if result_url.startswith("http"):
                results.append(SearchResult(title=title, url=result_url, snippet=snippet))
            if len(results) >= limit:
                break
        return results

    def fetch_text(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type and not response.text:
            return ""
        return html_to_text(response.text)

    @staticmethod
    def _clean_duck_url(url: str) -> str:
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            uddg = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(uddg)
        return url
