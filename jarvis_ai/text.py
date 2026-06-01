from __future__ import annotations

import re
from html.parser import HTMLParser


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|=>|::|[^\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_text(text: str, *, size: int = 900, overlap: int = 120) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


def html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return "\n".join(parser.parts)
