from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from .assistant import LocalCodingAssistant
from .web import WebResearcher


TRUSTED_SOURCES = {
    "w3schools": "w3schools.com",
    "geeksforgeeks": "geeksforgeeks.org",
    "wikipedia": "wikipedia.org",
    "pytorch": "pytorch.org",
    "tensorflow": "tensorflow.org",
}

DEFAULT_TOPICS = [
    "python programming basics",
    "python file handling",
    "python error handling",
    "python object oriented programming",
    "data structures algorithms python",
    "machine learning basics",
    "pytorch tensors neural network tutorial",
    "tensorflow keras model training tutorial",
    "computer vision opencv python",
    "speech recognition python",
]


@dataclass
class LearningStats:
    searched: int = 0
    fetched: int = 0
    skipped: int = 0
    chunks: int = 0
    tokens: int = 0

    def summary(self) -> str:
        return (
            f"searched={self.searched}, fetched={self.fetched}, skipped={self.skipped}, "
            f"chunks={self.chunks}, tokens={self.tokens}"
        )


class InternetLearningEngine:
    def __init__(
        self,
        assistant: LocalCodingAssistant,
        data_dir: Path | str = "data",
        *,
        delay_seconds: float = 1.0,
    ) -> None:
        self.assistant = assistant
        self.data_dir = Path(data_dir)
        self.web = WebResearcher(timeout=16)
        self.delay_seconds = delay_seconds
        self.sources_path = self.data_dir / "learned_sources.json"
        self.learned_sources = self._load_sources()

    def learn_topic(
        self,
        topic: str,
        *,
        source_keys: list[str] | None = None,
        results_per_source: int = 2,
        max_pages: int = 10,
    ) -> LearningStats:
        stats = LearningStats()
        keys = source_keys or list(TRUSTED_SOURCES)
        for key in keys:
            domain = TRUSTED_SOURCES.get(key.lower(), key.lower())
            query = f"site:{domain} {topic}"
            results = self.web.search(query, limit=results_per_source)
            stats.searched += len(results)
            for result in results:
                if stats.fetched >= max_pages:
                    self._save_sources()
                    return stats
                if not self._allowed(result.url, keys):
                    stats.skipped += 1
                    continue
                if result.url in self.learned_sources:
                    stats.skipped += 1
                    continue
                try:
                    page_text = self.web.fetch_text(result.url)
                except requests.RequestException:
                    stats.skipped += 1
                    continue
                if len(page_text.split()) < 80:
                    stats.skipped += 1
                    continue
                text = f"{result.title}\n{result.snippet}\n\n{page_text[:12000]}"
                chunks, tokens = self.assistant.ingest_text(
                    text,
                    source=result.url,
                    title=f"{key}: {result.title}",
                )
                self.learned_sources[result.url] = {
                    "title": result.title,
                    "topic": topic,
                    "domain": domain,
                    "chunks": chunks,
                    "tokens": tokens,
                    "time": int(time.time()),
                }
                stats.fetched += 1
                stats.chunks += chunks
                stats.tokens += tokens
                self._save_sources()
                time.sleep(self.delay_seconds)
        return stats

    def learn_default_curriculum(self, *, pages_per_topic: int = 5) -> LearningStats:
        total = LearningStats()
        for topic in DEFAULT_TOPICS:
            stats = self.learn_topic(topic, max_pages=pages_per_topic)
            total.searched += stats.searched
            total.fetched += stats.fetched
            total.skipped += stats.skipped
            total.chunks += stats.chunks
            total.tokens += stats.tokens
        return total

    def learn_pytorch_tutorials(
        self,
        *,
        index_url: str = "https://docs.pytorch.org/tutorials/index.html",
        max_pages: int = 8,
        dataset_dir: Path | str = "datasets/training_corpus/pytorch_tutorials",
    ) -> LearningStats:
        return self.learn_url_collection(
            index_url,
            allowed_domain="docs.pytorch.org",
            max_pages=max_pages,
            dataset_dir=Path(dataset_dir),
            title_prefix="PyTorch tutorial",
        )

    def learn_url_collection(
        self,
        index_url: str,
        *,
        allowed_domain: str,
        max_pages: int,
        dataset_dir: Path,
        title_prefix: str,
    ) -> LearningStats:
        stats = LearningStats()
        urls = [index_url]
        try:
            index_html = self.web.session.get(index_url, timeout=self.web.timeout).text
        except requests.RequestException:
            return stats
        for href in re.findall(r'href=["\']([^"\']+)["\']', index_html, flags=re.IGNORECASE):
            absolute = urljoin(index_url, href.split("#", 1)[0])
            parsed = urlparse(absolute)
            if self._skip_doc_url(parsed.path):
                continue
            if parsed.netloc == allowed_domain and parsed.path.endswith((".html", "/")):
                if absolute not in urls:
                    urls.append(absolute)
            if len(urls) >= max_pages:
                break

        dataset_dir.mkdir(parents=True, exist_ok=True)
        for url in urls[:max_pages]:
            stats.searched += 1
            if url in self.learned_sources:
                stats.skipped += 1
                continue
            parsed_url = urlparse(url)
            if parsed_url.netloc != allowed_domain or self._skip_doc_url(parsed_url.path):
                stats.skipped += 1
                continue
            try:
                text = self.web.fetch_text(url)
            except requests.RequestException:
                stats.skipped += 1
                continue
            if len(text.split()) < 80:
                stats.skipped += 1
                continue
            title = f"{title_prefix}: {url}"
            file_path = dataset_dir / f"{self._safe_filename(url)}.md"
            file_path.write_text(f"# {title}\n\nSource: {url}\n\n{text[:20000]}", encoding="utf-8")
            chunks, tokens = self.assistant.ingest_text(text[:20000], source=url, title=title)
            self.learned_sources[url] = {
                "title": title,
                "topic": title_prefix,
                "domain": allowed_domain,
                "dataset_file": str(file_path),
                "chunks": chunks,
                "tokens": tokens,
                "time": int(time.time()),
            }
            stats.fetched += 1
            stats.chunks += chunks
            stats.tokens += tokens
            self._save_sources()
            time.sleep(self.delay_seconds)
        return stats

    def _allowed(self, url: str, source_keys: list[str]) -> bool:
        host = urlparse(url).netloc.lower()
        allowed_domains = [TRUSTED_SOURCES.get(key.lower(), key.lower()) for key in source_keys]
        return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)

    def _load_sources(self) -> dict[str, dict]:
        if not self.sources_path.exists():
            return {}
        return json.loads(self.sources_path.read_text(encoding="utf-8"))

    def _save_sources(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sources_path.write_text(
            json.dumps(self.learned_sources, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_filename(url: str) -> str:
        parsed = urlparse(url)
        value = (parsed.netloc + parsed.path).strip("/").replace("/", "_")
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:140] or "page"

    @staticmethod
    def _skip_doc_url(path: str) -> bool:
        lower = path.lower()
        skip_names = ("genindex", "search", "_sources", "_static")
        return any(name in lower for name in skip_names)
