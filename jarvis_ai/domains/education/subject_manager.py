from __future__ import annotations

import re
from pathlib import Path

from ...assistant import LocalCodingAssistant
from ...text import tokenize


class SubjectManager:
    STOPWORDS = {"what", "is", "are", "the", "of", "a", "an", "in", "for", "to", "and", "or", "with"}

    def __init__(self, root: Path | str, assistant: LocalCodingAssistant) -> None:
        self.root = Path(root)
        self.assistant = assistant
        self.root.mkdir(parents=True, exist_ok=True)

    def create_subject(self, subject: str) -> str:
        path = self.root / self._safe_name(subject)
        path.mkdir(parents=True, exist_ok=True)
        return f"Created subject folder: {path}"

    def create_chapter(self, subject: str, chapter: str) -> str:
        path = self.root / self._safe_name(subject) / self._safe_name(chapter)
        path.mkdir(parents=True, exist_ok=True)
        return f"Created chapter folder: {path}"

    def save_note(self, subject: str, chapter: str, title: str, content: str) -> str:
        chapter_path = self.root / self._safe_name(subject) / self._safe_name(chapter)
        chapter_path.mkdir(parents=True, exist_ok=True)
        file_path = chapter_path / f"{self._safe_name(title)}.md"
        file_path.write_text(content, encoding="utf-8")
        chunks, tokens = self.assistant.ingest_text(
            content,
            source=str(file_path),
            title=f"{subject}/{chapter}/{title}",
        )
        return f"Saved note: {file_path} ({tokens} tokens, {chunks} chunks learned)."

    def answer_from_subject(self, question: str, *, subject: str | None = None) -> str | None:
        file_answer = self._answer_from_files(question, subject=subject)
        if file_answer:
            return file_answer

        query = f"{subject or ''} {question}".strip()
        hits = self.assistant.rag.search(query, limit=4)
        if not hits or hits[0][0] < 0.22:
            return None
        lines = ["I found this in local knowledge:"]
        for score, doc in hits[:3]:
            source_text = (doc.title + doc.source).lower()
            if "knowledge" not in source_text:
                continue
            if subject and subject.lower() not in source_text:
                continue
            lines.append(f"- {doc.title}: {doc.text[:420].strip()} (score={score:.2f})")
        return "\n".join(lines) if len(lines) > 1 else None

    def _answer_from_files(self, question: str, *, subject: str | None = None) -> str | None:
        question_tokens = self._important_tokens(question)
        if not question_tokens:
            return None
        roots = [self.root / self._safe_name(subject)] if subject else [self.root]
        matches: list[tuple[int, Path, str]] = []
        for root in roots:
            if not root.exists():
                continue
            for file_path in root.rglob("*.md"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                tokens = set(tokenize(file_path.stem.replace("_", " ") + " " + text))
                score = len(question_tokens.intersection(tokens))
                if score:
                    matches.append((score, file_path, text))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        best_score, best_path, best_text = matches[0]
        if best_score < 2:
            return None
        relative = best_path.relative_to(self.root)
        excerpt = self._best_excerpt(best_text, question_tokens)
        return f"From local subject knowledge `{relative}`:\n{excerpt}"

    @staticmethod
    def _best_excerpt(text: str, question_tokens: set[str]) -> str:
        sections = SubjectManager._markdown_sections(text)
        if sections:
            scored_sections: list[tuple[int, str]] = []
            for section in sections:
                section_lines = section.splitlines()
                heading_tokens = set(tokenize(section_lines[0])) if section_lines else set()
                body_tokens = set(tokenize(section))
                score = len(question_tokens.intersection(body_tokens))
                score += 2 * len(question_tokens.intersection(heading_tokens))
                if score:
                    scored_sections.append((score, section))
            if scored_sections:
                _, section = max(scored_sections, key=lambda item: item[0])
                return section[:900].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        scored: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            score = len(question_tokens.intersection(tokenize(line)))
            if score:
                scored.append((score, index))
        if not scored:
            return text[:700].strip()
        _, index = max(scored, key=lambda item: item[0])
        start = max(0, index - 2)
        end = min(len(lines), index + 5)
        return "\n".join(lines[start:end])[:900].strip()

    @staticmethod
    def _markdown_sections(text: str) -> list[str]:
        sections: list[list[str]] = []
        current: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("## ") and current:
                sections.append(current)
                current = [line]
            elif line:
                current.append(line)
        if current:
            sections.append(current)
        return ["\n".join(section) for section in sections]

    def bootstrap(self) -> str:
        subjects = {
            "mathematics": {
                "numbers": "Natural numbers, integers, rational numbers, and real numbers are number systems used for counting, measurement, and algebra.",
                "algebra": "Algebra uses variables and equations to represent unknown values and relationships.",
                "geometry": "Geometry studies shapes, angles, area, perimeter, volume, and spatial relationships.",
            },
            "computer_science": {
                "programming": "Programming means writing instructions that a computer can execute.",
                "data_structures": "Data structures organize data for efficient access and modification.",
                "machine_learning": "Machine learning trains models from data so they can make predictions or decisions.",
            },
        }
        created = 0
        for subject, chapters in subjects.items():
            for chapter, content in chapters.items():
                self.save_note(subject, chapter, "overview", content)
                created += 1
        return f"Bootstrapped {created} subject chapter notes."

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip().lower()).strip("_")
        return safe or "untitled"

    @classmethod
    def _important_tokens(cls, text: str) -> set[str]:
        return {token for token in tokenize(text) if token not in cls.STOPWORDS and token.strip()}
