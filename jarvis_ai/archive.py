from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class AnswerArchive:
    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "answers.jsonl"

    def save_answer(
        self,
        question: str,
        answer: str,
        *,
        source: str = "jarvis",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(row) for row in rows if row.strip()]
