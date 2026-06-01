"""Persistent local memory for Jarvis interactions and conversation history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "memory.jsonl"

    def remember(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "content": content,
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(row) for row in rows if row.strip()]

    def get_chat_history(self, limit: int = 10) -> list[dict[str, str]]:
        """
        Retrieves a rolling list of recent chat turns formatted for LLM ingestion.
        Alternates between 'user' (for commands) and 'assistant' (for responses).
        """
        if not self.path.exists():
            return []
        
        # Read the file and parse all entries
        rows = self.path.read_text(encoding="utf-8").splitlines()
        parsed_entries = []
        for row in rows:
            if not row.strip():
                continue
            try:
                parsed_entries.append(json.loads(row))
            except json.JSONDecodeError:
                continue

        # Filter and construct turns in chronological order
        chat_turns: list[dict[str, str]] = []
        for entry in parsed_entries:
            kind = entry.get("kind")
            content = entry.get("content", "").strip()
            if not content:
                continue
                
            if kind == "command":
                chat_turns.append({"role": "user", "content": content})
            elif kind == "response":
                chat_turns.append({"role": "assistant", "content": content})
                
        # Return the last N turns
        return chat_turns[-limit:]
