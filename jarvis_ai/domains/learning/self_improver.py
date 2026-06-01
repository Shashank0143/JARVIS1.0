from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ...text import tokenize


class SelfImprover:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.data_dir = self.workspace / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_path = self.data_dir / "reinforcement_feedback.json"
        self.router_weights_path = self.data_dir / "router_weights.json"

        self.weights = self._load_router_weights()

    def evaluate_response(self, question: str, response: str) -> dict[str, Any]:
        """Automatically evaluates own responses (Requirement 1, self-evaluation)."""
        tokens_q = set(tokenize(question))
        tokens_r = set(tokenize(response))

        overlap = len(tokens_q.intersection(tokens_r))
        word_count = len(response.split())

        relevance = min(10.0, 3.0 + overlap * 1.5)
        safety = 10.0 if not any(w in response.lower() for w in {"hack", "malware", "illegal", "bypass"}) else 3.0
        coherence = 10.0 if word_count >= 15 and response.count("\n") >= 1 else 6.0
        overall = round((relevance + safety + coherence) / 3, 2)

        return {
            "score": overall,
            "metrics": {
                "relevance": relevance,
                "safety": safety,
                "coherence": coherence,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def record_feedback(self, command: str, routed_intent: str, reward: float) -> str:
        """Applies simulated Policy Gradient reinforcement learning to update router weights."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "intent": routed_intent,
            "reward": reward,
        }
        with self.feedback_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")

        learning_rate = 0.05
        current_weight = self.weights.get(routed_intent, 0.5)
        new_weight = max(0.1, min(1.0, current_weight + learning_rate * reward))
        self.weights[routed_intent] = round(new_weight, 4)

        self._save_router_weights()
        return f"RL Pipeline updated weight for intent '{routed_intent}': {current_weight} -> {new_weight}"

    def auto_generate_knowledge(self) -> str:
        """Requirement 1: scans conversation logs and generates structured subject files."""
        memory_path = self.data_dir / "memory.jsonl"
        if not memory_path.exists():
            return "No local conversation logs found yet. Ingest some code first!"

        learned_snippets: list[str] = []
        try:
            with memory_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("kind") == "command" and len(row.get("content", "")) > 10:
                        learned_snippets.append(row["content"])
        except Exception as exc:
            return f"Error reading logs: {exc}"

        if not learned_snippets:
            return "No high-quality technical commands or questions extracted from local logs."

        summary_content = [
            "# Auto-Generated System Knowledge Manual",
            f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Frequently Ingested Developer Workflows",
        ]
        for idx, snippet in enumerate(learned_snippets[-8:], 1):
            summary_content.append(f"- **Workflow {idx}**: {snippet}")

        knowledge_dir = self.workspace / "knowledge" / "computer_science" / "data_structures"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        overview_path = knowledge_dir / "overview.md"
        overview_path.write_text("\n".join(summary_content), encoding="utf-8")

        return f"Structured knowledge generated successfully: {overview_path.relative_to(self.workspace)}"

    def _load_router_weights(self) -> dict[str, float]:
        if not self.router_weights_path.exists():
            return {
                "machine": 0.98,
                "vision": 0.98,
                "learn": 0.98,
                "education": 0.95,
                "deep_learning": 0.90,
                "coding": 0.75,
                "conversation": 0.95,
            }
        try:
            return json.loads(self.router_weights_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_router_weights(self) -> None:
        self.router_weights_path.write_text(
            json.dumps(self.weights, indent=2), encoding="utf-8"
        )
