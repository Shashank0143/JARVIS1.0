"""Intent routing for Jarvis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intent:
    name: str
    confidence: float


class IntentRouter:
    machine_prefixes = (
        "open ",
        "create file ",
        "create folder ",
        "write ",
        "append ",
        "update line ",
        "replace ",
        "delete ",
        "confirm delete ",
        "erase ",
        "confirm erase ",
        "execute ",
        "confirm execute ",
    )

    def route(self, text: str) -> Intent:
        lowered = text.lower().strip()
        
        # Priority custom routes for newly added agent domains
        if lowered.startswith("run agent ") or lowered.startswith("agent "):
            return Intent("agent", 0.98)
        if lowered.startswith("automl ") or "automl" in lowered:
            return Intent("automl", 0.98)
        if lowered.startswith(("security audit ", "security-audit ", "audit ")):
            return Intent("security", 0.98)
        if lowered.startswith(("run tests ", "test-loop ", "self-heal ")):
            return Intent("testing", 0.98)
        if lowered.startswith(("self-improve ", "auto-knowledge ", "evaluate ")):
            return Intent("self_improve", 0.98)

        if lowered.startswith(("write ", "create ", "generate ", "make ", "build ")):
            if self._looks_like_coding_request(lowered):
                return Intent("coding", 0.9)

        if lowered.startswith(self.machine_prefixes):
            return Intent("machine", 0.98)
        if lowered in {"camera", "look", "see", "what do you see"}:
            return Intent("vision", 0.98)
        if lowered.startswith("learn "):
            return Intent("learn", 0.98)
        if lowered.startswith(("subject ", "chapter ", "note ", "bootstrap subjects")):
            return Intent("education", 0.95)
        if lowered.startswith(("train neural", "train tensorflow", "deep learning", "neural status", "tensorflow", "pytorch")):
            return Intent("deep_learning", 0.9)
        if self._is_conversation(lowered):
            return Intent("conversation", 0.95)
        if any(word in lowered for word in {"formula", "calculate", "calculation", "algebra", "geometry", "trigonometry", "calculus", "statistics"}):
            return Intent("education", 0.82)
        if any(
            word in lowered
            for word in {
                "api",
                "backend",
                "class",
                "code",
                "css",
                "database",
                "error",
                "express",
                "fastapi",
                "frontend",
                "function",
                "html",
                "javascript",
                "node",
                "program",
                "python",
                "react",
                "sql",
                "typescript",
                "website",
                "algorithm",
            }
        ):
            return Intent("coding", 0.75)
        return Intent("knowledge", 0.5)

    @staticmethod
    def _is_conversation(text: str) -> bool:
        normalized = text.replace("?", "").replace(".", "").strip().lower()
        phrases = {
            "hi",
            "hello",
            "hey",
            "how are you",
            "who are you",
            "what can you do",
            "how can you help me",
            "how can you help",
            "what can you help with",
            "thanks",
            "thank you",
            "what is your name",
            "whats your name",
            "introduce yourself",
            "greetings",
            "who made you",
            "what are you",
        }
        if normalized in phrases:
            return True
        if any(
            phrase in normalized
            for phrase in {
                "how can you help",
                "what can you do",
                "what can you create",
            }
        ):
            return True
        words = normalized.split()
        if len(words) <= 4 and any(word in words for word in {"you", "your", "jarvis"}):
            return True
        return False

    @staticmethod
    def _looks_like_coding_request(text: str) -> bool:
        if any(marker in text for marker in {" with ", " as content ", " content "}):
            return False
        return any(
            word in text
            for word in {
                "api",
                "backend",
                "code",
                "database",
                "fastapi",
                "frontend",
                "html",
                "javascript",
                "program",
                "python",
                "react",
                "sql",
                "website",
            }
        )
