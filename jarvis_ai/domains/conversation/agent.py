from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...memory import MemoryStore
from ...rag import RagStore


@dataclass
class ConversationResult:
    answer: str
    confidence: float


class ConversationAgent:
    def __init__(self, memory: MemoryStore, rag: RagStore, model: Any = None) -> None:
        self.memory = memory
        self.rag = rag
        self.model = model
        self.responses = {
            "hello": "Hello! I am Jarvis, your local AI assistant. How can I help you today?",
            "hi": "Hi there! I am ready to help. Tell me what you need.",
            "hey": "Hey! I am listening. What can I do for you?",
            "how are you": "I am doing great, running locally and fully charged to assist you!",
            "how are you?": "I am doing great, running locally and fully charged to assist you!",
            "who are you": "I am Jarvis, your intelligent local assistant, capable of coding, file management, web research, self-improving, and running advanced multi-agent pipelines.",
            "what can you do": "I can assist with coding, perform math calculations, manage files and execute commands, search and learn from the web, run cooperative agent tasks, execute AutoML, and perform security audits.",
            "how can you help me": "I can help you write basic code, create frontend/backend/database starter projects, explain errors, run math, manage local files, learn from documents, and build small project templates. Try: `write basic python code`, `write database sql schema`, or `create backend fastapi project in python called my_api`.",
            "how can you help": "I can help you write basic code, create frontend/backend/database starter projects, explain errors, run math, manage local files, learn from documents, and build small project templates. Try: `write basic python code`, `write database sql schema`, or `create backend fastapi project in python called my_api`.",
            "what can you help with": "I can help you write basic code, create frontend/backend/database starter projects, explain errors, run math, manage local files, learn from documents, and build small project templates. Try: `write basic python code`, `write database sql schema`, or `create backend fastapi project in python called my_api`.",
            "thanks": "You are very welcome! Let me know if you need anything else.",
            "thank you": "You are very welcome! Let me know if you need anything else.",
            "what is your name": "I am Jarvis, your personal local assistant.",
            "whats your name": "I am Jarvis, your personal local assistant.",
            "what is your name?": "I am Jarvis, your personal local assistant.",
            "whats your name?": "I am Jarvis, your personal local assistant.",
            "introduce yourself": "Hello! I am Jarvis, a state-of-the-art local assistant. I combine RAG, neural models, multi-agent frameworks, machine execution controllers, and self-improving reinforcement learning to give you a powerful local workspace agent.",
            "greetings": "Greetings! I am Jarvis. How can I assist you today?",
            "who made you": "I am Jarvis, customized and trained to assist you in this workspace.",
            "what are you": "I am Jarvis, an advanced local-first AI coding and intelligent workspace assistant.",
        }

    def can_handle(self, text: str) -> bool:
        normalized = self._normalize(text)
        if normalized in self.responses:
            return True
        casual = {"hello", "hi", "hey", "thanks", "thank you"}
        return normalized in casual or len(normalized.split()) <= 4 and any(
            word in normalized for word in {"you", "your", "jarvis"}
        )

    def answer(self, text: str) -> ConversationResult:
        normalized = self._normalize(text)
        if normalized in self.responses:
            return ConversationResult(self.responses[normalized], 0.95)
        if any(
            phrase in normalized
            for phrase in {"how can you help", "what can you help with", "what can you do"}
        ):
            return ConversationResult(self.responses["how can you help me"], 0.95)

        # Retrieve relevant contextual facts from memory
        hits = self.rag.search(text, limit=4)
        context_list = []
        for _, doc in hits:
            context_list.append(f"Source: {doc.title} ({doc.source})\n{doc.text}")
        context = "\n\n".join(context_list)

        # Retrieve conversation history
        chat_history = self.memory.get_chat_history(limit=8)
        history_str = ""
        for turn in chat_history:
            role = "User" if turn["role"] == "user" else "Jarvis"
            history_str += f"{role}: {turn['content']}\n"

        if self.model is not None:
            # Construct a fully conversational chat prompt for the LLM
            prompt = (
                "You are JARVIS, an elite conversational AI software engineer and system assistant.\n"
                "Engage in natural, friendly, and highly intelligent conversation with the user. Keep your tone\n"
                "helpful and professional. If context is provided, ground your answer in it, but synthesize naturally.\n\n"
            )
            if context:
                prompt += f"CONTEXT FROM LOCAL MEMORY:\n{context}\n\n"
            if history_str:
                prompt += f"CONVERSATION HISTORY:\n{history_str}\n"
            
            prompt += f"User: {text}\nJarvis:"
            
            generated = self.model.complete(prompt, max_tokens=300, temperature=0.7).strip()
            if generated.startswith("Jarvis:"):
                generated = generated[len("Jarvis:"):].strip()
            
            if self._looks_useful(generated):
                return ConversationResult(generated, 0.9)

        # Static fallback if model completes poorly or is not set
        if hits and hits[0][0] > 0.35:
            _, doc = hits[0]
            return ConversationResult(
                f"I remember this from my local memory: {doc.text[:300].strip()}",
                0.65,
            )

        return ConversationResult("I am here and fully ready. What can I help you build or explain today?", 0.5)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().replace("?", "").replace(".", "").strip().split())

    @staticmethod
    def _looks_useful(text: str) -> bool:
        words = text.split()
        if len(words) < 8:
            return False
        if len(words) > 25 and not any(mark in text for mark in {".", ":", ";", "\n"}):
            return False
        unique_ratio = len(set(word.lower() for word in words)) / max(len(words), 1)
        if unique_ratio < 0.55:
            return False
        punctuation_ratio = sum(1 for char in text if char in "{}[]()<>?*=|") / max(len(text), 1)
        return punctuation_ratio <= 0.14
