from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env files
import json
import random
from collections import Counter, defaultdict

load_dotenv()  # Load environment variables from .env file
from pathlib import Path

from .text import tokenize
from .transformer_model import LocalTransformerCodeModel


class SelfTrainingCodeModel:
    def __init__(self, order: int = 3) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order = order
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocab: Counter[str] = Counter()
        self.examples_seen = 0

    def train_text(self, text: str) -> int:
        tokens = tokenize(text)
        if len(tokens) < self.order:
            return 0
        padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
        for index in range(len(padded) - self.order + 1):
            prefix = tuple(padded[index : index + self.order - 1])
            next_token = padded[index + self.order - 1]
            self.transitions[prefix][next_token] += 1
            self.vocab[next_token] += 1
        self.examples_seen += 1
        return len(tokens)

    def complete(self, prompt: str, *, max_tokens: int = 180, temperature: float = 0.25) -> str:
        if not self.transitions:
            return ""
        tokens = tokenize(prompt)
        prefix = (["<s>"] * (self.order - 1) + tokens)[-(self.order - 1) :]
        generated: list[str] = []
        for _ in range(max_tokens):
            options = self.transitions.get(tuple(prefix))
            if not options:
                prefix = list(random.choice(tuple(self.transitions.keys())))
                options = self.transitions[tuple(prefix)]
            token = self._sample(options, temperature)
            if token == "</s>":
                break
            generated.append(token)
            prefix = (prefix + [token])[-(self.order - 1) :]
        return self._detokenize(generated)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "examples_seen": self.examples_seen,
            "transitions": [
                {"prefix": list(prefix), "next": dict(counter)}
                for prefix, counter in self.transitions.items()
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, order: int = 3) -> "SelfTrainingCodeModel":
        model = cls(order=order)
        if not path.exists():
            return model
        payload = json.loads(path.read_text(encoding="utf-8"))
        model.order = int(payload.get("order", order))
        model.examples_seen = int(payload.get("examples_seen", 0))
        model.transitions = defaultdict(Counter)
        model.vocab = Counter()
        for row in payload.get("transitions", []):
            prefix = tuple(row["prefix"])
            counter = Counter(row["next"])
            model.transitions[prefix] = counter
            model.vocab.update(counter)
        return model

    @staticmethod
    def _sample(options: Counter[str], temperature: float) -> str:
        if temperature <= 0.05:
            return options.most_common(1)[0][0]
        items = list(options.items())
        weights = [count ** (1.0 / max(temperature, 0.05)) for _, count in items]
        return random.choices([token for token, _ in items], weights=weights, k=1)[0]

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        text = " ".join(tokens)
        for old, new in {
            " .": ".",
            " ,": ",",
            " :": ":",
            " ;": ";",
            " )": ")",
            "( ": "(",
            " ]": "]",
            "[ ": "[",
            " }": "}",
            "{ ": "{",
            " = = ": " == ",
            "! =": "!=",
            "< =": "<=",
            "> =": ">=",
        }.items():
            text = text.replace(old, new)
        return text


class HybridLocalCodeModel:

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.ngram_path = data_dir / "local_code_model.json"
        self.transformer = LocalTransformerCodeModel(
            model_path=data_dir / "transformer_code_model.pt",
            tokenizer_path=data_dir / "transformer_tokenizer.json",
            config_path=data_dir / "transformer_config.json",
        )
        self.ngram = SelfTrainingCodeModel.load(self.ngram_path)

    def train_text(self, text: str) -> int:
        tokens = self.ngram.train_text(text)
        self.transformer.train_text(text)
        return tokens

    def train_transformer_corpus(
        self,
        corpus_path: Path,
        *,
        epochs: int = 3,
        batch_size: int = 12,
        learning_rate: float = 3e-4,
        steps_per_epoch: int = 150,
    ) -> dict[str, float | int]:
        return self.transformer.train_corpus(
            corpus_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            steps_per_epoch=steps_per_epoch,
        )

    def complete(self, prompt: str, *, max_tokens: int = 180, temperature: float = 0.25) -> str:
        # High-quality fallback using Hugging Face Inference API with the token
        import os
        import requests
        
        token = os.environ.get("HF_TOKEN", os.getenv("HUGGING_TOKEN"))
        if token:
            try:
                # We use Qwen/Qwen2.5-Coder-7B-Instruct as it is incredibly good at coding & general chats
                api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-7B-Instruct"
                headers = {"Authorization": f"Bearer {token}"}
                
                # If the prompt is the conversational raw prompt, let's strip headers for direct chat
                clean_prompt = prompt
                if prompt.startswith("Question:"):
                    # Extract the question for better chat response if it is conversational
                    lines = prompt.split("\n\n")
                    if len(lines) >= 3:
                        question = lines[0].replace("Question:", "").strip()
                        context = lines[1].replace("Context:", "").strip()
                        # If context is not empty, include it, otherwise just ask the question
                        if context:
                            clean_prompt = f"Use the following context to answer the question:\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                        else:
                            clean_prompt = question

                payload = {
                    "inputs": clean_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False
                    }
                }
                import time
                max_retries = 3
                for attempt in range(max_retries):
                    response = requests.post(api_url, json=payload, headers=headers, timeout=15)
                    if response.status_code == 200:
                        res_json = response.json()
                        if isinstance(res_json, list) and len(res_json) > 0:
                            text = res_json[0].get("generated_text", "")
                            if text.strip():
                                return text.strip()
                        elif isinstance(res_json, dict):
                            text = res_json.get("generated_text", "")
                            if text.strip():
                                return text.strip()
                        break
                    elif response.status_code == 503:
                        # Model is loading - parse estimated time or default to 2 seconds
                        try:
                            err_data = response.json()
                            wait_sec = min(float(err_data.get("estimated_time", 2.0)), 4.0)
                        except Exception:
                            wait_sec = 2.0
                        time.sleep(wait_sec)
                    else:
                        break
            except Exception:
                pass

        if self.transformer.trained:
            try:
                local_res = self.transformer.complete(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.7),
                )
                if local_res.strip():
                    return local_res
            except Exception:
                pass
        return self.ngram.complete(prompt, max_tokens=max_tokens, temperature=temperature)

    @property
    def transformer_available(self) -> bool:
        return self.transformer.available

    @property
    def transformer_trained(self) -> bool:
        return self.transformer.trained

    def save(self, path: Path | None = None) -> None:
        self.ngram.save(path or self.ngram_path)

    @classmethod
    def load(cls, data_dir: Path) -> "HybridLocalCodeModel":
        return cls(data_dir)
