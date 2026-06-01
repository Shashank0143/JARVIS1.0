from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .text import tokenize

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


@dataclass
class TransformerConfig:
    block_size: int = 128
    embedding_size: int = 192
    heads: int = 6
    layers: int = 4
    dropout: float = 0.1
    max_vocab: int = 12000


class CodeTokenizer:
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id = token_to_id or {token: index for index, token in enumerate(self.specials)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def fit(self, texts: list[str], *, max_vocab: int) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        kept = [token for token, _ in ranked[: max(0, max_vocab - len(self.specials))]]
        self.token_to_id = {token: index for index, token in enumerate(self.specials + kept)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def encode(self, text: str, *, add_bounds: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]
        if add_bounds:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.id_to_token.get(index, "<unk>")
            for index in ids
            if index not in {self.pad_id, self.bos_id, self.eos_id}
        ]
        text = " ".join(tokens)
        replacements = {
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
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def to_json(self) -> dict[str, Any]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CodeTokenizer":
        return cls(token_to_id={str(key): int(value) for key, value in payload["token_to_id"].items()})

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.token_to_id)


if torch is not None:

    class TinyTransformerLM(nn.Module):
        def __init__(self, vocab_size: int, config: TransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(vocab_size, config.embedding_size)
            self.position_embedding = nn.Embedding(config.block_size, config.embedding_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_size,
                nhead=config.heads,
                dim_feedforward=config.embedding_size * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.norm = nn.LayerNorm(config.embedding_size)
            self.head = nn.Linear(config.embedding_size, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch, length = input_ids.shape
            positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            mask = torch.triu(
                torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            hidden = self.blocks(hidden, mask=mask)
            return self.head(self.norm(hidden))

else:
    TinyTransformerLM = None


class LocalTransformerCodeModel:
    def __init__(self, model_path: Path, tokenizer_path: Path, config_path: Path) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = None
        if torch is not None and self.model_path.exists() and self.tokenizer is not None:
            try:
                self._load_model()
            except Exception as exc:
                print(f"Warning: Could not load trained weights from {self.model_path} ({exc}). Operating in fallback mode.")

    @property
    def available(self) -> bool:
        return torch is not None

    @property
    def trained(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def train_text(self, text: str) -> int:
        return len(tokenize(text))

    def train_corpus(
        self,
        corpus_path: Path,
        *,
        epochs: int = 3,
        batch_size: int = 12,
        learning_rate: float = 3e-4,
        steps_per_epoch: int = 150,
    ) -> dict[str, float | int]:
        if torch is None:
            raise RuntimeError("PyTorch is required for transformer training. Install torch locally first.")

        file_paths = []
        if corpus_path.is_file():
            file_paths = [corpus_path]
        else:
            for file_path in corpus_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".py", ".json", ".js", ".ts"}:
                    file_paths.append(file_path)

        if not file_paths:
            raise ValueError(f"No training files found in {corpus_path}")

        def iter_chunks():
            for file_path in file_paths:
                current_chunk = []
                with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        current_chunk.append(line)
                        if len(current_chunk) >= 500:
                            yield "".join(current_chunk)
                            current_chunk = []
                if current_chunk:
                    yield "".join(current_chunk)

        if self.tokenizer is not None and len(self.tokenizer.token_to_id) > len(self.tokenizer.specials):
            print(f"Reusing existing tokenizer vocabulary with {len(self.tokenizer)} tokens.")
        else:
            self.tokenizer = CodeTokenizer()
            self.tokenizer.fit(iter_chunks(), max_vocab=self.config.max_vocab)

        token_ids: list[int] = []
        for file_path in file_paths:
            token_ids.append(self.tokenizer.bos_id)
            current_chunk = []
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    current_chunk.append(line)
                    if len(current_chunk) >= 500:
                        token_ids.extend(self.tokenizer.encode("".join(current_chunk), add_bounds=False))
                        current_chunk = []
            if current_chunk:
                token_ids.extend(self.tokenizer.encode("".join(current_chunk), add_bounds=False))
            token_ids.append(self.tokenizer.eos_id)

        if len(token_ids) < self.config.block_size + 2:
            raise ValueError("Corpus is too small for transformer training.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinyTransformerLM(len(self.tokenizer), self.config).to(device)

        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=device)
                model.load_state_dict(checkpoint["model"])
                print("Loaded existing trained model weights for incremental fine-tuning.")
            except Exception as exc:
                print(f"Could not load existing weights for fine-tuning ({exc}). Training from scratch.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        data = torch.tensor(token_ids, dtype=torch.long, device=device)
        losses: list[float] = []

        model.train()
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                starts = torch.randint(0, len(data) - self.config.block_size - 1, (batch_size,), device=device)
                x = torch.stack([data[start : start + self.config.block_size] for start in starts])
                y = torch.stack([data[start + 1 : start + self.config.block_size + 1] for start in starts])
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        self.model = model.eval()
        self._save_model()
        avg_loss = sum(losses[-min(len(losses), 50) :]) / min(len(losses), 50)
        return {
            "tokens": len(token_ids),
            "vocab": len(self.tokenizer),
            "epochs": epochs,
            "steps": epochs * steps_per_epoch,
            "loss": round(avg_loss, 4),
        }

    def complete(self, prompt: str, *, max_tokens: int = 180, temperature: float = 0.7) -> str:
        if torch is None or not self.trained:
            return ""
        device = next(self.model.parameters()).device
        ids = self.tokenizer.encode(prompt)[-self.config.block_size :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        self.model.eval()
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_tokens):
                window = input_ids[:, -self.config.block_size :]
                logits = self.model(window)[:, -1, :] / max(temperature, 0.05)
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                if next_id == self.tokenizer.eos_id:
                    break
                generated.append(next_id)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                    dim=1,
                )
        return self.tokenizer.decode(generated)

    def _load_model(self) -> None:
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = TinyTransformerLM(len(self.tokenizer), self.config)
        model.load_state_dict(checkpoint["model"])
        self.model = model.eval()

    def _save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, self.model_path)
        self.tokenizer_path.write_text(json.dumps(self.tokenizer.to_json()), encoding="utf-8")
        self.config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @staticmethod
    def _read_corpus(path: Path) -> list[str]:
        if path.is_file():
            return [path.read_text(encoding="utf-8", errors="ignore")]
        texts: list[str] = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".py", ".json", ".js", ".ts"}:
                texts.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        return texts

    @staticmethod
    def _load_config(path: Path) -> TransformerConfig:
        if not path.exists():
            return TransformerConfig()
        return TransformerConfig(**json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _load_tokenizer(path: Path) -> CodeTokenizer | None:
        if not path.exists():
            return None
        return CodeTokenizer.from_json(json.loads(path.read_text(encoding="utf-8")))
