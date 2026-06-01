"""Optional TensorFlow neural network backend.

TensorFlow currently has no wheel for this project's Python 3.14 runtime. This
class is ready for a TensorFlow-compatible interpreter and fails clearly here.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..text import tokenize


class TensorFlowNeuralNetwork:
    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "tensorflow_brain"
        self.vocab_path = self.model_dir / "vocab.json"
        self.tensorflow = self._load_tensorflow()
        self.model = None
        self.vocab: dict[str, int] = {}
        if self.available and self.vocab_path.exists():
            self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
            model_file = self.model_dir / "model.keras"
            if model_file.exists():
                self.model = self.tensorflow.keras.models.load_model(model_file)

    @property
    def available(self) -> bool:
        return self.tensorflow is not None

    @property
    def trained(self) -> bool:
        return self.model is not None

    def train_classifier(self, examples: list[tuple[str, str]], *, epochs: int = 10) -> str:
        if not self.available:
            return (
                "TensorFlow is not available in this Python environment. "
                "Use Python 3.11/3.12 with TensorFlow installed, or use the active PyTorch backend."
            )
        labels = sorted({label for _, label in examples})
        tokens = sorted({token for text, _ in examples for token in tokenize(text)})
        self.vocab = {token: index for index, token in enumerate(tokens)}
        label_to_id = {label: index for index, label in enumerate(labels)}
        x = [self._vectorize(text) for text, _ in examples]
        y = [label_to_id[label] for _, label in examples]

        tf = self.tensorflow
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(len(self.vocab),)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(len(labels), activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.int32), epochs=epochs, verbose=0)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.model_dir / "model.keras")
        self.vocab_path.write_text(json.dumps(self.vocab), encoding="utf-8")
        (self.model_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        self.model = model
        return f"TensorFlow neural network trained with {len(examples)} examples and {len(self.vocab)} tokens."

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * len(self.vocab)
        for token in tokenize(text):
            index = self.vocab.get(token)
            if index is not None:
                vector[index] += 1.0
        return vector

    @staticmethod
    def _load_tensorflow():
        try:
            import tensorflow as tf
        except ImportError:
            return None
        return tf
