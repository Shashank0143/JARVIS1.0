from __future__ import annotations

from pathlib import Path

from ...assistant import LocalCodingAssistant
from ...models.tensorflow_network import TensorFlowNeuralNetwork


class DeepLearningBrain:
    def __init__(self, assistant: LocalCodingAssistant) -> None:
        self.assistant = assistant
        self.tensorflow_network = TensorFlowNeuralNetwork(assistant.data_dir)
        self.tensorflow_available = self._has_module("tensorflow")
        self.torch_available = self._has_module("torch")

    def status(self) -> str:
        tf = "available" if self.tensorflow_available else "not installed"
        torch = "available" if self.torch_available else "not installed"
        active = "PyTorch Transformer" if self.torch_available else "fallback local model"
        return f"Deep learning status: TensorFlow={tf}, PyTorch={torch}, active={active}."

    def train_tensorflow_demo(self) -> str:
        examples = [
            ("hello how are you", "conversation"),
            ("open chrome", "machine"),
            ("create file notes", "machine"),
            ("what is algebra", "education"),
            ("quadratic formula", "education"),
            ("python error traceback", "coding"),
            ("train neural network", "deep_learning"),
        ]
        return self.tensorflow_network.train_classifier(examples, epochs=20)

    def train(self, corpus: Path, *, epochs: int = 3, steps_per_epoch: int = 150) -> str:
        if self.torch_available:
            stats = self.assistant.train_transformer_corpus(
                corpus,
                epochs=epochs,
                batch_size=12,
                learning_rate=3e-4,
                steps_per_epoch=steps_per_epoch,
            )
            return (
                "Neural model trained with local PyTorch Transformer: "
                f"{stats['tokens']} tokens, {stats['steps']} steps, loss={stats['loss']}."
            )
        return "No deep learning backend is available. Install PyTorch or TensorFlow locally."

    @staticmethod
    def _has_module(name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(name) is not None
