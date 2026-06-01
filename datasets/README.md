# Datasets

Put training data here. Jarvis can train its local PyTorch neural model from text,
Markdown, Python, JavaScript, TypeScript, and JSON files in this folder.

Recommended layout:

```text
datasets/
  training_corpus/
    conversation/
    coding/
    mathematics/
    pytorch_tutorials/
```

Train from this dataset:

```powershell
.venv\Scripts\python.exe main.py train-neural datasets\training_corpus --epochs 2 --steps-per-epoch 50
```

Learn official PyTorch tutorials into the dataset:

```powershell
.venv\Scripts\python.exe main.py learn-pytorch --pages 8
```
# AiDATASET

```OpenAi/gsm8k
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")
```
```OpenAi/gsm8k socratic
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "socratic")
```

```Roman1111111/claude-opus-4.6-10000x
from datasets import load_dataset
ds = load_dataset("Roman1111111/claude-opus-4.6-10000x")
```

```PsiBotAI/SynData
from datasets import load_dataset
ds = load_dataset("PsiBotAI/SynData")
```

```AlienKevin/SWE-ZERO-12M-trajectories
from datasets import load_dataset
ds = load_dataset("AlienKevin/SWE-ZERO-12M-trajectories")
```
