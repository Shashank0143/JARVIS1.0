# JARVIS Local Coding AI

This project is a local-only coding assistant. It does not integrate any hosted third-party AI model or AI API.

What it includes:

- A central `JarvisCore` that routes commands to machine, coding, voice, vision, and memory classes
- Domain folders under `jarvis_ai/domains/` for conversation, education, and deep learning
- Architecture folders under `jarvis_ai/core_system/`, `jarvis_ai/models/`, `jarvis_ai/storage/`, and `jarvis_ai/services/`
- Structured subject memory under `knowledge/<subject>/<chapter>/<file>.md`
- A dependency-free fallback language model stored in `data/local_code_model.json`
- An optional local Transformer checkpoint stored in `data/transformer_code_model.pt`
- A RAG knowledge store stored in `data/rag_store.json`
- Local memory stored in `data/memory.jsonl`
- Saved answers stored in `data/answers.jsonl`
- Trusted internet learning history stored in `data/learned_sources.json`
- Local code/document ingestion
- Internet search/page ingestion for coding questions
- A command-line interface for training and asking questions
- File/folder/app control with delete confirmation safeguards
- Voice input and speech output
- Camera frame analysis through local OpenCV

Important limitation: this is not a GPT-scale model. Training a real large language model from scratch needs massive datasets, GPUs, and weeks/months of compute. This implementation gives you a complete local pipeline that can learn from your own code/docs and retrieved web pages without calling external AI services.

## Usage

Train on this project:

```powershell
.venv\Scripts\python.exe main.py train .
```

Ask with web research:

```powershell
.venv\Scripts\python.exe main.py ask "How do I fix ModuleNotFoundError in Python?"
```

Ask only from local knowledge:

```powershell
.venv\Scripts\python.exe main.py ask "Explain this project structure" --no-web
```

Train the optional local Transformer:

```powershell
.venv\Scripts\python.exe main.py train-transformer corpus --epochs 3 --steps-per-epoch 150
```

Interactive mode:

```powershell
.venv\Scripts\python.exe main.py shell
```

Start mini Jarvis:

```powershell
.venv\Scripts\python.exe main.py start
.venv\Scripts\python.exe main.py start --auto-learn
```

Start with microphone, camera, and spoken responses:

```powershell
.venv\Scripts\python.exe main.py start --voice --camera --speak
```

Run one machine command:

```powershell
.venv\Scripts\python.exe main.py do "open notepad"
.venv\Scripts\python.exe main.py do "open C:\Users"
.venv\Scripts\python.exe main.py do "create file notes.txt"
.venv\Scripts\python.exe main.py do "write notes.txt with hello from Jarvis"
.venv\Scripts\python.exe main.py do "update line 1 in notes.txt with updated text"
.venv\Scripts\python.exe main.py do "delete notes.txt"
.venv\Scripts\python.exe main.py do "confirm delete notes.txt"
.venv\Scripts\python.exe main.py do "learn this project is my local Jarvis"
.venv\Scripts\python.exe main.py do "learn internet python file handling"
.venv\Scripts\python.exe main.py do "learn curriculum"
.venv\Scripts\python.exe main.py do "bootstrap subjects"
.venv\Scripts\python.exe main.py do "subject mathematics"
.venv\Scripts\python.exe main.py do "chapter algebra in mathematics"
.venv\Scripts\python.exe main.py do "note quadratic formula in mathematics/algebra with The quadratic formula solves ax squared plus bx plus c equals zero."
.venv\Scripts\python.exe main.py do "What is algebra?"
.venv\Scripts\python.exe main.py do "neural status"
```

Learn from trusted internet sources directly:

```powershell
.venv\Scripts\python.exe main.py learn-web "python file handling" --pages 5
.venv\Scripts\python.exe main.py learn-web --curriculum --pages 3
.venv\Scripts\python.exe main.py learn-pytorch --pages 8
.venv\Scripts\python.exe main.py train-neural datasets\training_corpus --epochs 2 --steps-per-epoch 50
```

Ask what the camera sees:

```powershell
.venv\Scripts\python.exe main.py start --camera
# then type: what do you see
```

## How It Learns

`train` ingests files, adds chunks to the RAG store, and updates the local fallback model. `train-transformer` trains a local Transformer from a curated corpus folder. `ask` can optionally search the web, ingest relevant pages, retrieve the best context, and generate an answer using only local model state.

`learn <text>` saves user-provided knowledge into local memory and the RAG store. Camera observations are also learned automatically. Every answer is archived into `data/answers.jsonl`, and coding answers are fed back into local RAG/model state so Jarvis can reuse them later.

`learn internet <topic>` searches trusted sources, fetches useful pages, cleans them, stores source URLs, indexes chunks into RAG, and trains the local fallback model state. The default trusted sources are W3Schools, GeeksforGeeks, Wikipedia, PyTorch, and TensorFlow.

`learn curriculum` runs a small built-in curriculum covering Python, files, errors, OOP, data structures, ML, PyTorch, TensorFlow, OpenCV, and speech recognition.

Subject learning is stored as real files. For example, mathematics notes live under:

```text
knowledge/mathematics/algebra/overview.md
knowledge/mathematics/algebra/quadratic_formula.md
```

Jarvis checks these subject folders before global web/RAG memory, so basic study questions do not get polluted by unrelated internet results.

## Architecture

Main classes:

- `JarvisCore`: central router and coordinator
- `IntentRouter`: decides whether a request is conversation, machine, education, coding, vision, learning, or deep learning
- `ConversationAgent`: handles normal conversation locally
- `SubjectManager`: creates subject/chapter folders and answers from local notes
- `DeepLearningBrain`: reports/trains local neural backends
- `TensorFlowNeuralNetwork`: optional TensorFlow classifier backend for TensorFlow-compatible Python versions
- `LocalCodingAssistant`: RAG, local model, web learning, answer archive
- `MachineController`: app/file/folder control with delete confirmation

Current neural backend on this machine:

- PyTorch is installed and used by the local Transformer.
- TensorFlow is not installable in this Python 3.14 environment because pip has no matching TensorFlow wheel for it.
- A TensorFlow backend class exists at `jarvis_ai/models/tensorflow_network.py` and will work in a Python 3.11/3.12 TensorFlow environment.

Useful neural commands:

```powershell
.venv\Scripts\python.exe main.py do "neural status"
.venv\Scripts\python.exe main.py do "train neural"
.venv\Scripts\python.exe main.py do "train tensorflow"
```

## Dataset Training

Put your own training data in:

```text
datasets/training_corpus/
```

Jarvis trains its local PyTorch Transformer from this folder:

```powershell
.venv\Scripts\python.exe main.py train-neural datasets\training_corpus --epochs 2 --steps-per-epoch 50
```

Learn official PyTorch tutorials into the dataset and RAG memory:

```powershell
.venv\Scripts\python.exe main.py learn-pytorch --pages 8
```

Inside Jarvis:

```text
Jarvis> learn pytorch tutorials
Jarvis> train neural
```

## AiDATASET Import

`datasets/README.md` lists Hugging Face dataset IDs under `AiDATASET`. Import them into the local training corpus before neural training:

```powershell
.venv\Scripts\python.exe main.py import-aidataset --list
.venv\Scripts\python.exe main.py import-aidataset gsm8k --max-rows 100
.venv\Scripts\python.exe main.py train datasets\training_corpus\aidataset
.venv\Scripts\python.exe main.py train-neural datasets\training_corpus --epochs 1 --steps-per-epoch 20 --batch-size 4
```

Available keys:

```text
gsm8k
gsm8k-socratic
claude-opus
syndata
swe-zero
```

Large datasets are streamed and capped by `--max-rows` so Jarvis does not accidentally download millions of examples.

## Safety

Delete and erase commands do not run immediately. Jarvis first returns a confirmation instruction. Use `confirm delete <path>` or `confirm erase <file>` only when you really want the action.

Voice recognition uses the installed `SpeechRecognition` package. It tries local Sphinx recognition and does not use online speech services by default. Install `pocketsphinx` for offline voice input. To explicitly allow online speech recognition, set `JARVIS_ALLOW_CLOUD_STT=1`.

## Local Transformer

The Transformer backend is implemented in `jarvis_ai/transformer_model.py`. It requires a local PyTorch install, but it does not download or call any hosted AI model. If PyTorch or a trained checkpoint is missing, JARVIS automatically falls back to the dependency-free model.

Add curated coding examples to `corpus/` before training. Good examples include solved bugs, project documentation, API usage notes, clean code snippets, and explanations of why the solution works.
