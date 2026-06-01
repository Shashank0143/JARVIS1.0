# Curated Coding Corpus Seed

This seed corpus teaches the local model concise coding problem-solving patterns.
Add more high-quality code, documentation, bug fixes, and explanations to this
folder before transformer training.

## Python import errors

When Python raises `ModuleNotFoundError`, check the package name, the active
virtual environment, and the project import path. Install missing packages into
the same interpreter that runs the script. For local modules, run the command
from the project root or use package-relative imports.

```python
from pathlib import Path

project_root = Path(__file__).resolve().parent
```

## CLI command pattern

Use `argparse` subcommands for clear command-line tools. Each command should
parse arguments, call one focused function, and print a concise result.

```python
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
train = subparsers.add_parser("train")
train.add_argument("path")
```

## Debugging workflow

Read the exact error first. Find the first traceback line that belongs to the
project. Reproduce the failure with the smallest command. Make one change, run
tests or compilation, and only then refactor.

## RAG coding assistant behavior

A coding assistant should retrieve relevant source files and documentation,
quote the evidence source, and provide a solution grounded in that context.
When the model is uncertain, it should say what is missing instead of inventing
APIs or files.
