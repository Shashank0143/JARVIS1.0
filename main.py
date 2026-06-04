from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis_ai import JarvisCore, LocalCodingAssistant
from jarvis_ai.dataset_importer import AiDatasetImporter
from jarvis_ai.gpu import initialize_cuda, get_gpu_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only coding AI with self-training and RAG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train/index local code or documents.")
    train.add_argument("path", type=Path, help="File or directory to ingest.")

    learn_web = subparsers.add_parser(
        "learn-web",
        help="Learn from trusted internet sources into local RAG/model state.",
    )
    learn_web.add_argument("topic", nargs="?", default="", help="Topic to learn. Omit with --curriculum.")
    learn_web.add_argument("--pages", type=int, default=5, help="Maximum pages to ingest.")
    learn_web.add_argument(
        "--curriculum",
        action="store_true",
        help="Learn the default coding/AI curriculum from trusted sources.",
    )

    learn_pytorch = subparsers.add_parser(
        "learn-pytorch",
        help="Learn official PyTorch tutorials into datasets/training_corpus.",
    )
    learn_pytorch.add_argument("--pages", type=int, default=8, help="Maximum PyTorch tutorial pages.")

    train_neural = subparsers.add_parser(
        "train-neural",
        help="Train the local PyTorch Transformer from a dataset folder.",
    )
    train_neural.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("datasets/training_corpus"),
        help="Dataset folder. Defaults to datasets/training_corpus.",
    )
    train_neural.add_argument("--epochs", type=int, default=2)
    train_neural.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4 for RTX 3050 Laptop, use 8-16 for desktop GPUs)")
    train_neural.add_argument("--steps-per-epoch", type=int, default=50)
    train_neural.add_argument("--learning-rate", type=float, default=3e-4)

    import_ai = subparsers.add_parser(
        "import-aidataset",
        help="Import AiDATASET entries from datasets/README.md into local training_corpus.",
    )
    import_ai.add_argument(
        "dataset",
        nargs="?",
        default="gsm8k",
        help="Dataset key or 'all'. Use --list to show keys. Defaults to gsm8k.",
    )
    import_ai.add_argument("--list", action="store_true", help="List known AiDATASET keys.")

    transformer = subparsers.add_parser(
        "train-transformer",
        help="Train the optional local Transformer on a curated corpus.",
    )
    transformer.add_argument(
        "corpus",
        type=Path,
        nargs="?",
        default=Path("corpus"),
        help="Curated corpus file or directory. Defaults to ./corpus.",
    )
    transformer.add_argument("--epochs", type=int, default=3)
    transformer.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4 for RTX 3050 Laptop with FP16 mixed precision)")
    transformer.add_argument("--steps-per-epoch", type=int, default=150)
    transformer.add_argument("--learning-rate", type=float, default=3e-4)

    ask = subparsers.add_parser("ask", help="Ask a coding question.")
    ask.add_argument("question", help="Coding problem or error to solve.")
    ask.add_argument(
        "--no-web",
        action="store_true",
        help="Use only local trained/indexed knowledge.",
    )

    do = subparsers.add_parser("do", help="Run one Jarvis machine/coding/vision command.")
    do.add_argument("instruction", help="Command to execute.")

    start = subparsers.add_parser("start", help="Start mini Jarvis.")
    start.add_argument("--voice", action="store_true", help="Enable microphone input.")
    start.add_argument("--camera", action="store_true", help="Enable camera processing.")
    start.add_argument("--speak", action="store_true", help="Speak responses aloud.")
    start.add_argument(
        "--auto-learn",
        action="store_true",
        help="Run a small trusted-source curriculum learning pass on startup.",
    )

    # New subcommands for expanded JARVIS features
    agent = subparsers.add_parser("agent", help="Run cooperative multi-agent tasks.")
    agent.add_argument("agent_or_instruction", help="Sub-agent name OR full pipeline instruction.")
    agent.add_argument("instruction", nargs="?", default=None, help="Instruction if a specific sub-agent was specified.")

    automl = subparsers.add_parser("automl", help="Train and optimize neural models via AutoML.")
    automl.add_argument("dataset", type=Path, nargs="?", default=Path("datasets/training_corpus"), help="Dataset path.")
    automl.add_argument("--epochs", type=int, default=1, help="AutoML training epochs.")

    sec_audit = subparsers.add_parser("security-audit", help="Run vulnerability audit scans.")
    sec_audit.add_argument("path", type=Path, nargs="?", default=Path("."), help="Path to file or directory to scan.")

    test_loop = subparsers.add_parser("test-loop", help="Execute test generation and autonomous self-healing.")
    test_loop.add_argument("target_file", type=Path, help="Target python file.")
    test_loop.add_argument("test_file", type=Path, nargs="?", default=None, help="Optional test file path.")

    self_improve = subparsers.add_parser("self-improve", help="Self-evaluation and reinforcement learning pass.")
    self_improve.add_argument("--auto-knowledge", action="store_true", help="Extract memory into structured markdown.")

    subparsers.add_parser("shell", help="Start an interactive assistant shell.")
    return parser


def train_path(assistant: LocalCodingAssistant, path: Path) -> None:
    if path.is_dir():
        files, chunks, tokens = assistant.ingest_directory(path)
        print(f"Indexed {files} files, {chunks} chunks, trained on {tokens} tokens.")
        return
    if path.is_file():
        chunks, tokens = assistant.ingest_file(path)
        print(f"Indexed {chunks} chunks, trained on {tokens} tokens.")
        return
    raise SystemExit(f"Path not found: {path}")


def run_shell(assistant: LocalCodingAssistant) -> None:
    print("Jarvis local coding assistant. Type 'exit' to quit.")
    while True:
        question = input("\nAsk> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print()
        print(assistant.answer(question))


def run_jarvis(
    core: JarvisCore,
    *,
    voice: bool = False,
    camera: bool = False,
    speak: bool = False,
    auto_learn: bool = False,
) -> None:
    print("======================================================================")
    print("       ██████╗  █████╗ ██████╗ ██╗   ██╗██╗███████╗")
    print("       ╚════██╗██╔══██╗██╔══██╗██║   ██║██║██╔════╝")
    print("        █████╔╝███████║██████╔╝██║   ██║██║███████╗")
    print("        ╚═══██╗██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║")
    print("       ██████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║")
    print("       ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝")
    print("             Local-Only Coding & Intelligent Assistant")
    print("======================================================================")
    print("Greeting: Hello! I am Jarvis, your personal local assistant.")
    print("Capabilities:")
    print("  • Local RAG Knowledge & Document Indexing")
    print("  • PyTorch Transformer Neural Model Training (GPU Accelerated)")
    print("  • Collaborative Multi-Agent Pipelines")
    print("  • AutoML Model Optimization")
    print("  • System Security Auditing & Code Vulnerability Scans")
    print("  • Autonomous Self-Healing Testing Loops")
    print("  • Mathematics & Precise Arithmetic Execution")
    print("  • Voice Interaction & Real-Time Camera Vision")
    print("======================================================================\n")
    print("Mini Jarvis started. Type 'exit' to quit.")
    if voice or speak:
        print(core.enable_voice())
    if camera:
        print(core.enable_vision())
    if auto_learn:
        print("Auto-learning from trusted sources...")
        print(core.learn_curriculum(pages_per_topic=1))

    while True:
        if voice:
            print("\nListening...")
            command = core.voice.listen_once()
            print(f"You: {command}")
        else:
            command = input("\nJarvis> ").strip()
        if command.lower() in {"exit", "quit", "stop"}:
            break
        response = core.handle(command)
        print(response)
        if speak and core.voice:
            core.voice.speak(response[:700])


def main() -> None:
    # Initialize CUDA and GPU support at startup
    print("\n" + "="*60)
    print("Initializing GPU/CUDA support...")
    initialize_cuda()
    gpu_info = get_gpu_info()
    if gpu_info.get("available"):
        print(f"✓ GPU Training Enabled")
        print(f"  Devices: {gpu_info.get('device_count')}")
        for device in gpu_info.get("devices", []):
            print(f"    - {device['name']} ({device['total_memory_gb']:.1f}GB)")
    else:
        print("ℹ CPU Mode (GPU not available)")
    print("="*60 + "\n")
    
    if len(sys.argv) == 1:
        run_jarvis(JarvisCore())
        return

    args = build_parser().parse_args()
    assistant = LocalCodingAssistant()

    if args.command == "train":
        train_path(assistant, args.path)
    elif args.command == "learn-web":
        core = JarvisCore()
        if args.curriculum:
            print(core.learn_curriculum(pages_per_topic=args.pages))
        elif args.topic:
            print(core.learn_from_internet(args.topic, pages=args.pages))
        else:
            raise SystemExit("Provide a topic or use --curriculum.")
    elif args.command == "learn-pytorch":
        core = JarvisCore()
        print(core.learn_pytorch_tutorials(pages=args.pages))
    elif args.command == "train-neural":
        try:
            stats = assistant.train_transformer_corpus(
                args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                steps_per_epoch=args.steps_per_epoch,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            "PyTorch neural model trained: "
            f"{stats['tokens']} tokens, {stats['vocab']} vocab, "
            f"{stats['steps']} steps, loss={stats['loss']}."
        )
    elif args.command == "import-aidataset":
        importer = AiDatasetImporter()
        if args.list:
            print("Available AiDATASET keys: " + ", ".join(importer.keys()))
            return
        keys = importer.keys() if args.dataset == "all" else [args.dataset]
        unknown = [key for key in keys if key not in importer.keys()]
        if unknown:
            raise SystemExit(f"Unknown dataset key(s): {', '.join(unknown)}")
        paths = importer.import_many(keys)
        for path in paths:
            print(f"Imported: {path}")
    elif args.command == "train-transformer":
        try:
            stats = assistant.train_transformer_corpus(
                args.corpus,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                steps_per_epoch=args.steps_per_epoch,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            "Transformer trained: "
            f"{stats['tokens']} tokens, {stats['vocab']} vocab, "
            f"{stats['steps']} steps, loss={stats['loss']}."
        )
    elif args.command == "ask":
        print(assistant.answer(args.question, use_web=not args.no_web))
    elif args.command == "do":
        core = JarvisCore()
        print(core.handle(args.instruction))
    elif args.command == "start":
        core = JarvisCore()
        run_jarvis(
            core,
            voice=args.voice,
            camera=args.camera,
            speak=args.speak,
            auto_learn=args.auto_learn,
        )
    elif args.command == "agent":
        core = JarvisCore()
        if args.instruction:
            cmd = f"agent {args.agent_or_instruction} {args.instruction}"
        else:
            cmd = f"agent {args.agent_or_instruction}"
        print(core.handle(cmd))
    elif args.command == "automl":
        core = JarvisCore()
        path = Path(args.dataset)
        content = "Simulated dataset sequence."
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="ignore")
        elif path.is_dir():
            file_contents = []
            for f in path.rglob("*"):
                if f.is_file() and f.suffix in (".py", ".txt", ".json", ".md"):
                    try:
                        file_contents.append(f.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass
            if file_contents:
                content = "\n".join(file_contents)
        print(core.automl.run_automl_search(content, epochs=args.epochs))
    elif args.command == "security-audit":
        core = JarvisCore()
        print(core.handle(f"security-audit {args.path}"))
    elif args.command == "test-loop":
        core = JarvisCore()
        if args.test_file:
            print(core.handle(f"self-heal {args.target_file} {args.test_file}"))
        else:
            print(core.handle(f"run tests {args.target_file}"))
    elif args.command == "self-improve":
        core = JarvisCore()
        if args.auto_knowledge:
            print(core.handle("auto-knowledge"))
        else:
            print(core.handle("self-improve"))
    elif args.command == "shell":
        run_jarvis(JarvisCore())


if __name__ == "__main__":
    main()
