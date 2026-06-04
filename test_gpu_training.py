#!/usr/bin/env python
"""Quick GPU test script for JARVIS."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_gpu_training():
    """Test GPU training with a small dataset."""
    print("\n" + "="*70)
    print("JARVIS GPU Training Test")
    print("="*70 + "\n")

    try:
        from jarvis_ai import initialize_cuda, get_gpu_info
        from jarvis_ai.assistant import LocalCodingAssistant
    except ImportError as exc:
        print(f"✗ Failed to import JARVIS modules: {exc}")
        return False

    # Initialize GPU
    print("1. Initializing GPU/CUDA...")
    initialize_cuda()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info.get("available"):
        print("   ℹ GPU not available, will use CPU for testing")
    else:
        print("   ✓ GPU available and initialized")
        print(f"   Device: {gpu_info['device_count']} GPU(s)")

    # Create assistant
    print("\n2. Creating JARVIS assistant...")
    try:
        assistant = LocalCodingAssistant()
        print("   ✓ Assistant created successfully")
    except Exception as exc:
        print(f"   ✗ Failed to create assistant: {exc}")
        return False

    # Test with small training corpus
    print("\n3. Testing transformer training (small corpus)...")
    try:
        # Create a small test corpus
        test_corpus = Path("datasets/training_corpus")
        if not test_corpus.exists():
            print("   ℹ Test corpus not found, skipping training test")
            print("   Note: Create datasets/training_corpus with .py or .md files for testing")
        else:
            stats = assistant.train_transformer_corpus(
                test_corpus,
                epochs=1,
                batch_size=4,
                learning_rate=3e-4,
                steps_per_epoch=10,
            )
            print(f"   ✓ Training successful!")
            print(f"     - Tokens: {stats['tokens']}")
            print(f"     - Vocab: {stats['vocab']}")
            print(f"     - Loss: {stats['loss']:.4f}")

    except Exception as exc:
        print(f"   ✗ Training failed: {exc}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference
    print("\n4. Testing inference...")
    try:
        prompt = "def hello"
        result = assistant.answer(prompt, use_web=False)
        print(f"   ✓ Inference successful!")
        print(f"     Question: '{prompt}'")
        print(f"     Response: {result[:100]}...")
    except Exception as exc:
        print(f"   ✗ Inference failed: {exc}")
        return False

    print("\n" + "="*70)
    print("✓ All GPU tests passed successfully!")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    success = test_gpu_training()
    sys.exit(0 if success else 1)
