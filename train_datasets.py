from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

os.environ["HF_TOKEN"] = os.getenv("HUGGING_TOKEN")

from jarvis_ai.dataset_importer import AiDatasetImporter
from jarvis_ai import LocalCodingAssistant

def main():
    datasets = [
        "openai",
    ]
    
    print("Importing datasets...")
    importer = AiDatasetImporter()
    paths = importer.import_many(datasets)
    for path in paths: 
        print(f"Imported: {path}")
    
    print("\nTraining model with imported datasets...")
    assistant = LocalCodingAssistant()
    stats = assistant.train_transformer_corpus(
        Path("datasets/training_corpus"),
        epochs=10,
        batch_size=12,
        learning_rate=3e-4,
        steps_per_epoch=150,
    )
    print(f"Trained: {stats['tokens']} tokens, {stats['vocab']} vocab, {stats['steps']} steps, loss={stats['loss']}")

if __name__ == "__main__":
    main()