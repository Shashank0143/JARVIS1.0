from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datasets import Image


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    config: str | None = None
    split: str = "train"


AIDATASET_SPECS = {
    #Greetings
    "SAGI-1": DatasetSpec("SAGI-1","SAGI-1/Greetings_DPO_dataset_V1"),
    "Goekdeniz-Guelmez": DatasetSpec("Goekdeniz-Guelmez", "Goekdeniz-Guelmez/Openai-function-invocations-20k-with-greetings"),
    "Goekdeniz-Guelmez-70k": DatasetSpec("Goekdeniz-Guelmez", "Goekdeniz-Guelmez/JOSIE_Wizard_Vicuna_unfiltered_de_with_greetings_70k_v2"),
    "RayBoustany": DatasetSpec("RayBoustany","RayBoustany/1200_rows_dataset_siren_greetings_thanks_augmented"),
    "nabinnvidia-data": DatasetSpec("nabinnvidia", "nabinnvidia/multi-lingual-greetings", "data"),
    "nabinnvidia-chat_format": DatasetSpec("nabinnvidia", "nabinnvidia/multi-lingual-greetings", "chat_format"),
    # Maths
    "gsm8k": DatasetSpec("gsm8k", "openai/gsm8k", "main"),
    "gsm8k-socratic": DatasetSpec("gsm8k-socratic", "openai/gsm8k", "socratic"),
    "applied-ai-018": DatasetSpec("applied-ai-018","applied-ai-018/Mathematics"),
    "premio-ai-dedup": DatasetSpec("premio-ai", "premio-ai/TheArabicPile_Mathematics", "dedup"),
    "premio-ai-original": DatasetSpec("premio-ai", "premio-ai/TheArabicPile_Mathematics", "original"),
    "ZixuanKe": DatasetSpec("ZixuanKe", "ZixuanKe/posttrain_tokenized_dm_mathematics_sup_qwen2.5_32b_instr"),
    "learningarena": DatasetSpec("learningarena", "learningarena/Mathematics"),
    "timaeus": DatasetSpec("timaeus","timaeus/dsir-pile-1m-filtered-no-github-or-dm_mathematics"),

    # Python
    "claude-opus": DatasetSpec("claude-opus", "Roman1111111/claude-opus-4.6-10000x"),
    "syndata": DatasetSpec("syndata", "PsiBotAI/SynData"),
    "swe-zero": DatasetSpec("swe-zero", "AlienKevin/SWE-ZERO-12M-trajectories"),
    "zero-to-cad": DatasetSpec("zero-to-cad", "ADSKAILab/Zero-To-CAD-1m"),
    "open-mm-rl": DatasetSpec("open-mm-rl", "TuringEnterprises/Open-MM-RL"),
    "physical-vantage-bench": DatasetSpec("physical-vantage-bench", "nvidia/PhysicalAI-VANTAGE-Bench","vqa","test"),
    "3d-arena": DatasetSpec("3d-arena", "3d-arena/3d-arena"),
    "h4iku": DatasetSpec("h4iku", "h4iku/coconut_javascript2010"),
    "iamtarun": DatasetSpec("iamtarun", "iamtarun/python_code_instructions_18k_alpaca"),
    "dylanhogg": DatasetSpec("dylanhogg","dylanhogg/awesome-python"),
    "Jackrong": DatasetSpec("Jackrong", "Jackrong/Competitive-Programming-python-blend"),
    "gss1147": DatasetSpec("gss1147", "gss1147/god_level_python_dataset_25k"),
    "Fraser": DatasetSpec("Fraser", "Fraser/python-state-changes"),
    "sia-precision-education": DatasetSpec("sia-precision-education", "sia-precision-education/pile_python"),
    "h4iku-python": DatasetSpec("h4iku","h4iku/coconut_python2010"),
    "angie-chen55": DatasetSpec("angie-chen55","angie-chen55/python-github-code")

}


class AiDatasetImporter:
    def __init__(self, output_root: Path | str = "datasets/training_corpus/aidataset") -> None:
        self.output_root = Path(output_root)

    def import_dataset(self, spec: DatasetSpec) -> Path:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the `datasets` package before importing AiDATASET entries.") from exc

        self.output_root.mkdir(parents=True, exist_ok=True)
        output_path = self.output_root / f"{self._safe_name(spec.key)}.md"
        dataset = load_dataset(
            spec.name,
            spec.config,
            split=spec.split,
            streaming=True,
        )

        try:
            if dataset.features and "image" in dataset.features:
                dataset = dataset.cast_column(
                    "image",
                    Image(decode=False)
                )
        except AttributeError:
            pass

        written = 0
        with output_path.open("w", encoding="utf-8") as file:
            file.write(f"# AiDATASET: {spec.key}\n\n")
            file.write(f"Source: {spec.name}\n\n")
            for row in dataset:
                text = self._row_to_text(row)
                if len(text.split()) < 8:
                    continue
                written += 1
                file.write(f"## Example {written}\n\n{text}\n\n")
        if written == 0:
            raise RuntimeError(f"No usable rows imported from {spec.name}.")
        return output_path

    def import_many(self, keys: list[str]) -> list[Path]:
        paths: list[Path] = []
        for key in keys:
            spec = AIDATASET_SPECS[key]
            paths.append(self.import_dataset(spec))
        return paths

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(AIDATASET_SPECS)

    @staticmethod
    def _row_to_text(row: dict[str, Any]) -> str:
        preferred = [
            "question",
            "answer",
            "solution",
            "prompt",
            "completion",
            "messages",
            "text",
            "instruction",
            "response",
            "trajectory",
        ]
        parts: list[str] = []
        used = set()
        for key in preferred:
            if key in row:
                parts.append(f"{key}: {AiDatasetImporter._value_to_text(row[key])}")
                used.add(key)
        for key, value in row.items():
            if key in used:
                continue
            text = AiDatasetImporter._value_to_text(value)
            if text:
                parts.append(f"{key}: {text}")
        return "\n\n".join(parts)

    # @staticmethod
    # def _value_to_text(value: Any) -> str:
    #     if value is None:
    #         return ""
    #     if isinstance(value, str):
    #         return value.strip()
    #     if isinstance(value, (int, float, bool)):
    #         return str(value)
    #     return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _value_to_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                return str(value)
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )
            except UnicodeDecodeError:
                return str(value)

        return str(value)
    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"
