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

    #Language
    "cfilt": DatasetSpec("cfilt","cfilt/iitb-english-hindi"), #128Mb

    # Maths
    "gsm8k": DatasetSpec("gsm8k", "openai/gsm8k", "main"),
    "gsm8k-socratic": DatasetSpec("gsm8k-socratic", "openai/gsm8k", "socratic"),
    "applied-ai-018": DatasetSpec("applied-ai-018","applied-ai-018/Mathematics"), #26Gb
    "ZixuanKe": DatasetSpec("ZixuanKe", "ZixuanKe/posttrain_tokenized_dm_mathematics_sup_qwen2.5_32b_instr"), #4.11Gb
    "learningarena": DatasetSpec("learningarena", "learningarena/Mathematics"), #182Mb
    "timaeus": DatasetSpec("timaeus","timaeus/dsir-pile-1m-filtered-no-github-or-dm_mathematics"),#981Mb

    # Python
    "claude-opus": DatasetSpec("claude-opus", "Roman1111111/claude-opus-4.6-10000x"),
    "syndata": DatasetSpec("syndata", "PsiBotAI/SynData"),#29.3Tb
    "swe-zero": DatasetSpec("swe-zero", "AlienKevin/SWE-ZERO-12M-trajectories"),#Above 50Gb
    "zero-to-cad": DatasetSpec("zero-to-cad", "ADSKAILab/Zero-To-CAD-1m"),
    "open-mm-rl": DatasetSpec("open-mm-rl", "TuringEnterprises/Open-MM-RL"),
    "physical-vantage-bench": DatasetSpec("physical-vantage-bench", "nvidia/PhysicalAI-VANTAGE-Bench","vqa","test"),#21.7Gb
    "3d-arena": DatasetSpec("3d-arena", "3d-arena/3d-arena"),#26.2Gb
    "h4iku": DatasetSpec("h4iku", "h4iku/coconut_javascript2010"),#4.74Gb
    "h4iku-preprocessed": DatasetSpec("h4iku-preprocessed", "h4iku-preprocessed/coconut_javascript2010_preprocessed"),#404Mb
    "iamtarun": DatasetSpec("iamtarun", "iamtarun/python_code_instructions_18k_alpaca"),
    "dylanhogg": DatasetSpec("dylanhogg","dylanhogg/awesome-python"),
    "Jackrong": DatasetSpec("Jackrong", "Jackrong/Competitive-Programming-python-blend"),#5.97Gb
    "gss1147": DatasetSpec("gss1147", "gss1147/god_level_python_dataset_25k"),
    "Fraser": DatasetSpec("Fraser", "Fraser/python-state-changes"),#1.21Gb
    "sia-precision-education": DatasetSpec("sia-precision-education", "sia-precision-education/pile_python"),#4.34GB
    "angie-chen55": DatasetSpec("angie-chen55","angie-chen55/python-github-code"),#20.2Gb

    #OpenAi
    "openai": DatasetSpec("openai","openai/gsm8k","main"),
    "openai-socratic": DatasetSpec("openai-socratic","openai/gsm8k","socratic"),
    "openai-axis": DatasetSpec("openai-axis","openai/summarize_from_feedback","axis"),
    "openai-comparisons": DatasetSpec("openai-comparisons","openai/summarize_from_feedback","comparisons"),
    "CarperAI": DatasetSpec("CarperAI","CarperAI/openai_summarize_tldr"), #123Mb
    "CarperAI-comparisons": DatasetSpec("CarperAI-comparisons","CarperAI/openai_summarize_comparisons"),
    "Birchlabs-best": DatasetSpec("Birchlabs-best","Birchlabs/openai-prm800k-phase2_test-stepwise-best"),
    "Birchlabs-critique": DatasetSpec("Birchlabs-critique","Birchlabs/openai-prm800k-phase2_test-stepwise-critique"),
    "EleutherAI": DatasetSpec("EleutherAI","EleutherAI/lambada_openai"),
    "EleutherAI-de": DatasetSpec("EleutherAI-de","EleutherAI/lambada_openai","de"),
    "EleutherAI-en": DatasetSpec("EleutherAI-en","EleutherAI/lambada_openai","en"),
    "openai-webgpt": DatasetSpec("openai-webgpt","openai/webgpt_comparisons"),
    "rubend18": DatasetSpec("rubend18","rubend18/DALL-E-Prompts-OpenAI-ChatGPT"),
    "carlosejimenez": DatasetSpec("carlosejimenez","carlosejimenez/wikitext-103-raw-v1_sents_min_len10_max_len30_openai_clip-vit-base-patch32"), #5.29Gb
    "sl-alex": DatasetSpec("sl-alex","sl-alex/openai-prm800k-solutions-only"),

    #Claude
    "Roman1111111-claude-sonnet": DatasetSpec("Roman1111111-claude-sonnet","Roman1111111/claude-sonnet-4.6-120000x"),
    "AnodeAI": DatasetSpec("AnodeAI","AnodeAI/ClaudeOpus4.6_promots"),
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
