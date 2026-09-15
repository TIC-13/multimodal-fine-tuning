from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from vlm_ft.config.training import ModelFamily, SUPPORTED_FAMILIES


@dataclass
class EvalModelConfig:
    family: ModelFamily
    model_id: str
    checkpoint: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = "flash_attention_2"
    trust_remote_code: bool = True


@dataclass
class EvalDataConfig:
    path: str
    split: str = "test"
    image_root: str | None = None
    max_length: int = 4096


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    do_sample: bool = False


@dataclass
class EvalConfig:
    model: EvalModelConfig
    data: EvalDataConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output_dir: str = "outputs/eval/run"


def _merge_dataclass(cls: type, raw: dict[str, Any] | None, defaults: Any | None = None):
    base = {}
    if defaults is not None:
        base.update(defaults.__dict__)
    if raw:
        base.update(raw)
    return cls(**base)


def load_eval_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping")

    model_raw = raw.get("model") or {}
    family = model_raw.get("family")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported model.family: {family!r}")

    return EvalConfig(
        model=_merge_dataclass(EvalModelConfig, model_raw),
        data=_merge_dataclass(EvalDataConfig, raw.get("data")),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation")),
        output_dir=raw.get("output_dir", "outputs/eval/run"),
    )
