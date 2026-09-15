from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_ENCODER_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "qkv",
    "proj",
]

@dataclass(frozen=True)
class ModulePaths:
    encoder: tuple[str, ...]
    projector: tuple[str, ...]
    llm: tuple[str, ...]

@dataclass
class LoadedModel:
    model: Any
    processor: Any
    family: str

class VLMModel(Protocol):
    family: str

    def load_model_and_processor(
        self,
        model_id: str,
        *,
        torch_dtype: Any,
        attn_implementation: str | None = None,
        quantization_config: Any | None = None,
        trust_remote_code: bool = True,
        device_map: str | dict | None = "auto",
    ) -> LoadedModel: ...

    def module_paths(self) -> ModulePaths: ...

    def default_lora_targets(self) -> list[str]: ...

    def default_encoder_lora_targets(self) -> list[str]: ...

def load_vlm(
    model_id: str,
    family: str,
    *,
    torch_dtype: Any,
    attn_implementation: str | None = None,
    quantization_config: Any | None = None,
    trust_remote_code: bool = True,
    device_map: str | dict | None = "auto",
) -> LoadedModel:
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return LoadedModel(model=model, processor=processor, family=family)
