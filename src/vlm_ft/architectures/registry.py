from __future__ import annotations

from vlm_ft.architectures.base import VLMModel
from vlm_ft.architectures.models.gemma4 import Gemma4Model
from vlm_ft.architectures.models.medgemma import MedGemmaModel
from vlm_ft.architectures.models.qwen3_5 import Qwen35Model


_REGISTRY: dict[str, VLMModel] = {
    MedGemmaModel.family: MedGemmaModel(),
    Gemma4Model.family: Gemma4Model(),
    Qwen35Model.family: Qwen35Model(),
}

def list_families() -> list[str]:
    return sorted(_REGISTRY)

def get_model(family: str) -> VLMModel:
    try:
        return _REGISTRY[family]
    except KeyError as exc:
        known = ", ".join(list_families())
        raise KeyError(f"Unknown model family {family!r}. Known: {known}") from exc

def register_model(model: VLMModel) -> None:
    _REGISTRY[model.family] = model
