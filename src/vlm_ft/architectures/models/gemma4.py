from __future__ import annotations

from typing import Any

from vlm_ft.architectures.base import (
    DEFAULT_ENCODER_LORA_TARGETS,
    DEFAULT_LORA_TARGETS,
    LoadedModel,
    ModulePaths,
    load_vlm,
)


class Gemma4Model:
    family = "gemma4"

    def load_model_and_processor(
        self,
        model_id: str,
        *,
        torch_dtype: Any,
        attn_implementation: str | None = None,
        quantization_config: Any | None = None,
        trust_remote_code: bool = True,
        device_map: str | dict | None = "auto",
    ) -> LoadedModel:
        return load_vlm(
            model_id,
            self.family,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

    def module_paths(self) -> ModulePaths:
        return ModulePaths(
            encoder=("vision_tower", "model.vision_tower"),
            projector=("embed_vision", "model.embed_vision"),
            llm=("language_model", "model.language_model", "lm_head", "model.lm_head"),
        )

    def default_lora_targets(self) -> list[str]:
        return list(DEFAULT_LORA_TARGETS)

    def default_encoder_lora_targets(self) -> list[str]:
        return list(DEFAULT_ENCODER_LORA_TARGETS)
