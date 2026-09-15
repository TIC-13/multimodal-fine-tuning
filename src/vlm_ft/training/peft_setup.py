from __future__ import annotations

import logging
from typing import Any, Literal

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from vlm_ft.architectures.base import ModulePaths, VLMModel
from vlm_ft.architectures.dtype import resolve_torch_dtype
from vlm_ft.config.training import LoraConfigArgs, Method, QuantizationConfig, TrainableConfig
from vlm_ft.training.freeze import classify_parameter

logger = logging.getLogger(__name__)

LoraScope = Literal["llm", "encoder"]


def build_bnb_config(method: Method, quantization: QuantizationConfig) -> BitsAndBytesConfig | None:
    if method != "qlora":
        return None

    compute_dtype = resolve_torch_dtype(quantization.compute_dtype)
    if quantization.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization.quant_type,
            bnb_4bit_use_double_quant=quantization.use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if quantization.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization.bits: {quantization.bits}. Use 4 or 8.")


def _as_suffix_list(value: str | list[str] | None, auto_fn) -> list[str]:
    if value == "auto" or value is None:
        return auto_fn()
    if isinstance(value, str):
        return [m.strip() for m in value.split(",") if m.strip()]
    return list(value)


def discover_lora_targets(
    model: Any,
    paths: ModulePaths,
    scope: LoraScope,
    suffixes: list[str],
) -> list[str]:
    suffix_set = set(suffixes)
    found: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if classify_parameter(name, paths) != scope:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in suffix_set:
            found.append(name)
    return found


def resolve_target_modules(
    model: Any,
    lora: LoraConfigArgs,
    architecture: VLMModel,
    paths: ModulePaths,
    trainable: TrainableConfig,
) -> list[str]:
    targets: list[str] = []
    llm_count = 0
    encoder_count = 0

    if trainable.llm:
        llm_suffixes = _as_suffix_list(lora.target_modules, architecture.default_lora_targets)
        llm_targets = discover_lora_targets(model, paths, "llm", llm_suffixes)
        llm_count = len(llm_targets)
        targets.extend(llm_targets)

    if lora.encoder:
        encoder_suffixes = _as_suffix_list(
            lora.encoder_target_modules,
            architecture.default_encoder_lora_targets,
        )
        encoder_targets = discover_lora_targets(model, paths, "encoder", encoder_suffixes)
        encoder_count = len(encoder_targets)
        targets.extend(encoder_targets)

    logger.info("LoRA targets llm=%s encoder=%s total=%s", llm_count, encoder_count, len(targets))
    if not targets:
        raise ValueError(
            "No LoRA target modules found. Enable trainable.llm and/or lora.encoder, "
            "or pass explicit target_modules / encoder_target_modules."
        )
    return targets


def build_lora_config(
    model: Any,
    lora: LoraConfigArgs,
    architecture: VLMModel,
    paths: ModulePaths,
    trainable: TrainableConfig,
) -> LoraConfig:
    targets = resolve_target_modules(model, lora, architecture, paths, trainable)
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )


def apply_peft(
    model: Any,
    method: Method,
    lora: LoraConfigArgs,
    architecture: VLMModel,
    paths: ModulePaths,
    trainable: TrainableConfig,
) -> Any:
    if method == "full":
        return model

    if method == "qlora":
        model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(model, lora, architecture, paths, trainable)
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model
