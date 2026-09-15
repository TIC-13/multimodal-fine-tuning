from __future__ import annotations

import logging
from typing import Iterable

from vlm_ft.config.training import TrainableConfig
from vlm_ft.architectures.base import ModulePaths

logger = logging.getLogger(__name__)


def _iter_named_parameters(model):
    return model.named_parameters()


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def classify_parameter(name: str, paths: ModulePaths) -> str:
    if _matches_prefix(name, paths.projector):
        return "projector"
    if _matches_prefix(name, paths.encoder):
        return "encoder"
    if _matches_prefix(name, paths.llm):
        return "llm"
    return "other"


def apply_trainable_modules(model, paths: ModulePaths, trainable: TrainableConfig) -> dict[str, int]:
    counts = {"encoder": 0, "projector": 0, "llm": 0, "other": 0, "trainable": 0, "frozen": 0}

    for name, param in _iter_named_parameters(model):
        kind = classify_parameter(name, paths)
        counts[kind] += param.numel()
        if kind == "encoder":
            param.requires_grad = trainable.encoder
        elif kind == "projector":
            param.requires_grad = trainable.projector
        elif kind == "llm":
            param.requires_grad = trainable.llm
        else:
            param.requires_grad = trainable.llm

        if param.requires_grad:
            counts["trainable"] += param.numel()
        else:
            counts["frozen"] += param.numel()

    logger.info(
        "Freeze plan llm=%s projector=%s encoder=%s | trainable_params=%s frozen_params=%s",
        trainable.llm,
        trainable.projector,
        trainable.encoder,
        counts["trainable"],
        counts["frozen"],
    )
    return counts


def enable_non_lora_trainable(model, paths: ModulePaths, trainable: TrainableConfig) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            continue
        kind = classify_parameter(name, paths)
        if kind == "encoder":
            param.requires_grad = trainable.encoder
        elif kind == "projector":
            param.requires_grad = trainable.projector
        elif kind == "llm":
            param.requires_grad = False
        else:
            param.requires_grad = False
