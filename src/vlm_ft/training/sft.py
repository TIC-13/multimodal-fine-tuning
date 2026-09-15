from __future__ import annotations

import logging
from typing import Any

from trl import SFTConfig, SFTTrainer

from vlm_ft.config.training import TrainConfig
from vlm_ft.data.collate import build_collate_fn

logger = logging.getLogger(__name__)


def build_sft_config(cfg: TrainConfig, *, has_eval: bool = False) -> SFTConfig:
    t = cfg.training
    kwargs: dict[str, Any] = dict(
        output_dir=t.output_dir,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        logging_steps=t.logging_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        report_to=t.report_to,
        seed=t.seed,
        dataloader_num_workers=t.dataloader_num_workers,
        remove_unused_columns=t.remove_unused_columns,
        push_to_hub=t.push_to_hub,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if has_eval:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = t.eval_steps or t.save_steps
        kwargs["load_best_model_at_end"] = t.load_best_model_at_end
        kwargs["metric_for_best_model"] = t.metric_for_best_model
        kwargs["greater_is_better"] = False

    try:
        return SFTConfig(max_length=cfg.data.max_length, **kwargs)
    except TypeError:
        try:
            return SFTConfig(max_seq_length=cfg.data.max_length, **kwargs)
        except TypeError:
            return SFTConfig(**kwargs)


def build_trainer(
    cfg: TrainConfig,
    model: Any,
    processor: Any,
    train_dataset,
    eval_dataset=None,
) -> SFTTrainer:
    args = build_sft_config(cfg, has_eval=eval_dataset is not None)
    collate_fn = build_collate_fn(
        processor=processor,
        family=cfg.model.family,
        max_length=cfg.data.max_length,
    )

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    try:
        return SFTTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        try:
            return SFTTrainer(tokenizer=processor, **trainer_kwargs)
        except TypeError:
            logger.warning("SFTTrainer did not accept processing_class/tokenizer; continuing without it")
            return SFTTrainer(**trainer_kwargs)
