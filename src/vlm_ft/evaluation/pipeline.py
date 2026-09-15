from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vlm_ft.data.dataset import load_processed_dataset
from vlm_ft.evaluation.config import EvalConfig, load_eval_config
from vlm_ft.evaluation.generate import generate_response
from vlm_ft.evaluation.labels import extract_gold_labels, parse_diagnosis, parse_malignancy
from vlm_ft.evaluation.load import load_eval_model
from vlm_ft.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class _EvaluationPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.cfg: EvalConfig = load_eval_config(config_path)

    def run(self) -> Path:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model family=%s id=%s checkpoint=%s", self.cfg.model.family, self.cfg.model.model_id, self.cfg.model.checkpoint)
        model, processor = load_eval_model(self.cfg.model)

        logger.info("Loading %s split from %s", self.cfg.data.split, self.cfg.data.path)
        dataset = load_processed_dataset(
            self.cfg.data.path,
            split=self.cfg.data.split,
            image_root=self.cfg.data.image_root,
        )

        predictions: list[dict[str, Any]] = []
        gold_diagnosis: list[str] = []
        pred_diagnosis: list[str] = []
        gold_malignancy: list[str | None] = []
        pred_malignancy: list[str | None] = []

        for idx, example in enumerate(dataset):
            messages = example["messages"]
            gold = extract_gold_labels(messages)
            generation = generate_response(
                model,
                processor,
                messages,
                self.cfg.generation,
                max_length=self.cfg.data.max_length,
            )
            pred_diag = parse_diagnosis(generation)
            pred_mal = parse_malignancy(generation)

            gold_diagnosis.append(gold["diagnosis"])
            pred_diagnosis.append(pred_diag)
            gold_malignancy.append(gold["malignancy"])
            pred_malignancy.append(pred_mal)

            predictions.append(
                {
                    "index": idx,
                    "gold_diagnosis": gold["diagnosis"],
                    "pred_diagnosis": pred_diag,
                    "gold_malignancy": gold["malignancy"],
                    "pred_malignancy": pred_mal,
                    "generation": generation,
                }
            )
            if (idx + 1) % 10 == 0:
                logger.info("Processed %s / %s samples", idx + 1, len(dataset))

        metrics = compute_metrics(gold_diagnosis, pred_diagnosis, gold_malignancy, pred_malignancy)

        predictions_path = output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        metrics_path = output_dir / "eval_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logger.info("Evaluation complete -> %s", metrics_path)
        logger.info(
            "Diagnosis accuracy=%.4f macro_f1=%.4f unparsed=%s",
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["num_unparsed"],
        )
        return output_dir


def run_evaluation(config_path: str | Path) -> Path:
    return _EvaluationPipeline(config_path).run()
