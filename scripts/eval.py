from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vlm_ft.evaluation.pipeline import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on a processed dataset split")
    parser.add_argument("config", type=Path, help="Path to eval YAML config")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
