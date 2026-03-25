"""
main.py — CLI entrypoint

Usage :
    python main.py --config config/config.yaml
    python main.py --method hybrid
    python main.py --steps preprocess extract --input data/preprocessed.parquet
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import FaceShadePipeline, load_config


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Shade Clustering — L'Oréal Fil Rouge")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "preprocess", "extract", "cluster", "evaluate"],
        default=["all"],
    )
    parser.add_argument("--method", type=str, choices=["text", "image", "hybrid"], default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)

    if args.method:
        config["clustering"]["method"] = args.method
    if args.output:
        config["data"]["output_path"] = args.output

    pipeline = FaceShadePipeline(config)
    steps = args.steps
    run_all = "all" in steps

    if args.input:
        df = pd.read_parquet(args.input)
    elif run_all or "preprocess" in steps:
        df = pipeline.load_data()
    else:
        logger.error("--input requis si --steps ne commence pas par 'preprocess'.")
        sys.exit(1)

    if run_all or "preprocess" in steps:
        df = pipeline.preprocess(df)
    if run_all or "extract" in steps:
        df = pipeline.extract_features(df)
    if run_all or "cluster" in steps:
        df = pipeline.cluster(df)
    if run_all or "evaluate" in steps:
        metrics = pipeline.evaluate(df)
        pipeline.evaluator.report(metrics)
    if run_all or "cluster" in steps:
        pipeline.save(df)

    logger.info("Terminé.")


if __name__ == "__main__":
    main()

