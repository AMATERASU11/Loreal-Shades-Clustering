"""
src/pipeline.py
---------------
Classe FaceShadePipeline : orchestre le pipeline de bout en bout.

Usage :
    from pipeline import FaceShadePipeline, load_config
    config = load_config("config/config.yaml")
    pipeline = FaceShadePipeline(config)
    pipeline.run()
"""
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from cleaning.preprocessing import ShadeTextProcessor
from swatch_extractor import SwatchExtractor
from clustering.clusterer import ShadeClusterer
from clustering.evaluator import ClusterEvaluator

logger = logging.getLogger(__name__)


class FaceShadePipeline:
    """
    Orchestre le pipeline complet de clustering des teintes Face.

    Peut s'utiliser en entier (run()) ou étape par étape pour le debug.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.preprocessor = ShadeTextProcessor(config)
        self.extractor = SwatchExtractor(config)
        self.clusterer = ShadeClusterer(config)
        self.evaluator = ClusterEvaluator(config)

    def run(self) -> pd.DataFrame:
        """Lance le pipeline complet."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("DÉMARRAGE — Face Shade Clustering")
        logger.info("=" * 60)

        df = self.load_data()
        df = self.preprocess(df)
        df = self.extract_features(df)
        df = self.cluster(df)
        metrics = self.evaluate(df)
        self.evaluator.report(metrics)
        self.save(df)

        logger.info("Pipeline terminé en %.1f secondes.", time.time() - t0)
        return df

    def load_data(self) -> pd.DataFrame:
        path = self.config["data"]["parquet_path"]
        logger.info("Chargement depuis : %s", path)
        df = pd.read_parquet(path)
        logger.info("DataFrame chargé : %d lignes, %d colonnes", *df.shape)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("--- Étape 1 : Preprocessing ---")
        return self.preprocessor.fit_transform(df)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("--- Étape 2 : Extraction swatches (Lab) ---")
        return self.extractor.transform(df)

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("--- Étape 3 : Clustering ---")
        return self.clusterer.fit_predict(df)

    def evaluate(self, df: pd.DataFrame) -> dict:
        logger.info("--- Étape 4 : Évaluation ---")
        if "shade_cluster_id" in df.columns:
            return self.evaluator.evaluate(df)
        logger.info("Pas de ground truth — métriques internes seulement.")
        return self.evaluator.evaluate_internal(df)

    def save(self, df: pd.DataFrame) -> None:
        out_path = self.config["data"]["output_path"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Résultat sauvegardé : %s", out_path)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
