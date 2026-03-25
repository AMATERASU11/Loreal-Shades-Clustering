"""
src/preprocessing.py
---------------------
Classe ShadeTextProcessor : prépare le DataFrame avant clustering.
"""
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils.text_utils import clean_shade_text, build_shade_key
from utils.image_utils import build_image_path, can_open_image

logger = logging.getLogger(__name__)
tqdm.pandas()


class ShadeTextProcessor:
    """
    Prépare le DataFrame pour le clustering des teintes.

    Responsabilités :
    - Filtrer sur le segment (category_level_2_name + mots-clés)
    - Ajouter les colonnes image_path, image_exists, image_load_ok
    - Nettoyer les shade_name et construire les clés de clustering texte

    Usage :
        processor = ShadeTextProcessor(config)
        df_ready = processor.fit_transform(df_raw)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.images_dir: str = config["data"]["images_dir"]
        self.segment_cfg: dict = config["segment"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complet de preprocessing."""
        logger.info("Preprocessing — %d lignes initiales", len(df))

        df = self._filter_segment(df)
        logger.info("Après filtrage segment : %d lignes", len(df))

        df = self._add_image_columns(df)
        logger.info(
            "Images valides : %d / %d",
            int(df["image_load_ok"].sum()),
            len(df),
        )

        df = self._clean_shade_names(df)
        logger.info("Nettoyage shade_name terminé.")

        return df

    def _filter_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_col = "category_level_2_name"
        cat3_col = "category_level_3_name"
        title_col = "title"

        target_cat = self.segment_cfg.get("category_level_2", "Face")
        keywords = self.segment_cfg.get("keywords", [])

        df_face = df[df[cat_col] == target_cat].copy()

        if keywords:
            pattern = "|".join(keywords)
            mask_cat3 = df_face[cat3_col].astype(str).str.lower().str.contains(
                pattern, na=False
            )
            mask_title = df_face[title_col].astype(str).str.lower().str.contains(
                pattern, na=False
            )
            df_face = df_face[mask_cat3 | mask_title].copy()

        return df_face.reset_index(drop=True)

    def _add_image_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        img_col = "image_filename"
        if img_col not in df.columns:
            logger.warning("Colonne '%s' absente — images désactivées.", img_col)
            df["image_path"] = np.nan
            df["image_exists"] = 0
            df["image_load_ok"] = 0
            return df

        df["image_path"] = df[img_col].apply(
            lambda f: build_image_path(f, self.images_dir)
        )
        df["image_exists"] = df["image_path"].apply(
            lambda p: int(p is not None and os.path.exists(str(p)))
        )
        df["image_load_ok"] = df["image_path"].progress_apply(
            lambda p: int(can_open_image(p))
        )

        n_corrupt = int(
            ((df["image_exists"] == 1) & (df["image_load_ok"] == 0)).sum()
        )
        logger.info("Images corrompues/illisibles : %d", n_corrupt)
        return df

    def _clean_shade_names(self, df: pd.DataFrame) -> pd.DataFrame:
        shade_col = "shade_name"
        if shade_col not in df.columns:
            logger.warning("Colonne '%s' absente.", shade_col)
            df["shade_clean"] = ""
            df["shade_key"] = "missing"
            return df

        df["shade_clean"] = df[shade_col].apply(clean_shade_text)
        df["shade_key"] = df["shade_clean"].apply(build_shade_key)

        pct_missing = 100 * (df["shade_key"] == "missing").mean()
        logger.info("Teintes 'missing' : %.1f%%", pct_missing)
        return df
