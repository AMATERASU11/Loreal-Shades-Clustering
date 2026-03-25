"""
src/clusterer.py
----------------
Classe ShadeClusterer : sous-clustering des teintes dans chaque cluster_id.

Méthodes : "text", "image", "hybrid"
"""
import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from utils.colors_utils import normalize_lab_for_clustering, delta_e_cie76

logger = logging.getLogger(__name__)


class ShadeClusterer:
    """
    Sous-clustering des teintes à l'intérieur de chaque cluster_id.

    Usage :
        clusterer = ShadeClusterer(config)
        df = clusterer.fit_predict(df)   # ajoute shade_cluster_id_pred
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        cfg = config["clustering"]
        self.method: str = cfg.get("method", "hybrid")
        self.k_max: int = cfg.get("k_max", 8)
        self.min_group_size: int = cfg.get("min_group_size", 3)
        self.random_state: int = cfg.get("random_state", 42)
        self.delta_e_thr: float = cfg.get("delta_e_threshold", 8.0)
        self.text_weight: float = cfg.get("text_weight", 0.5)
        self.image_weight: float = cfg.get("image_weight", 0.5)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique le clustering sur tout le DataFrame."""
        df = df.copy()
        df["shade_cluster_id_pred"] = -1

        counts = df.groupby("cluster_id").size()
        candidates = counts[counts >= self.min_group_size].index.tolist()

        logger.info(
            "Clustering [%s] sur %d cluster_id", self.method, len(candidates)
        )

        for cid in tqdm(candidates, desc="Shade clustering"):
            mask = df["cluster_id"] == cid
            sub = df[mask].copy()
            labels = self._cluster_one(sub)
            if labels is not None:
                df.loc[mask, "shade_cluster_id_pred"] = labels

        # Produits non traités (< min_group_size) → label 0
        df.loc[df["shade_cluster_id_pred"] == -1, "shade_cluster_id_pred"] = 0

        logger.info("Clustering terminé.")
        return df

    def _cluster_one(self, sub: pd.DataFrame) -> np.ndarray:
        if self.method == "text":
            return self._cluster_text(sub)
        if self.method == "image":
            return self._cluster_image(sub)
        if self.method == "hybrid":
            return self._cluster_hybrid(sub)
        raise ValueError(f"Méthode inconnue : {self.method}")

    # ------------------------------------------------------------------
    # Méthode texte
    # ------------------------------------------------------------------

    def _cluster_text(self, sub: pd.DataFrame) -> np.ndarray:
        if "shade_key" not in sub.columns:
            return np.zeros(len(sub), dtype=int)
        labels, _ = pd.factorize(sub["shade_key"].fillna("missing"))
        return labels.astype(int)

    # ------------------------------------------------------------------
    # Méthode image
    # ------------------------------------------------------------------

    def _cluster_image(self, sub: pd.DataFrame) -> np.ndarray:
        labels = np.full(len(sub), -1, dtype=int)
        
        # Filtrer à la fois image_load_ok ET pas de NaN sur L,a,b
        valid_mask = (sub["image_load_ok"] == 1) & sub[["L", "a", "b"]].notna().all(axis=1)
        valid_idx = valid_mask.values
        X = sub.loc[valid_mask, ["L", "a", "b"]].values

        if len(X) < 2:
            labels[:] = 0
            return labels

        X_weighted = normalize_lab_for_clustering(X)
        k = self._choose_k(X_weighted)
        if k == 1:
            labels[valid_idx] = 0
        else:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels[valid_idx] = km.fit_predict(X_weighted)

        # Fallback texte pour les produits sans image
        no_img_mask = ~valid_idx
        if no_img_mask.any():
            text_labels = self._cluster_text(sub)
            labels[no_img_mask] = text_labels[no_img_mask]

        return labels

    # ------------------------------------------------------------------
    # Méthode hybride
    # ------------------------------------------------------------------

    def _cluster_hybrid(self, sub: pd.DataFrame) -> np.ndarray:
        n = len(sub)
        if n < 2:
            return np.zeros(n, dtype=int)

        sim_text = self._build_text_similarity_matrix(sub)
        sim_image = self._build_image_similarity_matrix(sub)
        sim_hybrid = self.text_weight * sim_text + self.image_weight * sim_image

        return self._threshold_clustering(sim_hybrid, threshold=0.5)

    def _build_text_similarity_matrix(self, sub: pd.DataFrame) -> np.ndarray:
        n = len(sub)
        keys = sub["shade_key"].fillna("missing").values
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                sim[i, j] = 1.0 if keys[i] == keys[j] else 0.0
        return sim

    def _build_image_similarity_matrix(self, sub: pd.DataFrame) -> np.ndarray:
        n = len(sub)
        sim = np.zeros((n, n), dtype=float)
        lab_values = sub[["L", "a", "b"]].values
        has_color = ~np.isnan(lab_values).any(axis=1)

        for i in range(n):
            for j in range(n):
                if i == j:
                    sim[i, j] = 1.0
                elif has_color[i] and has_color[j]:
                    de = delta_e_cie76(lab_values[i], lab_values[j])
                    sim[i, j] = max(0.0, 1.0 - de / self.delta_e_thr)
        return sim

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _choose_k(self, X: np.ndarray) -> int:
        X_rounded = np.round(X, 2)
        n_unique = len(np.unique(X_rounded, axis=0))
        if n_unique <= 1:
            return 1
        return min(self.k_max, n_unique, len(X))

    @staticmethod
    def _threshold_clustering(sim: np.ndarray, threshold: float) -> np.ndarray:
        """
        Clustering par composantes connexes (union-find) sur une matrice
        de similarité. Deux éléments sont reliés si sim >= threshold.
        """
        n = sim.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        roots = [find(i) for i in range(n)]
        unique_roots = sorted(set(roots))
        root_to_label = {r: idx for idx, r in enumerate(unique_roots)}
        return np.array([root_to_label[r] for r in roots], dtype=int)
