"""
src/evaluator.py
----------------
Classe ClusterEvaluator : métriques de qualité du clustering.

Métriques :
- ARI (Adjusted Rand Index)
- Pureté
- Delta E moyen intra-cluster
- Delta E moyen inter-cluster
- Ratio inter/intra
- Silhouette score
- Cohérence texte/image
- Distribution tailles
- % clusters multi-pays (KPI business)
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Évalue la qualité du clustering de teintes.

    Usage avec ground truth :
        evaluator = ClusterEvaluator(config)
        metrics = evaluator.evaluate(df)
        evaluator.report(metrics)

    Usage sans ground truth :
        metrics = evaluator.evaluate_internal(df)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        cfg = config.get("evaluation", {})
        self.min_ari: float = cfg.get("min_ari_score", 0.5)
        self.min_purity: float = cfg.get("min_purity", 0.7)
        self.min_silhouette: float = cfg.get("min_silhouette", 0.3)
        self.min_text_consistency: float = cfg.get("min_text_consistency", 70.0)
        self.min_ratio_inter_intra: float = cfg.get("min_ratio_inter_intra", 2.0)

    def evaluate(
        self,
        df: pd.DataFrame,
        true_col: str = "shade_cluster_id",
        pred_col: str = "shade_cluster_id_pred",
    ) -> Dict[str, float]:
        """Évaluation complète avec ground truth."""
        metrics = {}

        valid = df[[true_col, pred_col]].dropna()
        valid = valid[valid[true_col] != -1]

        if len(valid) < 10:
            logger.warning("Pas assez de données avec ground truth.")
            return self.evaluate_internal(df, pred_col)

        y_true = valid[true_col].astype(str).values
        y_pred = valid[pred_col].astype(str).values

        metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
        metrics["purity"] = float(self._compute_purity(y_true, y_pred))
        metrics["n_samples"] = int(len(valid))
        metrics["n_clusters_true"] = int(len(np.unique(y_true)))
        metrics["n_clusters_pred"] = int(len(np.unique(y_pred)))

        metrics.update(self.evaluate_internal(df, pred_col))
        return metrics

    def evaluate_internal(
        self,
        df: pd.DataFrame,
        pred_col: str = "shade_cluster_id_pred",
    ) -> Dict[str, float]:
        """Métriques internes (sans ground truth)."""
        metrics = {}

        df_valid = df.dropna(subset=["L", "a", "b"]).copy()

        # Delta E intra
        if all(c in df.columns for c in ["L", "a", "b", pred_col]):
            metrics["mean_intra_delta_e"] = float(
                self._mean_intra_cluster_delta_e(df_valid, pred_col)
            )

        # Delta E inter + ratio
        if all(c in df.columns for c in ["L", "a", "b", pred_col, "cluster_id"]):
            inter = self._mean_inter_cluster_delta_e(df_valid, pred_col)
            metrics["mean_inter_delta_e"] = float(inter)
            intra = metrics.get("mean_intra_delta_e", 1e-6)
            metrics["ratio_inter_intra"] = float(inter / max(intra, 1e-6))

        # Silhouette score
        if all(c in df.columns for c in ["L", "a", "b", pred_col, "cluster_id"]):
            metrics["silhouette_score"] = float(
                self._compute_silhouette(df_valid, pred_col)
            )

        # Cohérence texte/image
        if all(c in df.columns for c in ["shade_key", pred_col, "cluster_id"]):
            metrics["text_consistency_pct"] = float(
                self._text_consistency(df_valid, pred_col)
            )

        # Distribution tailles
        if pred_col in df.columns and "cluster_id" in df.columns:
            sizes = df.groupby(["cluster_id", pred_col]).size()
            metrics["mean_cluster_size"] = float(sizes.mean())
            metrics["median_cluster_size"] = float(sizes.median())
            metrics["pct_singleton_clusters"] = float(100 * (sizes == 1).mean())

        # Couverture
        metrics["pct_color_extracted"] = float(
            100 * df["L"].notna().mean()
        )

        # % clusters multi-pays
        if "country_name" in df.columns and pred_col in df.columns:
            metrics["pct_clusters_multi_country"] = float(
                self._pct_multi_country_clusters(df, pred_col)
            )

        return metrics

    def report(self, metrics: Dict[str, float]) -> None:
        """Affiche un rapport lisible avec verdict pass/fail."""
        print("\n" + "=" * 55)
        print("         RAPPORT — SHADE CLUSTERING")
        print("=" * 55)

        # Ground truth metrics
        if "ari" in metrics:
            ari = metrics["ari"]
            status = "✓ PASS" if ari >= self.min_ari else "✗ FAIL"
            print(f"  ARI                        : {ari:.4f}  {status}")

        if "purity" in metrics:
            pur = metrics["purity"]
            status = "✓ PASS" if pur >= self.min_purity else "✗ FAIL"
            print(f"  Pureté                     : {pur:.4f}  {status}")

        # Métriques colorimétriques
        print()
        if "mean_intra_delta_e" in metrics:
            de = metrics["mean_intra_delta_e"]
            status = "✓" if de < 10 else "✗"
            print(f"  Delta E moyen intra        : {de:.2f}  {status}  (< 10 = bon)")

        if "mean_inter_delta_e" in metrics:
            print(f"  Delta E moyen inter        : {metrics['mean_inter_delta_e']:.2f}")

        if "ratio_inter_intra" in metrics:
            ratio = metrics["ratio_inter_intra"]
            status = "✓" if ratio >= self.min_ratio_inter_intra else "✗"
            print(f"  Ratio inter/intra          : {ratio:.2f}  {status}  (> 2 = bon)")

        if "silhouette_score" in metrics:
            sil = metrics["silhouette_score"]
            status = "✓" if sil >= self.min_silhouette else "✗"
            print(f"  Silhouette score           : {sil:.4f}  {status}  (> 0.3 = bon)")

        # Cohérence texte
        print()
        if "text_consistency_pct" in metrics:
            pct = metrics["text_consistency_pct"]
            status = "✓" if pct >= self.min_text_consistency else "✗"
            print(f"  Cohérence texte/image      : {pct:.1f}%  {status}  (> 70% = bon)")

        # Distribution
        print()
        if "mean_cluster_size" in metrics:
            print(
                f"  Taille moy. cluster        : {metrics['mean_cluster_size']:.1f} "
                f"(médiane : {metrics.get('median_cluster_size', 0):.1f})"
            )

        if "pct_singleton_clusters" in metrics:
            pct = metrics["pct_singleton_clusters"]
            status = "✓" if pct < 50 else "✗"
            print(f"  Clusters singleton (n=1)   : {pct:.1f}%  {status}  (< 50% = bon)")

        # Couverture
        print()
        if "pct_color_extracted" in metrics:
            print(f"  Couverture couleur         : {metrics['pct_color_extracted']:.1f}%")

        if "pct_clusters_multi_country" in metrics:
            print(f"  Clusters multi-pays        : {metrics['pct_clusters_multi_country']:.1f}%")

        print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n = len(y_pred)
        total_correct = 0
        for c in np.unique(y_pred):
            mask = y_pred == c
            true_in_cluster = y_true[mask]
            counts = np.unique(true_in_cluster, return_counts=True)[1]
            total_correct += counts.max()
        return total_correct / n if n > 0 else 0.0

    @staticmethod
    def _mean_intra_cluster_delta_e(df: pd.DataFrame, pred_col: str) -> float:
        delta_es = []
        df_valid = df[["L", "a", "b", pred_col, "cluster_id"]].dropna()

        for (cid, sub_id), group in df_valid.groupby(["cluster_id", pred_col]):
            if len(group) < 2:
                continue
            lab = group[["L", "a", "b"]].values[:20]
            for i in range(len(lab)):
                for j in range(i + 1, len(lab)):
                    delta_es.append(float(np.sqrt(np.sum((lab[i] - lab[j]) ** 2))))

        return float(np.mean(delta_es)) if delta_es else 0.0

    @staticmethod
    def _mean_inter_cluster_delta_e(df: pd.DataFrame, pred_col: str) -> float:
        inter_des = []
        for cid, group in df.groupby("cluster_id"):
            centroids = group.groupby(pred_col)[["L", "a", "b"]].mean()
            if len(centroids) < 2:
                continue
            vals = centroids.values
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    inter_des.append(float(np.sqrt(np.sum((vals[i] - vals[j]) ** 2))))
        return float(np.mean(inter_des)) if inter_des else 0.0

    @staticmethod
    def _compute_silhouette(df: pd.DataFrame, pred_col: str) -> float:
        # Garder uniquement les cluster_id avec plusieurs shade clusters
        df_multi = df.groupby("cluster_id").filter(
            lambda x: x[pred_col].nunique() > 1
        )
        if len(df_multi) < 100:
            return 0.0
        # Échantillonnage pour la vitesse
        sample = df_multi.sample(min(5000, len(df_multi)), random_state=42)
        try:
            return float(silhouette_score(
                sample[["L", "a", "b"]].values,
                sample[pred_col].values,
            ))
        except Exception:
            return 0.0

    @staticmethod
    def _text_consistency(df: pd.DataFrame, pred_col: str) -> float:
        df_text = df[df["shade_key"] != "missing"].copy()
        total_pairs = 0
        correct_pairs = 0
        for cid, group in df_text.groupby("cluster_id"):
            for key, subgroup in group.groupby("shade_key"):
                if len(subgroup) < 2:
                    continue
                preds = subgroup[pred_col].values
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        total_pairs += 1
                        if preds[i] == preds[j]:
                            correct_pairs += 1
        if total_pairs == 0:
            return 0.0
        return 100.0 * correct_pairs / total_pairs

    @staticmethod
    def _pct_multi_country_clusters(df: pd.DataFrame, pred_col: str) -> float:
        if "cluster_id" not in df.columns:
            return 0.0
        groups = df.groupby(["cluster_id", pred_col])["country_name"].nunique()
        total = len(groups)
        if total == 0:
            return 0.0
        return 100.0 * (groups > 1).sum() / total