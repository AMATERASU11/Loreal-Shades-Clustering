"""
Tests unitaires pour ShadeClusterer.
Lancez avec : pytest tests/test_clusterer.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clustering.clusterer import ShadeClusterer

BASE_CONFIG = {
    "clustering": {
        "method": "text",
        "k_max": 4,
        "min_group_size": 2,
        "random_state": 42,
        "delta_e_threshold": 8.0,
        "text_weight": 0.5,
        "image_weight": 0.5,
    }
}


def make_df(cluster_ids, shade_keys, L=None, a=None, b=None):
    n = len(cluster_ids)
    return pd.DataFrame({
        "cluster_id": cluster_ids,
        "shade_key": shade_keys,
        "image_load_ok": [1] * n,
        "L": L if L else [50.0] * n,
        "a": a if a else [128.0] * n,
        "b": b if b else [128.0] * n,
    })


class TestTextClustering:
    def setup_method(self):
        config = {**BASE_CONFIG, "clustering": {**BASE_CONFIG["clustering"], "method": "text"}}
        self.clusterer = ShadeClusterer(config)

    def test_same_key_same_cluster(self):
        df = make_df([1, 1, 1, 1], ["fair", "fair", "rose", "rose"])
        result = self.clusterer.fit_predict(df)
        labels = result["shade_cluster_id_pred"].values
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_output_column_exists(self):
        df = make_df([1, 1], ["a", "b"])
        result = self.clusterer.fit_predict(df)
        assert "shade_cluster_id_pred" in result.columns


class TestThresholdClustering:
    def test_high_sim_same_cluster(self):
        sim = np.array([[1.0, 0.9], [0.9, 1.0]])
        labels = ShadeClusterer._threshold_clustering(sim, 0.5)
        assert labels[0] == labels[1]

    def test_low_sim_different_clusters(self):
        sim = np.array([[1.0, 0.1], [0.1, 1.0]])
        labels = ShadeClusterer._threshold_clustering(sim, 0.5)
        assert labels[0] != labels[1]

    def test_three_nodes(self):
        sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        labels = ShadeClusterer._threshold_clustering(sim, 0.5)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_labels_contiguous(self):
        sim = np.eye(4)
        labels = ShadeClusterer._threshold_clustering(sim, 0.5)
        assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3]