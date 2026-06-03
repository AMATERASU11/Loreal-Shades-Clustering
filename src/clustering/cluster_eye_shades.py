import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage import color
import warnings
import time

warnings.filterwarnings("ignore")

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_gold.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_clustered.parquet")

# Plage de K à tester pour KMeans et GMM
K_RANGE = range(5, 31)

# DBSCAN : grille de paramètres
DBSCAN_EPS_RANGE = [3, 5, 8, 10, 12, 15]
DBSCAN_MIN_SAMPLES_RANGE = [5, 10, 20, 50]


def rgb_to_lab(rgb_array):
    """Convertit un array Nx3 RGB [0-255] en Lab."""
    rgb_norm = rgb_array.reshape(-1, 1, 3).astype(np.float64) / 255.0
    lab = color.rgb2lab(rgb_norm)
    return lab.reshape(-1, 3)


def prepare_data(df):
    """Prépare les données couleur pour le clustering."""
    # Extraire les smart_color en array
    colors_rgb = np.stack(df['smart_color'].values)

    # Convertir en Lab (meilleur espace perceptuel pour le clustering couleur)
    colors_lab = rgb_to_lab(colors_rgb)

    return colors_rgb, colors_lab


def benchmark_kmeans(X, k_range):
    """Benchmark KMeans sur une plage de K."""
    print("\n📊 Benchmark KMeans")
    print("-" * 60)

    results = []
    for k in k_range:
        t0 = time.time()
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        elapsed = time.time() - t0

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({
            'model': 'KMeans',
            'k': k,
            'silhouette': round(sil, 4),
            'calinski_harabasz': round(ch, 2),
            'davies_bouldin': round(db, 4),
            'n_clusters': k,
            'n_noise': 0,
            'time_s': round(elapsed, 2),
        })
        print(f"   K={k:2d} | Silhouette={sil:.4f} | CH={ch:>10.1f} | DB={db:.4f} | {elapsed:.1f}s")

    return results


def benchmark_gmm(X, k_range):
    """Benchmark Gaussian Mixture Model sur une plage de K."""
    print("\n📊 Benchmark GMM")
    print("-" * 60)

    results = []
    for k in k_range:
        t0 = time.time()
        model = GaussianMixture(n_components=k, random_state=42, n_init=3)
        labels = model.fit_predict(X)
        elapsed = time.time() - t0

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        bic = model.bic(X)
        aic = model.aic(X)

        results.append({
            'model': 'GMM',
            'k': k,
            'silhouette': round(sil, 4),
            'calinski_harabasz': round(ch, 2),
            'davies_bouldin': round(db, 4),
            'n_clusters': k,
            'n_noise': 0,
            'bic': round(bic, 2),
            'aic': round(aic, 2),
            'time_s': round(elapsed, 2),
        })
        print(f"   K={k:2d} | Silhouette={sil:.4f} | CH={ch:>10.1f} | DB={db:.4f} | BIC={bic:>12.1f} | {elapsed:.1f}s")

    return results


def benchmark_dbscan(X, eps_range, min_samples_range):
    """Benchmark DBSCAN sur une grille eps × min_samples."""
    print("\n📊 Benchmark DBSCAN")
    print("-" * 60)

    results = []
    for eps in eps_range:
        for ms in min_samples_range:
            t0 = time.time()
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            elapsed = time.time() - t0

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = n_noise / len(labels) * 100

            if n_clusters < 2:
                print(f"   eps={eps:>4} ms={ms:>3} | {n_clusters} cluster(s), {noise_pct:.1f}% bruit → skip")
                continue

            # Silhouette sur les non-bruit uniquement
            mask = labels != -1
            if mask.sum() < 2:
                continue

            sil = silhouette_score(X[mask], labels[mask])
            ch = calinski_harabasz_score(X[mask], labels[mask])
            db = davies_bouldin_score(X[mask], labels[mask])

            results.append({
                'model': 'DBSCAN',
                'k': f"eps={eps}_ms={ms}",
                'silhouette': round(sil, 4),
                'calinski_harabasz': round(ch, 2),
                'davies_bouldin': round(db, 4),
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': round(noise_pct, 1),
                'eps': eps,
                'min_samples': ms,
                'time_s': round(elapsed, 2),
            })
            print(f"   eps={eps:>4} ms={ms:>3} | K={n_clusters:>3} | bruit={noise_pct:>5.1f}% | Sil={sil:.4f} | CH={ch:>10.1f} | DB={db:.4f}")

    return results


def print_top_results(all_results):
    """Affiche un classement comparatif des meilleurs résultats."""
    df_res = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("🏆 CLASSEMENT COMPARATIF")
    print("=" * 70)

    # Top 5 par Silhouette (plus c'est haut, mieux c'est)
    print("\n📈 Top 5 — Silhouette Score (↑ mieux)")
    top_sil = df_res.nlargest(5, 'silhouette')[['model', 'k', 'n_clusters', 'silhouette', 'davies_bouldin', 'n_noise']]
    print(top_sil.to_string(index=False))

    # Top 5 par Davies-Bouldin (plus c'est bas, mieux c'est)
    print("\n📉 Top 5 — Davies-Bouldin Index (↓ mieux)")
    top_db = df_res.nsmallest(5, 'davies_bouldin')[['model', 'k', 'n_clusters', 'silhouette', 'davies_bouldin', 'n_noise']]
    print(top_db.to_string(index=False))

    # Top 5 par Calinski-Harabasz (plus c'est haut, mieux c'est)
    print("\n📈 Top 5 — Calinski-Harabasz Index (↑ mieux)")
    top_ch = df_res.nlargest(5, 'calinski_harabasz')[['model', 'k', 'n_clusters', 'silhouette', 'calinski_harabasz', 'n_noise']]
    print(top_ch.to_string(index=False))

    # Meilleur par modèle
    print("\n🥇 Meilleur par modèle (Silhouette)")
    for model_name in ['KMeans', 'GMM', 'DBSCAN']:
        sub = df_res[df_res['model'] == model_name]
        if len(sub) > 0:
            best = sub.loc[sub['silhouette'].idxmax()]
            noise_info = f" | bruit={best.get('noise_pct', 0):.1f}%" if best['model'] == 'DBSCAN' else ""
            print(f"   {model_name:8s} → K={best['n_clusters']:>3} | Sil={best['silhouette']:.4f} | DB={best['davies_bouldin']:.4f}{noise_info}")

    return df_res


def main():
    print("🔬 Benchmark Clustering Eye Shades")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fichier introuvable : {INPUT_FILE}")
        print("👉 Lancez d'abord l'inférence (step4_eye_inference_production.py)")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"📂 {len(df)} produits chargés")

    # Préparation des couleurs en Lab
    colors_rgb, colors_lab = prepare_data(df)
    print(f"🎨 Espace Lab : shape={colors_lab.shape}")
    print(f"   L: [{colors_lab[:,0].min():.1f}, {colors_lab[:,0].max():.1f}]")
    print(f"   a: [{colors_lab[:,1].min():.1f}, {colors_lab[:,1].max():.1f}]")
    print(f"   b: [{colors_lab[:,2].min():.1f}, {colors_lab[:,2].max():.1f}]")

    # --- BENCHMARKS ---
    all_results = []

    kmeans_results = benchmark_kmeans(colors_lab, K_RANGE)
    all_results.extend(kmeans_results)

    gmm_results = benchmark_gmm(colors_lab, K_RANGE)
    all_results.extend(gmm_results)

    dbscan_results = benchmark_dbscan(colors_lab, DBSCAN_EPS_RANGE, DBSCAN_MIN_SAMPLES_RANGE)
    all_results.extend(dbscan_results)

    # --- COMPARAISON ---
    df_results = print_top_results(all_results)

    # Sauvegarde du rapport
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "clustering_benchmark_eye.csv")
    df_results.to_csv(report_path, index=False)
    print(f"\n💾 Rapport sauvegardé : {report_path}")

    print("\n" + "=" * 70)
    print("✅ Benchmark terminé !")
    print("👉 Analysez les résultats et choisissez le meilleur modèle/K")


if __name__ == "__main__":
    main()
