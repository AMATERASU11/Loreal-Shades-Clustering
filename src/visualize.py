import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

df = pd.read_parquet('data/processed/output_shade_clusters.parquet')

# Prendre un cluster_id avec plusieurs shade clusters et plusieurs produits
stats = df.groupby('cluster_id').agg(
    n_products=('product_id', 'count'),
    n_shade_clusters=('shade_cluster_id_pred', 'nunique'),
    n_with_image=('image_load_ok', 'sum')
).reset_index()

# Chercher un cluster_id avec 3-6 shade clusters et beaucoup d images
good = stats[(stats['n_shade_clusters'].between(3,6)) & (stats['n_with_image'] >= 6)]
cid = good.sort_values('n_shade_clusters', ascending=False).iloc[0]['cluster_id']

sub = df[(df['cluster_id'] == cid) & (df['image_load_ok'] == 1)].copy()
sub = sub.dropna(subset=['L', 'a', 'b'])

print(f'cluster_id : {cid}')
print(f'Nombre de produits : {len(sub)}')
print(f'Nombre de shade clusters : {sub["shade_cluster_id_pred"].nunique()}')
print()

# Afficher quelques images par shade cluster
n_clusters = sub['shade_cluster_id_pred'].nunique()
fig, axes = plt.subplots(n_clusters, 5, figsize=(15, 3 * n_clusters))

for i, (sid, group) in enumerate(sub.groupby('shade_cluster_id_pred')):
    sample = group.head(5)
    for j in range(5):
        ax = axes[i, j] if n_clusters > 1 else axes[j]
        if j < len(sample):
            row = sample.iloc[j]
            try:
                img = Image.open(row['image_path']).convert('RGB')
                ax.imshow(img)
                ax.set_title(str(row['shade_name'])[:20], fontsize=7)
            except:
                ax.text(0.5, 0.5, 'Error', ha='center')
        ax.axis('off')
    # Label du cluster
    axes[i, 0].set_ylabel(f'Cluster {int(sid)}', fontsize=10, rotation=0, labelpad=60)

plt.suptitle(f'cluster_id={cid} — Shade clusters par couleur image', fontsize=13)
plt.tight_layout()
plt.savefig('data/processed/visualisation_shade_clusters.png', dpi=150, bbox_inches='tight')
print('Image sauvegardée : data/processed/visualisation_shade_clusters.png')

