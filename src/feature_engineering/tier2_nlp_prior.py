"""
Tier 2 — NLP Color Prior
═══════════════════════════════════════════════════════════════════
Extrait un prior couleur depuis le texte (shade_name + title)
et le convertit en 6 features numériques pour le XGBoost.

Logique pondérée recyclée de methode_4/pipeline.py :
  - shade_name nettoyé × 3 (poids triple)
  - title extrait COLOR_WORDS × 1
  - description COLOR_WORDS × 0.5 (si confirmé par shade_name)

Output — 6 features :
  [0]   has_color_prior     : 1.0 si couleur détectée, 0.0 sinon
  [1-5] de_prior_c0..c4     : ΔE(prior_lab, centre_cluster_i) pour i=0..4
                              0.0 si has_color_prior == 0 (neutre pour XGBoost)

Le prior est NEUTRE quand absent : le XGBoost apprend à ignorer les
features ΔE quand has_color_prior=0 (grace à l'indicateur binaire).
"""
import re
from typing import Optional

import numpy as np
import pandas as pd
from skimage import color as skcolor


# ═══════════════════════════════════════════════════════════════
#  DICTIONNAIRE COLOR_WORDS → LAB CENTROÏDE
# ═══════════════════════════════════════════════════════════════
# Valeurs Lab calibrées perceptuellement.
# Priorité : couleur la plus courante en cosmétiques.

COLOR_TO_LAB: dict[str, list[float]] = {
    # ── Rouges ──────────────────────────────────────────────
    "red":         [41.0,  52.0,  28.0],
    "rouge":       [41.0,  52.0,  28.0],
    "scarlet":     [38.0,  55.0,  32.0],
    "crimson":     [35.0,  50.0,  22.0],
    "ruby":        [36.0,  48.0,  20.0],
    "cherry":      [37.0,  46.0,  18.0],
    "cerise":      [37.0,  46.0,  18.0],
    "vermillion":  [46.0,  52.0,  38.0],
    "bordeaux":    [28.0,  32.0,  12.0],
    "burgundy":    [26.0,  30.0,  10.0],
    "maroon":      [28.0,  30.0,  10.0],
    "wine":        [28.0,  30.0,  10.0],
    "raspberry":   [40.0,  44.0,   8.0],
    "framboise":   [40.0,  44.0,   8.0],
    "strawberry":  [48.0,  52.0,  25.0],
    "fraise":      [48.0,  52.0,  25.0],
    "cranberry":   [32.0,  40.0,  12.0],
    # ── Roses / Pinks ────────────────────────────────────────
    "pink":        [65.0,  32.0,   2.0],
    "rose":        [65.0,  32.0,   2.0],
    "blush":       [73.0,  18.0,   5.0],
    "mauve":       [55.0,  20.0,  -5.0],
    "coral":       [62.0,  40.0,  28.0],
    "corail":      [62.0,  40.0,  28.0],
    "fuchsia":     [52.0,  58.0, -15.0],
    "magenta":     [48.0,  60.0, -20.0],
    "salmon":      [68.0,  28.0,  20.0],
    "peach":       [76.0,  18.0,  20.0],
    "pêche":       [76.0,  18.0,  20.0],
    "apricot":     [74.0,  20.0,  28.0],
    "abricot":     [74.0,  20.0,  28.0],
    # ── Nudes / Beiges / Marrons ─────────────────────────────
    "nude":        [72.0,  10.0,  12.0],
    "beige":       [78.0,   5.0,  14.0],
    "tan":         [68.0,   8.0,  18.0],
    "taupe":       [60.0,   5.0,   8.0],
    "sand":        [78.0,   4.0,  16.0],
    "sable":       [78.0,   4.0,  16.0],
    "brown":       [40.0,  14.0,  18.0],
    "marron":      [40.0,  14.0,  18.0],
    "chocolate":   [30.0,  14.0,  16.0],
    "chocolat":    [30.0,  14.0,  16.0],
    "caramel":     [55.0,  18.0,  28.0],
    "coffee":      [37.0,  12.0,  16.0],
    "café":        [37.0,  12.0,  16.0],
    "mocha":       [37.0,  12.0,  16.0],
    "espresso":    [22.0,   8.0,  10.0],
    "cinnamon":    [55.0,  22.0,  26.0],
    "ginger":      [58.0,  20.0,  28.0],
    "mahogany":    [32.0,  18.0,  14.0],
    "acajou":      [32.0,  18.0,  14.0],
    "chestnut":    [37.0,  16.0,  16.0],
    "amber":       [68.0,  18.0,  52.0],
    "ambre":       [68.0,  18.0,  52.0],
    "honey":       [72.0,  12.0,  44.0],
    "miel":        [72.0,  12.0,  44.0],
    "rosewood":    [42.0,  25.0,  10.0],
    "terracotta":  [52.0,  24.0,  26.0],
    "sienna":      [48.0,  24.0,  26.0],
    # ── Blancs / Crèmes ──────────────────────────────────────
    "white":       [96.0,  -1.0,   2.0],
    "blanc":       [96.0,  -1.0,   2.0],
    "cream":       [94.0,   1.0,   8.0],
    "crème":       [94.0,   1.0,   8.0],
    "creme":       [94.0,   1.0,   8.0],
    "ivory":       [94.0,   0.5,   6.0],
    "ivoire":      [94.0,   0.5,   6.0],
    "pearl":       [91.0,   0.0,   2.0],
    "perle":       [91.0,   0.0,   2.0],
    # ── Noirs / Gris ─────────────────────────────────────────
    "black":       [ 8.0,   0.5,   0.5],
    "noir":        [ 8.0,   0.5,   0.5],
    "charcoal":    [28.0,   0.0,   0.0],
    "graphite":    [40.0,   0.0,   0.0],
    "grey":        [55.0,   0.0,   0.0],
    "gray":        [55.0,   0.0,   0.0],
    "gris":        [55.0,   0.0,   0.0],
    "pewter":      [44.0,   0.0,   0.0],
    # ── Bleus ────────────────────────────────────────────────
    "blue":        [32.0,  12.0, -55.0],
    "bleu":        [32.0,  12.0, -55.0],
    "navy":        [20.0,   8.0, -35.0],
    "marine":      [20.0,   8.0, -35.0],
    "cobalt":      [28.0,  18.0, -60.0],
    "indigo":      [22.0,  14.0, -45.0],
    "periwinkle":  [50.0,  10.0, -32.0],
    "teal":        [47.0, -25.0, -18.0],
    "turquoise":   [57.0, -28.0, -14.0],
    "aqua":        [60.0, -24.0, -16.0],
    "cyan":        [62.0, -22.0, -22.0],
    "mint":        [78.0, -20.0,   5.0],
    "menthe":      [78.0, -20.0,   5.0],
    # ── Verts ────────────────────────────────────────────────
    "green":       [46.0, -35.0,  28.0],
    "vert":        [46.0, -35.0,  28.0],
    "emerald":     [42.0, -34.0,  16.0],
    "émeraude":    [42.0, -34.0,  16.0],
    "jade":        [50.0, -28.0,  10.0],
    "sage":        [60.0, -15.0,  10.0],
    "olive":       [50.0, -12.0,  24.0],
    "forest":      [36.0, -28.0,  18.0],
    "moss":        [50.0, -14.0,  18.0],
    "khaki":       [56.0,  -8.0,  20.0],
    # ── Jaunes / Oranges ─────────────────────────────────────
    "yellow":      [88.0,  -8.0,  78.0],
    "jaune":       [88.0,  -8.0,  78.0],
    "lemon":       [90.0, -10.0,  74.0],
    "citron":      [90.0, -10.0,  74.0],
    "lime":        [82.0, -18.0,  58.0],
    "mustard":     [67.0,  -2.0,  50.0],
    "moutarde":    [67.0,  -2.0,  50.0],
    "ochre":       [58.0,   8.0,  44.0],
    "orange":      [65.0,  34.0,  54.0],
    "gold":        [72.0,   8.0,  48.0],
    "or":          [72.0,   8.0,  48.0],
    # ── Violets / Pourpres ───────────────────────────────────
    "purple":      [37.0,  28.0, -28.0],
    "violet":      [37.0,  28.0, -28.0],
    "pourpre":     [37.0,  28.0, -28.0],
    "lilac":       [68.0,  18.0, -20.0],
    "lilas":       [68.0,  18.0, -20.0],
    "lavender":    [72.0,  12.0, -14.0],
    "lavande":     [72.0,  12.0, -14.0],
    "plum":        [33.0,  24.0, -18.0],
    "prune":       [33.0,  24.0, -18.0],
    "berry":       [37.0,  38.0, -10.0],
    # ── Métalliques ──────────────────────────────────────────
    "silver":      [72.0,   0.0,   0.0],
    "argent":      [72.0,   0.0,   0.0],
    "bronze":      [50.0,  12.0,  24.0],
    "copper":      [54.0,  20.0,  28.0],
    "cuivre":      [54.0,  20.0,  28.0],
    "champagne":   [84.0,   2.0,  10.0],
    "titanium":    [54.0,   0.0,   0.0],
    "platinum":    [78.0,   0.0,   0.0],
    # ── Autres ───────────────────────────────────────────────
    "dusty":       [65.0,   8.0,   5.0],
    "pastel":      [85.0,   8.0,   3.0],
    "neon":        [85.0,  -5.0,  55.0],
}

# Regex construite dynamiquement depuis les clés du dictionnaire
_COLOR_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in COLOR_TO_LAB.keys()) + r')\b',
    re.IGNORECASE
)

# Nettoyage shade_name (recycled from methode_4/pipeline.py clean_shade_name)
_RE_SHADE_CLEAN = [
    (re.compile(r'#\d+'),                                ''),    # #306
    (re.compile(r'\b[A-Z]?\d{3,}(-\d+m?l?)?\b', re.I), ''),    # codes 3+ chiffres
    (re.compile(r'\d+(\.\d+)?\s*(ml|oz|fl|g|cm)\b', re.I), ''), # tailles
    (re.compile(r'\(pack of \d+\)', re.I),               ''),    # (pack of N)
    (re.compile(r'\blot de \d+\b', re.I),                ''),    # lot de N
    (re.compile(r'\s+'),                                  ' '),   # espaces multiples
]


def _clean_shade(name: str) -> str:
    """Nettoyage léger du shade_name pour l'extraction couleur."""
    s = str(name).lower().strip()
    for pattern, repl in _RE_SHADE_CLEAN:
        s = pattern.sub(repl, s)
    return s.strip()


def _extract_color_lab(shade: str, title: str, desc: str) -> Optional[np.ndarray]:
    """
    Cherche un mot de couleur dans le texte pondéré :
      shade_name × 3 + title × 1 + desc × 0.5

    Retourne le Lab centroïde de la première couleur trouvée,
    ou None si aucune couleur détectée.
    """
    shade_clean = _clean_shade(shade)

    # Ordre de priorité : shade (×3), title, desc
    texts_by_priority = [
        shade_clean,   # priorité 1
        shade_clean,   # priorité 2 (pondération ×3)
        shade_clean,   # priorité 3
        title.lower(),
        desc.lower()[:200],
    ]

    for text in texts_by_priority:
        m = _COLOR_PATTERN.search(text)
        if m:
            word = m.group(0).lower()
            # Chercher dans le dictionnaire (insensible à la casse)
            lab = COLOR_TO_LAB.get(word)
            if lab is not None:
                return np.array(lab, dtype=np.float64)

    return None


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def _delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Distance CIE76."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


# ═══════════════════════════════════════════════════════════════
#  INTERFACE PUBLIQUE
# ═══════════════════════════════════════════════════════════════

def extract_features(
    row: pd.Series,
    centers_lab: np.ndarray,  # shape (K, 3), trié par poids décroissant
) -> np.ndarray:
    """
    Calcule les 6 features NLP pour une ligne du DataFrame.

    Args:
        row         : Ligne avec shade_name, title, description
        centers_lab : Centres KMeans Lab triés (K, 3)

    Returns:
        features : np.ndarray shape (6,)
          [0]   has_color_prior  (0.0 ou 1.0)
          [1-5] de_prior_ci      ΔE normalisé [0, 1] pour chaque cluster
                                 0.0 si has_color_prior == 0
    """
    K = len(centers_lab)

    shade = str(row.get("shade_name",  "") or "")
    title = str(row.get("title",       "") or "")
    desc  = str(row.get("description", "") or "")

    prior_lab = _extract_color_lab(shade, title, desc)

    features = np.zeros(1 + K, dtype=np.float32)

    if prior_lab is None:
        # has_color_prior = 0, tous les ΔE = 0 (neutre)
        return features

    features[0] = 1.0  # has_color_prior

    # ΔE normalisé : ΔE max théorique ≈ 170 (Lab space diagonal)
    # On normalise par 50 (seuil perceptuel raisonnable)
    DE_NORM = 50.0
    for i, center in enumerate(centers_lab):
        de = _delta_e(prior_lab, center)
        features[1 + i] = min(de / DE_NORM, 1.0)

    return features


def get_color_word(row: pd.Series) -> Optional[str]:
    """Retourne le mot de couleur détecté (pour debug/log)."""
    shade = _clean_shade(str(row.get("shade_name", "") or ""))
    title = str(row.get("title", "") or "").lower()
    desc  = str(row.get("description", "") or "").lower()[:200]

    for text in [shade, shade, shade, title, desc]:
        m = _COLOR_PATTERN.search(text)
        if m:
            return m.group(0).lower()
    return None
