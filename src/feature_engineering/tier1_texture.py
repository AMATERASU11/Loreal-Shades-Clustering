"""
Tier 1 — Texture Detector
═══════════════════════════════════════════════════════════════════
Détecte les textures complexes (glitter, shimmer, holographique...)
et les met en quarantaine.

Décision métier (irrévocable) :
  Option A — Quarantaine pure : les vernis à texture complexe
  ne sont PAS traités par KMeans/XGBoost.
  Raison : base transparente avec confettis, base noire + paillettes
  multicolores → l'extraction de la base détruirait le clustering.

Source recyclée : FINISH_WORDS de methode_4/pipeline.py

Retourne :
  'texture_complexe' → [TAG_TEXTURE] — exclu du pipeline
  'normal'           → passe au Tier 2
"""
import re
from typing import Literal

import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  PATTERNS DE QUARANTAINE
# ═══════════════════════════════════════════════════════════════

# Termes déclenchant la quarantaine — 1 match = exclu
# Recyclé de methode_4/pipeline.py FINISH_WORDS, enrichi
_QUARANTINE = re.compile(
    r'\b('
    # Paillettes / particules
    r'glitter|shimmer|sparkle|spangle|tinsel|sequin'
    # Effets optiques complexes
    r'|holograph(?:ic)?|holo|iridescent|duochrome|multichrome'
    r'|aurora|galaxy|chameleon|colour.changing|color.changing'
    # Effets magnétiques / cat eye
    r'|cat[\s\-]?eye|magnetic|magnet(?:ic)?'
    # Effets miroir / chrome
    r'|chrome|mirror|foil|metallic\s+foil'
    # Flocons
    r'|flake|flakie|jelly\s+flake|nail\s+art\s+flake'
    r')\b',
    re.IGNORECASE
)

# Termes NON problématiques (faux positifs potentiels → on les ignore)
# "metallic" seul : souvent extractible → pas en quarantaine
# "pearl" seul    : souvent extractible → pas en quarantaine
# "matte", "satin", "glossy", "cream", "sheer", "opaque" → toujours OK

# ── Exception : si shade_name contient uniquement un terme technique ──
# Ex : "Mirror" comme shade_name créatif sans autre contexte
# On vérifie que le terme est dans un contexte descriptif (title ou desc)
# et pas juste un nom d'artiste ou de collection
_CONTEXT_NEEDED = re.compile(
    r'\b(mirror|foil)\b',
    re.IGNORECASE
)

# Mots qui confirment que c'est bien une texture complexe (pas juste un nom)
_TEXTURE_CONTEXT = re.compile(
    r'\b(nail\s*polish|vernis|gel|polish|nail|effect|finish|coat|ongles?)\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════
#  INTERFACE PUBLIQUE
# ═══════════════════════════════════════════════════════════════

TextureType = Literal["texture_complexe", "normal"]


def classify(row: pd.Series) -> TextureType:
    """
    Classifie un produit en 'texture_complexe' ou 'normal'.

    Cherche dans shade_name + title + description (les 3 colonnes).
    1 match suffit pour la quarantaine.

    Args:
        row: Ligne du DataFrame (shade_name, title, description)

    Returns:
        'texture_complexe' | 'normal'
    """
    shade = str(row.get("shade_name",  "") or "")
    title = str(row.get("title",       "") or "")
    desc  = str(row.get("description", "") or "")[:300]  # tronqué pour perf

    # Chercher d'abord dans shade + title (plus fiables)
    shade_title = f"{shade} {title}"
    m = _QUARANTINE.search(shade_title)
    if m:
        word = m.group(0).lower()
        # Pour "mirror" et "foil" : vérifier contexte
        if _CONTEXT_NEEDED.match(word):
            if _TEXTURE_CONTEXT.search(shade_title):
                return "texture_complexe"
            # Pas de contexte → peut-être juste un nom de teinte
        else:
            return "texture_complexe"

    # Chercher dans la description (secondaire)
    if _QUARANTINE.search(desc):
        m_desc = _QUARANTINE.search(desc)
        word = m_desc.group(0).lower()
        if _CONTEXT_NEEDED.match(word):
            if _TEXTURE_CONTEXT.search(desc):
                return "texture_complexe"
        else:
            return "texture_complexe"

    return "normal"


def get_detected_keyword(row: pd.Series) -> str | None:
    """Retourne le mot-clé texture qui a déclenché la détection (debug)."""
    full = " ".join([
        str(row.get("shade_name",  "") or ""),
        str(row.get("title",       "") or ""),
        str(row.get("description", "") or "")[:300],
    ])
    m = _QUARANTINE.search(full)
    return m.group(0) if m else None


def apply_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la classification Tier 1 sur tout un DataFrame.
    Ajoute la colonne 'tier1_tag'.
    """
    df = df.copy()
    df["tier1_tag"] = df.apply(classify, axis=1)
    return df


def filter_normal(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les produits 'normal' après Tier 1."""
    df = apply_to_dataframe(df)
    n_texture = (df["tier1_tag"] == "texture_complexe").sum()
    print(f"Tier 1 — exclus : {n_texture} textures complexes "
          f"/ {len(df)} "
          f"({n_texture/len(df)*100:.1f}%)")
    return df[df["tier1_tag"] == "normal"].copy()
