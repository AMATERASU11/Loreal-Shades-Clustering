"""
Tier 1 — Texture Detector
═══════════════════════════════════════════════════════════════════
Détecte les textures et retourne 3 types :

  'texture_complexe' → quarantaine totale — impossible d'extraire une couleur
        glitter, holographique, iridescent, duochrome, multichrome,
        flake, foil, aurora, galaxy, chameleon, color-changing

  'cat_eye'          → traitement spécial dans Tier 4
        cat eye et produits magnétiques : ont une couleur de BASE extractable
        → NLP prior en premier, vision en fallback

  'normal'           → passe au Tier 2 / 3 / 4 sans restriction
        shimmer, sparkle, metallic, chrome, mirror : base colorée dominante,
        KMeans peut extraire la teinte principale

Décision métier :
  glitter/holo/duochrome = pas de couleur unique → quarantaine irrévocable
  cat eye = couleur de base bien définie (bleu, vert, rouge...) → NLP ou vision
  shimmer/chrome = finish sur couleur normale → vision normale
"""
import re
from typing import Literal

import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  PATTERN CAT EYE / MAGNÉTIQUE
# ═══════════════════════════════════════════════════════════════

_CAT_EYE = re.compile(
    r'\b('
    r'cat[\s\-]?eye'             # cat eye, cat-eye, cateye
    r'|magnetic|magnet(?:ic)?'   # magnetic, magnet, magnetical
    r'|\d+[dD]\s*cat'            # 9D cat, 7D cat, 5D cat
    r'|cat\s*\d+[dD]'            # cat 9D
    r'|aurora\s*cat|cat\s*aurora'  # aurora cat eye variants
    r'|magnetic\s*gel|gel\s*magnetic'
    r'|star\s*field|starfield'   # effet magnétique étoilé
    r')\b',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════
#  PATTERN QUARANTAINE DURE (pas de couleur unique extractable)
# ═══════════════════════════════════════════════════════════════

_QUARANTINE_HARD = re.compile(
    r'\b('
    # Paillettes / particules multicolores
    r'glitter|spangle|tinsel|sequin'
    # Effets optiques qui changent de couleur
    r'|holograph(?:ic)?|holo|iridescent|duochrome|multichrome'
    r'|aurora|galaxy|chameleon|colour.changing|color.changing'
    # Effets décoratifs opaques sur fond quelconque
    r'|foil|metallic\s+foil'
    # Flocons de couleurs mélangées
    r'|flake|flakie|jelly\s+flake|nail\s+art\s+flake'
    # Dégradé multi-couleurs (pas de couleur unique extractable)
    r'|ombre|gradient'
    r')\b',
    re.IGNORECASE
)

# "foil" seul peut être un nom de teinte créatif (ex: "Gold Foil" = juste doré)
# → on vérifie qu'il y a un contexte nail/polish/effect avant de quarantiner
_FOIL_ALONE = re.compile(r'^foil$', re.IGNORECASE)
_TEXTURE_CONTEXT = re.compile(
    r'\b(nail\s*polish|vernis|gel|polish|nail|effect|finish|coat|ongles?|art)\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════
#  INTERFACE PUBLIQUE
# ═══════════════════════════════════════════════════════════════

TextureType = Literal["texture_complexe", "cat_eye", "normal"]


def classify(row: pd.Series) -> TextureType:
    """
    Classifie un produit selon sa texture.

    Cherche dans shade_name + title + description (les 3 colonnes).
    Cat eye est vérifié EN PREMIER — un "Cat Eye Glitter" est cat_eye, pas quarantaine.

    Args:
        row: Ligne du DataFrame (shade_name, title, description)

    Returns:
        'cat_eye'          | 'texture_complexe' | 'normal'
    """
    shade = str(row.get("shade_name",  "") or "")
    title = str(row.get("title",       "") or "")
    desc  = str(row.get("description", "") or "")[:300]

    shade_title = f"{shade} {title}"
    full        = f"{shade_title} {desc}"

    # Cat eye prioritaire — on peut quand même extraire la couleur de base
    if _CAT_EYE.search(shade_title) or _CAT_EYE.search(desc):
        return "cat_eye"

    # Quarantaine dure — pas de couleur unique
    m = _QUARANTINE_HARD.search(shade_title)
    if m:
        word = m.group(0).strip().lower()
        # "foil" seul dans shade_name = peut-être juste un nom de teinte
        if _FOIL_ALONE.match(word):
            if not _TEXTURE_CONTEXT.search(shade_title):
                return "normal"
        return "texture_complexe"

    if _QUARANTINE_HARD.search(desc):
        m_desc = _QUARANTINE_HARD.search(desc)
        word   = m_desc.group(0).strip().lower()
        if _FOIL_ALONE.match(word):
            if not _TEXTURE_CONTEXT.search(desc):
                return "normal"
        return "texture_complexe"

    return "normal"


def get_detected_keyword(row: pd.Series) -> str | None:
    """Retourne le mot-clé texture/cat-eye détecté (debug)."""
    full = " ".join([
        str(row.get("shade_name",  "") or ""),
        str(row.get("title",       "") or ""),
        str(row.get("description", "") or "")[:300],
    ])
    m = _CAT_EYE.search(full) or _QUARANTINE_HARD.search(full)
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
    """
    Retourne les produits utilisables après Tier 1.

    Exclus  : texture_complexe uniquement
    Gardés  : normal + cat_eye
              (cat_eye → NLP prior dans predict(), couleur de base réelle)
    """
    df = apply_to_dataframe(df)
    n_texture = (df["tier1_tag"] == "texture_complexe").sum()
    n_cat_eye = (df["tier1_tag"] == "cat_eye").sum()
    n_normal  = (df["tier1_tag"] == "normal").sum()
    print(f"Tier 1 — exclus : {n_texture} textures complexes / {len(df)} "
          f"({n_texture/len(df)*100:.1f}%)")
    if n_cat_eye:
        print(f"         cat eye : {n_cat_eye} → NLP prior dans Tier 4")
    if n_normal:
        print(f"         normaux : {n_normal}")
    return df[df["tier1_tag"] != "texture_complexe"].copy()
