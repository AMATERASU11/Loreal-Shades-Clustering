"""
Tier 0 — Gatekeeper
═══════════════════════════════════════════════════════════════════
Détecte et exclut les produits Kit et Incolores avant toute extraction.

Sources recyclées :
  - Loreal-Shades-Clustering/src/incolor_kit_pipeline.py (regex patterns)
  - Méthode 4 filter_products() (logique anti-faux-positifs)

Corrections vs analyse_pre_m5.py :
  - Filtre kit : N >= 2 obligatoire (corrige "1 Pcs", "1 Vernis" faux positifs)
  - Anti-FP nail tips : "clear" dans un contexte nail tips = pas incolore

Détection kit hybride (NLP + ΔE visuel) :
  'kit_multicolor' → ≥2 mots couleur dans shade_name → exclu
  'kit_monocolor'  → 1 mot couleur, OU duo/trio + 0 couleur + pas de séparateur → gardé
  'kit_ambiguous'  → 0 mot couleur + quad/quintet/sextet ou N-count → ΔE tier3 décide
  'incolore'       → [TAG_INCOLORE] — exclu
  'normal'         → passe au Tier 1
"""
import re
from typing import Literal, Union

import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  PATTERNS
# ═══════════════════════════════════════════════════════════════

# ── Incolore : match exact sur shade_name ──
_INCOLOR_SHADE_EXACT = re.compile(
    r'^('
    r'clear|transparent|incolore|translucide|translucent'
    r'|top\s*coat|base\s*coat|gel\s*coat'
    r'|no\s*color|sans\s*couleur'
    r'|primer|hardener|strengthener|durcisseur'
    r'|sealant|finish|finition'
    r'|nail\s*treatment|soin|traitement'
    r')$',
    re.IGNORECASE
)

# ── Incolore : substring dans shade_name ──
_INCOLOR_SHADE_SUB = re.compile(
    r'\b('
    r'clear|transparent|incolore|translucide|translucent'
    r'|top\s*coat|base\s*coat|gel\s*coat'
    r'|no\s*color|sans\s*couleur'
    r'|primer|hardener|strengthener|durcisseur'
    r'|sealant|finition|nail\s*treatment|soin|traitement'
    r')\b',
    re.IGNORECASE
)

# ── Incolore : patterns dans le titre ──
_INCOLOR_TITLE = re.compile(
    r'\b('
    r'top\s*coat|base\s*coat|gel\s*coat'
    r'|nail\s*hardener|nail\s*strengthener|durcisseur'
    r'|nail\s*treatment|soin\s*ongles|traitement\s*ongles'
    r'|nail\s*primer|cuticle\s*oil|huile\s*cuticule'
    r'|nail\s*sealer|nail\s*sealant'
    r'|vernis\s*(?:de\s*)?(?:finition|protection|soin)'
    r')\b',
    re.IGNORECASE
)

# ── Mots de couleur → annulent une détection incolore ──
_COLOR_CANCEL = re.compile(
    r'\b('
    r'red|rouge|pink|rose|coral|corail|berry|wine|burgundy|bordeaux'
    r'|plum|prune|mauve|cherry|cerise|raspberry|framboise|strawberry'
    r'|fraise|cranberry|ruby|scarlet|crimson|magenta|fuchsia'
    r'|blue|bleu|green|vert|turquoise|teal|navy|marine|mint|menthe'
    r'|aqua|cyan|emerald|jade|sage|olive|forest'
    r'|yellow|jaune|orange|gold|or|peach|pêche|apricot|abricot'
    r'|amber|ambre|lemon|citron|lime|mustard|moutarde|honey|miel'
    r'|purple|violet|lilac|lilas|lavender|lavande|indigo'
    r'|brown|marron|beige|nude|tan|taupe|chocolate|chocolat'
    r'|caramel|coffee|café|mocha|espresso|cinnamon|ginger|auburn'
    r'|chestnut|mahogany|sand|sable'
    r'|silver|argent|bronze|copper|cuivre|champagne|pewter|titanium'
    r'|black|noir|white|blanc|cream|ivory|pearl|perle|grey|gray|gris|charcoal'
    r')\b',
    re.IGNORECASE
)

# ── Kit : duo/trio (candidats mono) ──
_KIT_DUO_TRIO = re.compile(
    r'\b(duo|trio)\b',
    re.IGNORECASE
)

# ── Kit : quad/quintet/sextet (multi-couleurs quasi-certain) ──
_KIT_QUAD_PLUS = re.compile(
    r'\b(quad|quintet|sextet)\b',
    re.IGNORECASE
)

# ── Séparateur de nuances multiples dans shade_name ──
_SHADE_SEPARATOR = re.compile(r'/| \+ | & | and ', re.IGNORECASE)

# ── Anti-FP : "1pc N Colors" = un seul produit avec N options de couleurs ──
#    Exemple : "1pc 36 Color Stepped Nail Polish Pen" → PAS un kit de 36
_SINGLE_UNIT_N_COLORS = re.compile(
    r'\b1\s*pcs?\s+\d+\s*(?:colors?|colours?|shades?)\b',
    re.IGNORECASE
)

# ── Kit : N + unité avec N >= 2 ──
#    Format : "3 pcs", "24 colors", "lot de 5", "set of 4", "4 vernis"
_KIT_N_UNIT = re.compile(
    r'\b(\d+)\s*(?:[-x])?\s*'
    r'(pcs?|pack|bottles?|colors?|colours?|shades?|pieces?|vernis|gels?'
    r'|polishs?|couleurs?|bouteilles?)\b',
    re.IGNORECASE
)

# ── Kit : coffret/bundle/lot/set explicites ──
_KIT_EXPLICIT = re.compile(
    r'\b('
    r'coffret|bundle|assortment|combo'
    r'|set\s+(?:of\s+)?\d+'
    r'|kit\s+(?:de\s+)?\d+'
    r'|lot\s+(?:de\s+)?\d+'
    r'|collection\s+(?:de\s+)?\d+'
    r')\b',
    re.IGNORECASE
)

# ── Anti-FP : nail tips / faux ongles ──
_NAIL_TIPS = re.compile(
    r'\b('
    r'nail\s*tips?|fake\s*nails?|faux\s*ongles?|press[\s\-]?on'
    r'|false\s*nails?|acrylic\s*nails?|nail\s*extension'
    r'|capsules?\s*(?:d.?)?ongles?|duck\s*(?:nail\s*)?tips?'
    r'|ballerina|coffin\s*(?:nail|tip)|almond\s*(?:nail|tip)'
    r'|nail\s*glue|builder\s*tip'
    r')\b',
    re.IGNORECASE
)

# ── Anti-FP : grand nombre seul = shade ID (> 50) ──
_BIG_NUMBER = re.compile(r'\b(\d+)\b')

# ── Seuil : nombre de produits min pour être un kit ──
KIT_MIN_COUNT = 2
KIT_MAX_SHADE_NUMBER = 50   # Au-dessus = shade ID, pas quantité


# ═══════════════════════════════════════════════════════════════
#  DÉTECTION KIT
# ═══════════════════════════════════════════════════════════════

_COLOR_SEG_SPLIT = re.compile(r',|/| \+ | & | and ', re.IGNORECASE)

def _count_color_segments(shade: str) -> int:
    """
    Compte le nombre de SEGMENTS colorés dans shade_name.

    Un segment = portion entre délimiteurs (, / + & and).
    Si un segment contient ≥1 mot couleur → compte comme 1 couleur distincte.

    "Rose Gold"     → 1 segment → 1  (nom composé d'une seule teinte)
    "Cherry Berry"  → 1 segment → 1  (idem)
    "Orange/rouge"  → 2 segments → 2  (deux teintes séparées par /)
    "Red, Blue"     → 2 segments → 2
    "Beige Nude"    → 1 segment → 1  (même teinte, deux descripteurs)
    """
    segments = _COLOR_SEG_SPLIT.split(shade)
    return sum(1 for seg in segments if _COLOR_CANCEL.search(seg))


def _is_kit(shade: str, title: str) -> Union[Literal["multicolor", "monocolor", "ambiguous"], bool]:
    """
    Détecte si le produit est un kit, et si oui, son type :
      'multicolor' → ≥2 mots couleur dans shade_name → exclure
      'monocolor'  → 1 mot couleur, OU duo/trio + 0 couleur + pas de séparateur → garder
      'ambiguous'  → 0 mot couleur + quad/quintet/sextet ou N-count → ΔE décide
      False        → pas un kit

    Règle duo/trio : "DND GEL DUO 736 WATERMELON" = 2 flacons même couleur → monocolor
    Règle quad+    : "Zoya Quad" = 4 couleurs distinctes → ambiguous (ΔE décide)
    """
    full = f"{shade} {title}"

    # Press-on nails / faux ongles → pas un kit de vernis
    if _NAIL_TIPS.search(full):
        return False

    # "1pc 36 Color Pen" → 1 seul produit avec N couleurs disponibles
    if _SINGLE_UNIT_N_COLORS.search(full):
        return False

    # shade_name avec ≥2 segments couleur séparés → kit multicolore sans ambiguïté
    # "Orange/rouge" → multicolor direct, sans avoir besoin d'un trigger titre
    if _count_color_segments(shade) >= 2:
        return "multicolor"

    detected = False
    is_duo_trio = False  # True seulement si duo/trio, pas quad/quintet/sextet/N-count

    if _KIT_DUO_TRIO.search(full):
        detected = True
        is_duo_trio = True

    if _KIT_QUAD_PLUS.search(full):
        detected = True
        is_duo_trio = False  # quad/quintet/sextet annule le flag mono

    if not detected:
        m = _KIT_EXPLICIT.search(full)
        if m:
            nums = [int(n) for n in _BIG_NUMBER.findall(m.group(0))]
            # Borne supérieure : "Collection 2022" → 2022 > 50 → pas un kit
            if nums and KIT_MIN_COUNT <= max(nums) <= KIT_MAX_SHADE_NUMBER:
                detected = True

    if not detected:
        # N_UNIT sur le titre uniquement (pas shade_name) pour éviter que les
        # IDs de nuance ("008 Vernis", "32 Gel Effect") déclenchent faussement
        for m in _KIT_N_UNIT.finditer(title):
            n = int(m.group(1))
            if KIT_MIN_COUNT <= n <= KIT_MAX_SHADE_NUMBER:
                detected = True
                break

    if not detected:
        return False

    # Kit confirmé — qualifier par nombre de SEGMENTS couleur dans shade_name
    n_colors = _count_color_segments(shade)
    if n_colors >= 2:
        return "multicolor"
    if n_colors == 1:
        return "monocolor"
    # 0 segments couleur : duo/trio sans séparateur → même nuance, 2 formats (gel+vernis)
    if is_duo_trio and not _SHADE_SEPARATOR.search(shade):
        return "monocolor"
    return "ambiguous"


# ═══════════════════════════════════════════════════════════════
#  DÉTECTION INCOLORE
# ═══════════════════════════════════════════════════════════════

def _is_incolore(shade: str, title: str) -> bool:
    """Retourne True si le produit est incolore/transparent."""
    # Nail tips → le "clear" ne signifie pas incolore
    if _NAIL_TIPS.search(f"{shade} {title}"):
        return False

    # Si shade_name contient un mot de couleur → pas incolore
    if _COLOR_CANCEL.search(shade):
        return False

    # Match exact shade_name
    shade_clean = shade.strip()
    if shade_clean and _INCOLOR_SHADE_EXACT.match(shade_clean):
        return True

    # Substring dans shade_name
    if _INCOLOR_SHADE_SUB.search(shade):
        return True

    # Terme dans le titre (sans ambiguïté de kit)
    if _INCOLOR_TITLE.search(title):
        return True

    return False


# ═══════════════════════════════════════════════════════════════
#  INTERFACE PUBLIQUE
# ═══════════════════════════════════════════════════════════════

ProductType = Literal[
    "normal",
    "kit_multicolor",   # kit avec ≥2 couleurs → exclu direct
    "kit_monocolor",    # pack N×même couleur   → traité comme normal
    "kit_ambiguous",    # kit sans couleur NLP  → ΔE decide dans predict()
    "incolore",
]


def classify(row: pd.Series) -> ProductType:
    """
    Classifie un produit.

    Args:
        row: Ligne du DataFrame (shade_name, title requis)

    Returns:
        'kit_multicolor' | 'kit_monocolor' | 'kit_ambiguous' | 'incolore' | 'normal'

    Note: incolore vérifié EN PREMIER — un "Top Coat Set" est incolore, pas kit_ambiguous.
    """
    shade = str(row.get("shade_name", "") or "").strip()
    title = str(row.get("title",      "") or "").strip()

    # Incolore prioritaire : "2Pcs Top Coat Set" → incolore, pas kit
    if _is_incolore(shade, title):
        return "incolore"

    kit = _is_kit(shade, title)
    if kit == "multicolor":
        return "kit_multicolor"
    if kit == "monocolor":
        return "kit_monocolor"
    if kit == "ambiguous":
        return "kit_ambiguous"

    return "normal"


def apply_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la classification Tier 0 sur tout un DataFrame.

    Ajoute la colonne 'tier0_tag'.
    """
    df = df.copy()
    df["tier0_tag"] = df.apply(classify, axis=1)
    return df


def filter_normal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les produits utilisables après Tier 0.

    Exclus  : kit_multicolor + incolore
    Gardés  : normal + kit_monocolor + kit_ambiguous
              (kit_ambiguous → ΔE décide dans predict())
    """
    df = apply_to_dataframe(df)
    _EXCLUDED = {"kit_multicolor", "incolore"}
    excluded = df[df["tier0_tag"].isin(_EXCLUDED)]
    n_multi    = (df["tier0_tag"] == "kit_multicolor").sum()
    n_mono     = (df["tier0_tag"] == "kit_monocolor").sum()
    n_ambig    = (df["tier0_tag"] == "kit_ambiguous").sum()
    n_incolore = (df["tier0_tag"] == "incolore").sum()
    print(f"Tier 0 — exclus : {n_multi} kits multicolores + {n_incolore} incolores "
          f"= {len(excluded)} / {len(df)} ({len(excluded)/len(df)*100:.1f}%)")
    if n_mono or n_ambig:
        print(f"         gardés : {n_mono} packs monochromes + {n_ambig} kits ambigus (→ ΔE)")
    return df[~df["tier0_tag"].isin(_EXCLUDED)].copy()
