"""
utils/text_utils.py
-------------------
Fonctions pures de nettoyage et normalisation des noms de teintes.
"""
import re
import numpy as np
import pandas as pd
from typing import Tuple


def clean_shade_text(s: str) -> str:
    """
    Nettoyage du shade_name : supprime les mesures, caractères spéciaux,
    mais CONSERVE les codes numériques de teinte (ex: 810, 7.2, 05).

    Examples:
        >>> clean_shade_text("Nu Muse (3.8 g)")
        'nu muse'
        >>> clean_shade_text("810 Fair")
        '810 fair'
    """
    if pd.isna(s):
        return ""
    s = str(s).lower()

    # Supprimer les groupes parenthèses contenant des mesures
    s = re.sub(
        r"\([^)]*(?:ml|oz|gr|gram|grams|fl\.?\s?oz|\d+[\.,]\d+\s*g|\d+\s*g)[^)]*\)",
        " ",
        s,
    )

    # Supprimer les mesures isolées restantes
    s = re.sub(r"\b(ml|oz|gr|gram|grams|fl\.?\s?oz)\b", " ", s)
    s = re.sub(r"(?<=\d)\s*\bg\b", " ", s)

    # Remplacer séparateurs courants
    s = re.sub(r"[/|,_-]+", " ", s)

    # Garder uniquement lettres, chiffres, points
    s = re.sub(r"[^a-z0-9\. ]+", " ", s)

    # Normaliser les espaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_code_words(s: str) -> Tuple[str, str]:
    """
    Sépare un code numérique de début (ex: "810 fair" -> ("810", "fair")).
    Si pas de code, retourne ("", s).
    """
    s = s.strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s+(.*)$", s)
    if m:
        return m.group(1), m.group(2).strip()
    return "", s


def build_shade_key(shade_clean: str) -> str:
    """
    Construit une clé de regroupement texte.
    Priorité au code numérique s'il existe, sinon les mots.
    """
    code, words = split_code_words(shade_clean)
    key = code if code else words
    return key if key else "missing"


def text_similarity_key(s1: str, s2: str) -> float:
    """
    Score de similarité Jaccard entre deux clés texte (0.0 à 1.0).
    """
    if s1 == s2:
        return 1.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)
