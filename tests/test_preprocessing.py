"""
Tests unitaires pour ShadeTextProcessor et les utils texte.
Lancez avec : pytest tests/test_preprocessing.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.text_utils import clean_shade_text, split_code_words, build_shade_key, text_similarity_key


class TestCleanShadeText:
    def test_removes_weight_in_parens(self):
        assert clean_shade_text("Nu Muse (3.8 g)") == "nu muse"

    def test_removes_oz(self):
        assert clean_shade_text("Fair (0.5 oz)") == "fair"

    def test_preserves_numeric_code(self):
        result = clean_shade_text("810 Fair")
        assert "810" in result and "fair" in result

    def test_handles_nan(self):
        assert clean_shade_text(np.nan) == ""
        assert clean_shade_text(None) == ""

    def test_lowercase(self):
        assert clean_shade_text("ROSE") == "rose"

    def test_n_code(self):
        result = clean_shade_text("N157 Nu Inattendu (3.8 g)")
        assert "n157" in result and "nu inattendu" in result

    def test_no_side_effects_on_words(self):
        # 'g' dans un mot ne doit pas être supprimé
        result = clean_shade_text("Rose Gold")
        assert "gold" in result


class TestSplitCodeWords:
    def test_numeric_code(self):
        code, words = split_code_words("810 fair")
        assert code == "810" and words == "fair"

    def test_no_code(self):
        code, words = split_code_words("nu muse")
        assert code == "" and words == "nu muse"

    def test_float_code(self):
        code, _ = split_code_words("7.2 naturel")
        assert code == "7.2"

    def test_empty(self):
        code, words = split_code_words("")
        assert code == "" and words == ""


class TestBuildShadeKey:
    def test_uses_code(self):
        assert build_shade_key("810 fair") == "810"

    def test_uses_words(self):
        assert build_shade_key("nu muse") == "nu muse"

    def test_missing_for_empty(self):
        assert build_shade_key("") == "missing"


class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity_key("nu muse", "nu muse") == 1.0

    def test_no_overlap(self):
        assert text_similarity_key("fair", "rose") == 0.0

    def test_partial(self):
        score = text_similarity_key("nu muse 19", "nu muse 20")
        assert 0.0 < score < 1.0
