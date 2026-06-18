"""Team-name normalization and results canonicalization."""
import pandas as pd

from worldcup2026.etl import clean


def test_normalize_known_variants():
    assert clean.normalize_team("USA") == "United States"
    assert clean.normalize_team("Czechia") == "Czech Republic"
    assert clean.normalize_team("Korea Republic") == "South Korea"
    assert clean.normalize_team("Türkiye") == "Turkey"
    assert clean.normalize_team("Côte d'Ivoire") == "Ivory Coast"
    assert clean.normalize_team("Curaçao") == "Curacao"


def test_normalize_results_schema():
    raw = pd.DataFrame({
        "date": ["2024-06-01", "2024-06-05"],
        "home_team": ["USA", "Brazil"],
        "away_team": ["Mexico", "Argentina"],
        "home_score": [1, 2],
        "away_score": [1, 3],
    })
    out = clean.normalize_results(raw)
    assert list(out.columns) == ["date", "home_team", "away_team", "home_score",
                                 "away_score", "tournament", "neutral"]
    assert str(out["date"].dtype).startswith("datetime")
    assert out.iloc[0]["home_team"] == "United States"
    assert out["neutral"].dtype == bool
