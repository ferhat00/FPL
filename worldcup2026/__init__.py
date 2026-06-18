"""World Cup 2026 winner-prediction pipeline.

A Groll-style random-forest + Monte-Carlo tournament simulator.

Layout
------
worldcup2026.config        configuration & path resolution
worldcup2026.persistence   parquet read/write helpers
worldcup2026.etl           data acquisition & cleaning
worldcup2026.features      feature engineering (Elo, Dixon-Coles, team attrs)
worldcup2026.model         training, validation, simulation, bracket logic
"""
from __future__ import annotations

__version__ = "0.1.0"
