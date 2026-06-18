"""Shared fixtures: a small synthetic dataset + team attributes."""
import pytest

from worldcup2026.etl.sources import synthesize_results
from worldcup2026.tournament import teams_table


@pytest.fixture(scope="session")
def small_results():
    # 2024-2026 only -> a few thousand matches, fast to train on
    return synthesize_results(start_year=2024, seed=7)


@pytest.fixture(scope="session")
def attrs():
    return teams_table()


@pytest.fixture(scope="session")
def small_cfg():
    return {
        "model": {"algorithm": "random_forest", "n_estimators": 25,
                  "min_samples_leaf": 5, "random_state": 1,
                  "rolling_window": 6, "time_decay_half_life_days": 1095},
        "simulation": {"n_simulations": 200, "bivariate_poisson_corr": 0.1,
                       "knockout_draw_extra_time_edge": 0.5, "random_state": 1},
    }
