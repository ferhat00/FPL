"""End-to-end model + Monte-Carlo invariants on a small synthetic dataset."""
import pandas as pd
import pytest

from worldcup2026.model.simulate import Simulator, run_simulation
from worldcup2026.model.train import build_lambda_table, train_model
from worldcup2026.tournament import all_teams


@pytest.fixture(scope="module")
def bundle(small_results, attrs, small_cfg):
    return train_model(small_results, attrs, small_cfg, persist=False)


def test_lambda_table_is_positive_and_complete(bundle):
    teams, lam = build_lambda_table(bundle)
    assert len(teams) == 48
    assert lam.shape == (48, 48)
    off_diag = lam[~(lam == 0)]
    assert (off_diag > 0).all()
    assert off_diag.max() <= 6.0


def test_simulation_probabilities_are_coherent(bundle):
    out = run_simulation(bundle, None, n=300, corr=0.1, et_edge=0.5, seed=2)
    s = out["summary"]
    # exactly one champion per simulation -> probabilities sum to 1
    assert s["P_champion"].sum() == pytest.approx(1.0, abs=1e-6)
    # 32 teams qualify each sim
    assert s["P_qualify_r32"].sum() == pytest.approx(32.0, abs=1e-6)
    # 12 group winners each sim
    assert s["P_group_winner"].sum() == pytest.approx(12.0, abs=1e-6)
    assert ((s[["P_champion", "P_final", "P_qualify_r32"]] >= 0).all().all())
    assert ((s[["P_champion", "P_final", "P_qualify_r32"]] <= 1).all().all())
    # monotonic funnel: champion <= final <= semi <= qualify
    assert (s["P_champion"] <= s["P_final"] + 1e-9).all()
    assert (s["P_final"] <= s["P_semi"] + 1e-9).all()


def test_expected_bracket_is_valid(bundle):
    out = run_simulation(bundle, None, n=50, corr=0.1, et_edge=0.5, seed=3)
    eb = out["expected_bracket"]
    assert eb["champion"] in all_teams()
    assert eb["runner_up"] in all_teams()
    assert len(eb["best_thirds"]) == 8
    assert len(eb["group_standings"]) == 12


def test_live_results_are_locked(bundle):
    # fix a group game 9-0 and confirm the simulator ingests it without error
    live = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-12"]),
        "home_team": ["Spain"], "away_team": ["Uruguay"],
        "home_score": [9], "away_score": [0],
        "tournament": ["FIFA World Cup 2026"], "neutral": [True],
    })
    sim = Simulator(bundle, live, seed=4)
    assert ("Spain", "Uruguay") in sim._fixed
    assert sim._fixed[("Spain", "Uruguay")] == (9, 0)
    out = sim.run(100)
    assert out["summary"]["P_champion"].sum() == pytest.approx(1.0, abs=1e-6)
