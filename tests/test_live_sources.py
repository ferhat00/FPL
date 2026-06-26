"""Live WC-2026 results feed: FIFA API parsing, provider toggle, merge priority."""
import pandas as pd

import worldcup2026.etl.sources as src


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _fifa_payload():
    """A few FIFA match records, incl. names needing normalization and a non-entrant."""
    return {"Results": [
        {"MatchStatus": 0, "Date": "2026-06-25T18:00:00Z",
         "Home": {"Score": 0, "TeamName": [{"Description": "Czechia"}]},
         "Away": {"Score": 3, "TeamName": [{"Description": "Mexico"}]}},
        {"MatchStatus": 0, "Date": "2026-06-26T20:00:00Z",
         "Home": {"Score": 3, "TeamName": [{"Description": "Türkiye"}]},
         "Away": {"Score": 2, "TeamName": [{"Description": "USA"}]}},
        {"MatchStatus": 1, "Date": "2026-06-27T18:00:00Z",   # not finished -> excluded
         "Home": {"Score": None, "TeamName": [{"Description": "Argentina"}]},
         "Away": {"Score": None, "TeamName": [{"Description": "Jordan"}]}},
        {"MatchStatus": 0, "Date": "2026-06-20T18:00:00Z",   # non-entrant -> excluded
         "Home": {"Score": 1, "TeamName": [{"Description": "Italy"}]},
         "Away": {"Score": 0, "TeamName": [{"Description": "Mexico"}]}},
    ]}


def test_fifa_live_parses_and_normalizes(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(_fifa_payload()))
    df = src.fetch_live_results_fifa()
    assert len(df) == 2                       # only finished entrant-vs-entrant games
    teams = set(df["home_team"]) | set(df["away_team"])
    assert {"Czech Republic", "Mexico", "Turkey", "United States"} <= teams
    assert "Italy" not in teams               # non-entrant dropped
    row = df[df["home_team"] == "Czech Republic"].iloc[0]
    assert (int(row["home_score"]), int(row["away_score"])) == (0, 3)


def test_fifa_live_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(src, "load_config",
                        lambda: {"sources": {"fetch_live_results": True,
                                             "live_results_provider": "none"}})
    assert len(src.fetch_live_results_fifa()) == 0


def test_fifa_live_network_failure_falls_back_to_empty(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr("requests.get", _boom)
    assert len(src.fetch_live_results_fifa()) == 0   # warns + empty, never raises


def test_dedupe_live_collapses_same_pair_regardless_of_date_or_orientation():
    a = pd.DataFrame([{"date": pd.Timestamp("2026-06-23"), "home_team": "Colombia",
                       "away_team": "DR Congo", "home_score": 9, "away_score": 9,
                       "tournament": "x", "neutral": True}])
    b = pd.DataFrame([{"date": pd.Timestamp("2026-06-24"), "home_team": "DR Congo",
                       "away_team": "Colombia", "home_score": 0, "away_score": 1,
                       "tournament": "y", "neutral": True}])
    out = src._dedupe_live([a, b])            # b is higher priority (last)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["home_team"] == "DR Congo" and (int(r["home_score"]), int(r["away_score"])) == (0, 1)


def test_fetch_live_results_fifa_overrides_history(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(_fifa_payload()))
    history = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-25"), "home_team": "Czech Republic",
        "away_team": "Mexico", "home_score": 5, "away_score": 5,   # stale/wrong
        "tournament": "FIFA World Cup", "neutral": True}])
    out = src.fetch_live_results(history)
    row = out[(out["home_team"] == "Czech Republic") & (out["away_team"] == "Mexico")]
    assert len(row) == 1
    assert (int(row.iloc[0]["home_score"]), int(row.iloc[0]["away_score"])) == (0, 3)  # FIFA wins
