"""Data sources with graceful degradation.

Every ``fetch_*`` returns a tidy DataFrame and never raises on a network/credential
failure — it logs a warning and falls back (snapshot or calibrated synthetic data),
so the whole pipeline runs offline. Live sources always take precedence when available.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import get_path, kaggle_available, load_config
from ..tournament import all_teams, load_tournament, teams_table
from .clean import normalize_results, normalize_team

log = logging.getLogger("worldcup2026.etl")

# A pool of additional national teams so the synthetic history has varied opponents.
# (Must not overlap the 48 WC entrants — overlap would allow a team-vs-itself match.)
_EXTRA_NATIONS = [
    "Italy", "Poland", "Serbia", "Denmark", "Wales", "Romania", "Hungary", "Greece",
    "Slovakia", "Slovenia", "Russia", "Ukraine", "Finland", "Iceland", "Northern Ireland",
    "Nigeria", "Cameroon", "Mali", "Burkina Faso", "Zambia", "Guinea", "Angola",
    "Chile", "Peru", "Venezuela", "Bolivia", "Costa Rica", "Honduras", "Jamaica",
    "China", "UAE", "Oman", "Bahrain", "Thailand", "Vietnam", "India",
]


# ---------------------------------------------------------------------------
# Historical match results
# ---------------------------------------------------------------------------
def fetch_results() -> pd.DataFrame:
    """Historical international results (canonical schema).

    Primary: Kaggle martj42 dataset. Fallback: calibrated synthetic history.
    """
    cfg = load_config()
    if cfg["sources"].get("use_kaggle", True) and kaggle_available():
        try:
            df = _kaggle_results()
            if df is not None and len(df) > 1000:
                log.info("Loaded %d historical matches from Kaggle.", len(df))
                return normalize_results(df)
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            log.warning("Kaggle results fetch failed (%s); using synthetic history.", exc)

    log.warning("Using SYNTHETIC historical results (offline fallback).")
    return synthesize_results()


def _kaggle_results() -> Optional[pd.DataFrame]:
    """Download martj42 results.csv via the Kaggle API."""
    cfg = load_config()["kaggle"]
    _ensure_kaggle_env()
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy import

    api = KaggleApi()
    api.authenticate()
    dest = get_path("cache") / "kaggle_results"
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(cfg["results_dataset"], path=str(dest), unzip=True)
    # the dataset contains results.csv (matches) among others
    candidates = list(dest.glob("results.csv")) or list(dest.glob("*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def _ensure_kaggle_env() -> None:
    import os
    kg = load_config()["kaggle"]
    if kg.get("username") and kg.get("key"):
        os.environ.setdefault("KAGGLE_USERNAME", kg["username"])
        os.environ.setdefault("KAGGLE_KEY", kg["key"])


def synthesize_results(start_year: int = 2002, seed: int = 42) -> pd.DataFrame:
    """Generate a calibrated synthetic match history.

    Goals are drawn from a Poisson model whose means depend on an Elo-style
    latent strength + home advantage, so downstream features (Elo, form) carry
    genuine predictive signal. Clearly flagged; overridden by real data.
    """
    rng = np.random.default_rng(seed)
    snap = teams_table()
    teams = list(dict.fromkeys(list(snap.index) + _EXTRA_NATIONS))  # dedup, keep order
    # latent strength: snapshot Elo for WC teams, sampled for the rest
    strength = {}
    for t in teams:
        if t in snap.index:
            strength[t] = float(snap.loc[t, "elo"])
        else:
            strength[t] = float(rng.normal(1700, 120))

    end_year = 2026
    rows = []
    for year in range(start_year, end_year + 1):
        n_matches = 700  # ~ per calendar year across all nations
        for _ in range(n_matches):
            h, a = rng.choice(teams, size=2, replace=False)
            neutral = bool(rng.random() < 0.35)
            ha = 0.0 if neutral else 60.0  # home advantage in Elo points
            diff = (strength[h] + ha - strength[a]) / 400.0
            # expected goals via logistic-ish mapping of Elo diff
            lam_h = float(np.clip(1.35 * np.exp(0.50 * diff), 0.12, 6.0))
            lam_a = float(np.clip(1.35 * np.exp(-0.50 * diff), 0.12, 6.0))
            gh, ga = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 28))
            rows.append({
                "date": f"{year}-{month:02d}-{day:02d}",
                "home_team": h, "away_team": a,
                "home_score": gh, "away_score": ga,
                "tournament": "Synthetic", "neutral": neutral,
            })
            # mild Elo drift so strengths evolve but stay near snapshot scale
            exp_h = 1.0 / (1.0 + 10 ** (-(strength[h] + ha - strength[a]) / 400))
            res_h = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
            k = 8.0
            strength[h] += k * (res_h - exp_h)
            strength[a] -= k * (res_h - exp_h)
    return normalize_results(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Elo ratings (current snapshot)
# ---------------------------------------------------------------------------
def fetch_elo() -> pd.DataFrame:
    """Current Elo per WC-2026 team. Tries eloratings.net, else snapshot."""
    cfg = load_config()["sources"]
    if cfg.get("scrape_eloratings", True):
        try:
            df = _scrape_eloratings()
            if df is not None and len(df) > 20:
                log.info("Scraped %d Elo ratings from eloratings.net.", len(df))
                return _restrict_to_wc(df, "elo")
        except Exception as exc:  # noqa: BLE001
            log.warning("eloratings.net scrape failed (%s); using snapshot.", exc)
    snap = teams_table()[["elo"]].reset_index().rename(columns={"index": "team"})
    return snap


def _scrape_eloratings() -> Optional[pd.DataFrame]:
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.eloratings.net/2026_World_Cup"
    r = requests.get(url, timeout=20, headers={"User-Agent": "wc2026/1.0"})
    r.raise_for_status()
    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:  # lxml not installed -> stdlib parser
        soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for tr in soup.select("tr"):
        cells = [c.get_text(strip=True) for c in tr.select("td")]
        if len(cells) >= 2:
            team = normalize_team(cells[0])
            try:
                elo = float(cells[1])
            except ValueError:
                continue
            rows.append({"team": team, "elo": elo})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# FIFA ranking, market values, macro — snapshot-backed
# ---------------------------------------------------------------------------
def fetch_fifa_ranking() -> pd.DataFrame:
    snap = teams_table()[["fifa_rank"]].reset_index().rename(columns={"index": "team"})
    return snap


def fetch_market_values() -> pd.DataFrame:
    """Squad market value (EUR m) + mean squad age. Snapshot fallback.

    Transfermarkt scraping is intentionally not hard-wired (very brittle / ToS);
    the snapshot in tournament_2026.yaml is used and can be refreshed manually.
    """
    snap = teams_table()[["market_value_m", "squad_age"]].reset_index()
    return snap.rename(columns={"index": "team"})


def fetch_worldbank() -> pd.DataFrame:
    """GDP per capita (current US$) + population for WC nations via World Bank API.

    Falls back to NaN columns (these are low-importance features per Groll et al.).
    """
    teams = all_teams()
    try:
        df = _worldbank_indicators()
        if df is not None and len(df) > 0:
            return df
    except Exception as exc:  # noqa: BLE001
        log.warning("World Bank fetch failed (%s); GDP/pop set to NaN.", exc)
    return pd.DataFrame({"team": teams, "gdp_per_capita": np.nan, "population": np.nan})


def _worldbank_indicators() -> Optional[pd.DataFrame]:
    import requests

    iso = _WB_ISO3
    codes_all = sorted(set(iso.values()))
    out: dict = {}
    for indicator, col in [("NY.GDP.PCAP.CD", "gdp_per_capita"), ("SP.POP.TOTL", "population")]:
        for i in range(0, len(codes_all), 25):       # chunk to keep URLs short
            chunk = codes_all[i:i + 25]
            url = (f"https://api.worldbank.org/v2/country/{';'.join(chunk)}/indicator/"
                   f"{indicator}?format=json&mrv=1&per_page=200")
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                continue
            for rec in payload[1]:
                code = rec.get("countryiso3code")
                val = rec.get("value")
                if code and val is not None:
                    out.setdefault(code, {})[col] = val
    rows = []
    for team, code in iso.items():
        rec = out.get(code, {})
        rows.append({"team": team,
                     "gdp_per_capita": rec.get("gdp_per_capita", np.nan),
                     "population": rec.get("population", np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Live WC-2026 results so far
# ---------------------------------------------------------------------------
_LIVE_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
              "tournament", "neutral"]


def _empty_live() -> pd.DataFrame:
    return normalize_results(pd.DataFrame(columns=_LIVE_COLS))


def _wc2026_from_history(history: pd.DataFrame) -> pd.DataFrame:
    """Extract the played WC-2026 matches out of the freshly-pulled history.

    Real results arrive in the historical feed (martj42 on Kaggle) labelled
    ``FIFA World Cup``. We isolate the 2026 edition by keeping World Cup games
    (not qualifiers) dated on/after the official opener and contested by two of
    the 48 entrants — so each Refresh auto-locks the latest scores with no
    manual step. The source ``tournament`` label is preserved, so concatenating
    these rows back into the historical table is a no-op after de-duplication.
    """
    start = pd.to_datetime(load_tournament()["meta"]["opening_match_date"])
    wc = set(all_teams())
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    tour = df["tournament"].astype(str).str.lower()
    mask = (
        tour.str.contains("world cup", na=False)
        & ~tour.str.contains("qualif", na=False)
        & (df["date"] >= start)
        & df["home_team"].isin(wc)
        & df["away_team"].isin(wc)
    )
    return normalize_results(df.loc[mask])


# Official FIFA public match API (no key required). Men's World Cup = comp "17".
_FIFA_MATCHES_URL = "https://api.fifa.com/api/v3/calendar/matches"
_FIFA_FINISHED = 0  # MatchStatus code for a completed (full-time) match


def _fifa_team_name(side: dict) -> Optional[str]:
    """Pull the team name out of a FIFA match ``Home``/``Away`` object."""
    name = side.get("TeamName")
    if isinstance(name, list) and name:
        return (name[0] or {}).get("Description")
    if isinstance(name, str):
        return name
    return side.get("ShortClubName") or None


def fetch_live_results_fifa() -> pd.DataFrame:
    """Played WC-2026 results from FIFA's official public match API (keyless).

    Authoritative and updated in real time, so it surfaces matches the Kaggle
    (martj42) historical feed has not yet ingested (which can lag by days). Only
    full-time matches between two of the 48 entrants are kept. Returns an empty
    canonical frame on any failure (offline / API change / disabled), so the
    pipeline transparently degrades to the Kaggle-derived feed.
    """
    cfg = load_config()["sources"]
    if (not cfg.get("fetch_live_results", True)
            or cfg.get("live_results_provider", "fifa") != "fifa"):
        return _empty_live()
    comp = str(cfg.get("fifa_competition_id", "17"))
    season = str(cfg.get("fifa_season_id", "285023"))
    try:
        import requests  # lazy import
        url = (f"{_FIFA_MATCHES_URL}?idCompetition={comp}&idSeason={season}"
               "&count=500&language=en")
        r = requests.get(url, timeout=25,
                         headers={"User-Agent": "wc2026/1.0", "Accept": "application/json"})
        r.raise_for_status()
        data = r.json().get("Results", []) or []
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        log.warning("FIFA live results fetch failed (%s); using Kaggle-derived feed.", exc)
        return _empty_live()

    wc = set(all_teams())
    rows = []
    for m in data:
        if m.get("MatchStatus") != _FIFA_FINISHED:
            continue
        home, away = m.get("Home") or {}, m.get("Away") or {}
        hs, as_ = home.get("Score"), away.get("Score")
        if hs is None or as_ is None:
            continue
        ht, at = _fifa_team_name(home), _fifa_team_name(away)
        if not ht or not at:
            continue
        rows.append({"date": (m.get("Date") or "")[:10],
                     "home_team": ht, "away_team": at,
                     "home_score": hs, "away_score": as_,
                     "tournament": "FIFA World Cup 2026", "neutral": True})
    if not rows:
        return _empty_live()
    df = normalize_results(pd.DataFrame(rows))
    df = df[df["home_team"].isin(wc) & df["away_team"].isin(wc)].reset_index(drop=True)
    log.info("Fetched %d finished WC-2026 results from the FIFA API.", len(df))
    return df


def _dedupe_live(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate live frames (low->high priority) and dedupe per fixture.

    The fixture key is the unordered team pair (NOT the date): any two WC-2026
    teams meet at most once, so this collapses the same match even when sources
    disagree on the calendar date (UTC vs local) or swap home/away. ``keep="last"``
    lets later (higher-priority) frames win — including their authoritative date.
    """
    if not frames:
        return _empty_live()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date", "home_team", "away_team"])
    if out.empty:
        return _empty_live()
    pair = out.apply(lambda r: "|".join(sorted([r["home_team"], r["away_team"]])), axis=1)
    out = (out.assign(_k=pair)
              .drop_duplicates(subset="_k", keep="last")
              .drop(columns="_k")
              .sort_values("date")
              .reset_index(drop=True))
    return out


def fetch_live_results(history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Actual WC-2026 results played to date (for live-tournament mode).

    Sources, lowest to highest priority (later wins on a per-fixture conflict):
      1. ``history`` — the freshly-pulled Kaggle (martj42) historical results;
         World Cup games on/after the 2026 opener between two entrants. Can lag.
      2. FIFA's official public match API (``fetch_live_results_fifa``) — keyless,
         real-time, authoritative; surfaces matches Kaggle hasn't ingested yet.
      3. A user-maintained CSV at ``data/wc2026_results_manual.csv`` (columns:
         date,home_team,away_team,home_score,away_score) — manual override.

    Returns an empty canonical frame when none yields a match (pre-tournament).
    """
    if not load_config()["sources"].get("fetch_live_results", True):
        return _empty_live()

    frames: List[pd.DataFrame] = []

    if history is not None and len(history):
        derived = _wc2026_from_history(history)
        if len(derived):
            log.info("Derived %d live WC-2026 results from the historical pull.", len(derived))
        frames.append(derived)

    fifa = fetch_live_results_fifa()
    if len(fifa):
        frames.append(fifa)

    manual = Path(get_path("raw")).parent / "wc2026_results_manual.csv"
    if manual.exists():
        try:
            mdf = pd.read_csv(manual)
            mdf["tournament"] = "FIFA World Cup 2026"
            if "neutral" not in mdf.columns:
                mdf["neutral"] = True
            mframe = normalize_results(mdf)
            log.info("Loaded %d manual WC-2026 results from %s (overrides derived).",
                     len(mframe), manual.name)
            frames.append(mframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read manual WC-2026 results (%s).", exc)

    return _dedupe_live(frames)


def _restrict_to_wc(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wc = set(all_teams())
    df = df[df["team"].isin(wc)].drop_duplicates("team")
    missing = wc - set(df["team"])
    if missing:  # backfill any missing WC team from snapshot
        snap = teams_table()
        extra = pd.DataFrame({"team": list(missing),
                              value_col: [float(snap.loc[t, value_col]) for t in missing]})
        df = pd.concat([df, extra], ignore_index=True)
    return df[["team", value_col]]


# ISO3 codes for World Bank lookups (WC-2026 nations).
_WB_ISO3 = {
    "Mexico": "MEX", "South Africa": "ZAF", "South Korea": "KOR", "Czech Republic": "CZE",
    "Canada": "CAN", "Bosnia and Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "CHE",
    "Brazil": "BRA", "Morocco": "MAR", "Haiti": "HTI", "Scotland": "GBR",
    "United States": "USA", "Paraguay": "PRY", "Australia": "AUS", "Turkey": "TUR",
    "Germany": "DEU", "Curacao": "CUW", "Ivory Coast": "CIV", "Ecuador": "ECU",
    "Netherlands": "NLD", "Japan": "JPN", "Sweden": "SWE", "Tunisia": "TUN",
    "Belgium": "BEL", "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL",
    "Spain": "ESP", "Cape Verde": "CPV", "Saudi Arabia": "SAU", "Uruguay": "URY",
    "France": "FRA", "Senegal": "SEN", "Iraq": "IRQ", "Norway": "NOR",
    "Argentina": "ARG", "Algeria": "DZA", "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "PRT", "DR Congo": "COD", "Uzbekistan": "UZB", "Colombia": "COL",
    "England": "GBR", "Croatia": "HRV", "Ghana": "GHA", "Panama": "PAN",
}
