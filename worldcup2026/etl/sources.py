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
from ..tournament import all_teams, teams_table
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
def fetch_live_results() -> pd.DataFrame:
    """Actual WC-2026 group results played to date.

    Primary path is a user-maintained CSV at data/wc2026_results_manual.csv
    (columns: date,home_team,away_team,home_score,away_score). This is the
    robust, source-agnostic way to feed real results in live-tournament mode.
    Returns an empty canonical frame if none exist (pre-tournament behaviour).
    """
    manual = Path(get_path("raw")).parent / "wc2026_results_manual.csv"
    if manual.exists():
        try:
            df = pd.read_csv(manual)
            df["tournament"] = "FIFA World Cup 2026"
            df["neutral"] = True
            out = normalize_results(df)
            log.info("Loaded %d live WC-2026 results from %s.", len(out), manual.name)
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read manual WC-2026 results (%s).", exc)
    return normalize_results(pd.DataFrame(
        columns=["date", "home_team", "away_team", "home_score", "away_score",
                 "tournament", "neutral"]))


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
