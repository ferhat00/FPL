#!/usr/bin/env python3
"""
Fantasy Premier League (FPL) Optimizer — 2025/26 season

What this does
--------------
• Pulls current-season data (players, prices, teams, fixtures) from the official FPL API.
• Optionally pulls each player's prior-season summary to estimate points-per-90 from last year.
• Projects near‑term points (default horizon = next 6 gameweeks) blending:
    - FPL's own expected points for the next GW (ep_next) when available,
    - Current-season production per 90 (if minutes exist),
    - Last-season per‑90 production (history_past),
    - Fixture difficulty for each player's team over the horizon,
    - Availability flags (injury/suspension chance).
• Solves a mixed‑integer program to pick a 15‑man squad under FPL constraints and an optimal starting XI
  in a valid formation, including captain choice (doubling the top projected XI scorer).

Constraints enforced (per FPL rules)
------------------------------------
• Budget: £100.0m for the initial squad of 15 (costs in API are in tenths; we convert to £m).
• Squad size: 15 = 2 GKP, 5 DEF, 5 MID, 3 FWD.
• Per‑club limit: max 3 players from any single Premier League club.
• Starting XI: exactly 11 with formation rules — DEF 3–5, MID 2–5, FWD 1–3; 1 GKP.

Outputs
-------
• Prints the chosen squad with price, club, projected horizon points, and whether they start.
• Shows the best formation, captain/vice, XI projection over the chosen horizon, and bank remaining.
• Saves CSV and a human‑readable summary to ./output/ .

Usage
-----
    pip install -U pandas numpy requests pulp tqdm
    python fpl_optimizer.py --horizon 6 --budget 100.0 --threads 8

Notes
-----
• This script calls the public FPL API endpoints directly; no login required.
• Network usage: it will parallel‑fetch element summaries for last‑season stats; you can disable that
  with --no-summaries to run faster (it will rely more on ep_next / current stats).
• Projections are simplistic by design and should be a starting point. Feel free to plug in your own
  model inside `project_points()`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import pulp
except ImportError as e:
    raise SystemExit("PuLP is required. Please `pip install pulp`.\n" + str(e))

FPL_BASE = "https://fantasy.premierleague.com/api"
ENDPOINTS = {
    "bootstrap": f"{FPL_BASE}/bootstrap-static/",
    "fixtures_future": f"{FPL_BASE}/fixtures/?future=1",
    # element-summary/{id}/ — per-player gw history and past seasons
    "element_summary": f"{FPL_BASE}/element-summary/{{player_id}}/",
}

CACHE_DIR = Path("./cache").resolve()
OUT_DIR = Path("./output").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------

def http_get_json(url: str, timeout: int = 20) -> dict:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "FPL-Optimizer/1.0"})
    r.raise_for_status()
    return r.json()


def load_bootstrap() -> dict:
    return http_get_json(ENDPOINTS["bootstrap"])  # teams, elements, element_types, events, etc.


def load_fixtures_future() -> List[dict]:
    return http_get_json(ENDPOINTS["fixtures_future"])  # list of fixtures not yet finished


def load_element_summary(player_id: int, use_cache: bool = True) -> dict:
    cache_fp = CACHE_DIR / f"element_summary_{player_id}.json"
    if use_cache and cache_fp.exists():
        try:
            return json.loads(cache_fp.read_text())
        except Exception:
            pass
    data = http_get_json(ENDPOINTS["element_summary"].format(player_id=player_id))
    try:
        cache_fp.write_text(json.dumps(data))
    except Exception:
        pass
    return data


# -----------------------------
# Data wrangling
# -----------------------------

def prepare_players_df(bootstrap: dict) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap["elements"])  # players
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    etypes = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name_short", "plural_name"]]

    # Normalize/rename for convenience
    elements.rename(
        columns={
            "id": "player_id",
            "element_type": "pos_id",
            "team": "team_id",
            "now_cost": "cost_tenths",
        },
        inplace=True,
    )

    df = elements[
        [
            "player_id",
            "first_name",
            "second_name",
            "web_name",
            "pos_id",
            "team_id",
            "status",
            "chance_of_playing_next_round",
            "cost_tenths",
            "selected_by_percent",
            "minutes",
            "total_points",
            "points_per_game",
            "ict_index",
            "influence",
            "creativity",
            "threat",
            "ep_next",
            "ep_this",
        ]
    ].copy()

    df = df.merge(teams, left_on="team_id", right_on="id", how="left", suffixes=("", "_team"))
    df.drop(columns=["id"], inplace=True)
    df = df.merge(etypes, left_on="pos_id", right_on="id", how="left", suffixes=("", "_etype"))
    df.drop(columns=["id"], inplace=True)

    # Friendly columns
    df["team_name"] = df["name"]
    df["team_short"] = df["short_name"]
    df["pos"] = df["singular_name_short"]  # GKP/DEF/MID/FWD
    df.drop(columns=["name", "short_name", "plural_name", "singular_name_short"], inplace=True)

    # Type conversions
    df["cost_m"] = df["cost_tenths"].astype(float) / 10.0
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["ep_next"] = pd.to_numeric(df["ep_next"], errors="coerce")
    df["ep_this"] = pd.to_numeric(df["ep_this"], errors="coerce")
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
    df["influence"] = pd.to_numeric(df["influence"], errors="coerce")
    df["creativity"] = pd.to_numeric(df["creativity"], errors="coerce")
    df["threat"] = pd.to_numeric(df["threat"], errors="coerce")

    df["full_name"] = df["first_name"].fillna("").str.strip() + " " + df["second_name"].fillna("").str.strip()
    df["display_name"] = np.where(df["web_name"].notna() & (df["web_name"].str.len() > 0), df["web_name"], df["full_name"].str.strip())

    return df


def build_team_fixture_table(fixtures_future: List[dict], horizon: int = 6) -> Dict[int, List[int]]:
    """Return mapping: team_id -> list of next `horizon` fixture difficulties (1..5), averaged per team.
    We average home/away difficulty per the role the team plays in each fixture.
    """
    # Build for each team a chronological list of upcoming difficulties
    by_team: Dict[int, List[Tuple[int, int]]] = {}
    for fx in fixtures_future:
        event = fx.get("event")
        if event is None:
            continue
        th = int(fx["team_h"])
        ta = int(fx["team_a"])
        dh = int(fx.get("team_h_difficulty", 3))
        da = int(fx.get("team_a_difficulty", 3))
        by_team.setdefault(th, []).append((event, dh))
        by_team.setdefault(ta, []).append((event, da))

    # Sort by event and cut horizon
    out: Dict[int, List[int]] = {}
    for tid, lst in by_team.items():
        lst_sorted = sorted(lst, key=lambda x: (x[0], x[1]))
        diffs = [d for _, d in lst_sorted[:horizon]]
        if len(diffs) < horizon:
            # pad with neutral difficulty 3 if schedule shorter
            diffs = diffs + [3] * (horizon - len(diffs))
        out[tid] = diffs
    return out


def fixture_multiplier(d: int) -> float:
    """Map FDR (1..5) to a rough multiplier; 1=easy, 5=hard."""
    mapping = {1: 1.10, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90}
    return mapping.get(int(d), 1.0)


def estimate_availability(status: str, chance_next: Optional[float]) -> float:
    if status == "a":
        return 1.0
    if status == "d":
        if chance_next is not None and not math.isnan(chance_next):
            return max(0.0, min(1.0, chance_next / 100.0))
        return 0.5
    if status in {"i", "s"}:  # injured, suspended
        return 0.0
    # 'u' (unavailable) or other
    return 0.2


def last_season_pp90(summary_json: dict) -> Optional[float]:
    past = summary_json.get("history_past") or []
    if not past:
        return None
    # pick most recent season by season_name sorting like '2024/25'
    def parse_season(s: str) -> Tuple[int, int]:
        try:
            a, b = s.split("/")
            return int("20" + a[-2:]), int("20" + b)
        except Exception:
            return (0, 0)

    past_sorted = sorted(past, key=lambda x: parse_season(x.get("season_name", "0000/00")), reverse=True)
    latest = past_sorted[0]
    minutes = latest.get("minutes", 0) or 0
    points = latest.get("total_points", 0) or 0
    if minutes <= 0:
        return None
    return float(points) / (minutes / 90.0)


def current_season_p90(row: pd.Series) -> Optional[float]:
    minutes = row.get("minutes", 0) or 0
    points = row.get("total_points", 0) or 0
    if minutes and minutes > 0:
        return float(points) / (minutes / 90.0)
    return None


def project_points(
    players: pd.DataFrame,
    team_fdr: Dict[int, List[int]],
    horizon: int = 6,
    use_summaries: bool = True,
    threads: int = 8,
) -> pd.DataFrame:
    """
    Compute projected points over the next `horizon` GWs and per-match projections.
    Adds columns: proj_per_match, proj_horizon, avail, fixture_factor, src
    """
    p = players.copy()

    # Fetch per-player summaries in parallel (for last-season pp90)
    last_pp90_map: Dict[int, Optional[float]] = {pid: None for pid in p["player_id"]}
    if use_summaries:
        ids = p["player_id"].tolist()
        with cf.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            futs = {ex.submit(load_element_summary, pid): pid for pid in ids}
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Fetching player summaries"):
                pid = futs[fut]
                try:
                    data = fut.result()
                    last_pp90_map[pid] = last_season_pp90(data)
                except Exception:
                    last_pp90_map[pid] = None

    p["last_pp90"] = p["player_id"].map(last_pp90_map)
    p["curr_p90"] = p.apply(current_season_p90, axis=1)

    # Availability
    p["chance_of_playing_next_round"] = pd.to_numeric(p["chance_of_playing_next_round"], errors="coerce")
    p["avail"] = p.apply(lambda r: estimate_availability(r["status"], r["chance_of_playing_next_round"]), axis=1)

    # Fixture factor: average multiplier over horizon
    def avg_fixture_mult(team_id: int) -> float:
        diffs = team_fdr.get(int(team_id), [3] * horizon)
        mults = [fixture_multiplier(d) for d in diffs[:horizon]]
        return float(np.mean(mults)) if mults else 1.0

    p["fixture_factor"] = p["team_id"].map(avg_fixture_mult)

    # Base per-match projection
    def per_match_row(r: pd.Series) -> Tuple[float, str]:
        # Use FPL's ep_next if available as a baseline per match (already considers next fixture)
        epn = r.get("ep_next")
        src = "ep_next"
        if pd.notna(epn) and epn is not None and epn > 0:
            base = float(epn)
            # For horizon >1, adjust by team fixture factor to avoid double counting too much
            # We lightly shrink toward 1.0 so ep_next isn't overemphasized
            f = 0.5 * r["fixture_factor"] + 0.5 * 1.0
            per_match = base * f * r["avail"]
            return per_match, src
        # else blend last and current p90s
        lp90 = r.get("last_pp90")
        cp90 = r.get("curr_p90")
        src = "blend_p90"
        if (cp90 is not None) and not math.isnan(cp90):
            # trust current more as minutes accumulate (cap at 900 mins ~ 10 full games)
            minutes = r.get("minutes", 0) or 0
            w = max(0.0, min(1.0, minutes / 900.0))
            if (lp90 is not None) and not math.isnan(lp90):
                p90 = w * cp90 + (1 - w) * lp90
            else:
                p90 = cp90
        else:
            p90 = lp90 if (lp90 is not None and not math.isnan(lp90)) else None

        if p90 is None:
            # very weak fallback: scale ICT to a pseudo per-match number per position
            ict = r.get("ict_index") or 0.0
            # heuristic baseline per position
            base_pos = {"GKP": 3.5, "DEF": 3.8, "MID": 4.5, "FWD": 4.2}.get(r["pos"], 4.0)
            p90 = base_pos * (1.0 + 0.02 * (float(ict) / 100.0))
            src = "ict_fallback"

        per_match = float(p90) * r["avail"] * r["fixture_factor"]
        return per_match, src

    vals = p.apply(per_match_row, axis=1, result_type="expand")
    p["proj_per_match"] = pd.to_numeric(vals[0], errors="coerce")
    p["proj_src"] = vals[1]
    p["proj_horizon"] = p["proj_per_match"] * float(horizon)

    # Clean NaNs
    p["proj_per_match"].fillna(0.0, inplace=True)
    p["proj_horizon"].fillna(0.0, inplace=True)

    return p


# -----------------------------
# Optimization model
# -----------------------------

def solve_optimal_squad(players: pd.DataFrame, budget_m: float = 100.0) -> Dict[str, any]:
    # Decision vars
    player_ids = players["player_id"].tolist()
    id_index = {pid: i for i, pid in enumerate(player_ids)}

    # Binary: pick in 15-man squad
    x = pulp.LpVariable.dicts("pick", player_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)
    # Binary: chosen in starting XI
    y = pulp.LpVariable.dicts("start", player_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)
    # Binary: captain among starters
    c = pulp.LpVariable.dicts("captain", player_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("FPL_Optimizer", pulp.LpMaximize)

    # Parameters
    cost = {pid: float(players.loc[players.player_id == pid, "cost_m"].values[0]) for pid in player_ids}
    team = {pid: int(players.loc[players.player_id == pid, "team_id"].values[0]) for pid in player_ids}
    pos = {pid: players.loc[players.player_id == pid, "pos"].values[0] for pid in player_ids}
    points = {pid: float(players.loc[players.player_id == pid, "proj_horizon"].values[0]) for pid in player_ids}

    # Objective: starting XI horizon points + captain doubles once more
    prob += (
        pulp.lpSum(y[pid] * points[pid] for pid in player_ids)
        + pulp.lpSum(c[pid] * points[pid] for pid in player_ids)
    )

    # Squad size and positions
    prob += pulp.lpSum(x[pid] for pid in player_ids) == 15, "squad_size"
    prob += pulp.lpSum(x[pid] for pid in player_ids if pos[pid] == "GKP") == 2, "gkp_count"
    prob += pulp.lpSum(x[pid] for pid in player_ids if pos[pid] == "DEF") == 5, "def_count"
    prob += pulp.lpSum(x[pid] for pid in player_ids if pos[pid] == "MID") == 5, "mid_count"
    prob += pulp.lpSum(x[pid] for pid in player_ids if pos[pid] == "FWD") == 3, "fwd_count"

    # Budget
    prob += pulp.lpSum(x[pid] * cost[pid] for pid in player_ids) <= float(budget_m), "budget"

    # Per-team limit
    teams = sorted(set(team.values()))
    for t in teams:
        prob += pulp.lpSum(x[pid] for pid in player_ids if team[pid] == t) <= 3, f"team_limit_{t}"

    # Starting XI constraints
    prob += pulp.lpSum(y[pid] for pid in player_ids) == 11, "xi_size"
    # Link start only if picked
    for pid in player_ids:
        prob += y[pid] <= x[pid], f"start_implies_pick_{pid}"
        prob += c[pid] <= y[pid], f"capt_implies_start_{pid}"
    prob += pulp.lpSum(c[pid] for pid in player_ids) == 1, "one_captain"

    # Formation ranges
    prob += pulp.lpSum(y[pid] for pid in player_ids if pos[pid] == "GKP") == 1, "xi_gkp=1"
    prob += 3 <= pulp.lpSum(y[pid] for pid in player_ids if pos[pid] == "DEF") <= 5, "xi_def_range"
    prob += 2 <= pulp.lpSum(y[pid] for pid in player_ids if pos[pid] == "MID") <= 5, "xi_mid_range"
    prob += 1 <= pulp.lpSum(y[pid] for pid in player_ids if pos[pid] == "FWD") <= 3, "xi_fwd_range"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

    picked = [pid for pid in player_ids if pulp.value(x[pid]) > 0.5]
    starters = [pid for pid in player_ids if pulp.value(y[pid]) > 0.5]
    captain = next((pid for pid in player_ids if pulp.value(c[pid]) > 0.5), None)

    # Summaries
    spent = sum(cost[pid] for pid in picked)
    xi_points = sum(points[pid] for pid in starters)
    cap_points = points.get(captain, 0.0) if captain is not None else 0.0
    total_obj = xi_points + cap_points

    return {
        "picked": picked,
        "starters": starters,
        "captain": captain,
        "spent": spent,
        "bank": float(budget_m) - spent,
        "xi_points": xi_points,
        "captain_points": cap_points,
        "objective": total_obj,
    }


# -----------------------------
# Pretty printing and saving
# -----------------------------

def format_team_output(players: pd.DataFrame, solution: Dict[str, any], horizon: int) -> Tuple[pd.DataFrame, str]:
    pmap = {pid: row for pid, row in players.set_index("player_id").iterrows()}

    def row_for(pid: int, start_flag: bool, captain_flag: bool) -> dict:
        r = pmap[pid]
        return {
            "player_id": pid,
            "name": r["display_name"],
            "pos": r["pos"],
            "team": r["team_short"],
            "price": r["cost_m"],
            "proj_per_match": r["proj_per_match"],
            "proj_{H}gw".format(H=horizon): r["proj_horizon"],
            "start": start_flag,
            "captain": captain_flag,
            "src": r["proj_src"],
        }

    picked = solution["picked"]
    starters = set(solution["starters"])
    captain = solution["captain"]

    rows = [row_for(pid, pid in starters, pid == captain) for pid in picked]
    df = pd.DataFrame(rows).sort_values(by=["start", "proj_{H}gw".format(H=horizon)], ascending=[False, False])

    # Build a textual summary
    xi = df[df["start"]]
    bench = df[~df["start"]]

    pos_counts = xi.groupby("pos").size().to_dict()
    formation = f"{pos_counts.get('DEF',0)}-{pos_counts.get('MID',0)}-{pos_counts.get('FWD',0)}"

    cap_name = df.loc[df["captain"], "name"].iloc[0] if (df["captain"].any()) else "(none)"

    summary = []
    summary.append("Optimal formation: " + formation)
    summary.append(f"Starting XI projected points (next {horizon} GW): {solution['xi_points']:.1f}")
    summary.append(f"Captain: {cap_name} (+{solution['captain_points']:.1f} extra)")
    summary.append(f"Total objective: {solution['objective']:.1f}")
    summary.append(f"Spend: £{solution['spent']:.1f}m | Bank: £{solution['bank']:.1f}m")

    # Nicely list starters & bench
    def fmt_player_row(r: pd.Series) -> str:
        return f"{r['name']} ({r['team']} {r['pos']} £{r['price']:.1f}m) — proj {r[f'proj_{horizon}gw']:.1f}"

    summary.append("\nStarting XI:")
    for _, r in xi.sort_values(by=["pos", f"proj_{horizon}gw"], ascending=[True, False]).iterrows():
        star = " (C)" if bool(r["captain"]) else ""
        summary.append("  - " + fmt_player_row(r) + star)

    summary.append("\nBench:")
    for _, r in bench.sort_values(by=["pos", f"proj_{horizon}gw"], ascending=[True, False]).iterrows():
        summary.append("  - " + fmt_player_row(r))

    return df, "\n".join(summary)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build an optimal FPL squad under official rules.")
    ap.add_argument("--horizon", type=int, default=6, help="Number of future GWs to project (default 6)")
    ap.add_argument("--budget", type=float, default=100.0, help="Initial squad budget in £m (default 100.0)")
    ap.add_argument("--threads", type=int, default=8, help="Parallel threads for player summaries (default 8)")
    ap.add_argument("--no-summaries", action="store_true", help="Skip fetching element summaries (faster; less accurate)")
    args = ap.parse_args()

    print("Loading FPL data...")
    bootstrap = load_bootstrap()
    fixtures = load_fixtures_future()

    players = prepare_players_df(bootstrap)
    team_fdr = build_team_fixture_table(fixtures, horizon=args.horizon)

    players_proj = project_points(
        players,
        team_fdr=team_fdr,
        horizon=args.horizon,
        use_summaries=(not args.no_summaries),
        threads=args.threads,
    )

    # Filter obvious non-options: zero-cost (shouldn't exist), totally unavailable, etc.
    candidates = players_proj[(players_proj["cost_m"] > 0) & (players_proj["avail"] > 0.0)].copy()

    print(f"Optimizing over {len(candidates)} candidates...")
    solution = solve_optimal_squad(candidates, budget_m=args.budget)

    roster_df, summary_txt = format_team_output(candidates, solution, horizon=args.horizon)

    # Save outputs
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_fp = OUT_DIR / f"optimal_squad_{ts}.csv"
    txt_fp = OUT_DIR / f"optimal_squad_{ts}.txt"
    roster_df.to_csv(csv_fp, index=False)
    txt_fp.write_text(summary_txt)

    print("\n=== OPTIMAL SQUAD ===")
    print(summary_txt)
    print(f"\nSaved: {csv_fp}")


if __name__ == "__main__":
    main()
