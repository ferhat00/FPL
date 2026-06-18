"""Group standings, FIFA tiebreakers, best-third selection and knockout resolution.

Pure tournament logic (no model dependency). The simulator injects a ``play``
callable; everything else here is deterministic FIFA rules.

Group ranking order (FIFA 2026): points -> goal difference -> goals for ->
(for sims) a stable strength tiebreak to avoid coin-flips. The eight best
third-placed teams are ranked by the same group criteria, then assigned to the
third-designated Round-of-32 slots avoiding same-group rematches.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

# play(team_a, team_b, knockout) -> (winner, loser, goals_a, goals_b)
PlayFn = Callable[[str, str, bool], Tuple[str, str, int, int]]


class GroupTable:
    """Accumulates points/goals for one group."""

    def __init__(self, teams: List[str]):
        self.teams = list(teams)
        self.pts = {t: 0 for t in teams}
        self.gf = {t: 0 for t in teams}
        self.ga = {t: 0 for t in teams}

    def add(self, a: str, b: str, ga_: int, gb: int) -> None:
        self.gf[a] += ga_; self.ga[a] += gb
        self.gf[b] += gb; self.ga[b] += ga_
        if ga_ > gb:
            self.pts[a] += 3
        elif ga_ < gb:
            self.pts[b] += 3
        else:
            self.pts[a] += 1; self.pts[b] += 1

    def standings(self, strength: Dict[str, float]) -> List[str]:
        """Return teams ranked best->worst."""
        def key(t: str):
            return (self.pts[t], self.gf[t] - self.ga[t], self.gf[t], strength.get(t, 0.0))
        return sorted(self.teams, key=key, reverse=True)

    def stats(self, t: str) -> Tuple[int, int, int]:
        return self.pts[t], self.gf[t] - self.ga[t], self.gf[t]


def select_best_thirds(third_stats: Dict[str, Tuple[int, int, int]],
                       strength: Dict[str, float], n: int = 8) -> List[str]:
    """Rank third-placed teams (pts, GD, GF, strength) and return the best ``n``."""
    teams = list(third_stats)
    teams.sort(key=lambda t: (*third_stats[t], strength.get(t, 0.0)), reverse=True)
    return teams[:n]


def assign_thirds(third_slots: List[str], thirds_ranked: List[str],
                  slot_groups: Dict[str, str], team_group: Dict[str, str]) -> Dict[str, str]:
    """Map third tokens (T1..Tn) -> team, avoiding same-group matchups where possible.

    ``third_slots`` are tokens in bracket order; ``slot_groups`` maps each token to
    the group of the *winner* it will face (so we avoid pairing a third with its
    own group's winner).
    """
    assignment = dict(zip(third_slots, thirds_ranked))
    tokens = list(third_slots)
    for i, tok in enumerate(tokens):
        team = assignment[tok]
        if team_group.get(team) == slot_groups.get(tok):
            # find a swap partner that resolves both clashes
            for j in range(len(tokens)):
                if j == i:
                    continue
                tok2 = tokens[j]
                t2 = assignment[tok2]
                if (team_group.get(t2) != slot_groups.get(tok)
                        and team_group.get(team) != slot_groups.get(tok2)):
                    assignment[tok], assignment[tok2] = t2, team
                    break
    return assignment


def _winner_slot_groups(round_of_32: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, str]]:
    """Identify third tokens and the group letter of the winner they face."""
    third_slots, slot_groups = [], {}
    for m in round_of_32:
        for pos, other in (("top", "bottom"), ("bottom", "top")):
            tok = m[pos]
            if tok.startswith("T"):
                third_slots.append(tok)
                opp = m[other]
                slot_groups[tok] = opp[1:] if opp[:1] in ("1", "2") else ""
    # keep T-order stable (T1, T2, ...)
    third_slots = sorted(set(third_slots), key=lambda s: int(s[1:]))
    return third_slots, slot_groups


def resolve_tournament(bracket_cfg: Dict[str, List[Dict[str, str]]],
                       slot_to_team: Dict[str, str],
                       play: PlayFn) -> Dict[str, object]:
    """Play out all knockout rounds; return per-round winners and the champion."""
    resolved: Dict[str, str] = dict(slot_to_team)   # token/"W:id" -> team
    rounds_out: Dict[str, List[Dict[str, object]]] = {}
    order = ["round_of_32", "round_of_16", "quarter_finals",
             "semi_finals", "final"]
    final_match = None
    for rnd in order:
        matches = bracket_cfg.get(rnd, [])
        rounds_out[rnd] = []
        for m in matches:
            a = resolved[m["top"]]
            b = resolved[m["bottom"]]
            winner, loser, ga_, gb = play(a, b, True)
            resolved["W:" + m["id"]] = winner
            rounds_out[rnd].append({"id": m["id"], "a": a, "b": b,
                                    "ga": ga_, "gb": gb, "winner": winner})
            if rnd == "final":
                final_match = (winner, loser)
    champion = final_match[0] if final_match else None
    runner_up = final_match[1] if final_match else None
    return {"champion": champion, "runner_up": runner_up, "rounds": rounds_out}


def third_slot_metadata(bracket_cfg: Dict[str, List[Dict[str, str]]]):
    """Public helper: (third_slots, slot_groups) for the configured R32."""
    return _winner_slot_groups(bracket_cfg["round_of_32"])
