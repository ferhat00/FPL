"""Group standings, FIFA tiebreakers, best-third selection and knockout resolution.

Pure tournament logic (no model dependency). The simulator injects a ``play``
callable; everything else here is deterministic FIFA rules.

Group ranking order (FIFA 2026): points -> goal difference -> goals for ->
(for sims) a stable strength tiebreak to avoid coin-flips. The eight best
third-placed teams are ranked by the same group criteria, then assigned to the
third-designated Round-of-32 slots honouring the official FIFA Annex C
eligibility lists (each slot accepts a third only from a fixed set of groups).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("worldcup2026.bracket")

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


def _match_thirds(third_slots: List[str],
                  candidates: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
    """Bijection slot -> team via most-constrained-slot-first DFS with backtracking.

    ``candidates[slot]`` lists the teams eligible for that slot in priority order
    (best-ranked third first). Deterministic: ties on availability break by the
    slot's index in ``third_slots``. Returns None if no perfect matching exists.
    """
    slot_index = {s: i for i, s in enumerate(third_slots)}
    used: set = set()
    result: Dict[str, str] = {}

    def solve(remaining: frozenset) -> bool:
        if not remaining:
            return True
        # pick the slot with the fewest still-available candidates (MRV)
        def avail(s: str) -> List[str]:
            return [t for t in candidates[s] if t not in used]
        slot = min(remaining, key=lambda s: (len(avail(s)), slot_index[s]))
        for team in avail(slot):            # priority = third-rank order
            used.add(team); result[slot] = team
            if solve(remaining - {slot}):
                return True
            used.discard(team); del result[slot]
        return False

    return result if solve(frozenset(third_slots)) else None


def assign_thirds(third_slots: List[str], thirds_ranked: List[str],
                  slot_eligible: Dict[str, List[str]],
                  team_group: Dict[str, str]) -> Dict[str, str]:
    """Map third tokens (T1..Tn) -> team honouring official slot eligibility.

    ``slot_eligible[token]`` is the list of group letters whose third-placed team
    may occupy that slot (FIFA Annex C). The 8 qualifying thirds are matched to
    the 8 slots so each third's group is eligible for its slot, preferring
    higher-ranked thirds. Falls back to a naive zip (and warns) only if the
    eligibility lists admit no perfect matching — which never happens with the
    official table.
    """
    candidates = {s: [t for t in thirds_ranked
                      if team_group.get(t) in slot_eligible.get(s, [])]
                  for s in third_slots}
    assignment = _match_thirds(third_slots, candidates)
    if assignment is None:
        log.warning("No eligible third-place assignment found; falling back to "
                    "rank order (check tournament bracket eligible_groups).")
        assignment = dict(zip(third_slots, thirds_ranked))
    return assignment


def _third_slot_eligibility(round_of_32: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Identify third tokens (T1..Tn) and each slot's eligible-group list."""
    third_slots, slot_eligible = [], {}
    for m in round_of_32:
        for pos in ("top", "bottom"):
            tok = m[pos]
            if isinstance(tok, str) and tok.startswith("T"):
                third_slots.append(tok)
                slot_eligible[tok] = list(m.get("eligible_groups", []))
    # keep T-order stable (T1, T2, ...)
    third_slots = sorted(set(third_slots), key=lambda s: int(s[1:]))
    return third_slots, slot_eligible


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
    """Public helper: (third_slots, slot_eligible) for the configured R32.

    ``slot_eligible[token]`` is the list of group letters whose third-placed team
    may fill that slot (official FIFA Annex C, read from the YAML).
    """
    return _third_slot_eligibility(bracket_cfg["round_of_32"])
