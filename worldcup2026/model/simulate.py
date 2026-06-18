"""Monte-Carlo tournament simulator.

Scorelines come from a bivariate-Poisson model (Karlis-Ntzoufras shared
component) on the RF-predicted goal means. The group stage is sampled in a
vectorised batch across all simulations; knockouts are resolved per simulation
through the configured bracket. In *live* mode, already-played group matches are
locked to their real scores and only the remainder is simulated.

Outputs per team: champion probability, probability of reaching each stage,
group-winner / group-qualification probabilities.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..tournament import bracket as bracket_cfg_fn
from ..tournament import groups as groups_fn
from . import bracket as bk
from .train import ModelBundle, build_lambda_table

log = logging.getLogger("worldcup2026.sim")

STAGES = ["group", "R32", "R16", "QF", "SF", "Final", "Champion"]
_ROUND_TO_STAGE = {"round_of_32": "R32", "round_of_16": "R16",
                   "quarter_finals": "QF", "semi_finals": "SF", "final": "Final"}


class Simulator:
    def __init__(self, bundle: ModelBundle, live_results: Optional[pd.DataFrame] = None,
                 corr: float = 0.10, et_edge: float = 0.5, seed: int = 42):
        self.teams, self.lam = build_lambda_table(bundle)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.strength = bundle.ctx.current_elo
        self.groups = groups_fn()
        self.bracket_cfg = bracket_cfg_fn()
        self.team_group = {t: g for g, ms in self.groups.items() for t in ms}
        self.third_slots, self.slot_groups = bk.third_slot_metadata(self.bracket_cfg)
        self.corr = float(corr)
        self.et_edge = float(et_edge)
        self.rng = np.random.default_rng(seed)
        self._fixed = self._index_live(live_results)
        # normalised strength for deterministic tiebreaks
        s = np.array([self.strength.get(t, 1500.0) for t in self.teams], dtype=float)
        self._strn = (s - s.min()) / (s.max() - s.min() + 1e-9)

    # -- live results ------------------------------------------------------
    def _index_live(self, live: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], Tuple[int, int]]:
        fixed: Dict[Tuple[str, str], Tuple[int, int]] = {}
        if live is None or len(live) == 0:
            return fixed
        for r in live.dropna(subset=["home_score", "away_score"]).itertuples():
            fixed[(r.home_team, r.away_team)] = (int(r.home_score), int(r.away_score))
            fixed[(r.away_team, r.home_team)] = (int(r.away_score), int(r.home_score))
        return fixed

    # -- scoreline samplers -----------------------------------------------
    def _bivpoisson(self, lam_a: np.ndarray, lam_b: np.ndarray):
        """Vectorised bivariate Poisson with a shared component."""
        lam0 = self.corr * np.minimum(lam_a, lam_b)
        y0 = self.rng.poisson(lam0)
        ga = self.rng.poisson(np.clip(lam_a - lam0, 1e-3, None)) + y0
        gb = self.rng.poisson(np.clip(lam_b - lam0, 1e-3, None)) + y0
        return ga, gb

    def _knockout(self, a: str, b: str, _ko: bool) -> Tuple[str, str, int, int]:
        i, j = self.idx[a], self.idx[b]
        lam0 = self.corr * min(self.lam[i, j], self.lam[j, i])
        y0 = self.rng.poisson(lam0)
        ga = self.rng.poisson(max(self.lam[i, j] - lam0, 1e-3)) + y0
        gb = self.rng.poisson(max(self.lam[j, i] - lam0, 1e-3)) + y0
        if ga == gb:  # extra time + penalties: edge to stronger side
            de = (self.strength.get(a, 1500) - self.strength.get(b, 1500)) / 400.0
            p = np.clip(0.5 + self.et_edge * de * 0.25, 0.05, 0.95)
            if self.rng.random() < p:
                return a, b, ga, gb
            return b, a, ga, gb
        return (a, b, ga, gb) if ga > gb else (b, a, ga, gb)

    # -- group stage (vectorised) -----------------------------------------
    def _simulate_groups(self, n: int):
        """Return winners, runners, third_team, third_key as (n, 12) arrays."""
        glabels = list(self.groups.keys())
        n_g = len(glabels)
        winners = np.empty((n, n_g), dtype=object)
        runners = np.empty((n, n_g), dtype=object)
        third_team = np.empty((n, n_g), dtype=object)
        third_key = np.zeros((n, n_g))

        for gi, g in enumerate(glabels):
            members = self.groups[g]
            m_idx = [self.idx[t] for t in members]
            pts = np.zeros((n, 4)); gf = np.zeros((n, 4)); ga = np.zeros((n, 4))
            for (la, lb) in combinations(range(4), 2):
                a, b = m_idx[la], m_idx[lb]
                fixed = self._fixed.get((members[la], members[lb]))
                if fixed is not None:
                    sa = np.full(n, fixed[0]); sb = np.full(n, fixed[1])
                else:
                    sa, sb = self._bivpoisson(np.full(n, self.lam[a, b]),
                                              np.full(n, self.lam[b, a]))
                gf[:, la] += sa; ga[:, la] += sb
                gf[:, lb] += sb; ga[:, lb] += sa
                pts[:, la] += np.where(sa > sb, 3, np.where(sa == sb, 1, 0))
                pts[:, lb] += np.where(sb > sa, 3, np.where(sb == sa, 1, 0))

            gd = gf - ga
            strn = np.array([self._strn[i] for i in m_idx])
            # composite ranking key (points >> gd >> gf >> strength)
            key = pts * 1e9 + (gd + 50) * 1e5 + gf * 1e1 + strn[None, :]
            order = np.argsort(-key, axis=1)  # best -> worst columns
            mem = np.array(members, dtype=object)
            winners[:, gi] = mem[order[:, 0]]
            runners[:, gi] = mem[order[:, 1]]
            third_pos = order[:, 2]
            third_team[:, gi] = mem[third_pos]
            rows = np.arange(n)
            third_key[:, gi] = (pts[rows, third_pos] * 1e9
                                + (gd[rows, third_pos] + 50) * 1e5
                                + gf[rows, third_pos] * 1e1
                                + strn[third_pos])
        return glabels, winners, runners, third_team, third_key

    # -- full run ----------------------------------------------------------
    def run(self, n: int = 20000) -> Dict[str, pd.DataFrame]:
        glabels, winners, runners, third_team, third_key = self._simulate_groups(n)
        gpos = {g: i for i, g in enumerate(glabels)}
        nteams = len(self.teams)

        champ = np.zeros(nteams)
        runner_up = np.zeros(nteams)
        reach = {st: np.zeros(nteams) for st in STAGES}
        group_winner = np.zeros(nteams)
        qualified = np.zeros(nteams)  # top-2 or best-third (reach R32)

        for s in range(n):
            # best-8 thirds this sim
            tk = third_key[s]
            best_g = np.argsort(-tk)[:8]
            thirds_ranked = [third_team[s, g] for g in best_g]
            assignment = bk.assign_thirds(self.third_slots, thirds_ranked,
                                          self.slot_groups, self.team_group)
            slot_to_team: Dict[str, str] = {}
            for gi, g in enumerate(glabels):
                slot_to_team[f"1{g}"] = winners[s, gi]
                slot_to_team[f"2{g}"] = runners[s, gi]
                group_winner[self.idx[winners[s, gi]]] += 1
                qualified[self.idx[winners[s, gi]]] += 1
                qualified[self.idx[runners[s, gi]]] += 1
            slot_to_team.update(assignment)
            for t in thirds_ranked:
                qualified[self.idx[t]] += 1

            res = bk.resolve_tournament(self.bracket_cfg, slot_to_team, self._knockout)

            # stage reach from participants
            for rname, matches in res["rounds"].items():
                st = _ROUND_TO_STAGE[rname]
                for m in matches:
                    reach[st][self.idx[m["a"]]] += 1
                    reach[st][self.idx[m["b"]]] += 1
            ch = res["champion"]; ru = res["runner_up"]
            if ch is not None:
                champ[self.idx[ch]] += 1
                reach["Champion"][self.idx[ch]] += 1
            if ru is not None:
                runner_up[self.idx[ru]] += 1

        reach["group"] = np.full(nteams, n, dtype=float)  # everyone plays the group stage
        reach["R32"] = qualified.copy()                   # reaching R32 == qualifying

        summary = pd.DataFrame({"team": self.teams})
        summary["group"] = [self.team_group.get(t) for t in self.teams]
        summary["P_champion"] = champ / n
        summary["P_final"] = reach["Final"] / n
        summary["P_semi"] = reach["SF"] / n
        summary["P_quarter"] = reach["QF"] / n
        summary["P_r16"] = reach["R16"] / n
        summary["P_qualify_r32"] = qualified / n
        summary["P_group_winner"] = group_winner / n
        summary["P_runner_up"] = runner_up / n
        summary["elo"] = [self.strength.get(t, 1500.0) for t in self.teams]
        summary = summary.sort_values("P_champion", ascending=False).reset_index(drop=True)

        # advancement matrix (team x stage)
        adv = pd.DataFrame({st: reach[st] / n for st in STAGES})
        adv.insert(0, "team", self.teams)
        adv = adv.sort_values("Champion", ascending=False).reset_index(drop=True)

        return {"summary": summary, "advancement": adv, "n_simulations": n}


    # -- expected (deterministic favourites) bracket ----------------------
    def _win_prob(self, a: str, b: str) -> Tuple[float, float, float]:
        """P(a win), P(draw), P(b win) from Poisson goal means (no noise)."""
        from scipy.stats import poisson
        i, j = self.idx[a], self.idx[b]
        ga = poisson.pmf(np.arange(11), self.lam[i, j])
        gb = poisson.pmf(np.arange(11), self.lam[j, i])
        m = np.outer(ga, gb)
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())

    def expected_bracket(self) -> Dict[str, object]:
        """Most-likely qualification + knockout path using expected results.

        Group order by expected points (round-robin win/draw probabilities),
        eight best thirds by expected points, knockout favourite = higher win
        probability. Honours locked live results.
        """
        glabels = list(self.groups.keys())
        standings = {}
        third_pool = []
        slot_to_team: Dict[str, str] = {}
        for g in glabels:
            members = self.groups[g]
            exp_pts = {}
            for t in members:
                pts = 0.0
                for o in members:
                    if o == t:
                        continue
                    fixed = self._fixed.get((t, o))
                    if fixed is not None:
                        pts += 3 if fixed[0] > fixed[1] else 1 if fixed[0] == fixed[1] else 0
                    else:
                        pa, pd_, pb = self._win_prob(t, o)
                        pts += 3 * pa + 1 * pd_
                exp_pts[t] = pts
            ordered = sorted(members, key=lambda t: (exp_pts[t], self.strength.get(t, 0)),
                             reverse=True)
            standings[g] = [(t, round(exp_pts[t], 2)) for t in ordered]
            slot_to_team[f"1{g}"] = ordered[0]
            slot_to_team[f"2{g}"] = ordered[1]
            third_pool.append((g, ordered[2], exp_pts[ordered[2]]))

        third_pool.sort(key=lambda x: (x[2], self.strength.get(x[1], 0)), reverse=True)
        best = third_pool[:8]
        thirds_ranked = [t for _, t, _ in best]
        assignment = bk.assign_thirds(self.third_slots, thirds_ranked,
                                      self.slot_groups, self.team_group)
        slot_to_team.update(assignment)

        def play(a, b, _ko):
            pa, _pd, pb = self._win_prob(a, b)
            return (a, b, 0, 0) if pa >= pb else (b, a, 0, 0)

        res = bk.resolve_tournament(self.bracket_cfg, slot_to_team, play)
        return {"group_standings": standings, "best_thirds": thirds_ranked,
                "knockout": res["rounds"], "champion": res["champion"],
                "runner_up": res["runner_up"], "slot_to_team": slot_to_team}


def run_simulation(bundle: ModelBundle, live_results: Optional[pd.DataFrame],
                   n: int, corr: float, et_edge: float, seed: int) -> Dict[str, object]:
    sim = Simulator(bundle, live_results, corr=corr, et_edge=et_edge, seed=seed)
    log.info("Running %d Monte-Carlo simulations...", n)
    out = sim.run(n)
    out["expected_bracket"] = sim.expected_bracket()
    return out
