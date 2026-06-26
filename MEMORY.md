# MEMORY.md — Decision Log

Read this at the start of every session. Never contradict a logged decision without flagging it first.

---

## 2026-06-17, World Cup 2026 prediction pipeline created
**What was decided:** Build a standalone `worldcup2026/` package (ETL → features → random forest →
Monte-Carlo simulation → Streamlit dashboard) that forecasts the 2026 World Cup winner. Lives alongside
the existing FPL fantasy optimizer, which is untouched.
**Why:** The user requested a full data-science pipeline for the tournament; the existing repo is fantasy
squad optimization and shares no logic.
**What was rejected:** Extending the FPL scripts (different domain); a single notebook (less maintainable
than a tested package).

## 2026-06-17, Modelling approach = Groll-style random forest on goals
**What was decided:** A single `RandomForestRegressor` predicts goals scored by a team in a match from
team/opponent covariates; applied twice per fixture → (λ_home, λ_away). Knockouts/groups are then
Monte-Carlo simulated via a bivariate-Poisson (Karlis-Ntzoufras) scoreline model with Dixon-Coles low-score
behaviour. Validation is `TimeSeriesSplit` only, plus a train-before/test-after backtest scored with RPS,
log-loss and Brier vs an Elo-only baseline.
**Why:** Groll et al. (2019, arXiv:1806.03208) is the published SOTA random-forest method for exactly this
problem and matches the user's "random forest + state-of-the-art features" requirement.
**What was rejected:** A direct 1X2 match-outcome classifier (can't produce knockout scorelines or extra
time); pure Poisson/Dixon-Coles regression (Groll shows RF outperforms it); gradient boosting (user asked
for random forest specifically — kept as an `extra_trees` config option only).

## 2026-06-17, Feature set = robust core + market value
**What was decided:** Features = dynamic Elo (+diff), FIFA rank, squad market value & age, GDP/population,
host & confederation flags, Dixon-Coles time-decayed rolling form (team + opponent), days rest.
**Why:** User chose "robust core + market value" over full Groll replication. These are Groll's
highest-importance covariates without the most brittle scrapers (per-player CL counts, bookmaker odds).
**What was rejected:** Full Groll replication (extra fragile scrapers); lean Elo-only (lower ceiling).
**How to apply:** Elo is the headline feature and must be on the same scale at train and predict time — the
Elo engine is seeded from the current snapshot so RF (which cannot extrapolate) doesn't saturate on strong
teams. See `features/engineering.py::build_training_matrix`.

## 2026-06-17, Data source = Kaggle, with offline synthetic fallback
**What was decided:** Primary data = Kaggle (martj42 results 1872-2026 + an Elo dataset), key read from
`config.yaml`. Every source degrades gracefully: if unavailable, a calibrated **synthetic** history
(Elo→Poisson goals) is generated so the whole pipeline + dashboard run offline. Live WC-2026 results are
fed via `worldcup2026/data/wc2026_results_manual.csv` (source-agnostic) in live-tournament mode.
**Why:** User chose Kaggle; robustness requires the pipeline to run with no key/offline for demos and CI.
**What was rejected:** Hard dependency on live scraping (brittle); bundling a large static dump (stale).
**How to apply:** Synthetic mode is clearly flagged in logs and the dashboard. Real results override it.

## 2026-06-17, Tournament config is the source of truth (VERIFY playoff slots)
**What was decided:** Groups A-L, team attributes, and the knockout bracket live in
`worldcup2026/data/tournament_2026.yaml`. Group fixtures are generated as a round-robin; the bracket is a
data-driven default (8 best thirds + fixed progression).
**Why:** Single human-verifiable artifact; editing the YAML re-wires the whole simulation.
**What was rejected:** Hardcoding fixtures in code (not reviewable).
**What needs attention:** Six slots (Czech Republic, Bosnia & Herzegovina, Turkey, Sweden, Iraq, DR Congo)
were decided by March-2026 play-offs — verify identities against the official FIFA source. The exact
winner-vs-third R32 pairings are a valid default and should be checked against FIFA's published bracket.

## 2026-06-26, Live WC-2026 results auto-derived from the Kaggle pull
**What was decided:** `fetch_live_results()` now derives the played WC-2026 matches from the freshly-pulled
historical table (World Cup games dated on/after `meta.opening_match_date`, between two of the 48 entrants)
instead of requiring a manual CSV. The `wc2026_results_manual.csv` path is retained as an optional override
that wins on a per-fixture conflict (for results Kaggle has not yet ingested).
**Why:** The dashboard showed "0 WC-2026 results" while 48 real group results were already sitting in the
Kaggle (martj42) history — live-tournament mode read only the (non-existent) manual CSV, so it re-simulated
already-played matches. Auto-deriving locks them with no manual step and updates every Refresh.
**What was rejected:** A one-off manual CSV snapshot (goes stale each matchday).
**How to apply:** This supersedes the manual-CSV-primary part of the 2026-06-17 "Data source = Kaggle"
decision. Derived rows keep their source `tournament` label, so unioning them back into history is a no-op
after de-dup (training table unchanged). Only the 48 rows with non-null scores are locked; future fixtures
with null scores are ignored by the simulator.

## 2026-06-26, Knockout bracket switched to the OFFICIAL FIFA 2026 structure + full-bracket dashboard
**What was decided:** Replaced the simplified default knockout map in `tournament_2026.yaml` with the
official FIFA 2026 bracket (match numbers M73-M104): the real winner-vs-runner-up ties, the non-sequential
Round-of-16 cross-links, and an `eligible_groups` list on each of the 8 winner-vs-third R32 ties (the FIFA
Annex C constraint on which group's third-placed team may fill each slot). `bracket.py::assign_thirds` now
solves a constrained bijection (most-constrained-slot-first DFS with backtracking, `_match_thirds`) honouring
those eligibility lists instead of the old "avoid the facing winner's group" heuristic;
`third_slot_metadata` now returns `(third_slots, slot_eligible)`. The dashboard renders the full projected
bracket Round-of-32 → Final in a 5-column layout (previously only QF/SF/Final).
**Why:** User requirement — the dashboard must show the full bracket from R32 onwards and "who plays whom"
must match the official bracket given group-qualifying projections. The structure was cross-verified against
Wikipedia's knockout-stage page and confirmed real R32 results (England[L] v Cape Verde[3rd-H], USA[D] v
Bosnia[3rd-B], Spain[H] v Austria[2-J], Argentina[J] v Uruguay[2-H]). Because `resolve_tournament` reads the
tree generically from YAML, the YAML rewrite drives BOTH the Monte-Carlo `run()` and `expected_bracket()`.
**What was rejected:** The exact 495-row Annex C lookup table (combination-dependent slot order) — chosen
"eligibility matching" tier instead: simpler, deterministic, and a test proves all C(12,8)=495 qualifying-
third combinations admit a perfect eligibility-respecting matching (fallback never fires). A Plotly bracket
tree was rejected in favour of the simpler 5-column layout.
**How to apply:** This resolves the 2026-06-17 "VERIFY playoff slots / check R32 winner-vs-third pairings"
flag for the bracket structure (group *identities* still subject to the playoff-slot caveat). Editing the
YAML still re-wires everything. The third-place play-off (FIFA M103) is intentionally not modelled.

## 2026-06-26, Dashboard group-stage view (actual + projected, side by side)
**What was decided:** Added a new pure module `worldcup2026/standings.py` (`results_by_group`,
`group_standings`, `third_place_ranking`, `r32_qualifiers`) that computes ACTUAL group-stage state — all
game results, full standings (Pld/W/D/L/GF/GA/GD/Pts), the 12-team third-place ranking (best 8 flagged), and
the 32 R32 qualifiers — from the played `wc2026_live` results, ranked by the same FIFA key the simulator uses
(pts→GD→GF→Elo) and reusing `model/bracket.py::select_best_thirds`. A new dashboard "📋 Group stage" section
(rendered after Refresh Data, no model needed) shows these and, when a simulation exists in session, the
model's PROJECTED-final standings/thirds/qualifiers (from `expected_bracket`) side by side.
**Why:** User asked the dashboard to also show group results, final positions, the 3rd-place table and who
qualifies to R32. Mid-tournament (48 of 72 games played, matchday 3 pending), the user chose to see both the
factual table and the model forecast together.
**What was rejected:** Driving the view only from the model projection (loses the factual results) or only
from actual results (can't give a definitive final until matchday 3) — "both side by side" keeps each honest.
Tying the section to Run Prediction (rejected: the actual side needs only `wc2026_live` + Elo, so it renders
straight after Refresh).
**How to apply:** Actual standings are PROVISIONAL until all 72 results are in and auto-finalise on the next
Refresh after the final matchday (2026-06-27); the UI labels this. Tiebreak uses Elo (no FIFA head-to-head),
consistent with the simulator. Strength for the actual table comes from the `team_attrs` parquet `elo` column.

## 2026-06-26, Live results now pulled from the official FIFA API (Kaggle martj42 lags)
**What was decided:** Added `etl/sources.py::fetch_live_results_fifa()` — the played WC-2026 results from
FIFA's official **keyless** public match API (`api.fifa.com/api/v3/calendar/matches`, men's World Cup
competition `17`, 2026 season `285023`). It is wired into `fetch_live_results` at higher priority than the
Kaggle-derived feed (priority: manual CSV > FIFA API > Kaggle history), with a new orientation-AND-date-
independent dedup (`_dedupe_live`, keyed on the unordered team pair, since any two WC-2026 teams meet at most
once). New config knobs in `config.example.yaml`: `sources.live_results_provider` (`fifa`|`none`),
`fifa_competition_id`, `fifa_season_id`.
**Why:** The dashboard showed only 48 of 72 group games (through 2026-06-23) even right after a Refresh,
because the automatic Kaggle (martj42) community dataset lags real-time by ~3 days — matchday 3 was missing.
The FIFA API is authoritative and real-time; a Refresh now pulls 60 finished games (Groups A-F complete) and
auto-updates as more finish. Verified end-to-end against live data.
**What was rejected:** TheSportsDB free tier (only 5 stale events for WC-2026); football-data.org (requires a
token); transcribing scores myself (cannot fabricate / high error). FIFA's API needs no key and is the
source of truth. Date-inclusive dedup was rejected after finding Colombia–DR Congo dated Jun 23 by Kaggle vs
Jun 24 by FIFA (same match) — pair-only dedup prevents the double-count.
**How to apply:** Degrades gracefully — any FIFA fetch failure (offline/API change) or
`live_results_provider: none` falls back to the Kaggle feed, so CI/offline runs are unaffected (tests mock
HTTP). If FIFA changes the 2026 season id, update `fifa_season_id` in config.
