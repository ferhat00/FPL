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
