# World Cup 2026 — Winner Prediction Pipeline

End-to-end data science for forecasting the **2026 FIFA World Cup** (48 teams, hosts USA/Canada/Mexico):
ETL → feature engineering → **random forest** → **Monte-Carlo** tournament simulation → **Streamlit**
dashboard. Method follows Groll et al. (2019, [arXiv:1806.03208](https://arxiv.org/abs/1806.03208)).

> Lives alongside the existing FPL optimizer (`fpl_optimizer*.py`), which is unrelated and untouched.

## Quick start (Windows / PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

# 1) configure (optional — runs offline with synthetic data if skipped)
Copy-Item config.example.yaml config.yaml   # then add your Kaggle username/key

# 2) run the pipeline (flags mirror the dashboard buttons)
python scripts/run_pipeline.py --refresh --train --backtest --simulate

# 3) launch the dashboard
streamlit run app/dashboard.py
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip`/`python` commands.

## The three actions (CLI flags = dashboard buttons)

| Button | CLI flag | What it does |
|---|---|---|
| 🔄 Refresh Data | `--refresh` | Pull historical + live results, Elo, market values, GDP/pop → Parquet |
| 🧠 Train & Validate | `--train` `--backtest` | Fit RF goals model (TimeSeriesSplit CV) + backtest (RPS/log-loss/calibration) |
| 🏆 Run Prediction | `--simulate` | Monte-Carlo the remaining tournament → champion & per-stage probabilities |

## Architecture

```
config.example.yaml            # keys + settings (copy to config.yaml)
worldcup2026/
  data/tournament_2026.yaml    # groups A-L, team attrs, knockout bracket (source of truth)
  etl/        sources.py ingest.py clean.py     # acquisition, orchestration, name normalization
  features/   elo.py engineering.py team_attrs.py
  model/      train.py validate.py simulate.py bracket.py
  config.py persistence.py tournament.py
scripts/run_pipeline.py        # CLI
app/dashboard.py               # Streamlit + Plotly
tests/                         # pytest (bracket, features, simulation, cleaning, tournament)
```

**Model.** A `RandomForestRegressor` predicts goals per team per match from dynamic Elo (+diff), FIFA rank,
squad market value & age, GDP/population, host & confederation flags, and Dixon-Coles time-decayed rolling
form. Applied twice per fixture → (λ_home, λ_away). A bivariate-Poisson layer samples scorelines; the
Monte-Carlo loop plays out groups (FIFA tiebreakers + 8 best thirds) and the knockout bracket, repeating
≥20k times for probabilities.

**Live mode.** Drop played results into `worldcup2026/data/wc2026_results_manual.csv`
(`date,home_team,away_team,home_score,away_score`); they are locked and only the remainder is simulated.

## Offline / synthetic mode

With no Kaggle key (or offline), the pipeline generates a **calibrated synthetic** match history
(Elo→Poisson goals) so everything runs and the dashboard is fully demonstrable. This is clearly flagged in
logs and the UI; real data always overrides it.

## ⚠️ Data to verify before trusting results

- **Six play-off slots** (Czech Republic, Bosnia & Herzegovina, Turkey, Sweden, Iraq, DR Congo) were decided
  by March-2026 play-offs — confirm the nations in `tournament_2026.yaml` against the official FIFA source.
- **R32 winner-vs-third pairings** are a valid default; check against FIFA's published bracket. Editing the
  YAML re-wires the simulation.
- The bundled `teams:` attributes (Elo, market value, …) are an **approximate snapshot** for offline use;
  a successful `--refresh` with live sources overrides them.

## Testing

```powershell
python -m pytest tests/ -q
```
