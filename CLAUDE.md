# CLAUDE.md - FPL Optimizer Codebase Guide

**Last Updated**: 2026-01-23
**License**: Apache 2.0
**Repository**: FPL (Fantasy Premier League) Optimizer

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [Codebase Structure](#codebase-structure)
3. [Architecture Evolution](#architecture-evolution)
4. [Key Technologies](#key-technologies)
5. [Data Flow & Processing](#data-flow--processing)
6. [Development Workflows](#development-workflows)
7. [File Conventions](#file-conventions)
8. [Optimization Constraints](#optimization-constraints)
9. [Machine Learning Components](#machine-learning-components)
10. [Best Practices for AI Assistants](#best-practices-for-ai-assistants)
11. [Common Tasks](#common-tasks)

---

## Repository Overview

This repository contains a Fantasy Premier League (FPL) squad optimizer that has evolved from simple heuristic-based optimization to advanced machine learning-enhanced portfolio optimization. The tool helps users build optimal FPL squads by:

- Fetching live player data from the official FPL API
- Predicting player performance using ML models (v2/v3)
- Solving mixed-integer linear programming problems to select optimal squads
- Respecting all FPL rules and constraints
- Optimizing for maximum projected points with risk management

**Primary Purpose**: Generate optimal 15-player FPL squads with starting XI and captain selection for upcoming gameweeks.

**Target Users**: FPL managers, data scientists, ML researchers interested in sports analytics.

---

## Codebase Structure

```
/home/user/FPL/
├── fpl_optimizer.py              # v1: Basic optimizer (548 lines)
├── fpl_optimizer_v2.py            # v2: ML-enhanced (909 lines)
├── fpl_optimizer_v3.py            # v3: Advanced ML + premium bias (1,114 lines)
├── fpl-lightgbm.ipynb             # Interactive Jupyter notebook (23 cells)
│
├── fpl_real_data.parquet          # Cached FPL data (54KB)
├── FPL Optimiser Real Transfers   # Transfer optimization data (JSON)
├── fpl_sklearn_simple             # Simple sklearn implementation (JSON)
│
├── models/
│   └── ensemble_models.pkl        # Trained ML models (3.1MB)
│
├── output/                        # Generated squad recommendations
│   ├── optimal_squad_*.csv        # v1 output
│   ├── optimal_squad_*.txt
│   ├── ml_optimal_squad_*.csv     # v2/v3 output
│   └── ml_optimal_squad_*.txt
│
├── cache/                         # API response cache (auto-generated)
│   └── element_summary_*.json
│
├── readme.md                      # User setup guide
├── LICENSE                        # Apache 2.0
├── .gitignore                     # Python + project-specific ignores
└── CLAUDE.md                      # This file
```

### Key Files

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `fpl_optimizer.py` | 548 | Basic optimizer with simple projections | Stable |
| `fpl_optimizer_v2.py` | 909 | ML-enhanced with feature engineering | Stable |
| `fpl_optimizer_v3.py` | 1,114 | Advanced ML with premium player bias | **Current** |
| `fpl-lightgbm.ipynb` | 23 cells | Interactive transfer recommendations | Active |

---

## Architecture Evolution

### Version 1: Basic Optimizer (fpl_optimizer.py)

**Philosophy**: Heuristic-based optimization with FPL's own expected points.

**Key Features**:
- Uses `ep_next` (expected points next GW) from FPL API
- Blends current season per-90 with last season performance
- Simple fixture difficulty multipliers (1=easy → 5=hard)
- Linear availability scoring
- Pure MILP optimization with PuLP

**Projection Formula**:
```python
proj_per_match = (
    0.4 * ep_next +
    0.3 * current_season_p90 +
    0.2 * last_season_p90 +
    0.1 * fixture_difficulty_adjustment
) * availability_factor
```

**Use Case**: Quick, reliable baseline optimizer for standard users.

---

### Version 2: ML-Enhanced (fpl_optimizer_v2.py)

**Philosophy**: Data-driven predictions with ensemble machine learning.

**New Features**:
- Ensemble models (XGBoost 60% + Random Forest 40%)
- Advanced feature engineering class (24 features)
- Risk-aware optimization with variance penalties
- ICT index (Influence, Creativity, Threat) integration
- Team strength indicators
- Differential scoring for low-ownership players

**ML Pipeline**:
1. Feature engineering with `AdvancedFeatureEngineer`
2. Time-series cross-validation (3 splits)
3. Ensemble prediction with weighted voting
4. Risk-adjusted optimization: `maximize(points - risk_factor * variance)`

**Research Foundation**: Based on academic papers showing 34%+ ROI with ML optimization.

---

### Version 3: Advanced ML + Premium Bias (fpl_optimizer_v3.py) ⭐

**Philosophy**: ML predictions enhanced with domain expertise and real-world player hierarchies.

**New Features**:
- **Premium player database**: 50+ hardcoded elite players
  - Haaland: 1.6x multiplier
  - Salah: 1.5x multiplier
  - Palmer, Saka, Son: 1.4x multiplier
- **Multi-factor bias system**:
  - Premium bias (40%): Hardcoded top performers
  - Popularity bias (30%): Ownership + form weighted
  - Elite team bias (30%): Top 6 teams get 15% boost
- **Captain potential scoring**: Identifies captain magnets
- **Enhanced output**: ⭐ symbols for premium players

**Bias Calculation**:
```python
final_multiplier = (
    premium_bias * 0.4 +
    popularity_bias * 0.3 +
    elite_team_bias * 0.3
)
predicted_points = ml_prediction * final_multiplier
```

**Use Case**: Most accurate predictions for competitive FPL managers. **This is the recommended version.**

---

### Jupyter Notebook: Transfer Optimizer (fpl-lightgbm.ipynb)

**Purpose**: Interactive tool for mid-season transfer decisions.

**Features**:
- User squad input functionality
- 2-transfer optimization with MILP
- Transfer impact analysis
- Value picks and alternatives
- Visual charts and analysis

**Use Case**: Weekly transfer planning for existing teams.

---

## Key Technologies

### Core Stack

| Technology | Purpose | Version Notes |
|------------|---------|---------------|
| **Python** | Language | 3.9+ recommended |
| **pandas** | Data manipulation | Required all versions |
| **numpy** | Numerical operations | Required all versions |
| **requests** | HTTP API calls | Required all versions |
| **pulp** | MILP optimization | Required all versions |
| **tqdm** | Progress bars | Required all versions |

### Machine Learning Stack (v2/v3 only)

| Library | Purpose | Notes |
|---------|---------|-------|
| **scikit-learn** | Base ML framework | RandomForest, StandardScaler |
| **xgboost** | Gradient boosting | Primary predictor (60% weight) |
| **joblib/pickle** | Model persistence | v3 uses pickle |

### External APIs

**FPL Official API**: `https://fantasy.premierleague.com/api/`

| Endpoint | Purpose | Cache |
|----------|---------|-------|
| `/bootstrap-static/` | Players, teams, events | Session |
| `/fixtures/` | Future fixtures & difficulty | Session |
| `/element-summary/{id}/` | Player history & stats | Persistent |
| `/event/{id}/live/` | Live gameweek data | No cache |

**API Access**: No authentication required. Rate limiting unknown but respected with caching.

---

## Data Flow & Processing

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION                                          │
├─────────────────────────────────────────────────────────────┤
│ • Fetch bootstrap-static (teams, players, positions)        │
│ • Fetch fixtures (future gameweeks)                         │
│ • Parallel fetch element summaries (cached)                 │
│ • Save to fpl_real_data.parquet                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPARATION                                         │
├─────────────────────────────────────────────────────────────┤
│ • Merge teams and positions                                 │
│ • Type conversions (costs, percentages, floats)            │
│ • Calculate games_played, display_name                      │
│ • Create unified player DataFrame                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING (v2/v3)                             │
├─────────────────────────────────────────────────────────────┤
│ • Calculate per-90 statistics                               │
│ • Team strength integration                                 │
│ • Position-specific baselines                               │
│ • Premium/popularity/elite biases (v3)                      │
│ • Availability and differential scoring                     │
│ • Total: 24 engineered features                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PROJECTION / PREDICTION                                  │
├─────────────────────────────────────────────────────────────┤
│ v1: Heuristic projection (ep_next + p90 blend)             │
│ v2: ML ensemble (XGBoost + RandomForest)                   │
│ v3: ML ensemble + bias multipliers                         │
│ Output: proj_horizon, proj_std (v2/v3)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. OPTIMIZATION                                             │
├─────────────────────────────────────────────────────────────┤
│ • Define decision variables (pick, start, captain)          │
│ • Apply FPL constraints (see below)                         │
│ • Objective: max(XI_points + captain_bonus - risk*var)     │
│ • Solve with CBC solver                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTPUT                                                   │
├─────────────────────────────────────────────────────────────┤
│ • Generate summary text                                     │
│ • Create CSV with player details                            │
│ • Save to output/ with timestamp                            │
│ • Display formation, captain, bench, projections           │
└─────────────────────────────────────────────────────────────┘
```

### Data Schema

**Player DataFrame Columns** (key fields):

```python
{
    'id': int,                      # FPL player ID
    'web_name': str,                # Short name (e.g., "Haaland")
    'display_name': str,            # Full name (e.g., "E. Haaland")
    'element_type': int,            # Position code (1=GKP, 2=DEF, 3=MID, 4=FWD)
    'pos': str,                     # Position label ("GKP", "DEF", "MID", "FWD")
    'team': int,                    # Team code
    'team_short': str,              # Team abbreviation (e.g., "MCI")
    'now_cost': int,                # Price in tenths (e.g., 110 = £11.0m)
    'cost_m': float,                # Price in millions (e.g., 11.0)
    'selected_by_percent': float,   # Ownership percentage
    'form': str,                    # Recent form (string)
    'form_numeric': float,          # Recent form (numeric)
    'ep_next': float,               # FPL expected points next GW
    'minutes': int,                 # Total minutes played this season
    'total_points': int,            # Total points this season
    'points_per_90': float,         # Points per 90 minutes
    'ict_index': float,             # ICT index (Influence, Creativity, Threat)
    'status': str,                  # Availability ("a" = available, "d" = doubtful, etc.)
    'chance_of_playing_next_round': int,  # 0-100
    'proj_horizon': float,          # Projected points over horizon (output)
    'proj_std': float,              # Standard deviation (v2/v3)
}
```

---

## Development Workflows

### Setting Up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd FPL

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -U pip
pip install -U pandas numpy requests pulp tqdm

# For ML versions (v2/v3)
pip install -U scikit-learn xgboost joblib

# For Jupyter notebook
pip install -U jupyter matplotlib seaborn lightgbm
```

### Running the Optimizers

**Version 1 (Basic)**:
```bash
python fpl_optimizer.py --horizon 6 --budget 100.0 --threads 8
```

**Version 3 (Recommended)**:
```bash
python fpl_optimizer_v3.py --horizon 6 --budget 100.0 --risk-factor 0.3 --threads 8
```

**Command-line Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--horizon` | int | 6 | Number of future gameweeks to project |
| `--budget` | float | 100.0 | Initial squad budget in millions |
| `--threads` | int | 8 | Parallel fetch threads for player summaries |
| `--risk-factor` | float | 0.3 | Risk penalty weight (v2/v3 only) |
| `--no-summaries` | flag | False | Skip last-season data (faster but less accurate) |

**Jupyter Notebook**:
```bash
jupyter notebook fpl-lightgbm.ipynb
```

### Git Workflow

**Current Branch**: `claude/claude-md-mkqzhc70hwixoo4d-KU3zy`

**Branch Naming Convention**: `claude/claude-md-<session-id>`

**Commit Guidelines**:
- Use descriptive commit messages
- Follow conventional commits format when possible
- Example: "Add advanced feature engineering to FPL notebook"

**Common Git Commands**:
```bash
# Check status
git status

# Stage changes
git add <files>

# Commit
git commit -m "Description of changes"

# Push to branch (always use -u flag)
git push -u origin <branch-name>

# Create PR (after pushing)
gh pr create --title "PR Title" --body "Description"
```

---

## File Conventions

### Naming Patterns

**Output Files**:
```
# v1 format
optimal_squad_YYYYMMDD_HHMMSS.csv
optimal_squad_YYYYMMDD_HHMMSS.txt

# v2/v3 format
ml_optimal_squad_YYYYMMDD_HHMMSS.csv
ml_optimal_squad_YYYYMMDD_HHMMSS.txt
```

**Cache Files**:
```
cache/element_summary_{player_id}.json
```

**Model Files**:
```
models/ensemble_models.pkl
```

### Directory Structure

| Directory | Purpose | Gitignored | Auto-created |
|-----------|---------|------------|--------------|
| `cache/` | API response cache | Partial (*.json) | Yes |
| `output/` | Squad recommendations | No | Yes |
| `models/` | Trained ML models | No | Yes |
| `.venv/` | Virtual environment | Yes | No |
| `__pycache__/` | Python bytecode | Yes | Auto |

### Code Style

**General Conventions**:
- PEP 8 compliant
- Type hints for function signatures (v1-v3)
- Docstrings for modules and complex functions
- 4-space indentation
- Max line length: ~100 characters (flexible)

**Import Order**:
1. Standard library (`import argparse`, `import datetime`)
2. Third-party (`import pandas`, `import numpy`)
3. Local modules (if any)

**Function Naming**:
- Snake_case for functions: `load_bootstrap()`, `project_points()`
- Classes in PascalCase: `AdvancedFeatureEngineer`, `EnsemblePredictor`

---

## Optimization Constraints

### FPL Official Rules (Enforced)

**Squad Composition**:
```python
SQUAD_SIZE = 15
POSITION_LIMITS = {
    'GKP': 2,  # Exactly 2 goalkeepers
    'DEF': 5,  # Exactly 5 defenders
    'MID': 5,  # Exactly 5 midfielders
    'FWD': 3,  # Exactly 3 forwards
}
TEAM_PLAYER_LIMIT = 3  # Max 3 players from any single club
BUDGET = 100.0  # £100m total budget
```

**Starting XI Rules**:
```python
STARTING_XI_SIZE = 11
FORMATION_CONSTRAINTS = {
    'GKP': 1,      # Exactly 1 (always)
    'DEF': (3, 5), # 3 to 5 defenders
    'MID': (2, 5), # 2 to 5 midfielders
    'FWD': (1, 3), # 1 to 3 forwards
}
# DEF + MID + FWD must equal 10
```

**Captain Rules**:
```python
CAPTAIN_COUNT = 1       # Exactly 1 captain
CAPTAIN_MULTIPLIER = 2  # Captain points are doubled
VICE_CAPTAIN = 1        # Exactly 1 vice-captain (backup)
```

### MILP Formulation (PuLP)

**Decision Variables**:
```python
# For each player i:
pick[i] = Binary      # 1 if player i in 15-man squad, 0 otherwise
start[i] = Binary     # 1 if player i in starting XI, 0 otherwise
captain[i] = Binary   # 1 if player i is captain, 0 otherwise
```

**Objective Function**:

**v1**:
```python
maximize: sum(start[i] * proj[i] for i in players)
        + sum(captain[i] * proj[i] for i in players)
```

**v2/v3 (Risk-Aware)**:
```python
maximize: sum(start[i] * proj[i] for i in players)
        + sum(captain[i] * proj[i] for i in players)
        - risk_factor * sum(start[i] * proj_std[i]^2 for i in players)
```

**Key Constraints**:
```python
# Squad size
sum(pick[i] for i in players) == 15

# Budget
sum(pick[i] * cost[i] for i in players) <= BUDGET

# Position limits
sum(pick[i] for i in GKPs) == 2
sum(pick[i] for i in DEFs) == 5
sum(pick[i] for i in MIDs) == 5
sum(pick[i] for i in FWDs) == 3

# Team diversity
sum(pick[i] for i in team_j) <= 3  (for each team j)

# Starting XI
sum(start[i] for i in players) == 11
start[i] <= pick[i]  (can only start if in squad)

# Formation constraints
sum(start[i] for i in GKPs) == 1
sum(start[i] for i in DEFs) >= 3
sum(start[i] for i in DEFs) <= 5
sum(start[i] for i in MIDs) >= 2
sum(start[i] for i in MIDs) <= 5
sum(start[i] for i in FWDs) >= 1
sum(start[i] for i in FWDs) <= 3

# Captain
sum(captain[i] for i in players) == 1
captain[i] <= start[i]  (captain must be in starting XI)
```

---

## Machine Learning Components

### Feature Engineering (v2/v3)

**AdvancedFeatureEngineer Class** generates 24 features:

**Basic Statistics** (4 features):
- `minutes_per_game`: Average minutes per appearance
- `points_per_90`: Points normalized per 90 minutes
- `form_numeric`: Recent form (last 5 GWs)
- `form_trend`: Form trajectory (improving/declining)

**ICT Metrics** (4 features):
- `influence_per_90`: Influence index per 90
- `creativity_per_90`: Creativity index per 90
- `threat_per_90`: Threat index per 90
- `ict_index_per_90`: Combined ICT per 90

**Value Metrics** (2 features):
- `value_ratio`: Points per million (total_points / cost_m)
- `value_per_million`: Expected value efficiency

**Team Strength** (3 features):
- `team_attack_strength`: Team's attacking power (1-5)
- `team_defence_strength`: Team's defensive solidity (1-5)
- `team_overall_strength`: Combined team quality (1-10)

**Position Context** (2 features):
- `position_baseline`: Expected points for position
- `performance_vs_baseline`: Player's points vs position average

**Defense** (1 feature):
- `defensive_potential`: Clean sheet likelihood (GKP/DEF)

**Meta Features** (4 features):
- `availability_score`: Injury/suspension risk (0-1)
- `differential_score`: Low ownership bonus
- `cost_m`: Player price in millions
- `selected_by_percent`: Ownership percentage

**Bias Features (v3 only)** (4 features):
- `premium_bias`: Hardcoded elite player multiplier (1.0-1.6)
- `popularity_bias`: Ownership + form weighted (1.0-1.4)
- `elite_team_bias`: Top 6 team bonus (1.0-1.15)
- `captain_potential`: Captain suitability score (0-1)

### Model Architecture (v2/v3)

**Ensemble Composition**:

**XGBoost Regressor** (60% weight):
```python
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.05,      # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    random_state=42
)
```

**Random Forest Regressor** (40% weight):
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42
)
```

**Voting Ensemble**:
```python
VotingRegressor([
    ('xgb', xgb_model),
    ('rf', rf_model)
], weights=[0.6, 0.4])
```

### Training Process

**Data Preparation**:
1. Filter players with `minutes > 0`
2. Engineer 24 features
3. Standardize with `StandardScaler`
4. Use `total_points` as target variable

**Cross-Validation**:
```python
TimeSeriesSplit(n_splits=3)
# Respects temporal ordering of gameweeks
```

**Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Feature importance extraction

**Model Persistence**:
```python
# Save
pickle.dump({
    'models': ensemble,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'feature_importance': importance_dict
}, open('models/ensemble_models.pkl', 'wb'))

# Load
model_data = pickle.load(open('models/ensemble_models.pkl', 'rb'))
```

### Premium Player Database (v3)

**Elite Players** (50+ players with multipliers):

**Tier 1 (1.5-1.6x)**:
- Haaland (MCI) → 1.6x
- Salah (LIV) → 1.5x

**Tier 2 (1.4x)**:
- Palmer (CHE), Saka (ARS), Son (TOT)

**Tier 3 (1.3x)**:
- Isak (NEW), Watkins (AVL), Foden (MCI), etc.

**Usage**: Multiplies ML predictions to account for consistent elite performance and underlying quality not fully captured by historical stats.

---

## Best Practices for AI Assistants

### Understanding Context

1. **Version Awareness**: Always clarify which version (v1/v2/v3) the user is working with
   - v1: For quick, simple optimizations
   - v2: For ML-based predictions
   - v3: For most accurate predictions with domain expertise

2. **Data Freshness**: Remember that FPL data is fetched live from the API
   - Player prices change throughout the season
   - Injuries/suspensions update regularly
   - Form metrics are dynamic

3. **Caching Strategy**: Be aware of the caching system
   - Element summaries are cached per player
   - ML models are cached in `models/`
   - Output files are never cached (new timestamp each run)

### Code Modification Guidelines

**DO**:
- Maintain backward compatibility when editing shared functions
- Test with small horizons first (e.g., `--horizon 3`)
- Use type hints for new functions
- Add comments for complex ML logic
- Validate against FPL constraints before solving

**DON'T**:
- Modify the MILP constraints without understanding FPL rules
- Remove caching mechanisms (they prevent API rate limiting)
- Change position codes (1=GKP, 2=DEF, 3=MID, 4=FWD are standard)
- Hardcode API endpoints (use `ENDPOINTS` dict)
- Ignore error handling for network requests

### Common Pitfalls

1. **Position Encoding**: Remember positions are 1-indexed, not 0-indexed
   ```python
   # Correct
   positions = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

   # Incorrect
   positions = {0: 'GKP', 1: 'DEF', 2: 'MID', 3: 'FWD'}
   ```

2. **Cost Conversion**: FPL API returns costs in tenths (110 = £11.0m)
   ```python
   # Correct
   cost_m = now_cost / 10.0

   # Incorrect
   cost_m = now_cost  # This would be £110m!
   ```

3. **Formation Validation**: Always ensure DEF + MID + FWD = 10 in starting XI
   ```python
   # Correct constraint
   sum(start[i] for i in DEF+MID+FWD) == 10
   ```

4. **Captain Points**: Remember captain points are already doubled in optimization
   ```python
   # The objective function includes:
   sum(captain[i] * proj[i] for i in players)  # Already accounts for 2x
   ```

### Debugging Tips

**API Issues**:
```python
# Add verbose logging
response = requests.get(url, timeout=20)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")
```

**Optimization Failures**:
```python
# Check problem status
prob.solve(pulp.PULP_CBC_CMD(msg=True))  # Enable solver output
print(f"Status: {pulp.LpStatus[prob.status]}")
```

**ML Prediction Issues**:
```python
# Validate features
print(f"Missing features: {set(expected_features) - set(df.columns)}")
print(f"NaN values: {df.isna().sum()}")
```

### Testing Workflows

**Quick Validation** (fast, for development):
```bash
python fpl_optimizer_v3.py --horizon 3 --no-summaries
```

**Full Run** (production):
```bash
python fpl_optimizer_v3.py --horizon 6 --budget 100.0 --risk-factor 0.3 --threads 8
```

**Check Output**:
```bash
# View latest result
ls -lt output/ | head -n 1
cat output/ml_optimal_squad_*.txt | tail -n 1
```

---

## Common Tasks

### Task 1: Add a New Feature to ML Model

**Location**: `fpl_optimizer_v3.py` → `AdvancedFeatureEngineer.engineer_features()`

**Steps**:
1. Calculate new feature in `engineer_features()` method
2. Add to feature list
3. Retrain model (delete `models/ensemble_models.pkl` to force retrain)
4. Test with `--horizon 3` first

**Example**:
```python
# Add "goals_per_90" feature
df['goals_per_90'] = np.where(
    df['minutes'] > 0,
    df['goals_scored'] * 90 / df['minutes'],
    0
)
```

### Task 2: Adjust Premium Player Multipliers

**Location**: `fpl_optimizer_v3.py` → `calculate_premium_bias()`

**Steps**:
1. Find the `PREMIUM_PLAYERS` dictionary (around line 400)
2. Update multiplier for existing player or add new entry
3. Run optimizer to see impact

**Example**:
```python
# Increase Haaland's bias
'Haaland': 1.7,  # Changed from 1.6

# Add new premium player
'Isak': 1.35,  # New entry
```

### Task 3: Change Optimization Objective

**Location**: `fpl_optimizer_v3.py` → `solve_optimal_squad()` or `RiskAwareOptimizer`

**Steps**:
1. Modify the `prob +=` line that defines the objective
2. Add or adjust penalties (e.g., risk, ownership diversity)
3. Test with different values

**Example**:
```python
# Original (v3)
prob += pulp.lpSum([...]) - risk_factor * pulp.lpSum([...])

# Modified: Add ownership diversity bonus
prob += (
    pulp.lpSum([...])  # Points
    - risk_factor * pulp.lpSum([...])  # Risk
    + 0.1 * pulp.lpSum([start[i] * (100 - df.loc[i, 'selected_by_percent'])
                        for i in player_ids])  # Ownership bonus
)
```

### Task 4: Export Squad to FPL Website Format

**Current**: CSV output is internal format
**Goal**: Export in FPL import-ready format

**Steps**:
1. Read output CSV
2. Map to FPL player IDs (already in `id` column)
3. Format as FPL expects (TBD based on FPL's format)
4. Save as `fpl_import_YYYYMMDD.csv`

### Task 5: Add Transfer Optimization

**See**: `fpl-lightgbm.ipynb` for reference implementation

**Steps**:
1. Accept current squad as input
2. Add transfer constraints to MILP
3. Penalize transfers (e.g., -4 points per transfer)
4. Optimize for net gain after transfer costs

**Constraint Example**:
```python
# Limit transfers
transfers_out = [transfer_out[i] for i in current_squad_ids]
prob += pulp.lpSum(transfers_out) <= max_transfers
```

### Task 6: Update for New Season

**When**: New FPL season starts (usually August)

**Checklist**:
1. Clear cache: `rm -rf cache/`
2. Delete old models: `rm models/*.pkl`
3. Test API endpoints (URL format may change)
4. Update premium player list if needed
5. Review new FPL rules (check official site)
6. Run full optimization to retrain models

### Task 7: Analyze Feature Importance

**After Training** (v2/v3):

```python
# Load model
model_data = pickle.load(open('models/ensemble_models.pkl', 'rb'))
importance = model_data['feature_importance']

# Print sorted importance
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature:30s}: {score:.4f}")
```

### Task 8: Compare v1 vs v2 vs v3 Predictions

**Script**:
```bash
# Run all versions with same horizon
python fpl_optimizer.py --horizon 6 > v1_output.txt
python fpl_optimizer_v2.py --horizon 6 > v2_output.txt
python fpl_optimizer_v3.py --horizon 6 > v3_output.txt

# Compare squads
diff v1_output.txt v2_output.txt
diff v2_output.txt v3_output.txt
```

**Analysis**: Look for differences in:
- Captain selection
- Premium player inclusion
- Overall projected points
- Budget utilization

---

## Additional Resources

### Official FPL API Documentation
- **Base URL**: `https://fantasy.premierleague.com/api/`
- **Unofficial Docs**: Search "FPL API" for community documentation
- **Rate Limiting**: Unknown, but caching is strongly recommended

### Academic References
- Mlčoch (2024): ML-enhanced FPL optimization
- Solomon (2019): Expected goals and fantasy points
- Bertsimas & Stellato (2021): Sports analytics with optimization

### Community
- r/FantasyPL (Reddit): Strategy discussions
- FPL Discord servers: Real-time data analysis
- GitHub: Search "FPL optimizer" for alternative approaches

---

## Changelog

### 2026-01-23 (Current State)
- **Latest commit**: 351fdd8 "Update fpl-lightgbm.ipynb"
- **Recent changes**: Advanced feature engineering, notebook development
- **Models**: ensemble_models.pkl (3.1MB, trained Aug 2025)
- **Active branch**: `claude/claude-md-mkqzhc70hwixoo4d-KU3zy`

### Version History
- **v3**: Premium bias system, 50+ elite players, enhanced output
- **v2**: ML ensemble (XGBoost + RF), feature engineering, risk-aware optimization
- **v1**: Basic optimizer, heuristic projections, simple constraints

---

## Notes for AI Assistants

1. **Always verify FPL constraints** before modifying optimization code
2. **Test with small horizons** (3-4 GWs) during development
3. **Preserve caching mechanisms** to avoid API rate limits
4. **Document new features** thoroughly in code and this file
5. **Consider backward compatibility** when changing data schemas
6. **Validate ML predictions** against realistic bounds (0-20 pts typical)
7. **Use v3 as the reference** for new features (most complete implementation)
8. **Update this file** when making significant architectural changes

---

## Contact & License

**License**: Apache 2.0 (see LICENSE file)
**Repository**: FPL Optimizer
**Last Updated**: 2026-01-23

For questions or contributions, refer to the README.md for setup instructions and basic usage.

---

**End of CLAUDE.md**
