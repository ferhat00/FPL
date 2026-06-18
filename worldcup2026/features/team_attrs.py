"""Access to per-team attribute table (live ETL output, snapshot fallback)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import persistence as pstore
from ..tournament import teams_table


def load_team_attrs() -> pd.DataFrame:
    """Team attribute table indexed by team. Processed parquet if present, else snapshot."""
    df = pstore.read_parquet("team_attrs", "processed")
    if df is None:
        df = teams_table()
    if df.index.name != "team":
        if "team" in df.columns:
            df = df.set_index("team")
    # derived transforms used as features
    if "gdp_per_capita" in df.columns:
        df["loggdp"] = np.log1p(df["gdp_per_capita"].astype(float))
    else:
        df["loggdp"] = np.nan
    if "population" in df.columns:
        df["logpop"] = np.log1p(df["population"].astype(float))
    else:
        df["logpop"] = np.nan
    df["host"] = df.get("host", False).astype(bool) if "host" in df.columns else False
    return df
