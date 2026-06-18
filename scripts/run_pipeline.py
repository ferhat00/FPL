#!/usr/bin/env python3
"""End-to-end CLI for the World Cup 2026 pipeline.

The three flags mirror the dashboard's three buttons:

    python scripts/run_pipeline.py --refresh                # ETL
    python scripts/run_pipeline.py --train --backtest       # model + validation
    python scripts/run_pipeline.py --simulate               # Monte-Carlo prediction
    python scripts/run_pipeline.py --refresh --train --simulate   # full run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow running as a plain script (repo root on path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldcup2026 import persistence as pstore
from worldcup2026.config import load_config
from worldcup2026.etl.ingest import run_etl
from worldcup2026.model.simulate import run_simulation
from worldcup2026.model.train import load_bundle, train_model
from worldcup2026.model.validate import backtest


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_processed():
    results = pstore.read_parquet("results", "processed")
    attrs = pstore.read_parquet("team_attrs", "processed")
    if results is None or attrs is None:
        raise SystemExit("No processed data found — run with --refresh first.")
    return results, attrs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="World Cup 2026 prediction pipeline")
    ap.add_argument("--refresh", action="store_true", help="run ETL (refresh data)")
    ap.add_argument("--train", action="store_true", help="train the RF goals model")
    ap.add_argument("--backtest", action="store_true", help="run time-holdout backtest")
    ap.add_argument("--simulate", action="store_true", help="Monte-Carlo the tournament")
    ap.add_argument("--n-sims", type=int, default=None, help="override simulation count")
    ap.add_argument("--cutoff", default="2018-01-01", help="backtest train/test cutoff date")
    ap.add_argument("-q", "--quiet", action="store_true", help="less logging")
    args = ap.parse_args(argv)

    if not any([args.refresh, args.train, args.backtest, args.simulate]):
        ap.print_help()
        return 0

    _setup_logging(not args.quiet)
    cfg = load_config()

    if args.refresh:
        summary = run_etl(refresh=True)
        print("\n[ETL]", summary)

    if args.train:
        results, attrs = _load_processed()
        bundle = train_model(results, attrs, cfg)
        print("\n[TRAIN] metrics:", bundle.metrics)
        print("[TRAIN] top features:",
              dict(list(bundle.importances.items())[:8]))

    if args.backtest:
        results, attrs = _load_processed()
        bt = backtest(results, attrs, cutoff=args.cutoff, cfg=cfg)
        print("\n[BACKTEST]", bt.get("metrics", bt))

    if args.simulate:
        results, attrs = _load_processed()
        bundle = load_bundle()
        if bundle is None:
            print("No model found — training first...")
            bundle = train_model(results, attrs, cfg)
        live = pstore.read_parquet("wc2026_live", "processed")
        scfg = cfg.get("simulation", {})
        n = args.n_sims or int(scfg.get("n_simulations", 20000))
        out = run_simulation(bundle, live, n=n,
                             corr=float(scfg.get("bivariate_poisson_corr", 0.10)),
                             et_edge=float(scfg.get("knockout_draw_extra_time_edge", 0.5)),
                             seed=int(scfg.get("random_state", 42)))
        summary = out["summary"]
        pstore.write_parquet(summary, "prediction_summary", "processed")
        pstore.write_parquet(out["advancement"], "prediction_advancement", "processed")
        print(f"\n[PREDICTION]  ({n:,} simulations)  most likely winner: "
              f"{summary.iloc[0]['team']}  ({summary.iloc[0]['P_champion']:.1%})\n")
        cols = ["team", "group", "P_champion", "P_final", "P_semi", "P_qualify_r32"]
        print(summary[cols].head(12).to_string(index=False))
        eb = out["expected_bracket"]
        print(f"\n[EXPECTED FINAL]  {eb['champion']}  def.  {eb['runner_up']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
