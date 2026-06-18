"""World Cup 2026 prediction dashboard (Streamlit + Plotly).

Three buttons mirror the pipeline:
    1. Refresh Data            -> ETL (Kaggle / live results / snapshot fallback)
    2. Train & Validate Model  -> RF goals model + TimeSeriesSplit CV + backtest
    3. Run Prediction          -> Monte-Carlo simulation of the remaining tournament

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldcup2026 import persistence as pstore
from worldcup2026.config import kaggle_available, load_config
from worldcup2026.etl.ingest import run_etl
from worldcup2026.model.simulate import run_simulation
from worldcup2026.model.train import load_bundle, train_model
from worldcup2026.model.validate import backtest

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")
CFG = load_config()


# --------------------------------------------------------------------------- helpers
def _processed():
    return (pstore.read_parquet("results", "processed"),
            pstore.read_parquet("team_attrs", "processed"))


def _status_bar():
    res, attrs = _processed()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical matches", f"{len(res):,}" if res is not None else "—")
    c2.metric("Last data refresh", (pstore.last_updated("results") or "never")[:19].replace("T", " "))
    live = pstore.read_parquet("wc2026_live", "processed")
    nlive = int(live["home_score"].notna().sum()) if live is not None and len(live) else 0
    c3.metric("WC-2026 results in", nlive)
    bundle_meta = pstore.read_meta("rf_goals_model", "models")
    c4.metric("Model trained", (bundle_meta.get("last_updated", "never"))[:19].replace("T", " "))
    if res is not None and (res["tournament"] == "Synthetic").any():
        st.info("⚠️ Running on **synthetic** history (no Kaggle key / offline). "
                "Add Kaggle credentials to `config.yaml` and refresh for real data.", icon="ℹ️")


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Kaggle key detected" if kaggle_available()
                   else "No Kaggle key — offline/synthetic mode")
n_sims = st.sidebar.slider("Monte-Carlo simulations", 2000, 100000,
                           int(CFG["simulation"]["n_simulations"]), step=2000)
live_mode = st.sidebar.toggle("Live tournament mode", value=True,
                              help="Lock played WC-2026 results and simulate only the remainder.")
fast_mode = st.sidebar.toggle("Fast train (fewer trees)", value=True,
                              help="200 trees instead of the configured count — quicker iteration.")
st.sidebar.divider()
st.sidebar.markdown(
    "**Method**: Groll-style random forest on goals → bivariate-Poisson "
    "Monte-Carlo bracket. See `MEMORY.md` for design decisions.")


# --------------------------------------------------------------------------- header
st.title("🏆 2026 FIFA World Cup — Winner Predictor")
st.caption("ETL → feature engineering → random forest → Monte-Carlo tournament simulation. "
           "Hosts: USA · Canada · Mexico — 48 teams, 12 groups.")
_status_bar()
st.divider()

b1, b2, b3 = st.columns(3)
refresh_clicked = b1.button("🔄  Refresh Data", use_container_width=True, type="primary")
train_clicked = b2.button("🧠  Train & Validate Model", use_container_width=True)
predict_clicked = b3.button("🏆  Run Prediction", use_container_width=True)


# --------------------------------------------------------------------------- 1. ETL
if refresh_clicked:
    with st.spinner("Fetching results, Elo, market values, live scores…"):
        summary = run_etl(refresh=True)
    st.session_state["etl_summary"] = summary
    st.success(f"Refreshed: {summary['n_matches']:,} matches "
               f"({summary['date_min']} → {summary['date_max']}), "
               f"{summary['n_live_wc2026']} WC-2026 results ingested.")
    _status_bar()


# --------------------------------------------------------------------------- 2. TRAIN
if train_clicked:
    res, attrs = _processed()
    if res is None:
        st.error("No data yet — click **Refresh Data** first.")
    else:
        cfg = {**CFG, "model": {**CFG["model"], **({"n_estimators": 200} if fast_mode else {})}}
        with st.spinner("Training random forest (TimeSeriesSplit CV)…"):
            bundle = train_model(res, attrs, cfg)
        with st.spinner("Backtesting on held-out matches…"):
            bt = backtest(res, attrs, cutoff="2018-01-01", cfg=cfg)
        st.session_state["bundle"] = bundle
        st.session_state["train_metrics"] = bundle.metrics
        st.session_state["importances"] = bundle.importances
        st.session_state["backtest"] = bt
        st.success("Model trained and validated.")

if "train_metrics" in st.session_state:
    m = st.session_state["train_metrics"]
    bt = st.session_state.get("backtest", {})
    st.subheader("Model validation")
    cols = st.columns(4)
    cols[0].metric("CV goals MAE", f"{m['cv_goals_mae']:.3f}")
    cols[1].metric("CV Poisson deviance", f"{m['cv_poisson_deviance']:.3f}")
    if "metrics" in bt:
        cols[2].metric("Backtest RPS (model)", f"{bt['metrics']['model']['rps']:.4f}",
                       help="Ranked Probability Score, lower is better.")
        cols[3].metric("RPS (Elo baseline)", f"{bt['metrics']['elo_baseline']['rps']:.4f}",
                       delta=f"{bt['metrics']['model']['rps'] - bt['metrics']['elo_baseline']['rps']:+.4f}",
                       delta_color="inverse")
    cc = st.columns(2)
    imp = pd.Series(st.session_state["importances"]).sort_values().tail(14)
    fig_imp = px.bar(imp, orientation="h", labels={"value": "importance", "index": "feature"},
                     title="Random-forest feature importances")
    fig_imp.update_layout(showlegend=False, height=420)
    cc[0].plotly_chart(fig_imp, use_container_width=True)
    if "metrics" in bt and len(bt.get("calibration", [])):
        cal = bt["calibration"]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="grey"), name="perfect"))
        fig_cal.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
                                     name="model"))
        fig_cal.update_layout(title="Calibration — P(team A win)", height=420,
                              xaxis_title="predicted", yaxis_title="observed")
        cc[1].plotly_chart(fig_cal, use_container_width=True)


# --------------------------------------------------------------------------- 3. PREDICT
if predict_clicked:
    bundle = st.session_state.get("bundle") or load_bundle()
    if bundle is None:
        st.error("No trained model — click **Train & Validate Model** first.")
    else:
        live = pstore.read_parquet("wc2026_live", "processed") if live_mode else None
        scfg = CFG["simulation"]
        with st.spinner(f"Running {n_sims:,} Monte-Carlo simulations…"):
            out = run_simulation(bundle, live, n=n_sims,
                                 corr=float(scfg["bivariate_poisson_corr"]),
                                 et_edge=float(scfg["knockout_draw_extra_time_edge"]),
                                 seed=int(scfg["random_state"]))
        st.session_state["sim_out"] = out
        pstore.write_parquet(out["summary"], "prediction_summary", "processed")

if "sim_out" in st.session_state:
    out = st.session_state["sim_out"]
    summary = out["summary"]
    eb = out["expected_bracket"]

    st.subheader(f"🏆 Predicted winner: {summary.iloc[0]['team']} "
                 f"({summary.iloc[0]['P_champion']:.1%})")

    top = summary.head(16)
    fig = px.bar(top, x="P_champion", y="team", orientation="h", color="P_champion",
                 color_continuous_scale="Viridis", title="Title-winning probability (top 16)")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=520, coloraxis_showscale=False)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([3, 2])
    adv = out["advancement"].head(20).set_index("team")[["R32", "R16", "QF", "SF", "Final", "Champion"]]
    fig_hm = px.imshow(adv, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                       title="Probability of reaching each stage (top 20)")
    fig_hm.update_layout(height=620, coloraxis_showscale=False)
    left.plotly_chart(fig_hm, use_container_width=True)

    right.markdown("#### Expected knockout path")
    right.caption("Deterministic favourites (most-likely bracket).")
    rounds = eb["knockout"]
    label = {"round_of_32": "Round of 32", "round_of_16": "Round of 16",
             "quarter_finals": "Quarter-finals", "semi_finals": "Semi-finals", "final": "Final"}
    for rname in ["quarter_finals", "semi_finals", "final"]:
        right.markdown(f"**{label[rname]}**")
        for mtch in rounds.get(rname, []):
            mark_a = "✅" if mtch["winner"] == mtch["a"] else ""
            mark_b = "✅" if mtch["winner"] == mtch["b"] else ""
            right.write(f"{mark_a} {mtch['a']} — {mtch['b']} {mark_b}")
    right.success(f"Expected champion: **{eb['champion']}**  (def. {eb['runner_up']})")

    st.subheader("Full probability table")
    show = summary.copy()
    for c in ["P_champion", "P_final", "P_semi", "P_quarter", "P_r16",
              "P_qualify_r32", "P_group_winner"]:
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show[["team", "group", "P_champion", "P_final", "P_semi",
                       "P_quarter", "P_r16", "P_qualify_r32", "P_group_winner", "elo"]],
                 use_container_width=True, height=460)
else:
    st.info("Click **Run Prediction** (after refreshing data and training) to simulate the tournament.")
