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
from worldcup2026 import standings as gs
from worldcup2026.config import kaggle_available, load_config
from worldcup2026.etl.ingest import run_etl
from worldcup2026.model.simulate import run_simulation
from worldcup2026.model.train import load_bundle, train_model
from worldcup2026.model.validate import backtest
from worldcup2026.tournament import groups as groups_fn

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


def _style_standings(df):
    """Highlight qualifiers: top-2 green, 3rd amber (by position index)."""
    def _row(row):
        if row.name <= 2:
            return ["background-color: #14532d; color: white"] * len(row)
        if row.name == 3:
            return ["background-color: #78350f; color: white"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row, axis=1)


def _projected_group_table(eb, g):
    """Projected final standings for one group from expected_bracket -> DataFrame."""
    rows = [{"team": t, "ExpPts": pts} for t, pts in eb["group_standings"][g]]
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(1, len(df) + 1, name="Pos")
    return df


def _group_stage_section():
    """Dedicated group-stage view: results, standings, 3rd-place table, R32 qualifiers.

    The *actual* side is computed from played results (needs only Refresh Data);
    the *projected* side is shown alongside when a simulation is in session.
    """
    live = pstore.read_parquet("wc2026_live", "processed")
    _, attrs = _processed()
    if live is None or len(live) == 0 or attrs is None:
        return  # nothing to show until data is refreshed

    grps = groups_fn()
    team_group = {t: g for g, ms in grps.items() for t in ms}
    strength = attrs["elo"].to_dict() if "elo" in attrs.columns else {}

    standings = gs.group_standings(live, grps, strength)
    thirds = gs.third_place_ranking(standings, strength)
    quals = gs.r32_qualifiers(standings, thirds)
    results = gs.results_by_group(live, grps)
    n_played = int(results["played"].sum()) if len(results) else 0
    n_total = len(results)

    eb = None
    if "sim_out" in st.session_state:
        eb = st.session_state["sim_out"].get("expected_bracket")

    st.divider()
    st.header("📋 Group stage")
    if n_played >= n_total and n_total:
        st.caption(f"Group stage complete — {n_total} games played. Final standings below.")
    else:
        st.caption(f"Played **{n_played} of {n_total}** group games — actual standings are "
                   "**provisional** until the final matchday. Projected = model forecast of the "
                   "remaining games.")
    if eb is None:
        st.info("Run **🏆 Run Prediction** (after training) to populate the *projected final* "
                "columns beside the actual ones.", icon="ℹ️")

    # ---- per-group: results + actual vs projected standings ----
    for g in grps:
        leaders = ", ".join(standings[g].loc[1:2, "team"].tolist())
        with st.expander(f"Group {g} — {leaders} leading", expanded=False):
            gr = results[results["group"] == g]
            res_rows = []
            for m in gr.itertuples():
                score = (f"{int(m.home_score)}–{int(m.away_score)}" if m.played
                         else "—")
                when = pd.to_datetime(m.date).strftime("%b %d") if pd.notna(m.date) else ""
                res_rows.append({"Match": f"{m.home} v {m.away}", "Score": score,
                                 "Date": when, "": "✔" if m.played else "🗓"})
            st.dataframe(pd.DataFrame(res_rows), hide_index=True, use_container_width=True)

            ca, cp = st.columns(2)
            ca.markdown("**Actual standings**")
            ca.dataframe(_style_standings(standings[g]), use_container_width=True)
            cp.markdown("**Projected final**")
            if eb is not None:
                cp.dataframe(_style_standings(_projected_group_table(eb, g)),
                             use_container_width=True)
            else:
                cp.caption("— run a prediction —")

    # ---- third-place finishers ----
    st.subheader("🥉 Third-place finishers")
    ta, tp = st.columns(2)
    ta.markdown("**Actual** (best 8 qualify)")
    show_thirds = thirds.copy()
    show_thirds["qualified"] = show_thirds["qualified"].map({True: "✅", False: ""})
    ta.dataframe(show_thirds.rename(columns={"qualified": "R32?"}),
                 hide_index=True, use_container_width=True)
    tp.markdown("**Projected** (best 8)")
    if eb is not None:
        tp.dataframe(pd.DataFrame({"rank": range(1, len(eb["best_thirds"]) + 1),
                                   "team": eb["best_thirds"],
                                   "group": [team_group.get(t) for t in eb["best_thirds"]]}),
                     hide_index=True, use_container_width=True)
    else:
        tp.caption("— run a prediction —")

    # ---- qualified for the Round of 32 ----
    st.subheader("✅ Qualified for the Round of 32")
    note = "final" if (n_played >= n_total and n_total) else "provisional"
    qa, qp = st.columns(2)
    qa.markdown(f"**Actual** ({note}) — {len(quals)} teams")
    qa.dataframe(quals[["team", "group", "route"]].sort_values(["route", "group"]),
                 hide_index=True, use_container_width=True, height=320)
    qp.markdown("**Projected**")
    if eb is not None:
        proj_rows = []
        for g, lst in eb["group_standings"].items():
            proj_rows.append({"team": lst[0][0], "group": g, "route": "Group winner"})
            proj_rows.append({"team": lst[1][0], "group": g, "route": "Runner-up"})
        for t in eb["best_thirds"]:
            proj_rows.append({"team": t, "group": team_group.get(t), "route": "3rd place"})
        qp.dataframe(pd.DataFrame(proj_rows).sort_values(["route", "group"]),
                     hide_index=True, use_container_width=True, height=320)
    else:
        qp.caption("— run a prediction —")


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


# --------------------------------------------------------------------------- 1b. GROUP STAGE
_group_stage_section()


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

    adv = out["advancement"].head(20).set_index("team")[["R32", "R16", "QF", "SF", "Final", "Champion"]]
    fig_hm = px.imshow(adv, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                       title="Probability of reaching each stage (top 20)")
    fig_hm.update_layout(height=620, coloraxis_showscale=False)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("#### Expected knockout bracket")
    st.caption("Deterministic favourites (most-likely path), Round of 32 → Final. "
               "Projected winner in **bold**; hover a tie for its FIFA match number.")
    rounds = eb["knockout"]
    order = ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]
    label = {"round_of_32": "Round of 32", "round_of_16": "Round of 16",
             "quarter_finals": "Quarter-finals", "semi_finals": "Semi-finals", "final": "Final"}
    cols = st.columns(5)
    for col, rname in zip(cols, order):
        col.markdown(f"**{label[rname]}**")
        for mtch in rounds.get(rname, []):
            a_win = mtch["winner"] == mtch["a"]
            a = f"**{mtch['a']}**" if a_win else mtch["a"]
            b = mtch["b"] if a_win else f"**{mtch['b']}**"
            col.markdown(f"{a}  \n{b}", help=mtch["id"])
            col.markdown("---")
    st.success(f"Expected champion: **{eb['champion']}**  (def. {eb['runner_up']})")

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
