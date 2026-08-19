"""
Streamlit front end for the mode-choice tool.

Run from the repo root:
    streamlit run app/streamlit_app.py

The model is estimated once on the synthetic data and cached. The sliders scale
each mode's travel time, per-trip cost, and monthly parking cost by a percentage
of its base value; the app re-applies the fixed estimated coefficients to the
adjusted attributes and shows how aggregate mode shares move.
"""

import pathlib
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modechoice import config
from modechoice.mnl import (
    estimate,
    predicted_shares,
    apply_adjustments,
    choice_probabilities,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_mode_choice.csv"
MODES = config.MODES

st.set_page_config(page_title="Mode Choice Share Tool", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def fit(_df):
    return estimate(_df)


df = load_data()
result = fit(df)
params = result.params

st.title("Mode choice share prediction tool")
st.caption(
    "Multinomial logit estimated on 1,000 synthetic travelers. "
    "Adjust an attribute for any mode and watch the predicted shares respond."
)

policy_tab, model_tab, person_tab = st.tabs(
    ["Policy sensitivity", "Model", "Individual traveler"]
)

# --------------------------------------------------------------------------- #
# Policy sensitivity
# --------------------------------------------------------------------------- #
with policy_tab:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Adjust attributes")
        st.caption("Percentage change from each mode's base value.")
        adjustments = {}
        for m in MODES:
            with st.expander(m, expanded=(m == "Drive")):
                adjustments[(m, "time")] = st.slider(
                    f"{m} travel time", -50, 100, 0, 5, key=f"{m}_time"
                ) / 100
                if config.HAS_COST[m]:
                    adjustments[(m, "cost")] = st.slider(
                        f"{m} travel cost", -50, 100, 0, 5, key=f"{m}_cost"
                    ) / 100
                if config.HAS_PARK[m]:
                    adjustments[(m, "park")] = st.slider(
                        f"{m} monthly parking", -100, 100, 0, 5, key=f"{m}_park"
                    ) / 100
        if st.button("Reset all"):
            for k in list(st.session_state.keys()):
                if k.endswith(("_time", "_cost", "_park")):
                    st.session_state[k] = 0
            st.rerun()

    base = predicted_shares(df, params)
    adj_df = apply_adjustments(df, adjustments)
    new = predicted_shares(adj_df, params)

    tbl = pd.DataFrame({"Base": base, "Scenario": new})
    tbl["Change (pp)"] = (tbl["Scenario"] - tbl["Base"]) * 100

    with right:
        st.subheader("Predicted mode shares")
        chart_df = (
            tbl[["Base", "Scenario"]]
            .reset_index()
            .melt("index", var_name="Case", value_name="Share")
            .rename(columns={"index": "Mode"})
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Mode:N", sort=MODES, title=None),
                xOffset="Case:N",
                y=alt.Y("Share:Q", axis=alt.Axis(format="%")),
                color=alt.Color("Case:N", scale=alt.Scale(range=["#9aa4b2", "#2563eb"])),
                tooltip=["Mode", "Case", alt.Tooltip("Share:Q", format=".1%")],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

        show = tbl.copy()
        show["Base"] = (show["Base"] * 100).round(1)
        show["Scenario"] = (show["Scenario"] * 100).round(1)
        show["Change (pp)"] = show["Change (pp)"].round(2)
        show.columns = ["Base %", "Scenario %", "Change (pp)"]
        st.dataframe(show, use_container_width=True)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
with model_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Log-likelihood", f"{result.loglik:,.1f}")
    c2.metric("Implied value of time", f"${result.value_of_time:,.1f}/hr")
    c3.metric("Converged", "yes" if result.converged else "no")

    st.subheader("Estimated coefficients")
    st.dataframe(result.table.round(4), use_container_width=True)

    st.subheader("Observed vs predicted shares")
    obs = df["chosen_mode"].value_counts(normalize=True).reindex(MODES)
    comp = pd.DataFrame({"Observed": obs, "Predicted": predicted_shares(df, params)})
    st.dataframe((comp * 100).round(1).rename(columns=lambda c: c + " %"),
                 use_container_width=True)

    st.caption(
        "Data are simulated from known coefficients "
        f"(b_time {config.TRUE_PARAMS['b_time']}, b_cost {config.TRUE_PARAMS['b_cost']}, "
        f"b_park {config.TRUE_PARAMS['b_park']}); the table above shows what "
        "estimation recovers from the sample."
    )

# --------------------------------------------------------------------------- #
# Individual traveler
# --------------------------------------------------------------------------- #
with person_tab:
    pid = st.selectbox("Person ID", df["person_id"].tolist())
    row = df[df["person_id"] == pid]
    st.write(f"Trip distance: {row['distance_mi'].iloc[0]:.1f} mi "
             f"| observed choice: {row['chosen_mode'].iloc[0]}")

    avail = [m for m in MODES if row[f"av_{m}"].iloc[0] == 1]
    base_p = choice_probabilities(row, params).iloc[0]
    adj_p = choice_probabilities(apply_adjustments(row, adjustments), params).iloc[0]
    pp = pd.DataFrame({"Base": base_p, "Scenario": adj_p}).loc[avail]
    st.dataframe((pp * 100).round(1).rename(columns=lambda c: c + " %"),
                 use_container_width=True)
    st.caption("Probabilities use the same slider settings from the Policy tab. "
               "Only modes available at this trip distance are shown.")
