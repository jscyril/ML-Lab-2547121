"""Streamlit live demo for the diabetes risk-screening pipeline."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, LABELS, RAW_FEATURES, TABLE_DIR

st.set_page_config(page_title="Mission Health — Diabetes Screening", page_icon="🩺", layout="wide")


HEALTHY_YOUNG = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
    "Sex": 0, "Age": 2, "Education": 6, "Income": 8,
}

TEST_CASES = {
    "Default video case": {
        **HEALTHY_YOUNG,
        "HighBP": 1, "HighChol": 1, "BMI": 34, "PhysActivity": 0,
        "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5,
        "Age": 8, "Education": 5, "Income": 5,
    },
    "Healthy young adult": HEALTHY_YOUNG,
    "Lifestyle risk, no known conditions": {
        **HEALTHY_YOUNG,
        "BMI": 31, "Smoker": 1, "PhysActivity": 0, "Fruits": 0,
        "Veggies": 0, "GenHlth": 2, "MentHlth": 5, "PhysHlth": 2,
        "Sex": 1, "Age": 5, "Education": 5, "Income": 6,
    },
    "Borderline negative": {
        **HEALTHY_YOUNG,
        "HighBP": 1, "BMI": 30, "PhysActivity": 0, "GenHlth": 3,
        "MentHlth": 1, "PhysHlth": 2, "Age": 6, "Education": 5, "Income": 6,
    },
    "Borderline positive": {
        **HEALTHY_YOUNG,
        "HighChol": 1, "BMI": 30, "PhysActivity": 0, "GenHlth": 3,
        "MentHlth": 1, "PhysHlth": 2, "Age": 7, "Education": 5, "Income": 6,
    },
    "Moderate risk with access barrier": {
        **HEALTHY_YOUNG,
        "HighBP": 1, "HighChol": 1, "BMI": 29, "PhysActivity": 0,
        "AnyHealthcare": 0, "NoDocbcCost": 1, "GenHlth": 3,
        "MentHlth": 7, "PhysHlth": 8, "Age": 7, "Education": 4, "Income": 3,
    },
    "High health burden": {
        **HEALTHY_YOUNG,
        "HighBP": 1, "HighChol": 1, "BMI": 39, "Smoker": 1,
        "Stroke": 1, "HeartDiseaseorAttack": 1, "PhysActivity": 0,
        "Fruits": 0, "Veggies": 0, "GenHlth": 5, "MentHlth": 15,
        "PhysHlth": 25, "DiffWalk": 1, "Sex": 1, "Age": 11,
        "Education": 3, "Income": 2,
    },
}

CASE_NOTES = {
    "Default video case": "Expected score about 0.726 — clear screen positive.",
    "Healthy young adult": "Expected score about 0.058 — clear screen negative.",
    "Lifestyle risk, no known conditions": "Expected score about 0.158 — lifestyle risk alone remains negative.",
    "Borderline negative": "Expected score about 0.413 — just below the 0.425 threshold.",
    "Borderline positive": "Expected score about 0.432 — just above the 0.425 threshold.",
    "Moderate risk with access barrier": "Expected score about 0.618 — screen positive with cost/access concerns.",
    "High health burden": "Expected score about 0.907 — strong screen positive.",
}


@st.cache_resource
def load_assets():
    model = joblib.load(ARTIFACT_DIR / "diabetes_screening_pipeline.joblib")
    metadata = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
    explanation = pd.read_csv(TABLE_DIR / "shap_demo_local.csv")
    return model, metadata, explanation


model, metadata, saved_explanation = load_assets()
st.title("Mission Health: diabetes risk screening")
st.caption(
    "A decision-support prototype trained on the 2015 US BRFSS survey. "
    "It prioritizes follow-up; it does not diagnose diabetes."
)

with st.sidebar:
    st.header("Responsible use")
    st.warning(
        "Do not deny care or make treatment decisions from this result. Confirm with clinical history, "
        "validated testing, and a qualified health professional."
    )
    st.write(f"Model: **{metadata['selected_model']}**")
    st.write(f"Screening threshold: **{metadata['operating_threshold']:.2f}** (chosen on validation data)")

scenario = st.selectbox("Choose a synthetic test case", options=list(TEST_CASES))
preset = TEST_CASES[scenario]
st.caption(CASE_NOTES[scenario])

with st.form(f"screening_form::{scenario}"):
    st.subheader("Synthetic respondent")
    col1, col2, col3 = st.columns(3)
    with col1:
        bmi = st.number_input(LABELS["BMI"], 10.0, 100.0, float(preset["BMI"]), 0.5, key=f"bmi::{scenario}")
        age = st.slider(LABELS["Age"], 1, 13, int(preset["Age"]), key=f"age::{scenario}")
        general_health = st.slider(LABELS["GenHlth"], 1, 5, int(preset["GenHlth"]), key=f"health::{scenario}")
        mental_days = st.slider(LABELS["MentHlth"], 0, 30, int(preset["MentHlth"]), key=f"mental::{scenario}")
        physical_days = st.slider(LABELS["PhysHlth"], 0, 30, int(preset["PhysHlth"]), key=f"physical::{scenario}")
        education = st.slider(LABELS["Education"], 1, 6, int(preset["Education"]), key=f"education::{scenario}")
        income = st.slider(LABELS["Income"], 1, 8, int(preset["Income"]), key=f"income::{scenario}")
    with col2:
        high_bp = st.checkbox(LABELS["HighBP"], bool(preset["HighBP"]), key=f"bp::{scenario}")
        high_chol = st.checkbox(LABELS["HighChol"], bool(preset["HighChol"]), key=f"chol::{scenario}")
        chol_check = st.checkbox(LABELS["CholCheck"], bool(preset["CholCheck"]), key=f"check::{scenario}")
        smoker = st.checkbox(LABELS["Smoker"], bool(preset["Smoker"]), key=f"smoker::{scenario}")
        stroke = st.checkbox(LABELS["Stroke"], bool(preset["Stroke"]), key=f"stroke::{scenario}")
        heart = st.checkbox(LABELS["HeartDiseaseorAttack"], bool(preset["HeartDiseaseorAttack"]), key=f"heart::{scenario}")
        walking = st.checkbox(LABELS["DiffWalk"], bool(preset["DiffWalk"]), key=f"walking::{scenario}")
    with col3:
        activity = st.checkbox(LABELS["PhysActivity"], bool(preset["PhysActivity"]), key=f"activity::{scenario}")
        fruits = st.checkbox(LABELS["Fruits"], bool(preset["Fruits"]), key=f"fruits::{scenario}")
        veggies = st.checkbox(LABELS["Veggies"], bool(preset["Veggies"]), key=f"veggies::{scenario}")
        alcohol = st.checkbox(LABELS["HvyAlcoholConsump"], bool(preset["HvyAlcoholConsump"]), key=f"alcohol::{scenario}")
        healthcare = st.checkbox(LABELS["AnyHealthcare"], bool(preset["AnyHealthcare"]), key=f"care::{scenario}")
        cost_barrier = st.checkbox(LABELS["NoDocbcCost"], bool(preset["NoDocbcCost"]), key=f"cost::{scenario}")
        sex = st.selectbox(
            LABELS["Sex"], options=[0, 1], index=int(preset["Sex"]),
            format_func=lambda value: "Female" if value == 0 else "Male",
            key=f"sex::{scenario}",
        )
    submitted = st.form_submit_button("Run screening", type="primary", use_container_width=True)

if submitted:
    record = {
        "HighBP": int(high_bp), "HighChol": int(high_chol), "CholCheck": int(chol_check),
        "BMI": bmi, "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart), "PhysActivity": int(activity),
        "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(alcohol),
        "AnyHealthcare": int(healthcare), "NoDocbcCost": int(cost_barrier),
        "GenHlth": general_health, "MentHlth": mental_days, "PhysHlth": physical_days,
        "DiffWalk": int(walking), "Sex": sex, "Age": age,
        "Education": education, "Income": income,
    }
    probability = float(model.predict_proba(pd.DataFrame([record], columns=RAW_FEATURES))[0, 1])
    threshold = float(metadata["operating_threshold"])
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Model risk score", f"{probability:.1%}", help="A ranking score, not a calibrated disease probability.")
        if probability >= threshold:
            st.error("Screen positive — prioritize confirmatory assessment")
        else:
            st.success("Screen negative — routine care still applies")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Decision threshold: {threshold:.1%}. False negatives and false positives remain possible.")
    with right:
        st.subheader("How to interpret this output")
        st.write(
            "The score combines survey-reported health, access, lifestyle, and demographic indicators. "
            "Associations are not causes. The US 2015 survey population may not represent another country, "
            "clinic, year, or individual."
        )
        st.write(
            "For the default video record, see the saved local SHAP explanation below. "
            "Changing inputs requires a new SHAP computation; this demo intentionally avoids presenting "
            "a stale explanation as if it described a changed record."
        )

st.divider()
st.subheader("Default-record local SHAP explanation")
st.caption("Positive contributions push the default synthetic record toward a higher score; negative contributions push it lower.")
chart = saved_explanation.head(10).set_index("feature")[["shap_contribution"]]
st.bar_chart(chart, horizontal=True)
