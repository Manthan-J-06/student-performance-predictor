"""
app.py
------
Streamlit app for the AI-Driven Student Performance Prediction System.

Two modes, chosen by the user up front:

1. "I have current test scores" -> uses a regression model trained on
   EVERY feature, including G1/G2, to predict the actual final grade
   (0-20). Because previous grades really are the strongest real-world
   predictor of a student's outcome (confirmed by feature-importance
   analysis on this dataset), they dominate the number -- but every
   other answer still nudges the prediction, and a SHAP-based breakdown
   shows exactly how many grade points each individual answer added or
   subtracted for THIS specific student. This makes "every question
   matters" visible and honest instead of hidden inside a black box.

2. "I don't have scores yet" -> uses a classifier trained WITHOUT G1/G2,
   for genuine early-warning use before any grades exist. This is the
   model where behavioral/attendance factors move the needle the most.

Run with:  python -m streamlit run app/app.py
(run this command from the project ROOT folder, not from inside app/)
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import encode_features  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 760px; }
    h1 { font-size: 2rem !important; margin-bottom: 0.2rem !important; }
    .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #f4f6fa; border-radius: 8px 8px 0 0; padding: 10px 18px; }
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: 600; padding: 0.65rem 0;
        background-color: #2E5EAA; color: white; border: none; margin-top: 1rem;
    }
    .stButton>button:hover { background-color: #244a87; }
    div[data-testid="stMetric"] { background-color: #f4f6fa; padding: 1rem; border-radius: 10px; }
    .section-note { color: #888; font-size: 0.85rem; margin-top: -8px; margin-bottom: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

FIXED_DEFAULTS = {
    "school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "nursery": "yes", "traveltime": 1,
}

# Friendly labels for the SHAP breakdown so raw column names never show up.
FRIENDLY_LABELS = {
    "G1": "First-term score", "G2": "Second-term score", "failures": "Past failures",
    "absences": "Absences", "studytime": "Study time", "age": "Age", "goout": "Socializing frequency",
    "health": "General health", "Walc": "Weekend drinking", "Dalc": "Weekday drinking",
    "Medu": "Mother's education", "Fedu": "Father's education", "famrel": "Family relationship quality",
    "freetime": "Free time", "schoolsup_yes": "Extra academic support", "famsup_yes": "Family support",
    "paid_yes": "Extra paid classes", "activities_yes": "Extracurricular activities",
    "higher_yes": "Wants higher education", "internet_yes": "Internet access", "romantic_yes": "In a relationship",
    "traveltime": "Travel time", "address_U": "Urban address", "famsize_LE3": "Small family",
}


def friendly(col):
    if col in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[col]
    return col.replace("_", " ").replace("yes", "").strip().title()


@st.cache_resource
def load_grade_model():
    model = joblib.load(os.path.join(MODEL_DIR, "regression_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "regression_scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "regression_feature_columns.pkl"))
    residual_std = joblib.load(os.path.join(MODEL_DIR, "regression_residual_std.pkl"))
    try:
        explainer = joblib.load(os.path.join(MODEL_DIR, "regression_shap_explainer.pkl"))
    except (ModuleNotFoundError, ImportError):
        # shap isn't installed -- app still works, just without the
        # per-prediction breakdown chart. Run: pip install shap
        explainer = None
    return model, scaler, feature_columns, residual_std, explainer


@st.cache_resource
def load_early_warning_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model_early_warning.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_early_warning.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns_early_warning.pkl"))
    return model, scaler, feature_columns


st.title("🎓 Student Performance Predictor")
st.markdown(
    '<div class="subtitle">Predicts pass/fail risk using history, habits, '
    "and (if available) current test scores.</div>",
    unsafe_allow_html=True,
)

try:
    grade_model, grade_scaler, grade_cols, residual_std, explainer = load_grade_model()
    ew_model, ew_scaler, ew_cols = load_early_warning_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python src/train_model.py` and "
        "`python src/train_regression_model.py` from the project root first."
    )
    st.stop()

mode = st.radio(
    "Does this student already have current test scores?",
    ["Yes — use their scores", "No — predict before scores exist"],
    horizontal=True,
)
has_scores = mode.startswith("Yes")

if has_scores:
    st.caption(
        "Previous grades are the strongest real-world predictor of a student's "
        "final outcome — so they'll dominate the result below. Every other answer "
        "still shifts the prediction; the breakdown after you predict shows exactly "
        "how much, in grade points, each factor contributed for this student."
    )

tab_academic, tab_lifestyle, tab_optional = st.tabs(["📊 Academic", "🎒 Lifestyle & Family", "➕ Optional"])

with tab_academic:
    if has_scores:
        c1, c2 = st.columns(2)
        with c1:
            G1 = st.slider("First-term test score (out of 20)", 0, 20, 12)
        with c2:
            G2 = st.slider("Second-term test score (out of 20)", 0, 20, 12)
    else:
        st.markdown(
            '<div class="section-note">No scores yet — these are the strongest available signals.</div>',
            unsafe_allow_html=True,
        )

    failures = st.select_slider(
        "Classes failed in the past",
        options=["None", "1", "2", "3", "4 or more"], value="None",
    )
    failures_map = {"None": 0, "1": 1, "2": 2, "3": 3, "4 or more": 4}

    absences = st.slider("Days absent this term", 0, 75, 4)

    studytime = st.select_slider(
        "Weekly study time outside class",
        options=["Under 2 hrs", "2–5 hrs", "5–10 hrs", "Over 10 hrs"], value="2–5 hrs",
    )
    studytime_map = {"Under 2 hrs": 1, "2–5 hrs": 2, "5–10 hrs": 3, "Over 10 hrs": 4}

with tab_lifestyle:
    c3, c4 = st.columns(2)
    with c3:
        age = st.slider("Student's age", 15, 22, 17)
        goout = st.select_slider(
            "How often does the student socialize with friends?",
            options=["Rarely", "Sometimes", "Regularly", "Often", "Almost daily"], value="Regularly",
        )
        health = st.select_slider(
            "Overall health",
            options=["Poor", "Below average", "Average", "Good", "Excellent"], value="Good",
        )
    with c4:
        weekend_social = st.select_slider(
            "Weekend social drinking (self-reported)",
            options=["None", "Occasional", "Moderate", "Frequent", "Heavy"], value="None",
        )
        Medu = st.select_slider(
            "Mother's education level",
            options=["None", "Primary", "Middle school", "High school", "University"], value="Middle school",
        )
        Fedu = st.select_slider(
            "Father's education level",
            options=["None", "Primary", "Middle school", "High school", "University"], value="Middle school",
        )
    scale5 = {"Rarely": 1, "Sometimes": 2, "Regularly": 3, "Often": 4, "Almost daily": 5,
              "Poor": 1, "Below average": 2, "Average": 3, "Good": 4, "Excellent": 5,
              "None": 1, "Occasional": 2, "Moderate": 3, "Frequent": 4, "Heavy": 5}
    edu_map = {"None": 0, "Primary": 1, "Middle school": 2, "High school": 3, "University": 4}

with tab_optional:
    st.markdown(
        '<div class="section-note">Each of these shifts the prediction slightly — skip anything you don\'t know.</div>',
        unsafe_allow_html=True,
    )
    applies = st.multiselect(
        "Which of these apply to the student?",
        [
            "Gets extra tutoring or academic support",
            "Takes part in extracurricular activities",
            "Has internet access at home",
            "Plans to pursue higher education",
            "Currently in a relationship",
        ],
        default=["Has internet access at home", "Plans to pursue higher education"],
    )
    c5, c6 = st.columns(2)
    with c5:
        freetime = st.select_slider(
            "Free time after school", options=["Very little", "Little", "Moderate", "A lot", "Very much"], value="Moderate",
        )
    with c6:
        famrel = st.select_slider(
            "Quality of family relationships", options=["Poor", "Below average", "Average", "Good", "Excellent"], value="Good",
        )
    scale5b = {"Very little": 1, "Little": 2, "Moderate": 3, "A lot": 4, "Very much": 5,
               "Poor": 1, "Below average": 2, "Average": 3, "Good": 4, "Excellent": 5}

st.write("")
predict_clicked = st.button("🔮 Predict Performance", type="primary")

if predict_clicked:
    schoolsup = "yes" if "Gets extra tutoring or academic support" in applies else "no"
    activities = "yes" if "Takes part in extracurricular activities" in applies else "no"
    internet = "yes" if "Has internet access at home" in applies else "no"
    higher = "yes" if "Plans to pursue higher education" in applies else "no"
    romantic = "yes" if "Currently in a relationship" in applies else "no"

    raw_dict = dict(
        **FIXED_DEFAULTS,
        age=age, Medu=edu_map[Medu], Fedu=edu_map[Fedu], studytime=studytime_map[studytime],
        failures=failures_map[failures], schoolsup=schoolsup, famsup=schoolsup, paid=schoolsup,
        activities=activities, higher=higher, internet=internet, romantic=romantic,
        famrel=scale5b[famrel], freetime=scale5b[freetime], goout=scale5[goout],
        Dalc=1, Walc=scale5[weekend_social], health=scale5[health], absences=absences,
    )

    st.divider()
    st.subheader("Result")

    if has_scores:
        raw_dict["G1"] = G1
        raw_dict["G2"] = G2
        raw_input = pd.DataFrame([raw_dict])
        encoded = encode_features(raw_input).reindex(columns=grade_cols, fill_value=0)
        scaled = pd.DataFrame(grade_scaler.transform(encoded), columns=encoded.columns)

        predicted_score = float(grade_model.predict(scaled)[0])
        predicted_score = max(0, min(20, predicted_score))
        pass_probability = float(norm.cdf((predicted_score - 10) / residual_std))

        r1, r2 = st.columns([1, 1])
        with r1:
            if predicted_score >= 10:
                st.success(f"✅ Predicted grade: {predicted_score:.1f} / 20 — likely PASS")
            else:
                st.error(f"⚠️ Predicted grade: {predicted_score:.1f} / 20 — AT RISK")
        with r2:
            st.metric("Estimated probability of passing", f"{pass_probability:.0%}")
        st.progress(float(pass_probability))

        # --- SHAP breakdown for this specific prediction ---
        if explainer is not None:
            st.markdown("#### What drove this prediction")
            shap_vals = explainer.shap_values(scaled)[0]
            base_value = float(np.array(explainer.expected_value).flatten()[0])
            contrib = pd.Series(shap_vals, index=grade_cols)
            contrib = contrib[contrib.abs() > 0.01].sort_values(key=abs, ascending=False).head(8)

            if len(contrib) > 0:
                breakdown_df = pd.DataFrame({
                    "Factor": [friendly(c) for c in contrib.index],
                    "Effect on predicted grade": contrib.values.round(2),
                }).set_index("Factor")
                st.bar_chart(breakdown_df, horizontal=True)
                st.caption(
                    f"Average student in this dataset scores {base_value:.1f}/20. "
                    f"Each bar shows how many grade points that answer added or "
                    f"subtracted for THIS student, landing at {predicted_score:.1f}/20."
                )
        else:
            st.caption(
                "Install the `shap` package (`pip install shap`) to see a "
                "per-student breakdown of what drove this prediction."
            )
    else:
        raw_input = pd.DataFrame([raw_dict])
        encoded = encode_features(raw_input).reindex(columns=ew_cols, fill_value=0)
        scaled = pd.DataFrame(ew_scaler.transform(encoded), columns=encoded.columns)

        prediction = ew_model.predict(scaled)[0]
        probability = ew_model.predict_proba(scaled)[0][1]

        r1, r2 = st.columns([1, 1])
        with r1:
            if prediction == 1:
                st.success("✅ Likely to PASS")
            else:
                st.error("⚠️ AT RISK of failing")
        with r2:
            st.metric("Probability of passing", f"{probability:.0%}")
        st.progress(float(probability))

        if prediction == 0:
            st.info(
                "💡 The biggest levers here are **past failures** and **absences** — "
                "an attendance plan or early tutoring tend to move the needle most "
                "for at-risk students, well before final grades are in."
            )

    st.caption(
        "This is a statistical estimate based on patterns in historical data — "
        "not a certainty, and not a judgment of the student."
    )

st.divider()
st.caption(
    "AI-Driven Student Performance Prediction System | scikit-learn + SHAP + Streamlit | "
    "Dataset: UCI Student Performance Data Set (Cortez & Silva, 2008)"
)
