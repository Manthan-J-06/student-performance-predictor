3# 🎓 AI-Driven Student Performance Prediction System

Predicts whether a student is likely to **pass or is at risk of failing**, using machine learning trained on real academic, behavioral, and demographic data — with a live Streamlit app and per-student explainability powered by SHAP.
---

## Overview

This project explores whether a student's outcome can be predicted from data that schools already collect — attendance, study habits, family background, and (optionally) current test scores — using the [UCI Student Performance Dataset](https://doi.org/10.24432/C5TG7T) (Cortez & Silva, 2008).

It ships with **two models**, because they answer two different real questions:

| Model | Question it answers | Inputs used | Accuracy |
|---|---|---|---|
| 🔮 **Early-Warning Classifier** | "Is this student at risk, *before* any grades exist?" | Attendance, study time, past failures, background — **no test scores** | 65.8% |
| 📈 **Grade-Aware Regressor** | "Given everything we know, including current scores, what will their final grade be?" | Everything, including 1st/2nd term test scores | R² = 0.82 (avg. error ≈ 1.3 / 20) |

The app lets you pick which situation applies and uses the right model for it.

---


## Why two models instead of one?

A single model trained on current-term grades essentially learns "if 2nd term score ≥ 10, predict pass" and mostly ignores everything else — because grades genuinely are that predictive of a student's next grade. That's a real, documented property of this dataset, not a bug.

Rather than hide that, this project:
- Trains an **early-warning model without grades**, so it's forced to rely on attendance/behavior — genuinely useful before grades exist.
- Trains a **grade-aware model with everything**, and uses **SHAP** to show, per student, exactly how many grade points *each individual factor* contributed to that prediction — so the influence of non-grade factors is visible and honest instead of buried.

---

## Features

- 📊 **Exploratory Data Analysis** — auto-generated charts (grade distribution, study time vs. grade, absences vs. grade, correlation heatmap, etc.)
- 🤖 **Model comparison** — Logistic Regression vs. Random Forest (classification), Linear vs. Random Forest (regression)
- 🧠 **Explainable predictions** — SHAP-based breakdown of what drove each individual prediction
- 🖥️ **Interactive Streamlit app** — clean, tabbed form; no raw dataset jargon
- 📦 **Fully reproducible pipeline** — from raw CSV to trained model to live app


## Project Structure

```
student_performance_project/
├── data/
│   ├── student-mat.csv              # Math course dataset (395 students)
│   └── student-por.csv              # Portuguese course dataset (649 students)
├── src/
│   ├── preprocess.py                # Data loading, cleaning, encoding, splits
│   ├── eda.py                       # Exploratory Data Analysis → outputs/
│   ├── train_model.py               # Trains early-warning classifier
│   └── train_regression_model.py    # Trains grade-aware regressor + SHAP explainer
├── app/
│   └── app.py                       # Streamlit web app
├── models/                          # Saved models, scalers, explainer (generated)
├── outputs/                         # Generated charts + comparison tables (generated)
├── requirements.txt
└── README.md
```

## How It Works

1. **Preprocessing** — categorical features one-hot encoded, numeric features standardized, target derived from final grade (pass = G3 ≥ 10).
2. **Two training pipelines** — one excludes current-term grades entirely (`train_model.py`), one includes everything (`train_regression_model.py`).
3. **Explainability** — `shap.TreeExplainer` computes per-student SHAP values for the regression model, so the app can show exactly how many grade points each answer contributed.
4. **App** — Streamlit form grouped into logical tabs (Academic / Lifestyle & Family / Optional), routes to the appropriate model based on whether current scores are known.

---

## Dataset

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `SHAP` · `Streamlit` · `matplotlib` / `seaborn`

## Future Work

- Try combining both Math and Portuguese datasets for more training data
- Add gradient boosting (XGBoost/LightGBM) for comparison
- Deploy publicly via Streamlit Community Cloud
- Add a teacher-facing dashboard view for class-wide trends

========================================================================
HOW TO ACCESS / USE THE APP
========================================================================

1. INSTALL DEPENDENCIES (one-time setup)
-----------------------------------------
Open a terminal in the project folder and run:

    pip install -r requirements.txt


2. TRAIN THE MODELS (one-time setup)
-------------------------------------
Run these two commands, in order, from the project ROOT folder:

    python src/train_model.py
    python src/train_regression_model.py

This creates the trained model files the app needs. You only need to
do this once (or again if you change the code/data).


3. LAUNCH THE APP
------------------
    streamlit run app/app.py

If you get "'streamlit' is not recognized", use this instead:

    python -m streamlit run app/app.py

This will automatically open the app in your browser at:

    http://localhost:8501


4. USING THE APP
------------------
*   First, tell it whether the student's current test scores are
    already known ("Yes" or "No") -- this decides which model is used.
*   Fill in the Academic, Lifestyle & Family, and Optional tabs.
*   Click "Predict Performance."
*   You'll get a Pass / At-Risk result, a confidence percentage, and
    (if scores were provided) a breakdown showing exactly which
    factors pushed the prediction up or down.

Note: This app runs locally on your own machine -- there is no public
website link. Anyone who wants to use it needs to clone this repo and
run the steps above on their own computer.

========================================================================