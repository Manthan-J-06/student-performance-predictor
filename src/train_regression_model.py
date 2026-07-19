"""
train_regression_model.py
--------------------------
Trains a model that predicts the ACTUAL final grade (G3, 0-20) as a
continuous number, using every available feature -- including previous
term grades (G1/G2) AND every behavioral/demographic feature.

Why regression instead of pass/fail classification for the main app:
A classifier trained with G1/G2 saturates near a fixed probability once
G1/G2 cross the pass threshold, making it look like nothing else matters.
A regression model predicting the exact score doesn't have that hard
threshold -- every input nudges the predicted number, which is what
you'd expect from a system that's supposed to weigh "each and every
question asked". We also compute SHAP values so the app can show,
for any individual prediction, exactly how much each answer pushed the
score up or down.

Run with:  python src/train_regression_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import build_regression_dataset

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"MAE  : {mae:.3f} (average error in grade points, out of 20)")
    print(f"RMSE : {rmse:.3f}")
    print(f"R^2  : {r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    X_train, X_test, y_train, y_test, scaler = build_regression_dataset()

    results = []

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lin_reg, X_test, y_test))

    rf = RandomForestRegressor(
        n_estimators=400, max_depth=6, max_features=0.3,
        min_samples_leaf=5, random_state=42,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest Regressor", rf, X_test, y_test))

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison (regression) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_regression.csv"), index=False)

    # Pick best model by R^2
    best_row = results_df.loc[results_df["r2"].idxmax()]
    best_model = rf if "Random Forest" in best_row["model"] else lin_reg
    print(f"\nBest regression model: {best_row['model']} (R^2={best_row['r2']:.3f})")

    # Residual std -- used by the app to turn a predicted score into a
    # rough pass-probability band (assumes ~normal residuals).
    residuals = y_test.values - best_model.predict(X_test)
    residual_std = float(np.std(residuals))
    print(f"Residual std (used for confidence estimate): {residual_std:.3f}")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, ax=ax)
        ax.set_title("Top 15 Feature Importances - Grade Regression Model")
        ax.set_xlabel("Importance")
        fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance_regression.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR}/feature_importance_regression.png")

    # Actual vs predicted scatter -- good report visual
    preds_test = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds_test, alpha=0.6)
    ax.plot([0, 20], [0, 20], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Final Grade (G3)")
    ax.set_ylabel("Predicted Final Grade")
    ax.set_title("Actual vs Predicted Grade")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/actual_vs_predicted.png")

    # Save artifacts for the app
    joblib.dump(best_model, os.path.join(MODEL_DIR, "regression_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "regression_scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "regression_feature_columns.pkl"))
    joblib.dump(residual_std, os.path.join(MODEL_DIR, "regression_residual_std.pkl"))
    print(f"\nSaved regression model + scaler + feature list + residual_std to {MODEL_DIR}/")

    # Build and save a SHAP explainer so the app can show, for any single
    # prediction, exactly how much each answer pushed the score up/down.
    import shap
    explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, os.path.join(MODEL_DIR, "regression_shap_explainer.pkl"))
    print(f"Saved: {MODEL_DIR}/regression_shap_explainer.pkl")


if __name__ == "__main__":
    main()
