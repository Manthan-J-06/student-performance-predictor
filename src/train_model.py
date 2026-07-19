"""
train_model.py
---------------
Trains and compares two models (Logistic Regression, Random Forest)
on the Student Performance dataset, evaluates them, saves the best
one to disk, and saves a feature-importance chart for your report.

Run with:  python src/train_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import build_dataset

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_feature_importance(model, feature_names, top_n=15, suffix=""):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest) - {suffix}")
    ax.set_xlabel("Importance")
    fname = f"feature_importance_{suffix}.png" if suffix else "feature_importance.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/{fname}")


def train_variant(drop_G1_G2: bool, label: str):
    """
    Trains + compares models for one variant of the problem, then saves
    the best model, scaler, and feature list under a name matching `label`.

    label is either "with_grades" or "early_warning".
    """
    X_train, X_test, y_train, y_test, scaler = build_dataset(drop_G1_G2=drop_G1_G2)

    results = []

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    results.append(evaluate(f"Logistic Regression ({label})", log_reg, X_test, y_test))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate(f"Random Forest ({label})", rf, X_test, y_test))

    results_df = pd.DataFrame(results)
    print(f"\n=== Model Comparison: {label} ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"model_comparison_{label}.csv"), index=False)

    plot_feature_importance(rf, X_train.columns, top_n=15, suffix=label)

    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_model = rf if "Random Forest" in best_row["model"] else log_reg
    print(f"Best model ({label}) by F1: {best_row['model']} (F1={best_row['f1']:.3f})")

    joblib.dump(best_model, os.path.join(MODEL_DIR, f"best_model_{label}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{label}.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, f"feature_columns_{label}.pkl"))
    print(f"Saved to {MODEL_DIR}/best_model_{label}.pkl\n")

    return results_df


def main():
    # Variant A: includes G1/G2 (current term grades already exist).
    # Easier problem, higher accuracy, but barely uses anything except grades
    # -- not very useful as an "early warning" tool since the grades already
    # tell you most of the answer.
    results_with_grades = train_variant(drop_G1_G2=False, label="with_grades")

    # Variant B: NO G1/G2. This is the genuinely useful "early warning"
    # version -- predicts risk using only attendance, study habits,
    # past failures, and background factors, before current grades exist.
    # Lower accuracy, but the model actually has to use every feature
    # meaningfully instead of shortcutting through G1/G2.
    results_early = train_variant(drop_G1_G2=True, label="early_warning")

    print("\n=== Summary: grade-inclusive vs early-warning ===")
    print("With G1/G2   -> best F1:", results_with_grades["f1"].max().round(3))
    print("Early warning -> best F1:", results_early["f1"].max().round(3))
    print(
        "\nThe drop in accuracy for the early-warning model is expected and "
        "worth discussing in your report: it trades some accuracy for being "
        "usable before grades exist, and for actually depending on the "
        "behavioral/demographic features rather than shortcutting through G1/G2."
    )


if __name__ == "__main__":
    main()
