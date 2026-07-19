"""
eda.py
------
Exploratory Data Analysis for the Student Performance dataset.
Generates plots and saves them to the outputs/ folder so you can
drop them straight into your report/presentation.

Run with:  python src/eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data, add_target_label

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def run_eda():
    df = load_data()
    df = add_target_label(df)

    # 1. Basic info
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- Missing values ---")
    print(df.isnull().sum().sum(), "missing values total")
    print("\n--- Target balance ---")
    print(df["pass"].value_counts(normalize=True).round(3))

    # 2. Final grade distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Final Grade (G3)")
    ax.set_xlabel("Final Grade (0-20)")
    save(fig, "01_grade_distribution.png")

    # 3. Study time vs final grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="studytime", y="G3", data=df, ax=ax)
    ax.set_title("Study Time vs Final Grade")
    ax.set_xlabel("Weekly Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
    ax.set_ylabel("Final Grade")
    save(fig, "02_studytime_vs_grade.png")

    # 4. Absences vs final grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="absences", y="G3", data=df, hue="pass", ax=ax)
    ax.set_title("Absences vs Final Grade")
    save(fig, "03_absences_vs_grade.png")

    # 5. Past failures vs pass rate
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x="failures", y="pass", data=df, ax=ax)
    ax.set_title("Past Failures vs Pass Rate")
    ax.set_ylabel("Proportion who Passed")
    save(fig, "04_failures_vs_passrate.png")

    # 6. Correlation heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    save(fig, "05_correlation_heatmap.png")

    # 7. Parental education vs grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="Medu", y="G3", data=df, ax=ax)
    ax.set_title("Mother's Education Level vs Final Grade")
    ax.set_xlabel("Mother's Education (0=none - 4=higher education)")
    save(fig, "06_medu_vs_grade.png")

    print(f"\nAll plots saved to '{OUTPUT_DIR}/'. Use these directly in your report.")


if __name__ == "__main__":
    run_eda()
