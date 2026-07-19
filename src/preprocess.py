"""
preprocess.py
--------------
Loads the raw UCI Student Performance dataset, cleans it, engineers a
target label, encodes categorical features, and returns train/test splits.

Dataset source: UCI Machine Learning Repository - Student Performance
(Cortez & Silva, 2008) -> student-mat.csv (Math course)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/student-mat.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. The UCI file uses ';' as a separator."""
    df = pd.read_csv(path, sep=";")
    return df


def add_target_label(df: pd.DataFrame, pass_threshold: int = 10) -> pd.DataFrame:
    """
    Create a binary target column 'pass' from the final grade G3.
    G3 ranges 0-20. pass = 1 if G3 >= pass_threshold else 0.
    """
    df = df.copy()
    df["pass"] = (df["G3"] >= pass_threshold).astype(int)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical (object-type) columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def get_train_test_split(
    df: pd.DataFrame,
    target_col: str = "pass",
    drop_cols=None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Split into train/test sets. By default drops G3 (used to build the
    target) and G1/G2 are KEPT since they are legitimate early-term
    predictors. Set drop_cols to also drop G1/G2 if you want a harder,
    "early warning before any grades exist" version of the problem.
    """
    if drop_cols is None:
        drop_cols = ["G3"]  # G3 is the raw grade the target was derived from

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test, None


def get_regression_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Train/test split for predicting the ACTUAL final grade (G3, 0-20) as a
    continuous number, using every available feature including G1/G2.
    This is used for the main app so that every answer -- not just a
    pass/fail threshold on grades -- visibly moves the prediction.
    """
    X = df.drop(columns=["G3", "pass"])
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test, None


def build_dataset(path: str = DATA_PATH, drop_G1_G2: bool = False):
    """
    Convenience function that runs the full pipeline and returns
    train/test splits ready for modeling.
    """
    df = load_data(path)
    df = add_target_label(df)
    df_encoded = encode_features(df)

    drop_cols = ["G3"]
    if drop_G1_G2:
        drop_cols += ["G1", "G2"]

    return get_train_test_split(df_encoded, drop_cols=drop_cols)


def build_regression_dataset(path: str = DATA_PATH):
    """
    Convenience function for the continuous-score regression pipeline.
    Uses ALL features (including G1/G2) to predict the exact final grade.
    """
    df = load_data(path)
    df = add_target_label(df)
    df_encoded = encode_features(df)
    return get_regression_split(df_encoded)


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    X_train, X_test, y_train, y_test, scaler = build_dataset()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Pass rate (train):", y_train.mean().round(3))
    print("Pass rate (test):", y_test.mean().round(3))
