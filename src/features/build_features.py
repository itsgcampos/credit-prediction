from __future__ import annotations

from typing import Any
import pandas as pd


LEAKAGE_COLUMNS = ["Rating", "Balance"]


def create_manual_features(df: pd.DataFrame) -> pd.DataFrame:

    df_feat = df.copy()

    if {"Balance", "Income"}.issubset(df_feat.columns):
        df_feat["debt_to_income"] = df_feat["Balance"] / (df_feat["Income"] + 1)

    if {"Cards", "Age"}.issubset(df_feat.columns):
        df_feat["cards_per_age"] = df_feat["Cards"] / (df_feat["Age"] + 1)

    return df_feat


def build_features_pipeline(
    df: pd.DataFrame,
    target: str = "Limit",
    mode: str = "train",
    feature_artifact: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    if mode not in {"train", "predict"}:
        raise ValueError("mode deve ser 'train' ou 'predict'.")

    df_feat = create_manual_features(df)

    feature_base = df_feat.copy()
    has_target = target in feature_base.columns

    # Identify categorical columns excluding target
    columns_for_types = feature_base.drop(columns=[target]) if has_target else feature_base
    categorical_cols = columns_for_types.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        encoded = pd.get_dummies(feature_base, columns=categorical_cols, drop_first=True)
    else:
        encoded = feature_base.copy()

    if mode == "train":
        artifact = {
            "target": target,
            "categorical_columns": categorical_cols,
            "encoded_columns": [col for col in encoded.columns if col != target],
            "leakage_columns": LEAKAGE_COLUMNS,
        }
        return encoded, artifact

    if feature_artifact is None:
        raise ValueError("feature_artifact é obrigatório em mode='predict'.")

    expected_columns = feature_artifact["encoded_columns"]


    if target in encoded.columns:
        encoded = encoded.drop(columns=[target])

    for col in expected_columns:
        if col not in encoded.columns:
            encoded[col] = 0

    extra_cols = [col for col in encoded.columns if col not in expected_columns]
    if extra_cols:
        encoded = encoded.drop(columns=extra_cols)

    encoded = encoded[expected_columns].copy()

    return encoded, feature_artifact


def select_model_features(
    df_processed: pd.DataFrame,
    target: str = "Limit",
    leakage_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:

    leakage_columns = leakage_columns or LEAKAGE_COLUMNS

    missing = [col for col in [target] if col not in df_processed.columns]
    if missing:
        raise ValueError(f"Coluna target ausente no dataframe processado: {missing}")

    X = df_processed.drop(columns=[target] + [col for col in leakage_columns if col in df_processed.columns]).copy()
    y = df_processed[target].copy()


    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X = X.astype({col: "int64" for col in bool_cols})

    X = X.apply(pd.to_numeric)

    return X, y
