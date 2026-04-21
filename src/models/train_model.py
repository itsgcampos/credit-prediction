from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.data.preprocess import clean_credit_data
from src.features.build_features import build_features_pipeline, select_model_features
from src.models.evaluate_model import evaluate_regression


def train_linear_regression(
    raw_path: str | Path,
    model_path: str | Path = "artifacts/model.joblib",
    feature_artifact_path: str | Path = "artifacts/feature_artifact.joblib",
    metrics_path: str | Path = "artifacts/train_metrics.json",
    processed_output_path: str | Path | None = "data/processed/credit_processed.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    df_raw = load_raw_data(raw_path)
    print(f"Base bruta carregada: {df_raw.shape}")

    df_clean = clean_credit_data(df_raw)
    print(f"Base após limpeza: {df_clean.shape}")

    df_processed, feature_artifact = build_features_pipeline(df_clean, mode="train")
    print(f"Base após feature engineering + encoding: {df_processed.shape}")

    if processed_output_path is not None:
        processed_path = Path(processed_output_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(processed_path, index=False)
        print(f"CSV tratado salvo em: {processed_path}")

    X, y = select_model_features(
        df_processed,
        target=feature_artifact["target"],
        leakage_columns=feature_artifact["leakage_columns"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = evaluate_regression(y_train, y_pred_train)
    test_metrics = evaluate_regression(y_test, y_pred_test)

    feature_artifact["model_features"] = X.columns.tolist()
    feature_artifact["test_size"] = test_size
    feature_artifact["random_state"] = random_state

    model_path = Path(model_path)
    feature_artifact_path = Path(feature_artifact_path)
    metrics_path = Path(metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(feature_artifact, feature_artifact_path)

    payload = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": X.columns.tolist(),
        "leakage_columns_removed": feature_artifact["leakage_columns"],
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Treino concluído.")
    print(f"Métricas de teste -> MAE: {test_metrics['mae']:.4f} | RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f}")

    return {
        "model_path": str(model_path),
        "feature_artifact_path": str(feature_artifact_path),
        "metrics_path": str(metrics_path),
        "processed_output_path": str(processed_output_path) if processed_output_path is not None else None,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    train_linear_regression("data/raw/Credit.csv")
