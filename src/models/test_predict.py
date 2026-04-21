from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.predict import predict_limits


def run_test_prediction(
    input_path: str | Path = "data/raw/Credit.csv",
    output_path: str | Path = "data/predictions/test_predictions.csv",
    model_path: str | Path = "artifacts/model.joblib",
    feature_artifact_path: str | Path = "artifacts/feature_artifact.joblib",
) -> pd.DataFrame:

    saved_path = predict_limits(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        feature_artifact_path=feature_artifact_path,
    )

    df_pred = pd.read_csv(saved_path)
    print(f"Arquivo de predição salvo em: {saved_path}")
    print(f"Shape do resultado: {df_pred.shape}")
    print(df_pred.head())

    return df_pred


if __name__ == "__main__":
    run_test_prediction()
