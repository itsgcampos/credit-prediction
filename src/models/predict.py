from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_ARTIFACT_PATH,
    DEFAULT_PREDICT_INPUT_PATH,
    DEFAULT_PREDICTIONS_PATH,
)
from src.data.load_data import load_raw_data
from src.data.preprocess import clean_credit_data
from src.features.build_features import build_features_pipeline


def predict_limits(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_artifact_path: str | Path = DEFAULT_FEATURE_ARTIFACT_PATH,
) -> Path:
    model = joblib.load(model_path)
    feature_artifact = joblib.load(feature_artifact_path)

    df_raw = load_raw_data(input_path)
    df_clean = clean_credit_data(df_raw, target=feature_artifact.get("target", "Limit"))

    df_output = df_clean.copy()

    X_encoded, _ = build_features_pipeline(
        df_clean,
        target=feature_artifact["target"],
        mode="predict",
        feature_artifact=feature_artifact,
    )

    model_features = feature_artifact["model_features"]
    X_model = X_encoded[model_features].copy()

    predictions = model.predict(X_model)
    df_output["predicted_limit"] = predictions

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output, index=False)

    return output


def predict(
    input_path: str | Path = DEFAULT_PREDICT_INPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = DEFAULT_PREDICTIONS_PATH,
) -> Path:
    """
    Função wrapper para fazer predições.
    
    Args:
        input_path: Caminho dos dados para prever
        model_path: Caminho do modelo treinado
        output_path: Caminho para salvar as predições
    
    Returns:
        Path do arquivo com predições salvo
    """
    return predict_limits(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        feature_artifact_path=DEFAULT_FEATURE_ARTIFACT_PATH,
    )


if __name__ == "__main__":
    saved = predict_limits(
        input_path="data/raw/Credit.csv",
        output_path="data/predictions/credit_predictions.csv",
    )
    print(f"Predições salvas em: {saved}")
    print(f"Predições salvas em: {saved}")
