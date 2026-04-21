from __future__ import annotations

import argparse
from pathlib import Path

from src.models.train_model import train_linear_regression
from src.models.predict import predict_limits
from src.models.test_predict import run_test_prediction


DEFAULT_RAW_PATH = "data/raw/Credit.csv"
DEFAULT_PROCESSED_PATH = "data/processed/credit_processed.csv"
DEFAULT_MODEL_PATH = "models/linear_regression.pkl"
DEFAULT_METRICS_PATH = "outputs/metrics/metrics.json"
DEFAULT_PREDICTIONS_PATH = "data/predictions/credit_predictions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orquestrador do projeto de predição de limite de crédito."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Treina o modelo e salva artefatos.")
    train_parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Caminho da base bruta CSV.")
    train_parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Caminho para salvar o modelo.")
    train_parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Caminho para salvar métricas.")
    train_parser.add_argument("--processed", default=DEFAULT_PROCESSED_PATH, help="Caminho para salvar o CSV tratado.")
    train_parser.add_argument("--test-size", type=float, default=0.20, help="Proporção do conjunto de teste.")
    train_parser.add_argument("--random-state", type=int, default=42, help="Seed do split.")

    predict_parser = subparsers.add_parser("predict", help="Gera predições a partir de uma base crua.")
    predict_parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Caminho da base de entrada CSV.")
    predict_parser.add_argument("--output", default=DEFAULT_PREDICTIONS_PATH, help="Caminho para salvar predições.")
    predict_parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Caminho do modelo treinado.")

    smoke_parser = subparsers.add_parser("test-predict", help="Executa um teste rápido do pipeline de predição.")
    smoke_parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Caminho da base de entrada CSV.")
    smoke_parser.add_argument("--output", default="data/predictions/test_predictions.csv", help="Caminho para salvar o resultado do teste.")
    smoke_parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Caminho do modelo treinado.")

    run_all_parser = subparsers.add_parser("run-all", help="Treina e depois gera predições.")
    run_all_parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Caminho da base bruta CSV.")
    run_all_parser.add_argument("--output", default=DEFAULT_PREDICTIONS_PATH, help="Caminho para salvar predições.")
    run_all_parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Caminho do modelo treinado.")
    run_all_parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Caminho para salvar métricas.")
    run_all_parser.add_argument("--processed", default=DEFAULT_PROCESSED_PATH, help="Caminho para salvar o CSV tratado.")
    run_all_parser.add_argument("--test-size", type=float, default=0.20, help="Proporção do conjunto de teste.")
    run_all_parser.add_argument("--random-state", type=int, default=42, help="Seed do split.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_linear_regression(
            raw_path=args.input,
            model_path=args.model,
            feature_artifact_path=args.feature_artifact,
            metrics_path=args.metrics,
            processed_output_path=args.processed,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        return

    if args.command == "predict":
        output = predict_limits(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            feature_artifact_path=args.feature_artifact,
        )
        print(f"Predições salvas em: {output}")
        return

    if args.command == "test-predict":
        run_test_prediction(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            feature_artifact_path=args.feature_artifact,
        )
        return

    if args.command == "run-all":
        train_linear_regression(
            raw_path=args.input,
            model_path=args.model,
            feature_artifact_path=args.feature_artifact,
            metrics_path=args.metrics,
            processed_output_path=args.processed,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        output = predict_limits(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            feature_artifact_path=args.feature_artifact,
        )
        print(f"Pipeline completo finalizado. Predições salvas em: {output}")
        return

    raise ValueError(f"Comando não suportado: {args.command}")


if __name__ == "__main__":
    main()
