import pandas as pd
import sys

from src.config import DEFAULT_RAW_PATH, DEFAULT_MODEL_PATH, DEFAULT_METRICS_PATH, DEFAULT_PROCESSED_PATH, DEFAULT_PREDICTIONS_PATH, DEFAULT_PREDICT_INPUT_PATH
from src.models.train_model import train
from src.models.predict import predict


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [train|predict]")
        print("  train  - Treina o modelo com dados brutos")
        print("  predict - Faz predições com dados de entrada")
        return

    cmd = sys.argv[1]

    if cmd == "train":
        print(f"Iniciando treinamento com dados de: {DEFAULT_RAW_PATH}")
        result = train(
            df=None,
            model_path=DEFAULT_MODEL_PATH,
            metrics_path=DEFAULT_METRICS_PATH,
            processed_path=DEFAULT_PROCESSED_PATH,
        )
        print(f"\n✓ Modelo salvo em: {result['model_path']}")
        print(f"✓ Métricas salvas em: {result['metrics_path']}")
        print(f"✓ Dados processados em: {result['processed_output_path']}")

    elif cmd == "predict":
        print(f"Iniciando predições com dados de: {DEFAULT_PREDICT_INPUT_PATH}")
        output = predict(
            input_path=DEFAULT_PREDICT_INPUT_PATH,
            model_path=DEFAULT_MODEL_PATH,
            output_path=DEFAULT_PREDICTIONS_PATH,
        )
        print(f"✓ Predições salvas em: {output}")

    else:
        print(f"Comando desconhecido: {cmd}")
        print("Use: train | predict")


if __name__ == "__main__":
    main()
