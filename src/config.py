"""
Configurações centralizadas do projeto de predição de crédito.
"""

from pathlib import Path

# Caminhos base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" /"predictions"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dados
DEFAULT_RAW_PATH = RAW_DATA_DIR / "Credit.csv"
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_DIR / "credit_processed.csv"
DEFAULT_PREDICT_INPUT_PATH = RAW_DATA_DIR / "credit_to_predict.csv"
DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "credit_predictions.csv"

# Artefatos do modelo
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DEFAULT_FEATURE_ARTIFACT_PATH = ARTIFACTS_DIR / "feature_artifact.joblib"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# Parâmetros do modelo
TEST_SIZE = 0.25
RANDOM_STATE = 42
TARGET_COLUMN = "Limit"
LEAKAGE_COLUMNS = ["Rating", "Balance"]
