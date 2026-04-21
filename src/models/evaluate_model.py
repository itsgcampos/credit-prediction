from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_to_frame(train_metrics: dict[str, float], test_metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "train", **train_metrics},
            {"dataset": "test", **test_metrics},
        ]
    )


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return output
