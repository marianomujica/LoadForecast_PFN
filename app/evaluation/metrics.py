from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, prediction) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    nonzero = actual != 0
    p90 = float(np.percentile(actual, 90))
    p90_mask = actual >= p90
    return {
        "r2": float(r2_score(actual, prediction)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mape": float(np.mean(np.abs((actual[nonzero] - prediction[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
        "mape_zero_excluded": int((~nonzero).sum()),
        "p90": p90,
        "mae_p90": float(mean_absolute_error(actual[p90_mask], prediction[p90_mask])),
        "bias_p90": float(np.mean(error[p90_mask])),
    }
