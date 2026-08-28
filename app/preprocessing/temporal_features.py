from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["month", "day", "hour", "temperature", "humidity", "month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos"]


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    result["month"] = timestamp.dt.month
    result["day"] = timestamp.dt.day
    result["hour"] = timestamp.dt.hour * 4 + timestamp.dt.minute // 15
    result["month_sin"] = np.sin(2 * np.pi * (timestamp.dt.month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (timestamp.dt.month - 1) / 12)
    result["day_sin"] = np.sin(2 * np.pi * (timestamp.dt.day - 1) / 31)
    result["day_cos"] = np.cos(2 * np.pi * (timestamp.dt.day - 1) / 31)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 96)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 96)
    return result
