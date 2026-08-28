from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass(frozen=True)
class Settings:
    data_energy_path: Path
    data_smn_path: Path
    output_path: Path
    data_start_date: str
    data_end_date: str
    holdout_date: str
    forbidden_after_date: str
    station: str
    frequency: str
    train_ratio: float
    test_ratio: float
    device: str
    random_state: int
    n_estimators: int
    ignore_pretraining_limits: bool
    fit_mode: str
    n_preprocessing_jobs: int
    prediction_batch_size: int
    max_historical_rows: int | None
    experiment_id: int

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def start(self):
        import pandas as pd
        return pd.Timestamp(self.data_start_date)

    @property
    def end(self):
        import pandas as pd
        return pd.Timestamp(self.data_end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

    @property
    def holdout_start(self):
        import pandas as pd
        return pd.Timestamp(self.holdout_date)


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("JIT_CONFIG", Path(__file__).resolve().parents[1] / "config" / "experiment_config.txt"))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = config_path.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(os.path.expandvars(value))
        return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    raw["data_energy_path"] = resolve(raw["data_energy_path"])
    raw["data_smn_path"] = resolve(raw["data_smn_path"])
    raw["output_path"] = resolve(raw["output_path"])
    if abs(float(raw["train_ratio"]) + float(raw["test_ratio"]) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO and TEST_RATIO must sum to 1.0")
    if raw["device"].lower() != "cpu":
        raise ValueError("This project is CPU-only; DEVICE must be cpu")
    if raw["frequency"] != "15min":
        raise ValueError("This project requires FREQUENCY=15min")
    if raw["data_end_date"] >= raw["holdout_date"]:
        raise ValueError("DATA_END_DATE must be earlier than HOLDOUT_DATE")
    if raw["experiment_id"] not in range(1, 7):
        raise ValueError("EXPERIMENT_ID must be between 1 and 6")
    return Settings(**raw)
