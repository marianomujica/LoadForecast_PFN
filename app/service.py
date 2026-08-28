from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
import importlib.metadata
import pandas as pd

from .experiments.registry import build_experiment
from .evaluation.metrics import calculate_metrics
from .evaluation.plots import create_plots
from .models.model_io import load_bundle, save_bundle
from .models.tabpfn_model import create_regressor, version as tabpfn_version
from .preprocessing.energy_loader import load_energy
from .preprocessing.weather_loader import load_weather

LOGGER = logging.getLogger(__name__)


def prepare_all(settings):
    energy = load_energy(settings.data_energy_path, settings.start, settings.end)
    weather = load_weather(settings.data_smn_path, settings.station, settings.start, settings.end)
    prepared = {}
    for experiment_id in range(1, 7):
        result = build_experiment(energy, weather, experiment_id, settings)
        path = settings.output_path / f"experiment_{experiment_id}" / f"dataset_experiment_{experiment_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_csv(path, index=False)
        prepared[experiment_id] = {"rows": len(result.data), "features": len(result.feature_names), "path": str(path), "discarded_days": result.discarded_days}
    settings.output_path.mkdir(parents=True, exist_ok=True)
    (settings.output_path / "prepare_summary.json").write_text(json.dumps(prepared, indent=2, default=str), encoding="utf-8")
    return prepared


def train_one(settings, experiment_id: int):
    energy = load_energy(settings.data_energy_path, settings.start, settings.end)
    weather = load_weather(settings.data_smn_path, settings.station, settings.start, settings.end)
    result = build_experiment(energy, weather, experiment_id, settings)
    output_dir = settings.output_path / f"experiment_{experiment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"dataset_experiment_{experiment_id}.csv"
    result.data.to_csv(dataset_path, index=False)
    model = create_regressor(settings)
    started = time.perf_counter()
    model.fit(result.train[result.feature_names], result.train["target"])
    elapsed = time.perf_counter() - started
    prediction = model.predict(result.test[result.feature_names])
    predictions = pd.DataFrame({"timestamp": result.test["timestamp"], "prediction": prediction, "actual": result.test["target"]})
    metrics = calculate_metrics(predictions["actual"], predictions["prediction"])
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "metrics.txt").write_text("\n".join(f"{key}: {value}" for key, value in metrics.items()), encoding="utf-8")
    metadata = {"experiment_id": experiment_id, "feature_names": result.feature_names, "feature_count": len(result.feature_names), "train_rows": len(result.train), "test_rows": len(result.test), "max_historical_rows": settings.max_historical_rows, "elapsed_seconds": elapsed, "python": sys.version, "platform": platform.platform(), "tabpfn": tabpfn_version(), "pandas": pd.__version__, "numpy": importlib.metadata.version("numpy"), "sklearn": importlib.metadata.version("scikit-learn"), "device": settings.device, "ignore_pretraining_limits": settings.ignore_pretraining_limits, "data_start": str(settings.start), "data_end": str(settings.end), "holdout_date": settings.holdout_date, "discarded_days": result.discarded_days}
    save_bundle(model, output_dir / f"model_experiment_{experiment_id}", metadata)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    create_plots(predictions, metrics, output_dir)
    return {"experiment_id": experiment_id, "metrics": metrics, "rows": len(result.data), "features": len(result.feature_names), "model": str(output_dir / f"model_experiment_{experiment_id}.tabpfn_fit")}
