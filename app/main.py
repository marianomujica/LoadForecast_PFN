from __future__ import annotations

import json
from pathlib import Path
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File

from .config import load_settings
from .service import prepare_all, train_one
from .models.model_io import load_bundle
from .models.tabpfn_model import version as tabpfn_version
from .preprocessing.energy_loader import load_energy
from .preprocessing.weather_loader import load_weather
from .experiments.registry import build_experiment

app = FastAPI(title="JIT 2026 TabPFN-3 Demand API", version="1.0.0")


def _settings():
    return load_settings()


@app.get("/health")
def health():
    settings = _settings()
    return {"status": "ok", "device": settings.device, "tabpfn_version": tabpfn_version(), "project": "JIT_2026"}


@app.post("/prepare-data")
def prepare_data():
    try:
        return prepare_all(_settings())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/{experiment_id}")
def train(experiment_id: int):
    if experiment_id not in range(1, 7):
        raise HTTPException(status_code=400, detail="experiment_id must be between 1 and 6")
    try:
        return train_one(_settings(), experiment_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train-all")
def train_all():
    try:
        return [train_one(_settings(), experiment_id) for experiment_id in range(1, 7)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/{experiment_id}")
def predict(experiment_id: int, demand_file: UploadFile | None = File(default=None), weather_file: UploadFile | None = File(default=None)):
    settings = _settings()
    model_path = settings.output_path / f"experiment_{experiment_id}" / f"model_experiment_{experiment_id}"
    dataset_path = model_path.parent / f"dataset_experiment_{experiment_id}.csv"
    if not model_path.with_suffix(".tabpfn_fit").exists() or not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Model and prepared dataset are required")
    import pandas as pd
    model = load_bundle(model_path)
    metadata = json.loads(model_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    if demand_file is not None and weather_file is not None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            (temporary_path / "demand.csv").write_bytes(demand_file.file.read())
            (temporary_path / "weather.txt").write_bytes(weather_file.file.read())
            energy = load_energy(temporary_path)
            weather = load_weather(temporary_path, settings.station, energy["timestamp"].min(), energy["timestamp"].max())
            prepared = build_experiment(energy, weather, experiment_id, settings).data
        data = prepared
    elif demand_file is not None or weather_file is not None:
        raise HTTPException(status_code=400, detail="demand_file and weather_file must be provided together")
    else:
        data = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    prediction = model.predict(data[feature_names])
    return {"experiment_id": experiment_id, "predictions": [{"timestamp": str(ts), "prediction": float(value)} for ts, value in zip(data["timestamp"], prediction, strict=True)]}


@app.get("/metrics/{experiment_id}")
def metrics(experiment_id: int):
    path = _settings().output_path / f"experiment_{experiment_id}" / "metrics.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    return {"experiment_id": experiment_id, "metrics": dict(line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines())}
