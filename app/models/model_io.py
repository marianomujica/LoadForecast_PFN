from __future__ import annotations

import json
from pathlib import Path
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model


def save_bundle(model, path: Path, metadata: dict) -> None:
    save_fitted_tabpfn_model(model, path.with_suffix(".tabpfn_fit"))
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_bundle(path: Path):
    return load_fitted_tabpfn_model(path.with_suffix(".tabpfn_fit"), device="cpu")
