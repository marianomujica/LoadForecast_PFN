from __future__ import annotations

import logging
import re
from pathlib import Path
import unicodedata
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _repair_text(value: object) -> str:
    text = str(value).strip().replace("\ufeff", "")
    for _ in range(2):
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == text:
            break
        text = repaired
    return text


def normalize_column_name(value: object) -> str:
    text = _repair_text(value)
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if "fecha" in compact and ("hora" in compact or "fecha" == compact):
        return "timestamp"
    if "potencia" in compact and "activa" in compact:
        return "active_power"
    if "energia" in compact and "reactiva" in compact:
        return "reactive_energy"
    if "energia" in compact and "activa" in compact:
        return "active_energy"
    return text.strip()


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read energy file {path}: {last_error}")


def load_energy(directory: str | Path, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No energy CSV files found in {directory}")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = _read_csv(path)
        original = list(frame.columns)
        frame.columns = [normalize_column_name(c) for c in frame.columns]
        LOGGER.info("Energy columns %s: %s -> %s", path.name, original, list(frame.columns))
        if "timestamp" not in frame.columns or "active_power" not in frame.columns:
            raise ValueError(f"Could not identify timestamp and active power in {path.name}: {original}")
        frames.append(frame[["timestamp", "active_power"]].copy())
    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], dayfirst=True, errors="coerce")
    data["active_power"] = pd.to_numeric(data["active_power"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if start is not None:
        data = data[data["timestamp"] >= start]
    if end is not None:
        data = data[data["timestamp"] <= end]
    if data.empty:
        raise ValueError("Energy data is empty after date filtering")
    LOGGER.info("Loaded %d energy rows from %s to %s", len(data), data.timestamp.min(), data.timestamp.max())
    return data.reset_index(drop=True)
