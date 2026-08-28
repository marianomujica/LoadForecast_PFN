from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_weather(directory: str | Path, station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = sorted(Path(directory).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No weather TXT files found in {directory}")
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_fwf(path, skiprows=[1], encoding="latin-1")
            frame.columns = [str(c).strip() for c in frame.columns]
            required = {"FECHA", "HORA", "TEMP", "HUM", "NOMBRE"}
            if not required.issubset(frame.columns):
                LOGGER.warning("Skipping weather file %s: columns=%s", path.name, list(frame.columns))
                continue
            frame["station"] = frame["NOMBRE"].astype(str).str.strip()
            frame = frame[frame["station"].eq(station)].copy()
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["FECHA"].astype(str).str.replace(r"\.0$", "", regex=True) + " " + frame["HORA"].astype(str).str.replace(r"\.0$", "", regex=True), format="%d%m%Y %H", errors="coerce")
            frame["temperature"] = pd.to_numeric(frame["TEMP"], errors="coerce")
            frame["humidity"] = pd.to_numeric(frame["HUM"], errors="coerce")
            frames.append(frame[["timestamp", "temperature", "humidity"]])
        except Exception as exc:
            LOGGER.warning("Could not process weather file %s: %s", path.name, exc)
    if not frames:
        raise ValueError(f"No weather observations found for station {station!r}")
    hourly = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    hourly = hourly.loc[(hourly.index >= start.floor("h")) & (hourly.index <= end.ceil("h"))]
    quarter = hourly.resample("15min").interpolate("time").ffill().bfill()
    return quarter.loc[(quarter.index >= start) & (quarter.index <= end)].reset_index()
