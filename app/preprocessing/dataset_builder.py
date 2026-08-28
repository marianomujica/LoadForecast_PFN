from __future__ import annotations

import logging
from dataclasses import dataclass
import pandas as pd
from .temporal_features import BASE_FEATURES, add_temporal_features

LOGGER = logging.getLogger(__name__)

EXPERIMENT_FEATURE_COUNTS = {1: 107, 2: 15, 3: 203, 4: 19, 5: 299, 6: 23}
VECTOR_EXPERIMENTS = {1, 3, 5}
STATISTICS_EXPERIMENTS = {2, 4, 6}


@dataclass(frozen=True)
class DatasetResult:
    data: pd.DataFrame
    feature_names: list[str]
    train: pd.DataFrame
    test: pd.DataFrame
    discarded_days: dict[str, str]


def _complete_days(power: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = power.copy()
    frame["date"] = frame["timestamp"].dt.normalize()
    counts = frame.groupby("date").size()
    invalid = counts[counts != 96]
    discarded = {str(day.date()): f"expected 96 rows, found {int(count)}" for day, count in invalid.items()}
    valid_dates = counts[counts == 96].index
    return frame[frame["date"].isin(valid_dates)].copy(), discarded


def _day_values(indexed: pd.Series, day: pd.Timestamp, days_back: int) -> list[float] | None:
    values = indexed.get(day - pd.Timedelta(days=days_back))
    if values is None:
        return None
    values = values.sort_index()
    if len(values) != 96:
        return None
    return values.to_list()


def build_dataset(
    energy: pd.DataFrame,
    weather: pd.DataFrame,
    experiment_id: int,
    train_ratio: float = 0.8,
    holdout_date: pd.Timestamp | str = "2026-04-19",
    forbidden_after_date: pd.Timestamp | str = "2026-04-19",
    max_historical_rows: int | None = None,
) -> DatasetResult:
    if experiment_id not in EXPERIMENT_FEATURE_COUNTS:
        raise ValueError(f"experiment_id must be 1..6, got {experiment_id}")
    demand = energy.copy()
    demand["timestamp"] = pd.to_datetime(demand["timestamp"])
    demand = demand.sort_values("timestamp").dropna(subset=["active_power"])
    holdout_day = pd.Timestamp(holdout_date).normalize()
    forbidden_day = pd.Timestamp(forbidden_after_date).normalize()
    demand = demand[
        (demand["timestamp"].dt.normalize() != holdout_day)
        & (demand["timestamp"].dt.normalize() <= forbidden_day)
    ]
    demand, discarded = _complete_days(demand)
    merged = demand.merge(weather, on="timestamp", how="left", validate="one_to_one")
    merged = add_temporal_features(merged).sort_values("timestamp").set_index("timestamp")
    max_back = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}[experiment_id]
    data = merged[BASE_FEATURES + ["active_power"]].rename(columns={"active_power": "target"}).copy()
    valid = pd.Series(True, index=data.index)
    power = merged["active_power"]
    day_index = data.index.normalize()
    daily_values = power.to_frame("value")
    daily_values["day"] = daily_values.index.normalize()
    daily_values["slot"] = daily_values.index.hour * 4 + daily_values.index.minute // 15
    daily_values = daily_values.pivot(index="day", columns="slot", values="value").reindex(columns=range(96))
    historical_columns: dict[str, pd.Series] = {}
    for back in range(1, max_back + 1):
        source_days = day_index - pd.Timedelta(days=back)
        if experiment_id in VECTOR_EXPERIMENTS:
            for slot in range(96):
                values = daily_values[slot].reindex(source_days)
                values.index = data.index
                historical_columns[f"power_D-{back}_{slot:02d}"] = values
            valid &= pd.DataFrame(
                {name: historical_columns[name] for name in historical_columns if name.startswith(f"power_D-{back}_")},
                index=data.index,
            ).notna().all(axis=1)
        else:
            daily_stats = power.groupby(power.index.normalize()).agg(["mean", "std", "max", "min"])
            stats = daily_stats.reindex(source_days)
            stats.index = data.index
            valid &= stats.notna().all(axis=1)
            for stat in ("mean", "std", "max", "min"):
                historical_columns[f"{stat}_D-{back}"] = stats[stat]
    if historical_columns:
        data = pd.concat([data, pd.DataFrame(historical_columns, index=data.index)], axis=1)
    data = data.loc[valid].reset_index(names="timestamp")
    if max_historical_rows is not None:
        if max_historical_rows < 2:
            raise ValueError("max_historical_rows must be at least 2")
        data = data.tail(max_historical_rows).reset_index(drop=True)
    feature_names = [column for column in data.columns if column not in ("timestamp", "target")]
    if len(feature_names) != EXPERIMENT_FEATURE_COUNTS[experiment_id]:
        raise ValueError(f"Experiment {experiment_id} generated {len(feature_names)} features; expected {EXPERIMENT_FEATURE_COUNTS[experiment_id]}")
    split = int(len(data) * train_ratio)
    return DatasetResult(data, feature_names, data.iloc[:split].copy(), data.iloc[split:].copy(), discarded)
