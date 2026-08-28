from __future__ import annotations

from app.preprocessing.dataset_builder import build_dataset


def build_experiment(energy, weather, experiment_id, settings):
    return build_dataset(
        energy,
        weather,
        experiment_id,
        train_ratio=settings.train_ratio,
        holdout_date=settings.holdout_date,
        forbidden_after_date=settings.forbidden_after_date,
        max_historical_rows=settings.max_historical_rows,
    )
