from __future__ import annotations

import importlib.metadata
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def create_regressor(settings):
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3,
        device=settings.device,
        n_estimators=settings.n_estimators,
        random_state=settings.random_state,
        ignore_pretraining_limits=settings.ignore_pretraining_limits,
        fit_mode=settings.fit_mode,
        memory_saving_mode=True,
        n_preprocessing_jobs=settings.n_preprocessing_jobs,
        show_progress_bar=True,
    )


def version() -> str:
    return importlib.metadata.version("tabpfn")
