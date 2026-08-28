from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from app.preprocessing.dataset_builder import EXPERIMENT_FEATURE_COUNTS, build_dataset


class DatasetBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range("2026-04-01", "2026-04-20 23:45", freq="15min")
        energy = pd.DataFrame({"timestamp": timestamps, "active_power": np.arange(len(timestamps), dtype=float)})
        weather = pd.DataFrame({"timestamp": timestamps, "temperature": 20.0, "humidity": 60.0})
        cls.energy = energy
        cls.weather = weather

    def test_dimensions_and_columns(self):
        for experiment_id, expected_count in EXPERIMENT_FEATURE_COUNTS.items():
            result = build_dataset(self.energy, self.weather, experiment_id)
            self.assertEqual(len(result.feature_names), expected_count)
            self.assertEqual(len(result.data.columns), expected_count + 2)
            self.assertEqual(len(result.feature_names), len(set(result.feature_names)))
            self.assertNotIn("timestamp", result.feature_names)
            self.assertNotIn("target", result.feature_names)

    def test_holdout_and_forbidden_dates_are_absent_from_training_data(self):
        result = build_dataset(self.energy, self.weather, 1)
        dates = result.data["timestamp"].dt.date
        self.assertNotIn(pd.Timestamp("2026-04-19").date(), dates.tolist())
        self.assertNotIn(pd.Timestamp("2026-04-20").date(), dates.tolist())

    def test_temporal_split_has_no_shuffle(self):
        result = build_dataset(self.energy, self.weather, 1)
        self.assertLess(result.train["timestamp"].max(), result.test["timestamp"].min())
        self.assertEqual(result.data["timestamp"].diff().dropna().dt.total_seconds().unique().tolist(), [900.0])

    def test_experiment_one_lag_is_previous_day(self):
        result = build_dataset(self.energy, self.weather, 1)
        row = result.data.loc[result.data["timestamp"].eq(pd.Timestamp("2026-04-10 10:15"))].iloc[0]
        expected = self.energy.loc[self.energy["timestamp"].dt.date.eq(pd.Timestamp("2026-04-09").date()), "active_power"].tolist()
        actual = [row[f"power_D-1_{slot:02d}"] for slot in range(96)]
        self.assertEqual(actual, expected)
        day_rows = result.data[result.data["timestamp"].dt.normalize().eq(pd.Timestamp("2026-04-10"))]
        self.assertEqual(day_rows.shape[0], 96)
        self.assertTrue(day_rows[[f"power_D-1_{slot:02d}" for slot in range(96)]].notna().all().all())
        self.assertEqual(day_rows[f"power_D-1_00"].nunique(), 1)

    def test_experiment_three_lags_keep_day_order(self):
        result = build_dataset(self.energy, self.weather, 5)
        row = result.data.loc[result.data["timestamp"].eq(pd.Timestamp("2026-04-10 10:15"))].iloc[0]
        for back in range(1, 4):
            expected = self.energy.loc[self.energy["timestamp"].dt.normalize().eq(pd.Timestamp("2026-04-10") - pd.Timedelta(days=back)), "active_power"].tolist()
            actual = [row[f"power_D-{back}_{slot:02d}"] for slot in range(96)]
            self.assertEqual(actual, expected)
        self.assertTrue(row[[f"power_D-{back}_{slot:02d}" for back in range(1, 4) for slot in range(96)]].notna().all())

    def test_experiment_three_uses_two_days(self):
        result = build_dataset(self.energy, self.weather, 3)
        self.assertEqual(len(result.feature_names), 203)
        self.assertTrue(result.data.filter(regex=r"^power_D-[12]_").notna().all().all())

    def test_experiment_six_uses_three_day_statistics(self):
        result = build_dataset(self.energy, self.weather, 6)
        self.assertEqual(len(result.feature_names), 23)
        self.assertEqual(set(result.data.filter(regex=r"^(mean|std|max|min)_D-[123]$").columns), set(result.feature_names[11:]))

    def test_statistics_are_correct(self):
        result = build_dataset(self.energy, self.weather, 2)
        row = result.data.loc[result.data["timestamp"].eq(pd.Timestamp("2026-04-10 10:15"))].iloc[0]
        values = self.energy.loc[self.energy["timestamp"].dt.normalize().eq(pd.Timestamp("2026-04-09")), "active_power"]
        self.assertAlmostEqual(row["mean_D-1"], values.mean())
        self.assertAlmostEqual(row["std_D-1"], values.std())
        self.assertAlmostEqual(row["max_D-1"], values.max())
        self.assertAlmostEqual(row["min_D-1"], values.min())

    def test_incomplete_day_is_reported_and_examples_using_it_removed(self):
        energy = self.energy.drop(index=self.energy.index[self.energy["timestamp"].eq(pd.Timestamp("2026-04-05 00:00"))])
        result = build_dataset(energy, self.weather, 1)
        self.assertIn("2026-04-05", result.discarded_days)
        self.assertNotIn(pd.Timestamp("2026-04-06").date(), result.data["timestamp"].dt.date.tolist())


if __name__ == "__main__":
    unittest.main()
