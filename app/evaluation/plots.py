from __future__ import annotations

from pathlib import Path
import plotly.graph_objects as go


def create_plots(frame, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart = go.Figure()
    chart.add_scatter(x=frame["timestamp"], y=frame["actual"], name="actual", mode="lines")
    chart.add_scatter(x=frame["timestamp"], y=frame["prediction"], name="prediction", mode="lines")
    chart.update_layout(title="Actual vs prediction", xaxis_title="timestamp", yaxis_title="active power", hovermode="x unified")
    chart.write_html(output_dir / "test_actual_vs_prediction.html", include_plotlyjs="cdn")
    metric_names = ["r2", "mae", "rmse", "mape", "mae_p90", "bias_p90"]
    bars = go.Figure(go.Bar(x=metric_names, y=[float(metrics[name]) for name in metric_names]))
    bars.update_layout(title="Test metrics", xaxis_title="metric", yaxis_title="value")
    bars.write_html(output_dir / "test_metrics.html", include_plotlyjs="cdn")
