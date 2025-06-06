"""
ML Performance Monitor - Track model performance and drift
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""

    model_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    context: dict[str, Any]


class PerformanceMonitor:
    """Monitor ML model performance over time"""

    def __init__(self):
        self.logger = logging.getLogger("ml.performance_monitor")
        self.metrics: list[PerformanceMetric] = []
        self.baselines: dict[str, float] = {}  # model_id -> baseline metric

    def record_metric(
        self,
        model_id: str,
        metric_name: str,
        metric_value: float,
        context: dict[str, Any] = None,
    ) -> None:
        """Record a performance metric"""
        metric = PerformanceMetric(
            model_id=model_id,
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=datetime.utcnow(),
            context=context or {},
        )

        self.metrics.append(metric)
        self.logger.debug(f"Recorded metric: {model_id}.{metric_name} = {metric_value}")

    def set_baseline(
        self, model_id: str, metric_name: str, baseline_value: float
    ) -> None:
        """Set baseline performance for comparison"""
        key = f"{model_id}.{metric_name}"
        self.baselines[key] = baseline_value
        self.logger.info(f"Set baseline: {key} = {baseline_value}")

    def get_current_performance(
        self, model_id: str, metric_name: str
    ) -> Optional[float]:
        """Get most recent performance metric"""
        recent_metrics = [
            m
            for m in self.metrics
            if m.model_id == model_id and m.metric_name == metric_name
        ]

        if recent_metrics:
            return max(recent_metrics, key=lambda x: x.timestamp).metric_value
        return None

    def detect_drift(
        self, model_id: str, metric_name: str, threshold: float = 0.05
    ) -> bool:
        """Detect performance drift from baseline"""
        current = self.get_current_performance(model_id, metric_name)
        baseline_key = f"{model_id}.{metric_name}"
        baseline = self.baselines.get(baseline_key)

        if current is None or baseline is None:
            return False

        drift_ratio = abs(current - baseline) / baseline
        return drift_ratio > threshold

    def get_performance_trend(
        self, model_id: str, metric_name: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """Get performance trend over time"""
        cutoff = datetime.utcnow() - timedelta(days=days)

        relevant_metrics = [
            m
            for m in self.metrics
            if (
                m.model_id == model_id
                and m.metric_name == metric_name
                and m.timestamp >= cutoff
            )
        ]

        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "value": m.metric_value,
                "context": m.context,
            }
            for m in sorted(relevant_metrics, key=lambda x: x.timestamp)
        ]

    def get_model_summary(self, model_id: str) -> dict[str, Any]:
        """Get comprehensive performance summary for a model"""
        model_metrics = [m for m in self.metrics if m.model_id == model_id]

        if not model_metrics:
            return {"model_id": model_id, "status": "no_data"}

        # Group by metric name
        metrics_by_name = {}
        for metric in model_metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric)

        summary = {
            "model_id": model_id,
            "last_updated": max(
                model_metrics, key=lambda x: x.timestamp
            ).timestamp.isoformat(),
            "metrics": {},
        }

        for metric_name, metric_list in metrics_by_name.items():
            latest = max(metric_list, key=lambda x: x.timestamp)
            baseline_key = f"{model_id}.{metric_name}"
            baseline = self.baselines.get(baseline_key)

            summary["metrics"][metric_name] = {
                "current_value": latest.metric_value,
                "baseline": baseline,
                "drift_detected": self.detect_drift(model_id, metric_name)
                if baseline
                else False,
                "total_measurements": len(metric_list),
            }

        return summary
