"""
Storage Monitoring and Metrics

Performance monitoring, metrics collection, and alerting for the storage system.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point"""

    timestamp: datetime
    value: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for storage operations"""

    operation_count: int = 0
    total_duration_ms: int = 0
    min_duration_ms: int = float("inf")
    max_duration_ms: int = 0
    error_count: int = 0
    success_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration"""
        if self.operation_count == 0:
            return 0.0
        return self.total_duration_ms / self.operation_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.operation_count == 0:
            return 0.0
        return self.success_count / self.operation_count

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.operation_count == 0:
            return 0.0
        return self.error_count / self.operation_count


class StorageMetrics:
    """Comprehensive storage metrics collection and monitoring"""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.logger = logging.getLogger(__name__)

        # Thread-safe metrics storage
        self._lock = threading.RLock()

        # Metrics storage
        self._metrics: dict[str, deque] = defaultdict(lambda: deque())
        self._performance_metrics: dict[str, PerformanceMetrics] = defaultdict(
            PerformanceMetrics
        )

        # Current gauges
        self._gauges: dict[str, float] = {}

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_metrics, daemon=True
        )
        self._cleanup_thread.start()

    def record_upload(
        self,
        data_type: str,
        file_size: int,
        duration_ms: int,
        success: bool,
        bucket: str = "primary",
    ) -> None:
        """Record file upload metrics"""
        with self._lock:
            timestamp = datetime.now()

            # Record basic metrics
            self._add_metric(
                "upload_count",
                1,
                {"data_type": data_type, "bucket": bucket, "success": str(success)},
            )

            self._add_metric(
                "upload_size_bytes",
                file_size,
                {"data_type": data_type, "bucket": bucket},
            )

            self._add_metric(
                "upload_duration_ms",
                duration_ms,
                {"data_type": data_type, "bucket": bucket},
            )

            # Update performance metrics
            perf_key = f"upload_{data_type}_{bucket}"
            perf = self._performance_metrics[perf_key]

            perf.operation_count += 1
            perf.total_duration_ms += duration_ms
            perf.min_duration_ms = min(perf.min_duration_ms, duration_ms)
            perf.max_duration_ms = max(perf.max_duration_ms, duration_ms)

            if success:
                perf.success_count += 1
            else:
                perf.error_count += 1

            self.logger.debug(
                f"Recorded upload metrics: {data_type}, {file_size} bytes, {duration_ms}ms, success={success}"
            )

    def record_query(
        self,
        data_type: str,
        duration_ms: int,
        records_returned: int,
        cache_hit: bool,
        partitions_scanned: int = 0,
    ) -> None:
        """Record query performance metrics"""
        with self._lock:
            timestamp = datetime.now()

            # Record query metrics
            self._add_metric(
                "query_count", 1, {"data_type": data_type, "cache_hit": str(cache_hit)}
            )

            self._add_metric(
                "query_duration_ms",
                duration_ms,
                {"data_type": data_type, "cache_hit": str(cache_hit)},
            )

            self._add_metric(
                "query_records_returned", records_returned, {"data_type": data_type}
            )

            if partitions_scanned > 0:
                self._add_metric(
                    "query_partitions_scanned",
                    partitions_scanned,
                    {"data_type": data_type},
                )

            # Update performance metrics
            perf_key = f"query_{data_type}"
            perf = self._performance_metrics[perf_key]

            perf.operation_count += 1
            perf.total_duration_ms += duration_ms
            perf.min_duration_ms = min(perf.min_duration_ms, duration_ms)
            perf.max_duration_ms = max(perf.max_duration_ms, duration_ms)
            perf.success_count += 1  # Assume success if we're recording metrics

            self.logger.debug(
                f"Recorded query metrics: {data_type}, {duration_ms}ms, {records_returned} records, cache_hit={cache_hit}"
            )

    def record_error(
        self, operation: str, error_type: str, data_type: str = None
    ) -> None:
        """Record error metrics"""
        with self._lock:
            tags = {"operation": operation, "error_type": error_type}

            if data_type:
                tags["data_type"] = data_type

            self._add_metric("error_count", 1, tags)

            self.logger.warning(f"Recorded error: {operation} - {error_type}")

    def set_gauge(
        self, metric_name: str, value: float, tags: dict[str, str] = None
    ) -> None:
        """Set a gauge metric value"""
        with self._lock:
            gauge_key = self._build_metric_key(metric_name, tags or {})
            self._gauges[gauge_key] = value

            self.logger.debug(f"Set gauge {metric_name} = {value}")

    def increment_counter(
        self, metric_name: str, value: float = 1.0, tags: dict[str, str] = None
    ) -> None:
        """Increment a counter metric"""
        with self._lock:
            self._add_metric(metric_name, value, tags or {})

    def record_timer(
        self, metric_name: str, duration_ms: int, tags: dict[str, str] = None
    ) -> None:
        """Record a timer metric"""
        with self._lock:
            self._add_metric(metric_name, duration_ms, tags or {})

    def get_performance_summary(
        self, operation_type: str = None
    ) -> dict[str, PerformanceMetrics]:
        """Get performance metrics summary"""
        with self._lock:
            if operation_type:
                return {
                    k: v
                    for k, v in self._performance_metrics.items()
                    if k.startswith(operation_type)
                }
            return dict(self._performance_metrics)

    def get_metrics(
        self, metric_name: str, since: datetime = None
    ) -> list[MetricPoint]:
        """Get metrics for a specific metric name"""
        with self._lock:
            if metric_name not in self._metrics:
                return []

            metrics = list(self._metrics[metric_name])

            if since:
                metrics = [m for m in metrics if m.timestamp >= since]

            return metrics

    def get_metric_summary(
        self, metric_name: str, since: datetime = None
    ) -> dict[str, Any]:
        """Get summary statistics for a metric"""
        metrics = self.get_metrics(metric_name, since)

        if not metrics:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else 0,
        }

    def get_cache_metrics(self) -> dict[str, float]:
        """Get cache performance metrics"""
        with self._lock:
            # Calculate cache hit rate from recent queries
            recent_time = datetime.now() - timedelta(hours=1)

            cache_hits = len(
                [
                    m
                    for m in self._metrics.get("query_count", [])
                    if m.timestamp >= recent_time and m.tags.get("cache_hit") == "True"
                ]
            )

            cache_misses = len(
                [
                    m
                    for m in self._metrics.get("query_count", [])
                    if m.timestamp >= recent_time and m.tags.get("cache_hit") == "False"
                ]
            )

            total_queries = cache_hits + cache_misses

            return {
                "cache_hit_rate": cache_hits / total_queries
                if total_queries > 0
                else 0.0,
                "cache_miss_rate": cache_misses / total_queries
                if total_queries > 0
                else 0.0,
                "total_queries_1h": total_queries,
                "cache_hits_1h": cache_hits,
                "cache_misses_1h": cache_misses,
            }

    def get_throughput_metrics(self, data_type: str = None) -> dict[str, float]:
        """Get throughput metrics"""
        with self._lock:
            recent_time = datetime.now() - timedelta(hours=1)

            # Filter metrics by data type if specified
            upload_metrics = self._metrics.get("upload_count", [])
            query_metrics = self._metrics.get("query_count", [])

            if data_type:
                upload_metrics = [
                    m for m in upload_metrics if m.tags.get("data_type") == data_type
                ]
                query_metrics = [
                    m for m in query_metrics if m.tags.get("data_type") == data_type
                ]

            # Count recent operations
            recent_uploads = len(
                [m for m in upload_metrics if m.timestamp >= recent_time]
            )
            recent_queries = len(
                [m for m in query_metrics if m.timestamp >= recent_time]
            )

            return {
                "uploads_per_hour": recent_uploads,
                "queries_per_hour": recent_queries,
                "total_operations_per_hour": recent_uploads + recent_queries,
            }

    def _add_metric(self, metric_name: str, value: float, tags: dict[str, str]) -> None:
        """Add a metric point (internal method)"""
        metric_point = MetricPoint(timestamp=datetime.now(), value=value, tags=tags)

        self._metrics[metric_name].append(metric_point)

        # Limit memory usage by keeping only recent metrics
        max_points = 10000  # Adjust based on memory constraints
        if len(self._metrics[metric_name]) > max_points:
            self._metrics[metric_name].popleft()

    def _build_metric_key(self, metric_name: str, tags: dict[str, str]) -> str:
        """Build a unique key for a metric with tags"""
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_str}]" if tag_str else metric_name

    def _cleanup_old_metrics(self) -> None:
        """Background thread to clean up old metrics"""
        while True:
            try:
                time.sleep(3600)  # Run every hour

                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                with self._lock:
                    for metric_name, metric_deque in self._metrics.items():
                        # Remove old metrics
                        while metric_deque and metric_deque[0].timestamp < cutoff_time:
                            metric_deque.popleft()

                self.logger.debug("Cleaned up old metrics")

            except Exception as e:
                self.logger.error(f"Error cleaning up metrics: {str(e)}")

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        with self._lock:
            if format == "json":
                import json

                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "performance_metrics": {
                        k: {
                            "operation_count": v.operation_count,
                            "avg_duration_ms": v.avg_duration_ms,
                            "success_rate": v.success_rate,
                            "error_rate": v.error_rate,
                        }
                        for k, v in self._performance_metrics.items()
                    },
                    "cache_metrics": self.get_cache_metrics(),
                    "throughput_metrics": self.get_throughput_metrics(),
                    "gauges": self._gauges,
                }

                return json.dumps(export_data, indent=2)

            else:
                raise ValueError(f"Unsupported export format: {format}")


# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations"""

    def __init__(
        self, metrics: StorageMetrics, metric_name: str, tags: dict[str, str] = None
    ):
        self.metrics = metrics
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = int((time.time() - self.start_time) * 1000)
            self.metrics.record_timer(self.metric_name, duration_ms, self.tags)
