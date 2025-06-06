"""
Base ML Manager - Abstract interface for all ML services
Integrates with Phase 1 (storage) and Phase 2 (analytics)
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from buffetbot.analytics.bigquery.manager import BigQueryAnalyticsManager
from buffetbot.storage.schemas.manager import SchemaManager

# Phase 1 & 2 Integration
from buffetbot.utils.cache import Cache


@dataclass
class MLServiceConfig:
    """Configuration for ML services"""

    service_name: str
    cost_per_hour: float = 0.0
    max_cost_per_day: float = 100.0
    enable_cost_monitoring: bool = True
    cache_predictions: bool = True
    cache_ttl_seconds: int = 3600


class BaseMLManager(ABC):
    """Abstract base class for all ML service managers"""

    def __init__(self, config: MLServiceConfig):
        self.config = config
        self.logger = logging.getLogger(f"ml.{config.service_name}")

        # Phase 1 & 2 Integration
        self.cache = Cache(cache_type="memory")
        self.schema_manager = SchemaManager()

        # BigQuery integration - optional for ML foundation
        self.bigquery_manager = None
        try:
            # Try to initialize with environment variables or defaults
            import os

            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "buffetbot-demo")
            self.bigquery_manager = BigQueryAnalyticsManager(project_id)
        except Exception as e:
            self.logger.warning(f"BigQuery integration unavailable: {e}")

        # Cost tracking
        self.total_cost = 0.0
        self.session_start = datetime.utcnow()

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the ML service"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass

    def track_cost(self, operation_cost: float) -> None:
        """Track ML operation costs"""
        self.total_cost += operation_cost
        self.logger.info(
            f"ML Cost: +${operation_cost:.4f}, Total: ${self.total_cost:.4f}"
        )

        # Cost limit check
        if self.total_cost > self.config.max_cost_per_day:
            self.logger.error(f"Daily cost limit exceeded: ${self.total_cost:.2f}")
            raise Exception(f"ML cost limit exceeded: ${self.total_cost:.2f}")

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary for this session"""
        duration = (datetime.utcnow() - self.session_start).total_seconds() / 3600
        return {
            "service": self.config.service_name,
            "total_cost": self.total_cost,
            "session_duration_hours": duration,
            "cost_per_hour": self.total_cost / max(duration, 0.001),
            "cost_limit": self.config.max_cost_per_day,
        }
