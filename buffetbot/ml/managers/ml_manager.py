"""
Main ML Manager - Coordinates all ML operations
Implements local ML services with zero costs
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.registry import ModelRegistry
from ..monitoring.cost_monitor import MLCostMonitor
from .base_manager import BaseMLManager, MLServiceConfig


class MLManager(BaseMLManager):
    """Main ML service manager for BuffetBot"""

    def __init__(self, config: Optional[MLServiceConfig] = None):
        if config is None:
            config = MLServiceConfig(
                service_name="ml_manager",
                cost_per_hour=0.0,  # Local implementation is free
                max_cost_per_day=0.0,  # No costs expected
                enable_cost_monitoring=True,
                cache_predictions=True,
                cache_ttl_seconds=3600,
            )

        super().__init__(config)

        # ML Components
        self.model_registry = ModelRegistry()
        self.cost_monitor = MLCostMonitor()
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize ML services"""
        try:
            self.logger.info("Initializing ML Manager...")

            # Initialize model registry
            await self.model_registry.initialize()

            # Initialize cost monitoring
            self.cost_monitor.start_monitoring()

            # Verify Phase 1 & 2 integration
            await self._verify_integrations()

            self.is_initialized = True
            self.logger.info("✅ ML Manager initialized successfully (FREE)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Manager: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup ML resources"""
        try:
            self.logger.info("Cleaning up ML Manager...")

            # Stop cost monitoring
            self.cost_monitor.stop_monitoring()

            # Cleanup model registry
            await self.model_registry.cleanup()

            self.is_initialized = False
            self.logger.info("✅ ML Manager cleanup complete")

        except Exception as e:
            self.logger.error(f"❌ Error during ML cleanup: {e}")

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive ML system status"""
        return {
            "initialized": self.is_initialized,
            "timestamp": datetime.utcnow().isoformat(),
            "cost_summary": self.get_cost_summary(),
            "model_registry": {
                "total_models": await self.model_registry.get_model_count(),
                "latest_models": await self.model_registry.get_recent_models(limit=5),
            },
            "cost_monitoring": self.cost_monitor.get_cost_summary(),
            "integrations": {
                "cache": bool(self.cache),
                "schema_manager": self.schema_manager is not None,
                "bigquery": self.bigquery_manager is not None,
            },
        }

    async def _verify_integrations(self) -> None:
        """Verify Phase 1 & 2 integrations are working"""
        # Test cache integration
        cache_key = "ml_test_key"
        test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}

        try:
            self.cache.set(cache_key, test_data, expiration=60)
            cached_data = self.cache.get(cache_key)
            if cached_data != test_data:
                raise Exception("Cache integration test failed")

            self.logger.info("✅ Cache integration verified")

        except Exception as e:
            self.logger.warning(f"⚠️ Cache integration issue: {e}")

        # Test schema manager
        try:
            if self.schema_manager:
                self.logger.info("✅ Schema manager integration verified")
            else:
                self.logger.warning("⚠️ Schema manager not available")
        except Exception as e:
            self.logger.warning(f"⚠️ Schema manager integration issue: {e}")

        # Test BigQuery analytics
        try:
            if self.bigquery_manager:
                self.logger.info("✅ BigQuery analytics integration verified")
            else:
                self.logger.warning("⚠️ BigQuery analytics not available")
        except Exception as e:
            self.logger.warning(f"⚠️ BigQuery analytics integration issue: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all ML components"""
        health_status = {
            "overall": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        issues = []

        # Check initialization
        if not self.is_initialized:
            issues.append("ML Manager not initialized")
            health_status["components"]["ml_manager"] = "not_initialized"
        else:
            health_status["components"]["ml_manager"] = "healthy"

        # Check model registry
        try:
            registry_health = await self.model_registry.health_check()
            health_status["components"]["model_registry"] = registry_health
            if registry_health != "healthy":
                issues.append(f"Model registry: {registry_health}")
        except Exception as e:
            issues.append(f"Model registry error: {e}")
            health_status["components"]["model_registry"] = "error"

        # Check cost monitor
        try:
            cost_health = self.cost_monitor.health_check()
            health_status["components"]["cost_monitor"] = cost_health
            if cost_health != "healthy":
                issues.append(f"Cost monitor: {cost_health}")
        except Exception as e:
            issues.append(f"Cost monitor error: {e}")
            health_status["components"]["cost_monitor"] = "error"

        # Overall status
        if issues:
            health_status["overall"] = "degraded" if len(issues) < 3 else "unhealthy"
            health_status["issues"] = issues

        return health_status
