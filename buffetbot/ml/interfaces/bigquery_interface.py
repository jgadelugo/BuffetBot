"""
BigQuery ML Interface - Upgrade path for cloud-based ML
STUB IMPLEMENTATION - No actual cloud connection (zero costs)
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .ml_interface import MLInterface, MLServiceCapabilities


class BigQueryMLInterface(MLInterface):
    """
    BigQuery ML upgrade interface (STUB - NO ACTUAL CLOUD CONNECTION)
    Provides interface compatibility for future cloud upgrade
    """

    def __init__(self, project_id: str = None):
        self.project_id = project_id or "demo-project"
        self.logger = logging.getLogger("ml.bigquery_interface")
        self.enabled = False  # Never actually enable cloud services

        self.logger.info(
            "BigQuery ML Interface initialized (STUB - no cloud connection)"
        )

    async def initialize(self) -> bool:
        """Initialize BigQuery ML (STUB - always returns False for safety)"""
        self.logger.info("BigQuery ML interface stub - no actual initialization")
        return False  # Always return False to prevent accidental cloud usage

    async def cleanup(self) -> None:
        """Cleanup resources (STUB)"""
        self.logger.info("BigQuery ML cleanup (stub)")

    def get_capabilities(self) -> MLServiceCapabilities:
        """Get BigQuery ML capabilities (theoretical)"""
        return MLServiceCapabilities(
            supports_training=True,
            supports_prediction=True,
            supports_batch_prediction=True,
            supports_hyperparameter_optimization=True,
            supports_feature_importance=False,
            supports_model_versioning=True,
            max_features=10000,
            max_training_samples=100000000,
            cost_per_training_hour=25.0,  # Estimated BigQuery ML cost
            cost_per_prediction=0.001,  # Estimated cost per prediction
        )

    async def train_model(
        self,
        training_data: pd.DataFrame,
        target_column: str,
        model_type: str,
        hyperparameters: dict[str, Any] = None,
    ) -> str:
        """Train model with BigQuery ML (STUB)"""
        raise NotImplementedError(
            "BigQuery ML training is not implemented in FREE tier. "
            "This is a stub interface for future cloud upgrade. "
            "To use BigQuery ML, enable billing and update implementation."
        )

    async def predict_single(
        self, model_id: str, features: dict[str, Union[float, int, str]]
    ) -> dict[str, Any]:
        """Make single prediction (STUB)"""
        raise NotImplementedError(
            "BigQuery ML prediction is not implemented in FREE tier. "
            "Use LocalMLManager for zero-cost predictions."
        )

    async def predict_batch(
        self, model_id: str, features_list: list[dict[str, Union[float, int, str]]]
    ) -> list[dict[str, Any]]:
        """Make batch predictions (STUB)"""
        raise NotImplementedError(
            "BigQuery ML batch prediction is not implemented in FREE tier. "
            "Use LocalMLManager for zero-cost batch predictions."
        )

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get model information (STUB)"""
        raise NotImplementedError(
            "BigQuery ML model info is not implemented in FREE tier."
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """List models (STUB)"""
        return []  # Return empty list for compatibility

    async def delete_model(self, model_id: str) -> bool:
        """Delete model (STUB)"""
        self.logger.warning(f"BigQuery ML delete model stub called for {model_id}")
        return False

    def get_cost_estimate(self, operation: str, **kwargs) -> dict[str, float]:
        """Get cost estimate for BigQuery ML operations"""

        # Provide theoretical cost estimates for planning
        if operation == "train":
            training_samples = kwargs.get("training_samples", 1000)
            # Estimate based on BigQuery ML pricing
            cost = max(0.25, training_samples / 1000000 * 5.0)  # Minimum $0.25
            return {
                "training_cost": cost,
                "storage_cost": 0.02,  # Estimated storage cost
                "total_cost": cost + 0.02,
            }

        elif operation == "predict":
            num_predictions = kwargs.get("num_predictions", 1)
            cost = num_predictions * 0.001  # $0.001 per prediction
            return {"prediction_cost": cost, "total_cost": cost}

        elif operation == "batch_predict":
            num_predictions = kwargs.get("num_predictions", 100)
            cost = num_predictions * 0.0005  # Batch discount
            return {"batch_prediction_cost": cost, "total_cost": cost}

        return {"total_cost": 0.0}

    def get_service_health(self) -> dict[str, Any]:
        """Get service health (STUB)"""
        return {
            "service_name": "BigQuery ML",
            "status": "not_enabled",
            "enabled": self.enabled,
            "cost_status": "free_tier_stub",
            "connection_status": "no_connection",
            "note": "This is a stub interface for future cloud upgrade",
            "upgrade_instructions": [
                "1. Enable BigQuery API in GCP Console",
                "2. Set up billing account",
                "3. Update BigQueryMLInterface implementation",
                "4. Configure authentication",
            ],
        }
