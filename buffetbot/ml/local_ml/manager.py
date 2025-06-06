"""
Local ML Manager - Complete local ML system integration
Manages models, training, predictions with zero cloud costs
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ML Foundation integration
from buffetbot.ml.managers.base_manager import BaseMLManager, MLServiceConfig
from buffetbot.ml.models.registry import ModelMetadata, ModelRegistry
from buffetbot.ml.monitoring.cost_monitor import MLCostMonitor

# Local ML components
from .models import LocalMLModel, ModelPerformance, ModelType, create_model
from .predictions import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionService,
)
from .training import (
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
    create_default_training_config,
)


class LocalMLManager(BaseMLManager):
    """
    Complete local ML management system
    Integrates training, prediction, and model management with zero costs
    """

    def __init__(self, config: MLServiceConfig = None):
        if config is None:
            config = MLServiceConfig(
                service_name="local_ml",
                cost_per_hour=0.0,  # Local ML is free
                max_cost_per_day=0.0,
                enable_cost_monitoring=True,
                cache_predictions=True,
            )

        super().__init__(config)

        # Initialize ML components
        self.model_registry = ModelRegistry()
        self.prediction_service = PredictionService()
        self.cost_monitor = MLCostMonitor()

        # Training pipeline
        self.training_pipeline: Optional[TrainingPipeline] = None

        # Active models
        self.active_models: dict[str, LocalMLModel] = {}

        self.logger.info("Local ML Manager initialized (zero costs)")

    async def initialize(self) -> bool:
        """Initialize the local ML service"""
        try:
            # Initialize base manager
            await super().initialize()

            # All local components are ready immediately
            self.logger.info("Local ML service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize local ML service: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup local ML resources"""
        try:
            # Clear active models
            self.active_models.clear()

            # Clear prediction service models
            for model_id in list(self.prediction_service.models.keys()):
                self.prediction_service.unregister_model(model_id)

            self.logger.info("Local ML service cleaned up")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def train_models(
        self,
        training_data: pd.DataFrame,
        target_column: str,
        model_types: list[ModelType] = None,
        optimize_hyperparameters: bool = True,
        quick_mode: bool = False,
    ) -> list[TrainingResult]:
        """
        Train multiple models and compare performance

        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column
            model_types: List of model types to train (default: all)
            optimize_hyperparameters: Whether to optimize hyperparameters
            quick_mode: Use faster settings for development
        """
        start_time = datetime.utcnow()

        # Prepare data
        if target_column not in training_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        X = training_data.drop(columns=[target_column])
        y = training_data[target_column]

        self.logger.info(
            f"Starting model training - Features: {len(X.columns)}, "
            f"Samples: {len(X)}, Target: {target_column}"
        )

        # Create training configuration
        if model_types is None:
            model_types = [
                ModelType.LINEAR_REGRESSION,
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST,
                ModelType.LIGHTGBM,
            ]

        training_config = create_default_training_config(model_types, quick_mode)

        # Initialize training pipeline
        self.training_pipeline = TrainingPipeline(training_config)

        # Train all models
        self.logger.info(f"Training {len(model_types)} model types...")
        results = self.training_pipeline.train_all_models(
            X, y, optimize_hyperparameters
        )

        # Register trained models
        for result in results:
            model_id = f"{result.model.config.model_type.value}_{int(datetime.utcnow().timestamp())}"

            # Create model metadata
            from buffetbot.ml.models.metadata import ModelMetrics, ModelStatus
            from buffetbot.ml.models.metadata import ModelType as MetadataModelType

            # Map local model types to metadata model types
            model_type_mapping = {
                ModelType.LINEAR_REGRESSION: MetadataModelType.SKLEARN,
                ModelType.RIDGE_REGRESSION: MetadataModelType.SKLEARN,
                ModelType.LASSO_REGRESSION: MetadataModelType.SKLEARN,
                ModelType.ELASTIC_NET: MetadataModelType.SKLEARN,
                ModelType.RANDOM_FOREST: MetadataModelType.SKLEARN,
                ModelType.GRADIENT_BOOSTING: MetadataModelType.SKLEARN,
                ModelType.XGBOOST: MetadataModelType.XGBOOST,
                ModelType.LIGHTGBM: MetadataModelType.LIGHTGBM,
            }

            # Create metrics object
            metrics = ModelMetrics(
                r2_score=result.performance.r2_score,
                rmse=result.performance.rmse,
                mae=result.performance.mae,
                mape=result.performance.mape,
                custom_metrics={
                    "cv_r2_mean": result.cross_validation_scores["mean_r2"],
                    "cv_r2_std": result.cross_validation_scores["std_r2"],
                },
            )

            metadata = ModelMetadata(
                model_id=model_id,
                model_name=result.model.config.model_type.value,
                version="1.0",
                model_type=model_type_mapping[result.model.config.model_type],
                status=ModelStatus.TRAINED,
                training_date=datetime.utcnow(),
                training_duration_minutes=int(
                    result.performance.training_time_seconds / 60
                ),
                training_data_size=len(training_data),
                feature_columns=result.model.feature_names,
                target_column=result.model.target_name,
                metrics=metrics,
                file_path=f"models/{model_id}.pkl",
                model_hash="",  # Will be set by registry
                cost_to_train=0.0,  # Local training is free
            )

            # Register model
            await self.model_registry.register_model(
                result.model, metadata, save_to_gcs=False  # Keep local for now
            )

            # Store active model
            self.active_models[model_id] = result.model

            # Register with prediction service
            self.prediction_service.register_model(model_id, result.model)

        training_time = (datetime.utcnow() - start_time).total_seconds()

        # Track costs (zero for local training)
        self.track_cost(0.0)

        self.logger.info(
            f"Model training completed in {training_time:.1f}s - "
            f"{len(results)} models trained"
        )

        return results

    async def predict_single(
        self,
        features: dict[str, Union[float, int, str]],
        model_id: Optional[str] = None,
        include_confidence_interval: bool = False,
        include_feature_importance: bool = False,
    ) -> PredictionResponse:
        """Make a single prediction"""

        request = PredictionRequest(
            features=features,
            model_id=model_id,
            include_confidence_interval=include_confidence_interval,
            include_feature_importance=include_feature_importance,
        )

        # Track costs (zero for local predictions)
        self.track_cost(0.0)

        return await self.prediction_service.predict_single(request)

    async def predict_batch(
        self,
        features_list: list[dict[str, Union[float, int, str]]],
        model_id: Optional[str] = None,
        parallel_processing: bool = True,
    ) -> BatchPredictionResponse:
        """Make batch predictions"""

        request = BatchPredictionRequest(
            features_list=features_list,
            model_id=model_id,
            parallel_processing=parallel_processing,
        )

        # Track costs (zero for local predictions)
        self.track_cost(0.0)

        return await self.prediction_service.predict_batch(request)

    def get_best_model(self, metric: str = "r2_score") -> Optional[TrainingResult]:
        """Get the best performing model from latest training"""
        if not self.training_pipeline:
            return None

        return self.training_pipeline.get_best_model(metric)

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        if not self.training_pipeline:
            return pd.DataFrame()

        return self.training_pipeline.get_model_comparison()

    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        return await self.model_registry.get_model_metadata(model_id)

    async def list_models(
        self, model_name: Optional[str] = None
    ) -> list[ModelMetadata]:
        """List all registered models"""
        return await self.model_registry.list_models(model_name)

    def get_prediction_stats(self) -> dict[str, Any]:
        """Get prediction service statistics"""
        return self.prediction_service.get_service_stats()

    def get_model_info(self, model_id: str = None) -> dict[str, Any]:
        """Get information about registered models"""
        return self.prediction_service.get_model_info(model_id)

    async def deploy_model(self, model_id: str, make_default: bool = False) -> bool:
        """Deploy a model for predictions"""
        try:
            if model_id not in self.active_models:
                # Try to load from registry
                metadata = await self.model_registry.get_model_metadata(model_id)
                if not metadata:
                    raise ValueError(f"Model {model_id} not found")

                # For now, we need the model to be in active models
                # In a full implementation, we'd load it from storage
                raise ValueError(f"Model {model_id} not in active models")

            model = self.active_models[model_id]

            # Register with prediction service (if not already registered)
            if model_id not in self.prediction_service.models:
                self.prediction_service.register_model(model_id, model)

            self.logger.info(f"Model {model_id} deployed for predictions")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_id}: {e}")
            return False

    async def undeploy_model(self, model_id: str) -> bool:
        """Remove a model from prediction service"""
        try:
            self.prediction_service.unregister_model(model_id)
            self.logger.info(f"Model {model_id} undeployed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to undeploy model {model_id}: {e}")
            return False

    def get_service_health(self) -> dict[str, Any]:
        """Get comprehensive service health status"""

        prediction_stats = self.get_prediction_stats()
        cost_summary = self.get_cost_summary()

        return {
            "service_name": self.config.service_name,
            "status": "healthy",
            "initialized": True,
            "total_cost": cost_summary["total_cost"],
            "active_models": len(self.active_models),
            "registered_models": prediction_stats["registered_models"],
            "total_predictions": prediction_stats["total_predictions"],
            "cache_hit_rate": prediction_stats.get("cache_hit_rate_percent", 0),
            "avg_prediction_time_ms": prediction_stats.get("avg_prediction_time_ms", 0),
            "last_training": getattr(self.training_pipeline, "results", None)
            and len(self.training_pipeline.results) > 0,
            "supports_hyperparameter_optimization": True,
            "supports_parallel_training": True,
            "supports_model_comparison": True,
            "cost_per_prediction": 0.0,  # Local is free
            "uptime_hours": (datetime.utcnow() - self.session_start).total_seconds()
            / 3600,
        }

    async def create_sample_training_data(
        self, num_samples: int = 1000, num_features: int = 10, noise_level: float = 0.1
    ) -> pd.DataFrame:
        """Create sample training data for testing"""

        np.random.seed(42)

        # Generate features
        feature_data = {}
        for i in range(num_features):
            if i < 3:
                # Price-like features
                feature_data[f"price_{i}"] = np.random.uniform(50, 200, num_samples)
            elif i < 6:
                # Volume-like features
                feature_data[f"volume_{i-3}"] = np.random.uniform(
                    1000, 50000, num_samples
                )
            else:
                # Technical indicators
                feature_data[f"indicator_{i-6}"] = np.random.uniform(-1, 1, num_samples)

        df = pd.DataFrame(feature_data)

        # Create target variable with some relationship to features
        target = (
            0.3 * df["price_0"]
            + 0.2 * df["price_1"]
            + 0.1 * df["volume_0"] / 1000
            + 0.05 * df.get("indicator_0", 0)
            + noise_level * np.random.normal(0, 1, num_samples)
        )

        df["target"] = target

        self.logger.info(
            f"Created sample training data: {num_samples} samples, "
            f"{num_features} features"
        )

        return df
