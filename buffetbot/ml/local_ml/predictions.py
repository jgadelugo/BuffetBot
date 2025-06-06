"""
Enhanced Prediction Service - Fast local predictions with caching
Real-time serving capabilities with zero cloud costs
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Phase 1 Integration
from buffetbot.utils.cache import Cache

from .models import LocalMLModel, ModelPerformance


@dataclass
class PredictionRequest:
    """Request for making predictions"""

    features: dict[str, Union[float, int, str]]
    model_id: Optional[str] = None
    include_confidence_interval: bool = False
    include_feature_importance: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert features to DataFrame for prediction"""
        return pd.DataFrame([self.features])


@dataclass
class PredictionResponse:
    """Response from prediction service"""

    prediction: float
    model_id: str
    model_type: str
    timestamp: datetime
    confidence_interval: Optional[dict[str, float]] = None
    feature_importance: Optional[dict[str, float]] = None
    prediction_time_ms: float = 0.0
    cached: bool = False


@dataclass
class BatchPredictionRequest:
    """Request for batch predictions"""

    features_list: list[dict[str, Union[float, int, str]]]
    model_id: Optional[str] = None
    include_confidence_intervals: bool = False
    parallel_processing: bool = True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert features list to DataFrame for batch prediction"""
        return pd.DataFrame(self.features_list)


@dataclass
class BatchPredictionResponse:
    """Response from batch prediction service"""

    predictions: list[float]
    model_id: str
    model_type: str
    timestamp: datetime
    total_predictions: int
    batch_processing_time_ms: float
    avg_prediction_time_ms: float
    cached_predictions: int = 0


class PredictionCache:
    """Smart caching system for predictions"""

    def __init__(self, cache_ttl_minutes: int = 60):
        self.cache = Cache(cache_type="memory")
        self.cache_ttl_seconds = cache_ttl_minutes * 60
        self.logger = logging.getLogger("ml.prediction_cache")

    def _generate_cache_key(self, features: dict[str, Any], model_id: str) -> str:
        """Generate cache key from features and model ID"""
        # Sort features for consistent keys
        sorted_features = json.dumps(features, sort_keys=True)
        cache_key = f"pred:{model_id}:{hash(sorted_features)}"
        return cache_key

    async def get_prediction(
        self, features: dict[str, Any], model_id: str
    ) -> Optional[PredictionResponse]:
        """Get cached prediction if available"""
        cache_key = self._generate_cache_key(features, model_id)

        cached_data = self.cache.get(cache_key)
        if cached_data:
            try:
                # Deserialize cached prediction
                response_data = json.loads(cached_data)
                response_data["timestamp"] = datetime.fromisoformat(
                    response_data["timestamp"]
                )
                response_data["cached"] = True
                return PredictionResponse(**response_data)
            except Exception as e:
                self.logger.warning(f"Failed to deserialize cached prediction: {e}")

        return None

    async def store_prediction(
        self, features: dict[str, Any], model_id: str, response: PredictionResponse
    ) -> None:
        """Store prediction in cache"""
        cache_key = self._generate_cache_key(features, model_id)

        # Serialize response
        response_data = asdict(response)
        response_data["timestamp"] = response.timestamp.isoformat()
        response_data["cached"] = False  # Reset cached flag for storage

        serialized_data = json.dumps(response_data, default=str)

        self.cache.set(cache_key, serialized_data, expiration=self.cache_ttl_seconds)


class PredictionService:
    """Enhanced prediction service with caching and performance optimization"""

    def __init__(self, cache_ttl_minutes: int = 60):
        self.models: dict[str, LocalMLModel] = {}
        self.cache = PredictionCache(cache_ttl_minutes)
        self.logger = logging.getLogger("ml.prediction_service")
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "avg_prediction_time_ms": 0.0,
            "last_prediction_time": None,
        }

    def register_model(self, model_id: str, model: LocalMLModel) -> None:
        """Register a trained model for predictions"""
        if not model.is_trained:
            raise ValueError(f"Model {model_id} must be trained before registration")

        self.models[model_id] = model
        self.logger.info(
            f"Model registered: {model_id} ({model.config.model_type.value})"
        )

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the service"""
        if model_id in self.models:
            del self.models[model_id]
            self.logger.info(f"Model unregistered: {model_id}")

    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Make a single prediction with caching"""
        start_time = datetime.utcnow()

        # Determine model to use
        model_id = request.model_id or self._get_default_model_id()
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        # Check cache first
        cached_response = await self.cache.get_prediction(request.features, model_id)
        if cached_response:
            self.prediction_stats["cache_hits"] += 1
            self.prediction_stats["total_predictions"] += 1
            self.logger.debug(f"Cache hit for model {model_id}")
            return cached_response

        # Make prediction
        features_df = request.to_dataframe()
        prediction = model.predict(features_df)[0]

        # Calculate prediction time
        prediction_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Create response
        response = PredictionResponse(
            prediction=float(prediction),
            model_id=model_id,
            model_type=model.config.model_type.value,
            timestamp=datetime.utcnow(),
            prediction_time_ms=prediction_time_ms,
        )

        # Add confidence interval if requested
        if request.include_confidence_interval:
            response.confidence_interval = self._calculate_confidence_interval(
                model, features_df, prediction
            )

        # Add feature importance if requested
        if request.include_feature_importance and hasattr(
            model, "get_feature_importance"
        ):
            try:
                importance = model.get_feature_importance()
                response.feature_importance = importance.to_dict()
            except:
                pass

        # Cache the response
        await self.cache.store_prediction(request.features, model_id, response)

        # Update stats
        self._update_prediction_stats(prediction_time_ms)

        self.logger.debug(
            f"Prediction made: {prediction:.6f} (model: {model_id}, "
            f"time: {prediction_time_ms:.2f}ms)"
        )

        return response

    async def predict_batch(
        self, request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        """Make batch predictions with optional parallel processing"""
        start_time = datetime.utcnow()

        # Determine model to use
        model_id = request.model_id or self._get_default_model_id()
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        if request.parallel_processing and len(request.features_list) > 10:
            # Process in parallel for large batches
            predictions = await self._predict_batch_parallel(request, model, model_id)
        else:
            # Process sequentially for small batches
            predictions = await self._predict_batch_sequential(request, model, model_id)

        # Calculate timing
        batch_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        avg_time_ms = batch_time_ms / len(request.features_list)

        response = BatchPredictionResponse(
            predictions=predictions,
            model_id=model_id,
            model_type=model.config.model_type.value,
            timestamp=datetime.utcnow(),
            total_predictions=len(predictions),
            batch_processing_time_ms=batch_time_ms,
            avg_prediction_time_ms=avg_time_ms,
        )

        self.logger.info(
            f"Batch prediction completed: {len(predictions)} predictions "
            f"in {batch_time_ms:.1f}ms (avg: {avg_time_ms:.2f}ms)"
        )

        return response

    async def _predict_batch_sequential(
        self, request: BatchPredictionRequest, model: LocalMLModel, model_id: str
    ) -> list[float]:
        """Process batch predictions sequentially"""
        predictions = []
        cached_count = 0

        for features in request.features_list:
            # Check cache
            cached_response = await self.cache.get_prediction(features, model_id)
            if cached_response:
                predictions.append(cached_response.prediction)
                cached_count += 1
            else:
                # Make prediction
                features_df = pd.DataFrame([features])
                prediction = model.predict(features_df)[0]
                predictions.append(float(prediction))

                # Cache result
                response = PredictionResponse(
                    prediction=float(prediction),
                    model_id=model_id,
                    model_type=model.config.model_type.value,
                    timestamp=datetime.utcnow(),
                )
                await self.cache.store_prediction(features, model_id, response)

        self.prediction_stats["cache_hits"] += cached_count
        self.prediction_stats["total_predictions"] += len(predictions)

        return predictions

    async def _predict_batch_parallel(
        self, request: BatchPredictionRequest, model: LocalMLModel, model_id: str
    ) -> list[float]:
        """Process batch predictions in parallel (for large batches)"""

        # For very large batches, process in chunks to optimize memory usage
        chunk_size = 1000
        all_predictions = []

        for i in range(0, len(request.features_list), chunk_size):
            chunk = request.features_list[i : i + chunk_size]
            chunk_df = pd.DataFrame(chunk)

            # Make batch prediction on chunk
            chunk_predictions = model.predict(chunk_df)
            all_predictions.extend(chunk_predictions.tolist())

            # Cache results (asynchronously)
            asyncio.create_task(
                self._cache_batch_predictions(chunk, chunk_predictions, model_id)
            )

        self.prediction_stats["total_predictions"] += len(all_predictions)
        return all_predictions

    async def _cache_batch_predictions(
        self, features_list: list[dict], predictions: list[float], model_id: str
    ):
        """Cache batch predictions asynchronously"""
        for features, prediction in zip(features_list, predictions):
            response = PredictionResponse(
                prediction=float(prediction),
                model_id=model_id,
                model_type=self.models[model_id].config.model_type.value,
                timestamp=datetime.utcnow(),
            )
            await self.cache.store_prediction(features, model_id, response)

    def _calculate_confidence_interval(
        self, model: LocalMLModel, features_df: pd.DataFrame, prediction: float
    ) -> dict[str, float]:
        """Calculate approximate confidence interval (basic implementation)"""

        # For tree-based models, use prediction variance
        if hasattr(model.model, "estimators_"):
            try:
                # Get predictions from all estimators
                estimator_predictions = [
                    estimator.predict(features_df)[0]
                    for estimator in model.model.estimators_
                ]

                std_dev = np.std(estimator_predictions)

                return {
                    "lower_95": prediction - 1.96 * std_dev,
                    "upper_95": prediction + 1.96 * std_dev,
                    "std_dev": std_dev,
                }
            except:
                pass

        # Fallback: use historical performance metrics
        if model.performance:
            rmse = model.performance.rmse
            return {
                "lower_95": prediction - 1.96 * rmse,
                "upper_95": prediction + 1.96 * rmse,
                "std_dev": rmse,
            }

        return {"lower_95": prediction, "upper_95": prediction, "std_dev": 0.0}

    def _get_default_model_id(self) -> str:
        """Get the default model ID (first registered model)"""
        if not self.models:
            raise ValueError("No models registered in prediction service")
        return list(self.models.keys())[0]

    def _update_prediction_stats(self, prediction_time_ms: float) -> None:
        """Update prediction statistics"""
        self.prediction_stats["total_predictions"] += 1
        self.prediction_stats["last_prediction_time"] = datetime.utcnow()

        # Update rolling average
        total = self.prediction_stats["total_predictions"]
        current_avg = self.prediction_stats["avg_prediction_time_ms"]
        self.prediction_stats["avg_prediction_time_ms"] = (
            current_avg * (total - 1) + prediction_time_ms
        ) / total

    def get_service_stats(self) -> dict[str, Any]:
        """Get prediction service statistics"""
        cache_hit_rate = 0.0
        if self.prediction_stats["total_predictions"] > 0:
            cache_hit_rate = (
                self.prediction_stats["cache_hits"]
                / self.prediction_stats["total_predictions"]
            ) * 100

        return {
            "total_predictions": self.prediction_stats["total_predictions"],
            "cache_hits": self.prediction_stats["cache_hits"],
            "cache_hit_rate_percent": cache_hit_rate,
            "avg_prediction_time_ms": self.prediction_stats["avg_prediction_time_ms"],
            "registered_models": len(self.models),
            "model_ids": list(self.models.keys()),
            "last_prediction_time": self.prediction_stats["last_prediction_time"],
        }

    def get_model_info(self, model_id: str = None) -> dict[str, Any]:
        """Get information about registered models"""
        if model_id:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            model = self.models[model_id]
            return {
                "model_id": model_id,
                "model_type": model.config.model_type.value,
                "feature_names": model.feature_names,
                "target_name": model.target_name,
                "performance": asdict(model.performance) if model.performance else None,
                "is_trained": model.is_trained,
            }
        else:
            # Return info for all models
            return {
                model_id: self.get_model_info(model_id)
                for model_id in self.models.keys()
            }
