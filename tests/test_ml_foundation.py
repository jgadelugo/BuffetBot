"""
Test ML Foundation components
"""
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from buffetbot.ml.managers.base_manager import BaseMLManager, MLServiceConfig
from buffetbot.ml.managers.ml_manager import MLManager
from buffetbot.ml.models.metadata import (
    ModelMetadata,
    ModelMetrics,
    ModelStatus,
    ModelType,
)
from buffetbot.ml.models.registry import ModelRegistry
from buffetbot.ml.monitoring.cost_monitor import MLCostMonitor
from buffetbot.ml.monitoring.performance import PerformanceMonitor
from buffetbot.ml.utils.data_prep import DataPreprocessor
from buffetbot.ml.utils.validation import MLValidator


class TestMLManager(BaseMLManager):
    """Test implementation of BaseMLManager"""

    async def initialize(self) -> bool:
        return True

    async def cleanup(self) -> None:
        pass


@pytest.mark.asyncio
async def test_base_ml_manager():
    """Test base ML manager functionality"""
    config = MLServiceConfig(
        service_name="test_service", cost_per_hour=10.0, max_cost_per_day=50.0
    )

    manager = TestMLManager(config)

    # Test initialization
    result = await manager.initialize()
    assert result is True

    # Test cost tracking
    manager.track_cost(5.0)
    assert manager.total_cost == 5.0

    # Test cost summary
    summary = manager.get_cost_summary()
    assert summary["total_cost"] == 5.0
    assert summary["service"] == "test_service"

    # Test cost limit
    with pytest.raises(Exception):
        manager.track_cost(50.0)  # Should exceed daily limit


@pytest.mark.asyncio
async def test_ml_manager():
    """Test main ML manager functionality"""
    manager = MLManager()

    # Test initialization
    result = await manager.initialize()
    assert result is True
    assert manager.is_initialized is True

    # Test system status
    status = await manager.get_system_status()
    assert status["initialized"] is True
    assert "cost_summary" in status
    assert "model_registry" in status

    # Test health check
    health = await manager.health_check()
    assert health["overall"] in ["healthy", "degraded"]

    # Cleanup
    await manager.cleanup()
    assert manager.is_initialized is False


def test_model_metadata():
    """Test model metadata creation and serialization"""
    metrics = ModelMetrics(accuracy=0.85, precision=0.80, recall=0.90, mse=0.05)

    metadata = ModelMetadata(
        model_id="test_model_1",
        model_name="price_predictor",
        version="1.0",
        model_type=ModelType.SKLEARN,
        status=ModelStatus.TRAINED,
        training_date=datetime.utcnow(),
        training_duration_minutes=30,
        training_data_size=1000,
        feature_columns=["price", "volume", "sma_5"],
        target_column="next_price",
        metrics=metrics,
        file_path="models/price_predictor/1.0/model.pkl",
        model_hash="abcd1234",
        tags=["production", "v1"],
        description="Test model for price prediction",
    )

    # Test serialization
    data_dict = metadata.to_dict()
    assert data_dict["model_name"] == "price_predictor"
    assert data_dict["model_type"] == "sklearn"
    assert data_dict["status"] == "trained"

    # Test deserialization
    restored = ModelMetadata.from_dict(data_dict)
    assert restored.model_name == metadata.model_name
    assert restored.model_type == metadata.model_type
    assert restored.status == metadata.status


@pytest.mark.asyncio
async def test_model_registry():
    """Test model registry functionality"""
    registry = ModelRegistry()
    await registry.initialize()

    # Create test model metadata
    metrics = ModelMetrics(accuracy=0.85)
    metadata = ModelMetadata(
        model_id="test_model_1",
        model_name="price_predictor",
        version="1.0",
        model_type=ModelType.SKLEARN,
        status=ModelStatus.TRAINED,
        training_date=datetime.utcnow(),
        training_duration_minutes=30,
        training_data_size=1000,
        feature_columns=["price", "volume"],
        target_column="next_price",
        metrics=metrics,
        file_path="models/price_predictor/1.0/model.pkl",
        model_hash="",
    )

    # Test registration (without actual model for testing)
    model_id = await registry.register_model(None, metadata, save_to_gcs=False)
    assert model_id == "test_model_1"

    # Test retrieval
    retrieved = await registry.get_model_metadata("test_model_1")
    assert retrieved is not None
    assert retrieved.model_name == "price_predictor"

    # Test listing
    models = await registry.list_models()
    assert len(models) >= 1

    # Test latest model
    latest = await registry.get_latest_model("price_predictor")
    assert latest is not None
    assert latest.model_id == "test_model_1"

    # Test status update
    success = await registry.update_model_status("test_model_1", ModelStatus.DEPLOYED)
    assert success is True

    # Test deployed models
    deployed = await registry.get_deployed_models()
    assert len(deployed) >= 1

    # Test prediction recording
    await registry.record_prediction("test_model_1")
    updated = await registry.get_model_metadata("test_model_1")
    assert updated.prediction_count == 1

    await registry.cleanup()


def test_cost_monitor():
    """Test cost monitoring functionality"""
    monitor = MLCostMonitor()
    monitor.start_monitoring()

    # Track some costs
    monitor.track_cost("bigquery_ml", "training", 15.50)
    monitor.track_cost("vertex_ai", "prediction", 2.25)
    monitor.track_cost("local_ml", "feature_engineering", 0.0)

    # Test daily cost
    daily_cost = monitor.get_daily_cost()
    assert daily_cost == 17.75

    # Test total cost
    total_cost = monitor.get_total_cost()
    assert total_cost == 17.75

    # Test service breakdown
    breakdown = monitor.get_service_breakdown()
    assert "bigquery_ml" in breakdown
    assert "vertex_ai" in breakdown
    assert "local_ml" in breakdown
    assert breakdown["bigquery_ml"] == 15.50

    # Test operation breakdown
    op_breakdown = monitor.get_operation_breakdown()
    assert "bigquery_ml.training" in op_breakdown
    assert "vertex_ai.prediction" in op_breakdown

    # Test cost summary
    summary = monitor.get_cost_summary()
    assert summary["total_cost"] == 17.75
    assert summary["monitoring_active"] is True

    # Test health check
    health = monitor.health_check()
    assert health == "healthy"

    monitor.stop_monitoring()


def test_performance_monitor():
    """Test performance monitoring functionality"""
    monitor = PerformanceMonitor()

    # Record some metrics
    monitor.record_metric("model_1", "accuracy", 0.85)
    monitor.record_metric("model_1", "accuracy", 0.87)
    monitor.record_metric("model_1", "mse", 0.05)
    monitor.record_metric("model_2", "accuracy", 0.92)

    # Set baseline
    monitor.set_baseline("model_1", "accuracy", 0.80)

    # Test current performance
    current = monitor.get_current_performance("model_1", "accuracy")
    assert current == 0.87

    # Test drift detection
    drift = monitor.detect_drift("model_1", "accuracy", threshold=0.05)
    assert drift is True  # 0.87 vs 0.80 baseline > 5% threshold

    # Test performance trend
    trend = monitor.get_performance_trend("model_1", "accuracy", days=7)
    assert len(trend) == 2  # Two accuracy recordings

    # Test model summary
    summary = monitor.get_model_summary("model_1")
    assert summary["model_id"] == "model_1"
    assert "accuracy" in summary["metrics"]
    assert summary["metrics"]["accuracy"]["current_value"] == 0.87


def test_data_preprocessor():
    """Test data preprocessing functionality"""
    preprocessor = DataPreprocessor()

    # Create sample time series data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 105,
            "low": np.random.randn(100) + 95,
            "close": np.random.randn(100) + 100,
        }
    )

    # Test time features
    df_with_time = preprocessor.create_time_features(df, "timestamp")
    assert "timestamp_hour" in df_with_time.columns
    assert "timestamp_weekday" in df_with_time.columns
    assert "timestamp_is_weekend" in df_with_time.columns

    # Test lag features
    df_with_lags = preprocessor.create_lag_features(df, "price", [1, 2, 3])
    assert "price_lag_1" in df_with_lags.columns
    assert "price_lag_2" in df_with_lags.columns

    # Test rolling features
    df_with_rolling = preprocessor.create_rolling_features(df, "price", [5, 10])
    assert "price_rolling_mean_5" in df_with_rolling.columns
    assert "price_rolling_std_10" in df_with_rolling.columns

    # Test technical indicators
    df_with_indicators = preprocessor.create_technical_indicators(df)
    assert "sma_5" in df_with_indicators.columns
    assert "price_range" in df_with_indicators.columns

    # Test target creation
    df_with_target = preprocessor.create_target_features(
        df, "price", prediction_horizon=1
    )
    assert "price_future_1" in df_with_target.columns
    assert "price_direction_1" in df_with_target.columns

    # Test complete preprocessing
    df_complete = preprocessor.prepare_time_series_data(
        df, target_col="price", datetime_col="timestamp"
    )
    assert df_complete.shape[1] > df.shape[1]  # Should have more columns


def test_ml_validator():
    """Test ML validation functionality"""
    validator = MLValidator()

    # Create test data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100),
            "price": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )

    # Test DataFrame validation
    result = validator.validate_dataframe(df, required_columns=["timestamp", "price"])
    assert result["valid"] is True
    assert result["stats"]["shape"] == (100, 4)

    # Test time series validation
    ts_result = validator.validate_time_series_data(df, "timestamp", "price")
    assert ts_result["valid"] is True
    assert "date_range_days" in ts_result["stats"]

    # Test feature-target validation
    X = df[["volume", "category"]]
    y = df["price"]
    ft_result = validator.validate_feature_target_split(X, y)
    assert ft_result["valid"] is True
    assert ft_result["stats"]["feature_count"] == 2

    # Test prediction validation
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise
    pred_result = validator.validate_model_predictions(y_true, y_pred)
    assert pred_result["valid"] is True

    # Test data leakage check
    leakage_result = validator.check_data_leakage(X, y, datetime_col="timestamp")
    assert "potential_leakage" in leakage_result


if __name__ == "__main__":
    # Run basic tests
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing ML Foundation Components")

    # Test cost monitor
    print("Testing Cost Monitor...")
    test_cost_monitor()
    print("✅ Cost Monitor tests passed")

    # Test performance monitor
    print("Testing Performance Monitor...")
    test_performance_monitor()
    print("✅ Performance Monitor tests passed")

    # Test data preprocessor
    print("Testing Data Preprocessor...")
    test_data_preprocessor()
    print("✅ Data Preprocessor tests passed")

    # Test validator
    print("Testing ML Validator...")
    test_ml_validator()
    print("✅ ML Validator tests passed")

    # Test metadata
    print("Testing Model Metadata...")
    test_model_metadata()
    print("✅ Model Metadata tests passed")

    print("🎉 All ML Foundation tests completed successfully!")
    print("Phase 3 Task 1 (ML Foundation) is ready!")
