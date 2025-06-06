"""
Comprehensive tests for Enhanced Local ML (Task 2)
Tests XGBoost, LightGBM, training pipeline, and prediction service
"""

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from buffetbot.ml.local_ml.manager import LocalMLManager

# Enhanced ML Components
from buffetbot.ml.local_ml.models import (
    LightGBMModel,
    LinearRegressionModel,
    LocalMLModel,
    ModelType,
    RandomForestModel,
    XGBoostModel,
    create_model,
)
from buffetbot.ml.local_ml.predictions import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionService,
)
from buffetbot.ml.local_ml.training import (
    TrainingConfig,
    TrainingPipeline,
    create_default_training_config,
)


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing"""
    np.random.seed(42)
    n_samples = 500

    # Create features
    data = {
        "price_0": np.random.uniform(50, 200, n_samples),
        "price_1": np.random.uniform(50, 200, n_samples),
        "volume_0": np.random.uniform(1000, 50000, n_samples),
        "volume_1": np.random.uniform(1000, 50000, n_samples),
        "indicator_0": np.random.uniform(-1, 1, n_samples),
        "indicator_1": np.random.uniform(-1, 1, n_samples),
        "moving_avg_5": np.random.uniform(45, 205, n_samples),
        "volatility": np.random.uniform(0.1, 2.0, n_samples),
    }

    df = pd.DataFrame(data)

    # Create target with realistic relationship
    target = (
        0.3 * df["price_0"]
        + 0.2 * df["price_1"]
        + 0.1 * df["volume_0"] / 1000
        + 0.05 * df["indicator_0"]
        + 0.1 * np.random.normal(0, 1, n_samples)
    )

    df["target"] = target
    return df


class TestEnhancedMLModels:
    """Test enhanced ML model implementations"""

    def test_xgboost_model_creation(self):
        """Test XGBoost model creation and configuration"""
        model = create_model(ModelType.XGBOOST, {"n_estimators": 50, "max_depth": 4})
        assert isinstance(model, XGBoostModel)
        assert model.config.model_type == ModelType.XGBOOST
        assert model.config.hyperparameters["n_estimators"] == 50

    def test_lightgbm_model_creation(self):
        """Test LightGBM model creation and configuration"""
        model = create_model(
            ModelType.LIGHTGBM, {"n_estimators": 75, "learning_rate": 0.05}
        )
        assert isinstance(model, LightGBMModel)
        assert model.config.model_type == ModelType.LIGHTGBM
        assert model.config.hyperparameters["learning_rate"] == 0.05

    def test_random_forest_model_creation(self):
        """Test Random Forest model creation"""
        model = create_model(
            ModelType.RANDOM_FOREST, {"n_estimators": 100, "max_depth": 8}
        )
        assert isinstance(model, RandomForestModel)
        assert model.config.model_type == ModelType.RANDOM_FOREST

    def test_xgboost_training_and_prediction(self, sample_training_data):
        """Test XGBoost training and prediction workflow"""
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.XGBOOST, {"n_estimators": 10})

        # Train model
        performance = model.train(X, y)

        # Check training results
        assert model.is_trained
        assert performance.r2_score > 0.5  # Should achieve reasonable accuracy
        assert performance.training_time_seconds > 0
        assert len(model.feature_names) == len(X.columns)

        # Test predictions
        predictions = model.predict(X.head(5))
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)
        assert all(importance >= 0)  # All importance scores should be non-negative

    def test_lightgbm_training_and_prediction(self, sample_training_data):
        """Test LightGBM training and prediction workflow"""
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.LIGHTGBM, {"n_estimators": 10, "verbose": -1})

        # Train model
        performance = model.train(X, y)

        # Check training results
        assert model.is_trained
        assert performance.r2_score > 0.5
        assert performance.training_time_seconds > 0

        # Test predictions
        predictions = model.predict(X.head(5))
        assert len(predictions) == 5

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)

    def test_cross_validation(self, sample_training_data):
        """Test cross-validation functionality"""
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.RANDOM_FOREST, {"n_estimators": 20})
        cv_scores = model.cross_validate(X, y)

        # Check CV results
        assert "mean_r2" in cv_scores
        assert "std_r2" in cv_scores
        assert cv_scores["mean_r2"] > 0.3  # Should achieve reasonable CV score
        assert cv_scores["std_r2"] >= 0  # Standard deviation should be non-negative


class TestTrainingPipeline:
    """Test enhanced training pipeline with hyperparameter optimization"""

    def test_training_config_creation(self):
        """Test training configuration creation"""
        config = create_default_training_config(
            model_types=[ModelType.LINEAR_REGRESSION, ModelType.XGBOOST],
            quick_mode=True,
        )

        assert len(config.model_types) == 2
        assert ModelType.LINEAR_REGRESSION in config.model_types
        assert ModelType.XGBOOST in config.model_types
        assert config.hyperparameter_trials == 10  # Quick mode
        assert config.optimization_timeout_minutes == 5  # Quick mode

    def test_training_pipeline_initialization(self):
        """Test training pipeline initialization"""
        config = create_default_training_config(quick_mode=True)
        pipeline = TrainingPipeline(config)

        assert pipeline.config == config
        assert len(pipeline.results) == 0

    def test_training_pipeline_execution(self, sample_training_data):
        """Test training pipeline with multiple models"""
        # Use quick mode for faster testing
        config = create_default_training_config(
            model_types=[ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST],
            quick_mode=True,
        )

        pipeline = TrainingPipeline(config)

        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        # Train models (without hyperparameter optimization for speed)
        results = pipeline.train_all_models(X, y, optimize_hyperparameters=False)

        # Check results
        assert len(results) == 2  # Two models trained
        assert all(result.model.is_trained for result in results)
        assert all(result.performance.r2_score > 0.3 for result in results)

        # Test model comparison
        comparison_df = pipeline.get_model_comparison()
        assert len(comparison_df) == 2
        assert "Model" in comparison_df.columns
        assert "R²_Score" in comparison_df.columns

        # Test best model selection
        best_model = pipeline.get_best_model("r2_score")
        assert best_model is not None
        assert best_model in results

    def test_hyperparameter_optimization(self, sample_training_data):
        """Test hyperparameter optimization with Optuna"""
        # Use very quick optimization for testing
        config = TrainingConfig(
            model_types=[ModelType.XGBOOST],
            hyperparameter_trials=3,  # Very few trials for speed
            optimization_timeout_minutes=1,  # Short timeout
            parallel_training=False,
        )

        pipeline = TrainingPipeline(config)
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        # Train with hyperparameter optimization
        results = pipeline.train_all_models(X, y, optimize_hyperparameters=True)

        assert len(results) == 1
        assert results[0].model.is_trained
        assert len(results[0].hyperparameters) > 0  # Should have optimized parameters


class TestPredictionService:
    """Test enhanced prediction service with caching"""

    @pytest.mark.asyncio
    async def test_prediction_service_initialization(self):
        """Test prediction service initialization"""
        service = PredictionService(cache_ttl_minutes=30)

        assert len(service.models) == 0
        assert service.prediction_stats["total_predictions"] == 0
        assert service.cache.cache_ttl_seconds == 30 * 60

    @pytest.mark.asyncio
    async def test_model_registration(self, sample_training_data):
        """Test model registration with prediction service"""
        service = PredictionService()

        # Train a model
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.LINEAR_REGRESSION)
        model.train(X, y)

        # Register model
        service.register_model("test_model", model)

        assert "test_model" in service.models
        assert service.models["test_model"] == model

        # Test model info
        model_info = service.get_model_info("test_model")
        assert model_info["model_id"] == "test_model"
        assert model_info["model_type"] == ModelType.LINEAR_REGRESSION.value
        assert model_info["is_trained"] is True

    @pytest.mark.asyncio
    async def test_single_prediction(self, sample_training_data):
        """Test single prediction functionality"""
        service = PredictionService()

        # Train and register model
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.LINEAR_REGRESSION)
        model.train(X, y)
        service.register_model("test_model", model)

        # Make prediction
        features = X.iloc[0].to_dict()
        request = PredictionRequest(features=features, model_id="test_model")

        response = await service.predict_single(request)

        assert isinstance(response.prediction, (int, float))
        assert response.model_id == "test_model"
        assert response.model_type == ModelType.LINEAR_REGRESSION.value
        assert response.prediction_time_ms > 0
        assert not response.cached  # First prediction shouldn't be cached

    @pytest.mark.asyncio
    async def test_prediction_caching(self, sample_training_data):
        """Test prediction caching functionality"""
        service = PredictionService(cache_ttl_minutes=60)

        # Train and register model
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.LINEAR_REGRESSION)
        model.train(X, y)
        service.register_model("test_model", model)

        # Make same prediction twice
        features = X.iloc[0].to_dict()
        request = PredictionRequest(features=features, model_id="test_model")

        # First prediction
        response1 = await service.predict_single(request)
        assert not response1.cached

        # Second prediction (should be cached)
        response2 = await service.predict_single(request)
        assert response2.cached
        assert response2.prediction == response1.prediction

        # Check cache hit statistics
        stats = service.get_service_stats()
        assert stats["cache_hits"] >= 1
        assert stats["cache_hit_rate_percent"] > 0

    @pytest.mark.asyncio
    async def test_batch_predictions(self, sample_training_data):
        """Test batch prediction functionality"""
        service = PredictionService()

        # Train and register model
        X = sample_training_data.drop("target", axis=1)
        y = sample_training_data["target"]

        model = create_model(ModelType.RANDOM_FOREST, {"n_estimators": 10})
        model.train(X, y)
        service.register_model("test_model", model)

        # Make batch predictions
        features_list = [X.iloc[i].to_dict() for i in range(5)]
        request = BatchPredictionRequest(
            features_list=features_list,
            model_id="test_model",
            parallel_processing=False,  # Sequential for testing
        )

        response = await service.predict_batch(request)

        assert len(response.predictions) == 5
        assert response.total_predictions == 5
        assert response.batch_processing_time_ms > 0
        assert response.avg_prediction_time_ms > 0


class TestLocalMLManager:
    """Test complete local ML manager integration"""

    @pytest.mark.asyncio
    async def test_local_ml_manager_initialization(self):
        """Test local ML manager initialization"""
        manager = LocalMLManager()

        initialized = await manager.initialize()
        assert initialized is True

        # Check health
        health = manager.get_service_health()
        assert health["status"] == "healthy"
        assert health["total_cost"] == 0.0  # Should be zero cost
        assert health["supports_hyperparameter_optimization"] is True

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_end_to_end_ml_workflow(self, sample_training_data):
        """Test complete end-to-end ML workflow"""
        manager = LocalMLManager()
        await manager.initialize()

        try:
            # Train multiple models
            results = await manager.train_models(
                training_data=sample_training_data,
                target_column="target",
                model_types=[ModelType.LINEAR_REGRESSION, ModelType.XGBOOST],
                optimize_hyperparameters=False,  # Skip optimization for speed
                quick_mode=True,
            )

            # Check training results
            assert len(results) == 2
            assert all(result.model.is_trained for result in results)

            # Test model comparison
            comparison = manager.get_model_comparison()
            assert len(comparison) == 2

            # Test best model selection
            best_model = manager.get_best_model()
            assert best_model is not None

            # Test single prediction
            features = sample_training_data.drop("target", axis=1).iloc[0].to_dict()
            prediction = await manager.predict_single(features)

            assert isinstance(prediction.prediction, (int, float))
            assert prediction.prediction_time_ms > 0

            # Test batch predictions
            features_list = [
                sample_training_data.drop("target", axis=1).iloc[i].to_dict()
                for i in range(3)
            ]
            batch_response = await manager.predict_batch(features_list)

            assert len(batch_response.predictions) == 3
            assert batch_response.total_predictions == 3

            # Test service statistics
            stats = manager.get_prediction_stats()
            assert stats["total_predictions"] >= 4  # 1 single + 3 batch

            # Check that costs remain zero
            cost_summary = manager.get_cost_summary()
            assert cost_summary["total_cost"] == 0.0

        finally:
            await manager.cleanup()

    @pytest.mark.asyncio
    async def test_sample_data_generation(self):
        """Test sample training data generation"""
        manager = LocalMLManager()
        await manager.initialize()

        try:
            # Generate sample data
            sample_data = await manager.create_sample_training_data(
                num_samples=100, num_features=5, noise_level=0.05
            )

            assert len(sample_data) == 100
            assert len(sample_data.columns) == 6  # 5 features + 1 target
            assert "target" in sample_data.columns

            # Test training on generated data
            results = await manager.train_models(
                training_data=sample_data,
                target_column="target",
                model_types=[ModelType.LINEAR_REGRESSION],
                quick_mode=True,
            )

            assert len(results) == 1
            assert results[0].model.is_trained

        finally:
            await manager.cleanup()


@pytest.mark.asyncio
async def test_performance_comparison():
    """Test that enhanced models outperform basic linear regression"""
    # Create synthetic data with non-linear relationships
    np.random.seed(42)
    n_samples = 1000

    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(-5, 5, n_samples)
    X3 = np.random.uniform(0, 1, n_samples)

    # Non-linear target function
    y = 2 * X1 + np.sin(X2) * 3 + X3**2 + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame(
        {"feature_1": X1, "feature_2": X2, "feature_3": X3, "target": y}
    )

    manager = LocalMLManager()
    await manager.initialize()

    try:
        # Train models
        results = await manager.train_models(
            training_data=data,
            target_column="target",
            model_types=[
                ModelType.LINEAR_REGRESSION,
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST,
            ],
            optimize_hyperparameters=False,
            quick_mode=True,
        )

        # Get performance comparison
        comparison = manager.get_model_comparison()

        # Linear regression should have lowest R² for non-linear data
        linear_r2 = comparison[comparison["Model"] == "linear_regression"][
            "R²_Score"
        ].iloc[0]
        forest_r2 = comparison[comparison["Model"] == "random_forest"]["R²_Score"].iloc[
            0
        ]
        xgb_r2 = comparison[comparison["Model"] == "xgboost"]["R²_Score"].iloc[0]

        # Tree-based models should outperform linear regression on non-linear data
        assert forest_r2 > linear_r2
        assert xgb_r2 > linear_r2

        print(f"\n📊 Performance Comparison:")
        print(f"Linear Regression R²: {linear_r2:.4f}")
        print(f"Random Forest R²: {forest_r2:.4f}")
        print(f"XGBoost R²: {xgb_r2:.4f}")
        print(f"🚀 Improvement over Linear: {max(forest_r2, xgb_r2) - linear_r2:.4f}")

    finally:
        await manager.cleanup()


if __name__ == "__main__":
    # Run specific test for demonstration
    asyncio.run(test_performance_comparison())
