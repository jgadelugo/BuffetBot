#!/usr/bin/env python3
"""
Enhanced ML Demo - Simplified Version
=====================================
Demonstrates BuffetBot's enhanced local ML capabilities without complex data dependencies.
This version focuses on proving the ML architecture works with controlled test data.

Phase 3 Task 2: Enhanced Local ML Models
Cost: $0 (all local computation)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# BuffetBot imports
from buffetbot.ml.local_ml.manager import LocalMLManager
from buffetbot.ml.local_ml.models import ModelType
from buffetbot.ml.local_ml.predictions import PredictionService
from buffetbot.ml.local_ml.training import TrainingPipeline


def generate_simple_test_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generate simple test data with proper features for ML training"""
    print(f"🔧 Generating {num_samples} samples of test market data...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate time series data
    start_date = datetime.now() - timedelta(days=num_samples)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]

    # Generate base price trend with noise
    base_trend = np.linspace(100, 150, num_samples)
    noise = np.random.normal(0, 5, num_samples)
    prices = base_trend + noise

    # Create features that correlate with price movement
    volumes = np.random.lognormal(10, 0.5, num_samples)
    rsi = (
        50
        + 30 * np.sin(np.linspace(0, 20, num_samples))
        + np.random.normal(0, 5, num_samples)
    )
    macd = np.random.normal(0, 2, num_samples)
    bollinger_upper = prices * 1.05
    bollinger_lower = prices * 0.95

    # Technical indicators
    sma_20 = pd.Series(prices).rolling(20).mean().fillna(method="bfill")
    ema_12 = pd.Series(prices).ewm(span=12).mean()

    # Market sentiment features
    sentiment = np.random.uniform(-1, 1, num_samples)
    vix = 20 + 10 * np.random.random(num_samples)

    # Price change (target variable)
    price_change = np.diff(prices, prepend=prices[0])
    next_price = np.append(prices[1:], prices[-1] * 1.01)  # Simple future price

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["TSLA"] * num_samples,
            "price": prices,
            "volume": volumes,
            "rsi": rsi,
            "macd": macd,
            "bollinger_upper": bollinger_upper,
            "bollinger_lower": bollinger_lower,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "sentiment": sentiment,
            "vix": vix,
            "price_change": price_change,
            "next_price": next_price,
            # Additional features for better ML training
            "high": prices * (1 + np.random.uniform(0, 0.05, num_samples)),
            "low": prices * (1 - np.random.uniform(0, 0.05, num_samples)),
            "open": prices + np.random.normal(0, 1, num_samples),
            "close": prices,
        }
    )

    # Add some derived features
    data["price_volatility"] = data["high"] - data["low"]
    data["volume_ma"] = data["volume"].rolling(10).mean().fillna(method="bfill")

    print(f"✅ Generated data with {len(data)} rows and {len(data.columns)} features")
    print(
        f"   Target variable range: ${data['next_price'].min():.2f} - ${data['next_price'].max():.2f}"
    )

    return data


async def demonstrate_single_model_training():
    """Demonstrate training a single model with proper data"""
    print("\n" + "=" * 70)
    print("🧠 SINGLE MODEL TRAINING DEMONSTRATION")
    print("=" * 70)

    # Initialize manager
    manager = LocalMLManager()
    await manager.initialize()

    # Generate training data
    data = generate_simple_test_data(500)

    # Prepare features and target
    feature_columns = [
        "price",
        "volume",
        "rsi",
        "macd",
        "sentiment",
        "vix",
        "price_volatility",
        "high",
        "low",
        "open",
        "close",
    ]

    X = data[feature_columns].values
    y = data["next_price"].values

    print(f"🔧 Training data shape: X={X.shape}, y={y.shape}")

    # Test XGBoost model
    print("\n📊 Training XGBoost Model...")

    try:
        from buffetbot.ml.local_ml.models import XGBoostModel

        model = XGBoostModel(
            feature_columns=feature_columns, target_column="next_price"
        )

        # Train the model
        model.train(X, y)

        # Make predictions
        predictions = model.predict(X[:10])
        actual = y[:10]

        # Calculate metrics
        metrics = model.get_performance_metrics(X, y)

        print(f"✅ XGBoost Training Complete!")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")

        print("\n📈 Sample Predictions vs Actual:")
        for i in range(5):
            print(f"   Prediction: ${predictions[i]:.2f}, Actual: ${actual[i]:.2f}")

        # Test feature importance
        importance = model.get_feature_importance()
        print("\n🔍 Top 5 Feature Importances:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:5]:
            print(f"   {feature}: {score:.4f}")

        return True

    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
        return False

    finally:
        await manager.cleanup()


async def demonstrate_prediction_service():
    """Demonstrate the prediction service capabilities"""
    print("\n" + "=" * 70)
    print("🔮 PREDICTION SERVICE DEMONSTRATION")
    print("=" * 70)

    # Initialize components
    manager = LocalMLManager()
    await manager.initialize()

    # Generate data and train a simple model
    data = generate_simple_test_data(300)
    feature_columns = ["price", "volume", "rsi", "sentiment", "vix"]

    try:
        from buffetbot.ml.local_ml.models import LinearRegressionModel

        # Train a simple model
        model = LinearRegressionModel(
            feature_columns=feature_columns, target_column="next_price"
        )
        X = data[feature_columns].values
        y = data["next_price"].values
        model.train(X, y)

        # Initialize prediction service
        prediction_service = PredictionService()

        # Register the model
        model_id = await prediction_service.register_model(model, "test_predictor")
        print(f"✅ Model registered with ID: {model_id}")

        # Test single prediction
        test_features = data[feature_columns].iloc[0].to_dict()
        prediction = await prediction_service.predict_single(model_id, test_features)

        print(f"\n📊 Single Prediction Test:")
        print(f"   Input features: {test_features}")
        print(f"   Predicted price: ${prediction['prediction']:.2f}")
        print(
            f"   Confidence interval: ${prediction['confidence_interval']['lower']:.2f} - ${prediction['confidence_interval']['upper']:.2f}"
        )

        # Test batch prediction
        batch_features = data[feature_columns].iloc[:5].to_dict("records")
        batch_predictions = await prediction_service.predict_batch(
            model_id, batch_features
        )

        print(f"\n📈 Batch Prediction Test (5 samples):")
        for i, pred in enumerate(batch_predictions):
            print(f"   Sample {i+1}: ${pred['prediction']:.2f}")

        # Test caching
        print(f"\n💾 Testing Prediction Caching...")
        start_time = datetime.now()
        cached_prediction = await prediction_service.predict_single(
            model_id, test_features
        )
        cache_time = (datetime.now() - start_time).total_seconds()
        print(f"   Cached prediction retrieved in {cache_time*1000:.2f}ms")
        print(f"   Cache hit: {cached_prediction == prediction}")

        # Get statistics
        stats = prediction_service.get_prediction_statistics(model_id)
        print(f"\n📊 Prediction Service Statistics:")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Average response time: {stats['avg_response_time_ms']:.2f}ms")

        return True

    except Exception as e:
        print(f"❌ Prediction service demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await manager.cleanup()


async def demonstrate_cost_monitoring():
    """Demonstrate cost monitoring capabilities"""
    print("\n" + "=" * 70)
    print("💰 COST MONITORING DEMONSTRATION")
    print("=" * 70)

    manager = LocalMLManager()
    await manager.initialize()

    try:
        # Simulate some ML operations with costs
        manager.track_cost(0.0)  # Training cost (free locally)
        manager.track_cost(0.0)  # Prediction cost (free locally)
        manager.track_cost(0.0)  # Feature engineering cost (free locally)

        # Get cost summary
        cost_summary = manager.get_cost_summary()
        print(f"📊 Cost Summary:")
        print(f"   Service: {cost_summary['service']}")
        print(f"   Total cost: ${cost_summary['total_cost']:.4f}")
        print(
            f"   Session duration: {cost_summary['session_duration_hours']:.2f} hours"
        )
        print(f"   Cost per hour: ${cost_summary['cost_per_hour']:.4f}")
        print(f"   Daily limit: ${cost_summary['cost_limit']:.2f}")

        print(f"\n💡 Local ML Benefits:")
        print(f"   ✅ Zero cloud costs")
        print(f"   ✅ Unlimited predictions")
        print(f"   ✅ No API rate limits")
        print(f"   ✅ Complete data privacy")
        print(f"   ✅ Instant availability")

        return True

    except Exception as e:
        print(f"❌ Cost monitoring demo failed: {e}")
        return False

    finally:
        await manager.cleanup()


async def main():
    """Run the simplified enhanced ML demonstration"""
    print("🚀 BUFFETBOT ENHANCED ML DEMONSTRATION")
    print("=====================================")
    print("Phase 3 Task 2: Enhanced Local ML Models")
    print("Cost: $0.00 (100% local computation)")
    print("Time: ~30 seconds")
    print()

    # Track demo results
    results = []

    # Single model training
    print("1️⃣  Testing single model training...")
    result1 = await demonstrate_single_model_training()
    results.append(("Single Model Training", result1))

    # Prediction service
    print("\n2️⃣  Testing prediction service...")
    result2 = await demonstrate_prediction_service()
    results.append(("Prediction Service", result2))

    # Cost monitoring
    print("\n3️⃣  Testing cost monitoring...")
    result3 = await demonstrate_cost_monitoring()
    results.append(("Cost Monitoring", result3))

    # Summary
    print("\n" + "=" * 70)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False

    print(
        f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_passed:
        print("\n🎉 Enhanced ML System Successfully Demonstrated!")
        print("   ✅ Local ML models working")
        print("   ✅ Prediction service operational")
        print("   ✅ Cost monitoring active")
        print("   ✅ Zero cloud costs maintained")
        print()
        print("🚀 Ready for Phase 3 Task 3: Feature Engineering Pipeline")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
