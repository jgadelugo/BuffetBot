#!/usr/bin/env python3
"""
BuffetBot ML Foundation Demo

Demonstrates the complete ML foundation system implemented in Phase 3 Task 1.
This runs completely FREE with zero cloud costs.

Usage:
    python examples/ml_foundation_demo.py
"""

import asyncio
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from buffetbot.ml.managers.ml_manager import MLManager
from buffetbot.ml.models.metadata import (
    ModelMetadata,
    ModelMetrics,
    ModelStatus,
    ModelType,
)
from buffetbot.ml.utils.data_prep import DataPreprocessor
from buffetbot.ml.utils.validation import MLValidator


async def main():
    """Demo the ML Foundation system"""

    print("🚀 BuffetBot ML Foundation Demo")
    print("=" * 50)
    print("💰 Cost Status: FREE - No cloud charges!")
    print("🧠 Using: Local ML with scikit-learn")
    print("=" * 50)

    # Initialize ML Manager
    print("\n1. Initializing ML Manager...")
    ml_manager = MLManager()
    success = await ml_manager.initialize()

    if not success:
        print("❌ Failed to initialize ML Manager")
        return

    print("✅ ML Manager initialized successfully")

    # Create sample market data
    print("\n2. Creating sample market data...")
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)  # For reproducible results

    # Simulate stock price with trend and noise
    price_trend = np.cumsum(np.random.randn(200) * 0.02) + 100
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price": price_trend,
            "volume": np.random.randint(1000, 50000, 200),
            "open": price_trend + np.random.randn(200) * 0.5,
            "high": price_trend + abs(np.random.randn(200)) * 0.8,
            "low": price_trend - abs(np.random.randn(200)) * 0.8,
            "close": price_trend + np.random.randn(200) * 0.3,
        }
    )

    print(f"✅ Created {len(df)} days of market data")
    print(
        f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )
    print(f"   Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")

    # Validate the data
    print("\n3. Validating market data...")
    validator = MLValidator()
    validation_result = validator.validate_time_series_data(df, "timestamp", "price")

    if validation_result["valid"]:
        print("✅ Data validation passed")
        print(f"   Shape: {validation_result['stats']['shape']}")
        print(f"   Date range: {validation_result['stats']['date_range_days']} days")
    else:
        print("❌ Data validation failed:")
        for issue in validation_result["issues"]:
            print(f"   - {issue}")

    # Preprocess the data
    print("\n4. Preprocessing data for ML...")
    preprocessor = DataPreprocessor()

    # Create features
    df_processed = preprocessor.prepare_time_series_data(
        df, target_col="price", datetime_col="timestamp", prediction_horizon=1
    )

    print(f"✅ Feature engineering complete")
    print(f"   Original features: {df.shape[1]}")
    print(f"   Engineered features: {df_processed.shape[1]}")

    # Prepare features and target
    feature_cols = [
        col
        for col in df_processed.columns
        if not col.startswith("price_future") and col != "timestamp"
    ]

    # Use only rows where we have complete data
    valid_rows = df_processed.dropna()
    X = valid_rows[feature_cols]
    y = valid_rows["price_future_1"]

    print(f"   Training samples: {len(X)}")
    print(f"   Features used: {len(feature_cols)}")

    # Train a simple model
    print("\n5. Training ML model (FREE - local computation)...")
    model = LinearRegression()

    # Split data for training/testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Track training cost (free for local)
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model training complete")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Training cost: $0.00 (local)")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MSE: {mse:.4f}")

    # Register the model
    print("\n6. Registering model in ML Registry...")

    metrics = ModelMetrics(
        mse=mse, r2_score=r2, custom_metrics={"training_time_seconds": training_time}
    )

    metadata = ModelMetadata(
        model_id="price_predictor_v1",
        model_name="price_predictor",
        version="1.0",
        model_type=ModelType.SKLEARN,
        status=ModelStatus.TRAINED,
        training_date=datetime.now(),
        training_duration_minutes=int(training_time / 60),
        training_data_size=len(X_train),
        feature_columns=feature_cols,
        target_column="price_future_1",
        metrics=metrics,
        file_path="models/price_predictor/1.0/model.pkl",
        model_hash="",
        cost_to_train=0.0,  # Free!
        tags=["demo", "price_prediction", "sklearn"],
        description="Demo price prediction model using linear regression",
    )

    model_id = await ml_manager.model_registry.register_model(
        model, metadata, save_to_gcs=False
    )

    print(f"✅ Model registered with ID: {model_id}")

    # Deploy the model
    print("\n7. Deploying model...")
    await ml_manager.model_registry.update_model_status(model_id, ModelStatus.DEPLOYED)
    print("✅ Model deployed successfully")

    # Make some predictions
    print("\n8. Making predictions...")
    for i in range(5):
        sample_features = X_test.iloc[i : i + 1]
        prediction = model.predict(sample_features)[0]
        actual = y_test.iloc[i]

        # Record prediction in registry
        await ml_manager.model_registry.record_prediction(model_id)

        # Track prediction cost (free for local)
        ml_manager.cost_monitor.track_cost("local_ml", "prediction", 0.0)

        print(f"   Prediction {i+1}: ${prediction:.2f} (Actual: ${actual:.2f})")

    print("✅ Predictions complete")

    # Show system status
    print("\n9. ML System Status...")
    status = await ml_manager.get_system_status()

    print(f"✅ ML System Status:")
    print(f"   Initialized: {status['initialized']}")
    print(f"   Total models: {status['model_registry']['total_models']}")
    print(f"   Total cost: ${status['cost_summary']['total_cost']:.2f}")
    print(f"   Cache available: {status['integrations']['cache']}")

    # Show cost summary
    cost_summary = ml_manager.cost_monitor.get_cost_summary()
    print(f"\n💰 Cost Summary:")
    print(f"   Total operations: {cost_summary['total_operations']}")
    print(f"   Daily cost: ${cost_summary['daily_cost']:.2f}")
    print(f"   Monthly cost: ${cost_summary['monthly_cost']:.2f}")
    print(f"   Service breakdown: {cost_summary['service_breakdown']}")

    # Performance monitoring
    print(f"\n📊 Performance Monitoring:")
    ml_manager.cost_monitor.track_cost(
        "performance_monitoring", "model_evaluation", 0.0
    )

    # Health check
    print("\n🔍 Health Check...")
    health = await ml_manager.health_check()
    print(f"   Overall health: {health['overall']}")
    print(f"   ML Manager: {health['components'].get('ml_manager', 'unknown')}")
    print(f"   Model Registry: {health['components'].get('model_registry', 'unknown')}")
    print(f"   Cost Monitor: {health['components'].get('cost_monitor', 'unknown')}")

    # Cleanup
    print("\n10. Cleaning up...")
    await ml_manager.cleanup()
    print("✅ Cleanup complete")

    print("\n" + "=" * 50)
    print("🎉 ML Foundation Demo Complete!")
    print("💡 Key Achievements:")
    print("   ✅ ML Manager fully operational")
    print("   ✅ Model trained and deployed")
    print("   ✅ Predictions generated")
    print("   ✅ Cost monitoring active")
    print("   ✅ Zero cloud costs incurred")
    print("   ✅ Full Phase 1 & 2 integration")
    print("\n🚀 Ready for Phase 3 Task 2 (Local ML Models)!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
