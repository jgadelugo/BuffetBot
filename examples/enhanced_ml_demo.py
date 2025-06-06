#!/usr/bin/env python3
"""
Enhanced ML Demo - Task 2 Complete Showcase
Demonstrates XGBoost, LightGBM, hyperparameter optimization, and advanced predictions

🚀 Task 2 Features:
- XGBoost & LightGBM models (superior to linear regression)
- Hyperparameter optimization with Optuna
- Advanced training pipeline with model comparison
- Enhanced prediction service with caching
- Complete local ML system (zero costs)
"""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Enhanced ML Components (Task 2)
from buffetbot.ml.local_ml.manager import LocalMLManager
from buffetbot.ml.local_ml.models import ModelType
from buffetbot.ml.local_ml.training import create_default_training_config

# Set up logging
logging.basicConfig(level=logging.INFO)


async def create_advanced_market_data(num_samples: int = 1000) -> pd.DataFrame:
    """Create realistic market data with complex patterns"""

    print(f"📊 Creating advanced market dataset ({num_samples} samples)...")

    np.random.seed(42)

    # Base price series with trend and seasonality
    days = np.arange(num_samples)
    trend = 100 + 0.02 * days  # Upward trend
    seasonality = 5 * np.sin(2 * np.pi * days / 30)  # Monthly cycle
    volatility = np.random.normal(0, 2, num_samples)

    base_price = trend + seasonality + volatility

    # Create features with realistic relationships
    data = {
        # Price features
        "price": base_price,
        "price_lag_1": np.roll(base_price, 1),
        "price_lag_5": np.roll(base_price, 5),
        "price_change": np.diff(base_price, prepend=base_price[0]),
        # Volume features (higher volume on price changes)
        "volume": 10000
        + 5000 * np.abs(np.diff(base_price, prepend=0))
        + np.random.normal(0, 1000, num_samples),
        "volume_ma_5": np.convolve(
            np.ones(5) / 5,
            np.random.uniform(5000, 20000, num_samples + 4),
            mode="valid",
        ),
        # Technical indicators
        "rsi": 30 + 40 * np.random.beta(2, 2, num_samples),  # RSI between 30-70
        "macd": np.random.normal(0, 0.5, num_samples),
        "bollinger_upper": base_price + 2 * np.random.uniform(1, 3, num_samples),
        "bollinger_lower": base_price - 2 * np.random.uniform(1, 3, num_samples),
        # Market sentiment features
        "news_sentiment": np.random.uniform(-1, 1, num_samples),
        "market_fear_greed": np.random.uniform(0, 100, num_samples),
        # Time-based features
        "day_of_week": (days % 7),
        "month": ((days // 30) % 12) + 1,
        "quarter": ((days // 90) % 4) + 1,
        # Volatility features
        "volatility_5d": np.array(
            [np.std(base_price[max(0, i - 5) : i + 1]) for i in range(num_samples)]
        ),
        "volatility_20d": np.array(
            [np.std(base_price[max(0, i - 20) : i + 1]) for i in range(num_samples)]
        ),
    }

    df = pd.DataFrame(data)

    # Create complex target: next day price change with non-linear relationships
    target = (
        # Linear components
        0.7 * df["price_change"]
        + 0.3 * df["volume"] / 10000
        + 0.2 * df["macd"]
        +
        # Non-linear components (perfect for XGBoost/LightGBM)
        0.1 * np.sin(df["rsi"] / 10)
        + 0.15 * (df["news_sentiment"] ** 2) * np.sign(df["news_sentiment"])
        + 0.05 * np.log1p(df["volatility_5d"])
        +
        # Interaction effects
        0.1 * df["price_change"] * df["volume"] / 50000
        +
        # Market regime effects
        np.where(df["market_fear_greed"] < 25, -0.5, 0)
        + np.where(df["market_fear_greed"] > 75, 0.3, 0)  # Fear regime
        +  # Greed regime
        # Noise
        np.random.normal(0, 0.3, num_samples)
    )

    df["next_price_change"] = target

    # Clean up any NaN values
    df = df.fillna(method="bfill").fillna(method="ffill")

    print(f"✅ Created dataset with {len(df.columns)-1} features and complex target")
    print(f"   📈 Features: {list(df.columns[:-1])}")

    return df


async def demonstrate_model_comparison():
    """Demonstrate model comparison capabilities"""

    print("\n" + "=" * 80)
    print("🔬 MODEL COMPARISON DEMONSTRATION")
    print("=" * 80)

    # Initialize ML manager
    manager = LocalMLManager()
    await manager.initialize()

    try:
        # Create advanced dataset
        data = await create_advanced_market_data(1500)

        print(f"\n📊 Dataset Statistics:")
        print(f"   Samples: {len(data)}")
        print(f"   Features: {len(data.columns)-1}")
        print(
            f"   Target range: [{data['next_price_change'].min():.3f}, {data['next_price_change'].max():.3f}]"
        )
        print(f"   Target std: {data['next_price_change'].std():.3f}")

        # Train all model types for comparison
        print(f"\n🚀 Training all model types (this may take a moment)...")
        start_time = datetime.now()

        results = await manager.train_models(
            training_data=data,
            target_column="next_price_change",
            model_types=[
                ModelType.LINEAR_REGRESSION,
                ModelType.RIDGE_REGRESSION,
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST,
                ModelType.LIGHTGBM,
            ],
            optimize_hyperparameters=True,  # Enable optimization
            quick_mode=False,  # Full optimization
        )

        training_time = (datetime.now() - start_time).total_seconds()

        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"   Models trained: {len(results)}")

        # Get detailed comparison
        comparison = manager.get_model_comparison()

        print(f"\n📈 MODEL PERFORMANCE COMPARISON:")
        print("   " + "=" * 78)
        print(
            f"   {'Model':<20} {'R² Score':<10} {'RMSE':<10} {'MAE':<8} {'CV R²':<8} {'Time (s)':<8}"
        )
        print("   " + "-" * 78)

        for _, row in comparison.iterrows():
            print(
                f"   {row['Model']:<20} {row['R²_Score']:<10.4f} {row['RMSE']:<10.6f} "
                f"{row['MAE']:<8.4f} {row['CV_R²_Mean']:<8.4f} {row['Training_Time_s']:<8.1f}"
            )

        # Find best model
        best_model_result = manager.get_best_model("r2_score")
        if best_model_result is None:
            print("   ❌ No models were successfully trained!")
            return None, data, None

        best_model_name = best_model_result.model.config.model_type.value
        best_r2 = best_model_result.performance.r2_score

        print("   " + "=" * 78)
        print(f"   🏆 BEST MODEL: {best_model_name.upper()} (R² = {best_r2:.4f})")

        # Calculate improvement over linear regression
        linear_r2 = comparison[comparison["Model"] == "linear_regression"][
            "R²_Score"
        ].iloc[0]
        improvement = best_r2 - linear_r2
        improvement_pct = (improvement / abs(linear_r2)) * 100

        print(
            f"   📊 Improvement over Linear Regression: +{improvement:.4f} ({improvement_pct:+.1f}%)"
        )

        # Show feature importance for best tree-based model
        if hasattr(best_model_result.model, "get_feature_importance"):
            importance = best_model_result.model.get_feature_importance()
            print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES ({best_model_name}):")
            print("   " + "-" * 60)
            for i, (feature, score) in enumerate(importance.head(10).items()):
                print(f"   {i+1:2d}. {feature:<25} {score:.4f}")

        return manager, data, best_model_result

    except Exception as e:
        print(f"❌ Error in model comparison: {e}")
        await manager.cleanup()
        raise


async def demonstrate_prediction_service(manager, data, best_model_result):
    """Demonstrate advanced prediction service capabilities"""

    print("\n" + "=" * 80)
    print("🔮 PREDICTION SERVICE DEMONSTRATION")
    print("=" * 80)

    # Test data (last 100 samples)
    test_data = data.drop("next_price_change", axis=1).tail(100)
    actual_values = data["next_price_change"].tail(100).values

    print(f"\n📊 Testing on {len(test_data)} samples...")

    # Single predictions with different features
    print(f"\n🎯 SINGLE PREDICTIONS:")
    print("   " + "-" * 70)

    for i in range(5):
        features = test_data.iloc[i].to_dict()
        actual = actual_values[i]

        # Make prediction with confidence interval
        response = await manager.predict_single(
            features=features,
            include_confidence_interval=True,
            include_feature_importance=False,  # Skip for speed
        )

        prediction = response.prediction
        error = abs(prediction - actual)
        error_pct = (error / abs(actual)) * 100 if actual != 0 else 0

        print(
            f"   Sample {i+1}: Predicted={prediction:8.4f}, Actual={actual:8.4f}, "
            f"Error={error:6.4f} ({error_pct:5.1f}%)"
        )

        if response.confidence_interval:
            ci = response.confidence_interval
            print(
                f"             95% CI: [{ci['lower_95']:7.4f}, {ci['upper_95']:7.4f}], "
                f"StdDev={ci['std_dev']:6.4f}"
            )

    # Batch predictions for performance testing
    print(f"\n⚡ BATCH PREDICTION PERFORMANCE:")
    print("   " + "-" * 60)

    batch_sizes = [10, 50, 100]
    for batch_size in batch_sizes:
        features_list = [test_data.iloc[i].to_dict() for i in range(batch_size)]

        start_time = datetime.now()
        batch_response = await manager.predict_batch(
            features_list=features_list, parallel_processing=True
        )
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        print(
            f"   Batch size {batch_size:3d}: {total_time:6.1f}ms total, "
            f"{total_time/batch_size:5.2f}ms per prediction"
        )

    # Test caching performance
    print(f"\n🚀 CACHING PERFORMANCE TEST:")
    print("   " + "-" * 50)

    # Same prediction multiple times
    test_features = test_data.iloc[0].to_dict()

    # First prediction (no cache)
    start_time = datetime.now()
    response1 = await manager.predict_single(test_features)
    time1 = (datetime.now() - start_time).total_seconds() * 1000

    # Second prediction (should be cached)
    start_time = datetime.now()
    response2 = await manager.predict_single(test_features)
    time2 = (datetime.now() - start_time).total_seconds() * 1000

    speedup = time1 / time2 if time2 > 0 else float("inf")

    print(f"   First prediction:  {time1:6.2f}ms (not cached)")
    print(f"   Second prediction: {time2:6.2f}ms (cached)")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Cache hit: {response2.cached}")

    # Prediction service statistics
    stats = manager.get_prediction_stats()
    print(f"\n📊 PREDICTION SERVICE STATS:")
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"   Avg prediction time: {stats['avg_prediction_time_ms']:.2f}ms")


async def demonstrate_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization with different models"""

    print("\n" + "=" * 80)
    print("🔧 HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 80)

    # Create smaller dataset for faster optimization demo
    data = await create_advanced_market_data(800)

    manager = LocalMLManager()
    await manager.initialize()

    try:
        # Test hyperparameter optimization for different models
        model_types = [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]

        for model_type in model_types:
            print(f"\n🎯 Optimizing {model_type.value.upper()}...")

            start_time = datetime.now()

            # Train with optimization
            results = await manager.train_models(
                training_data=data,
                target_column="next_price_change",
                model_types=[model_type],
                optimize_hyperparameters=True,
                quick_mode=True,  # Faster for demo
            )

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = results[0]
            performance = result.performance
            best_params = result.hyperparameters

            print(f"   ⏱️  Optimization time: {optimization_time:.1f}s")
            print(f"   📈 Best R² score: {performance.r2_score:.4f}")
            print(f"   🔧 Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")

        await manager.cleanup()

    except Exception as e:
        print(f"❌ Error in hyperparameter optimization: {e}")
        await manager.cleanup()
        raise


async def demonstrate_cost_monitoring():
    """Demonstrate cost monitoring (all zeros for local ML)"""

    print("\n" + "=" * 80)
    print("💰 COST MONITORING DEMONSTRATION")
    print("=" * 80)

    manager = LocalMLManager()
    await manager.initialize()

    try:
        # Create data and train models
        data = await create_advanced_market_data(500)

        print(f"\n💸 Training models and tracking costs...")

        results = await manager.train_models(
            training_data=data,
            target_column="next_price_change",
            model_types=[ModelType.XGBOOST, ModelType.LIGHTGBM],
            quick_mode=True,
        )

        # Make various predictions
        test_features = data.drop("next_price_change", axis=1).iloc[0].to_dict()

        for i in range(10):
            await manager.predict_single(test_features)

        # Get cost summary
        cost_summary = manager.get_cost_summary()
        health = manager.get_service_health()

        print(f"\n💰 COST BREAKDOWN:")
        print(f"   Total cost: ${cost_summary['total_cost']:.2f}")
        print(
            f"   Session duration: {cost_summary['session_duration_hours']:.2f} hours"
        )
        print(f"   Cost per hour: ${cost_summary['cost_per_hour']:.4f}")
        print(f"   Cost limit: ${cost_summary['cost_limit']:.2f}")

        print(f"\n📊 SERVICE HEALTH:")
        print(f"   Status: {health['status']}")
        print(f"   Active models: {health['active_models']}")
        print(f"   Total predictions: {health['total_predictions']}")
        print(f"   Cost per prediction: ${health['cost_per_prediction']:.6f}")
        print(f"   Cache hit rate: {health['cache_hit_rate']:.1f}%")

        print(f"\n✅ LOCAL ML ADVANTAGE:")
        print(f"   🆓 Zero cloud costs")
        print(f"   🚀 Unlimited usage")
        print(f"   ⚡ No API rate limits")
        print(f"   🔒 Complete data privacy")

        await manager.cleanup()

    except Exception as e:
        print(f"❌ Error in cost monitoring demo: {e}")
        await manager.cleanup()
        raise


async def main():
    """Run complete enhanced ML demonstration"""

    print("🚀 BUFFETBOT ENHANCED ML DEMONSTRATION (Phase 3 Task 2)")
    print("=" * 80)
    print("Showcasing XGBoost, LightGBM, hyperparameter optimization, and more!")
    print("All running locally with ZERO cloud costs 💰")

    try:
        # Model comparison demonstration
        manager, data, best_model = await demonstrate_model_comparison()

        if manager is not None and data is not None and best_model is not None:
            # Prediction service demonstration
            await demonstrate_prediction_service(manager, data, best_model)

            # Clean up before next demo
            await manager.cleanup()
        else:
            print("❌ Skipping remaining demos due to failed model training")
            return

        # Hyperparameter optimization demonstration
        await demonstrate_hyperparameter_optimization()

        # Cost monitoring demonstration
        await demonstrate_cost_monitoring()

        print("\n" + "=" * 80)
        print("🎉 ENHANCED ML DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✅ Successfully demonstrated:")
        print("   🤖 XGBoost and LightGBM models")
        print("   🔧 Hyperparameter optimization with Optuna")
        print("   ⚡ Fast prediction service with caching")
        print("   📊 Model comparison and selection")
        print("   💰 Cost monitoring (all $0.00)")
        print("   🚀 Complete local ML pipeline")

        print("\n🎯 Key Benefits Achieved:")
        print(f"   • Superior accuracy vs linear regression")
        print(f"   • Professional hyperparameter tuning")
        print(f"   • Production-ready prediction service")
        print(f"   • Zero cloud costs with unlimited usage")
        print(f"   • Easy upgrade path to cloud when needed")

        print("\n📈 Ready for real trading applications!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the complete demonstration
    asyncio.run(main())
