"""
Example: Using BuffetBot GCS Storage System

This example demonstrates how to use the GCS storage integration for storing
and retrieving market data, forecasts, and other BuffetBot data types.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from buffetbot.config.gcs_config import get_config

# Import BuffetBot storage components
from buffetbot.storage import GCSStorageManager, SchemaManager, ValidationResult
from buffetbot.storage.utils.config import GCSConfig
from buffetbot.storage.utils.monitoring import StorageMetrics
from buffetbot.storage.utils.security import SecurityContext, SecurityManager


def create_sample_market_data() -> list[dict[str, Any]]:
    """Create sample market data for testing"""
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "price": 150.25,
            "volume": 1000000,
            "market_cap": 2500000000000,
            "pe_ratio": 25.5,
            "eps": 5.89,
            "dividend_yield": 0.5,
            "rsi_14d": 65.2,
            "sma_20d": 148.75,
            "volatility_30d": 0.25,
            "beta": 1.2,
            "data_source": "yahoo_finance",
            "created_at": datetime.now(timezone.utc),
            "version": "v1.2.0",
        },
        {
            "symbol": "MSFT",
            "timestamp": datetime.now(timezone.utc),
            "price": 305.50,
            "volume": 800000,
            "market_cap": 2300000000000,
            "pe_ratio": 28.1,
            "eps": 10.88,
            "dividend_yield": 0.7,
            "rsi_14d": 58.7,
            "sma_20d": 302.15,
            "volatility_30d": 0.22,
            "beta": 0.9,
            "data_source": "yahoo_finance",
            "created_at": datetime.now(timezone.utc),
            "version": "v1.2.0",
        },
    ]


def create_sample_forecast_data() -> list[dict[str, Any]]:
    """Create sample forecast data for testing"""
    return [
        {
            "symbol": "AAPL",
            "model_name": "lstm_v2",
            "model_version": "2.1.0",
            "forecast_horizon_days": 30,
            "ensemble_method": "weighted_average",
            "predicted_price": 155.75,
            "confidence_interval_lower": 145.20,
            "confidence_interval_upper": 166.30,
            "confidence_score": 0.85,
            "ensemble_weights": '{"lstm": 0.4, "transformer": 0.3, "xgboost": 0.3}',
            "component_predictions": '{"lstm": 154.20, "transformer": 157.80, "xgboost": 155.10}',
            "model_accuracy": 0.78,
            "feature_importance": '{"price_ma_20": 0.25, "volume": 0.15, "rsi": 0.12}',
            "training_data_end_date": datetime.now(timezone.utc).date(),
            "prediction_date": datetime.now(timezone.utc).date(),
            "created_at": datetime.now(timezone.utc),
            "expires_at": None,
        }
    ]


async def demonstrate_schema_validation():
    """Demonstrate schema validation capabilities"""
    logger.info("=== Schema Validation Demo ===")

    # Initialize schema manager
    schema_manager = SchemaManager()

    # Test market data validation
    market_data = create_sample_market_data()

    logger.info("Validating market data...")
    validation_result = schema_manager.validate_data(
        data=market_data, data_type="market_data", version="latest"
    )

    if validation_result.is_valid:
        logger.info(
            f"✅ Market data validation passed in {validation_result.validation_duration_ms}ms"
        )
    else:
        logger.error(
            f"❌ Market data validation failed: {len(validation_result.errors)} errors"
        )
        for error in validation_result.errors:
            logger.error(f"  - {error.field}: {error.message}")

    # Test forecast data validation
    forecast_data = create_sample_forecast_data()

    logger.info("Validating forecast data...")
    validation_result = schema_manager.validate_data(
        data=forecast_data, data_type="forecasts", version="latest"
    )

    if validation_result.is_valid:
        logger.info(
            f"✅ Forecast data validation passed in {validation_result.validation_duration_ms}ms"
        )
    else:
        logger.error(
            f"❌ Forecast data validation failed: {len(validation_result.errors)} errors"
        )
        for error in validation_result.errors:
            logger.error(f"  - {error.field}: {error.message}")


def demonstrate_security_features():
    """Demonstrate security management features"""
    logger.info("=== Security Features Demo ===")

    # Initialize security manager
    security_manager = SecurityManager()

    # Test data encryption
    sensitive_data = "user_portfolio_data_12345"
    logger.info("Testing data encryption...")

    encrypted_data = security_manager.encrypt_data(sensitive_data)
    logger.info(f"✅ Data encrypted: {encrypted_data[:50]}...")

    decrypted_data = security_manager.decrypt_data(encrypted_data)
    logger.info(f"✅ Data decrypted: {decrypted_data}")

    # Test access control
    security_context = SecurityContext(
        user_id="analyst_001",
        roles=["analyst", "user"],
        permissions=["read_market_data", "read_forecasts"],
        session_id="session_12345",
        ip_address="192.168.1.100",
    )

    # Test access to market data (should be allowed)
    has_access = security_manager.check_access(
        context=security_context, resource="market_data/AAPL", operation="read"
    )
    logger.info(f"✅ Market data read access: {'Granted' if has_access else 'Denied'}")

    # Test access to admin functions (should be denied)
    has_access = security_manager.check_access(
        context=security_context, resource="admin/delete_all", operation="delete"
    )
    logger.info(f"✅ Admin delete access: {'Granted' if has_access else 'Denied'}")


def demonstrate_monitoring():
    """Demonstrate monitoring and metrics collection"""
    logger.info("=== Monitoring Demo ===")

    # Initialize metrics
    metrics = StorageMetrics()

    # Simulate some operations
    logger.info("Simulating storage operations...")

    # Record upload metrics
    metrics.record_upload(
        data_type="market_data",
        file_size=1024000,  # 1MB
        duration_ms=250,
        success=True,
        bucket="primary",
    )

    metrics.record_upload(
        data_type="forecasts",
        file_size=512000,  # 512KB
        duration_ms=180,
        success=True,
        bucket="primary",
    )

    # Record query metrics
    metrics.record_query(
        data_type="market_data",
        duration_ms=45,
        records_returned=1000,
        cache_hit=False,
        partitions_scanned=3,
    )

    metrics.record_query(
        data_type="market_data", duration_ms=12, records_returned=1000, cache_hit=True
    )

    # Get performance summary
    performance = metrics.get_performance_summary()
    logger.info("Performance Summary:")
    for operation, perf in performance.items():
        logger.info(f"  {operation}:")
        logger.info(f"    Operations: {perf.operation_count}")
        logger.info(f"    Avg Duration: {perf.avg_duration_ms:.1f}ms")
        logger.info(f"    Success Rate: {perf.success_rate:.1%}")

    # Get cache metrics
    cache_metrics = metrics.get_cache_metrics()
    logger.info(f"Cache Hit Rate: {cache_metrics['cache_hit_rate']:.1%}")

    # Get throughput metrics
    throughput = metrics.get_throughput_metrics()
    logger.info(f"Operations per hour: {throughput['total_operations_per_hour']}")


async def demonstrate_full_workflow():
    """Demonstrate a complete storage workflow"""
    logger.info("=== Full Workflow Demo ===")

    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration for project: {config.project_id}")

        # Note: This would normally initialize the actual GCS storage manager
        # For this demo, we'll just show the configuration
        logger.info("Configuration loaded successfully:")
        logger.info(f"  Data Bucket: {config.data_bucket}")
        logger.info(f"  Archive Bucket: {config.archive_bucket}")
        logger.info(f"  Region: {config.region}")
        logger.info(f"  Compression: {config.compression}")

        # In a real implementation, you would:
        # 1. Initialize GCSStorageManager with the config
        # 2. Store data using storage_manager.store_data()
        # 3. Retrieve data using storage_manager.retrieve_data()
        # 4. Monitor performance with metrics

        logger.info("✅ Full workflow demonstration completed")

    except Exception as e:
        logger.error(f"❌ Workflow error: {str(e)}")


async def main():
    """Main demonstration function"""
    logger.info("🚀 BuffetBot GCS Storage System Demo")
    logger.info("=" * 50)

    # Run all demonstrations
    await demonstrate_schema_validation()
    print()

    demonstrate_security_features()
    print()

    demonstrate_monitoring()
    print()

    await demonstrate_full_workflow()

    logger.info("=" * 50)
    logger.info("✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
