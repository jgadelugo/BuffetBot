"""
Forecast Data Schema Definitions

Apache Arrow schemas for ML model forecasts and predictions.
"""

from typing import Dict

import pyarrow as pa

# Forecast Schema v1.0.0 - Initial release
FORECAST_V1_0_0 = pa.schema(
    [
        # Prediction identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("forecast_horizon_days", pa.int32(), nullable=False),
        # Predictions
        pa.field(
            "predicted_price", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field("confidence_score", pa.float64(), nullable=True),
        # Metadata
        pa.field("prediction_date", pa.date32(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Forecast Schema v2.0.0 - Added confidence intervals and model metrics
FORECAST_V2_0_0 = pa.schema(
    [
        # Prediction identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("forecast_horizon_days", pa.int32(), nullable=False),
        # Predictions
        pa.field(
            "predicted_price", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field(
            "confidence_interval_lower",
            pa.decimal128(precision=10, scale=4),
            nullable=True,
        ),
        pa.field(
            "confidence_interval_upper",
            pa.decimal128(precision=10, scale=4),
            nullable=True,
        ),
        pa.field("confidence_score", pa.float64(), nullable=True),
        # Model metrics
        pa.field("model_accuracy", pa.float64(), nullable=True),
        pa.field("feature_importance", pa.string(), nullable=True),  # JSON string
        pa.field("training_data_end_date", pa.date32(), nullable=False),
        # Metadata
        pa.field("prediction_date", pa.date32(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("expires_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)

# Forecast Schema v2.1.0 - Added ensemble predictions
FORECAST_V2_1_0 = pa.schema(
    [
        # Prediction identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("forecast_horizon_days", pa.int32(), nullable=False),
        pa.field("ensemble_method", pa.string(), nullable=True),  # New field
        # Predictions
        pa.field(
            "predicted_price", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field(
            "confidence_interval_lower",
            pa.decimal128(precision=10, scale=4),
            nullable=True,
        ),
        pa.field(
            "confidence_interval_upper",
            pa.decimal128(precision=10, scale=4),
            nullable=True,
        ),
        pa.field("confidence_score", pa.float64(), nullable=True),
        # Ensemble-specific fields
        pa.field("ensemble_weights", pa.string(), nullable=True),  # JSON string
        pa.field("component_predictions", pa.string(), nullable=True),  # JSON string
        # Model metrics
        pa.field("model_accuracy", pa.float64(), nullable=True),
        pa.field("feature_importance", pa.string(), nullable=True),  # JSON string
        pa.field("training_data_end_date", pa.date32(), nullable=False),
        # Metadata
        pa.field("prediction_date", pa.date32(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("expires_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)

# Model Performance Schema - For tracking model accuracy over time
MODEL_PERFORMANCE_SCHEMA = pa.schema(
    [
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("evaluation_date", pa.date32(), nullable=False),
        pa.field("forecast_horizon_days", pa.int32(), nullable=False),
        # Performance metrics
        pa.field("mae", pa.float64(), nullable=True),  # Mean Absolute Error
        pa.field("mse", pa.float64(), nullable=True),  # Mean Squared Error
        pa.field("rmse", pa.float64(), nullable=True),  # Root Mean Squared Error
        pa.field("mape", pa.float64(), nullable=True),  # Mean Absolute Percentage Error
        pa.field("directional_accuracy", pa.float64(), nullable=True),
        # Prediction vs actual
        pa.field(
            "predicted_value", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field("actual_value", pa.decimal128(precision=10, scale=4), nullable=False),
        pa.field(
            "prediction_error", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        # Metadata
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Schema version mapping
FORECAST_SCHEMAS: dict[str, pa.Schema] = {
    "v1.0.0": FORECAST_V1_0_0,
    "v2.0.0": FORECAST_V2_0_0,
    "v2.1.0": FORECAST_V2_1_0,
}

# Model performance tracking
MODEL_PERFORMANCE_SCHEMAS: dict[str, pa.Schema] = {"v1.0.0": MODEL_PERFORMANCE_SCHEMA}

# Default to latest version
FORECAST_SCHEMA_LATEST = FORECAST_V2_1_0
LATEST_VERSION = "v2.1.0"


def get_forecast_schema(version: str = "latest") -> pa.Schema:
    """Get forecast schema by version"""
    if version == "latest":
        return FORECAST_SCHEMA_LATEST

    if version not in FORECAST_SCHEMAS:
        available_versions = list(FORECAST_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return FORECAST_SCHEMAS[version]


def get_model_performance_schema(version: str = "latest") -> pa.Schema:
    """Get model performance schema by version"""
    if version == "latest":
        return MODEL_PERFORMANCE_SCHEMA

    if version not in MODEL_PERFORMANCE_SCHEMAS:
        available_versions = list(MODEL_PERFORMANCE_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return MODEL_PERFORMANCE_SCHEMAS[version]


def get_available_versions() -> list:
    """Get list of available forecast schema versions"""
    return list(FORECAST_SCHEMAS.keys())
