"""
Options Data Schema Definitions

Apache Arrow schemas for options chain data, Greeks, and implied volatility.
"""

from typing import Dict

import pyarrow as pa

# Options Data Schema v1.0.0 - Initial release
OPTIONS_DATA_V1_0_0 = pa.schema(
    [
        # Contract identifiers
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_symbol", pa.string(), nullable=False),
        pa.field("expiration_date", pa.date32(), nullable=False),
        pa.field("strike_price", pa.decimal128(precision=10, scale=2), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),  # 'call' or 'put'
        # Pricing data
        pa.field("bid", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("ask", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("last_price", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("volume", pa.int32(), nullable=True),
        pa.field("open_interest", pa.int32(), nullable=True),
        # Greeks
        pa.field("delta", pa.float64(), nullable=True),
        pa.field("gamma", pa.float64(), nullable=True),
        pa.field("theta", pa.float64(), nullable=True),
        pa.field("vega", pa.float64(), nullable=True),
        pa.field("rho", pa.float64(), nullable=True),
        pa.field("implied_volatility", pa.float64(), nullable=True),
        # Metadata
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Options Data Schema v1.1.0 - Added more pricing metrics
OPTIONS_DATA_V1_1_0 = pa.schema(
    [
        # Contract identifiers
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_symbol", pa.string(), nullable=False),
        pa.field("expiration_date", pa.date32(), nullable=False),
        pa.field("strike_price", pa.decimal128(precision=10, scale=2), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),  # 'call' or 'put'
        # Pricing data
        pa.field("bid", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("ask", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("last_price", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field(
            "mark_price", pa.decimal128(precision=8, scale=4), nullable=True
        ),  # New field
        pa.field("volume", pa.int32(), nullable=True),
        pa.field("open_interest", pa.int32(), nullable=True),
        pa.field("bid_size", pa.int32(), nullable=True),  # New field
        pa.field("ask_size", pa.int32(), nullable=True),  # New field
        # Derived metrics
        pa.field(
            "bid_ask_spread", pa.decimal128(precision=8, scale=4), nullable=True
        ),  # New field
        pa.field(
            "intrinsic_value", pa.decimal128(precision=8, scale=4), nullable=True
        ),  # New field
        pa.field(
            "time_value", pa.decimal128(precision=8, scale=4), nullable=True
        ),  # New field
        # Greeks
        pa.field("delta", pa.float64(), nullable=True),
        pa.field("gamma", pa.float64(), nullable=True),
        pa.field("theta", pa.float64(), nullable=True),
        pa.field("vega", pa.float64(), nullable=True),
        pa.field("rho", pa.float64(), nullable=True),
        pa.field("implied_volatility", pa.float64(), nullable=True),
        # Metadata
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Volatility Surface Schema - For implied volatility analysis
VOLATILITY_SURFACE_SCHEMA = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("expiration_date", pa.date32(), nullable=False),
        pa.field("strike_price", pa.decimal128(precision=10, scale=2), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),
        # Volatility metrics
        pa.field("implied_volatility", pa.float64(), nullable=False),
        pa.field("historical_volatility", pa.float64(), nullable=True),
        pa.field("volatility_skew", pa.float64(), nullable=True),
        pa.field("volatility_smile", pa.float64(), nullable=True),
        # Time to expiration
        pa.field("days_to_expiration", pa.int32(), nullable=False),
        pa.field("time_to_expiration_years", pa.float64(), nullable=False),
        # Market context
        pa.field(
            "underlying_price", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field("risk_free_rate", pa.float64(), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        # Metadata
        pa.field("snapshot_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Options Chain Summary Schema - Aggregated metrics per expiration
OPTIONS_CHAIN_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("expiration_date", pa.date32(), nullable=False),
        # Call metrics
        pa.field("total_call_volume", pa.int64(), nullable=True),
        pa.field("total_call_open_interest", pa.int64(), nullable=True),
        pa.field("call_put_ratio", pa.float64(), nullable=True),
        # Put metrics
        pa.field("total_put_volume", pa.int64(), nullable=True),
        pa.field("total_put_open_interest", pa.int64(), nullable=True),
        # Volatility metrics
        pa.field("weighted_implied_volatility", pa.float64(), nullable=True),
        pa.field("volatility_skew", pa.float64(), nullable=True),
        # Max pain and key levels
        pa.field(
            "max_pain_strike", pa.decimal128(precision=10, scale=2), nullable=True
        ),
        pa.field("gamma_exposure", pa.float64(), nullable=True),
        # Market context
        pa.field(
            "underlying_price", pa.decimal128(precision=10, scale=4), nullable=False
        ),
        pa.field("days_to_expiration", pa.int32(), nullable=False),
        # Metadata
        pa.field("snapshot_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Schema version mapping
OPTIONS_DATA_SCHEMAS: dict[str, pa.Schema] = {
    "v1.0.0": OPTIONS_DATA_V1_0_0,
    "v1.1.0": OPTIONS_DATA_V1_1_0,
}

VOLATILITY_SURFACE_SCHEMAS: dict[str, pa.Schema] = {"v1.0.0": VOLATILITY_SURFACE_SCHEMA}

OPTIONS_CHAIN_SUMMARY_SCHEMAS: dict[str, pa.Schema] = {
    "v1.0.0": OPTIONS_CHAIN_SUMMARY_SCHEMA
}

# Default to latest version
OPTIONS_DATA_SCHEMA_LATEST = OPTIONS_DATA_V1_1_0
LATEST_VERSION = "v1.1.0"


def get_options_data_schema(version: str = "latest") -> pa.Schema:
    """Get options data schema by version"""
    if version == "latest":
        return OPTIONS_DATA_SCHEMA_LATEST

    if version not in OPTIONS_DATA_SCHEMAS:
        available_versions = list(OPTIONS_DATA_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return OPTIONS_DATA_SCHEMAS[version]


def get_volatility_surface_schema(version: str = "latest") -> pa.Schema:
    """Get volatility surface schema by version"""
    if version == "latest":
        return VOLATILITY_SURFACE_SCHEMA

    if version not in VOLATILITY_SURFACE_SCHEMAS:
        available_versions = list(VOLATILITY_SURFACE_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return VOLATILITY_SURFACE_SCHEMAS[version]


def get_options_chain_summary_schema(version: str = "latest") -> pa.Schema:
    """Get options chain summary schema by version"""
    if version == "latest":
        return OPTIONS_CHAIN_SUMMARY_SCHEMA

    if version not in OPTIONS_CHAIN_SUMMARY_SCHEMAS:
        available_versions = list(OPTIONS_CHAIN_SUMMARY_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return OPTIONS_CHAIN_SUMMARY_SCHEMAS[version]


def get_available_versions() -> list:
    """Get list of available options data schema versions"""
    return list(OPTIONS_DATA_SCHEMAS.keys())
