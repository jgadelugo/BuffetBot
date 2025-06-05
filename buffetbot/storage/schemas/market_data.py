"""
Market Data Schema Definitions

Apache Arrow schemas for market data storage with versioning support.
"""

from typing import Dict

import pyarrow as pa

# Market Data Schema v1.0.0 - Initial release
MARKET_DATA_V1_0_0 = pa.schema(
    [
        # Primary identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        # Price data
        pa.field("price", pa.decimal128(precision=10, scale=4), nullable=False),
        pa.field("volume", pa.int64(), nullable=True),
        pa.field("market_cap", pa.decimal128(precision=15, scale=2), nullable=True),
        # Financial metrics
        pa.field("pe_ratio", pa.float64(), nullable=True),
        pa.field("eps", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        # Technical indicators
        pa.field("rsi_14d", pa.float64(), nullable=True),
        pa.field("sma_20d", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field("volatility_30d", pa.float64(), nullable=True),
        # Metadata
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("version", pa.string(), nullable=False),
    ]
)

# Market Data Schema v1.1.0 - Added beta field
MARKET_DATA_V1_1_0 = pa.schema(
    [
        # Primary identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        # Price data
        pa.field("price", pa.decimal128(precision=10, scale=4), nullable=False),
        pa.field("volume", pa.int64(), nullable=True),
        pa.field("market_cap", pa.decimal128(precision=15, scale=2), nullable=True),
        # Financial metrics
        pa.field("pe_ratio", pa.float64(), nullable=True),
        pa.field("eps", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        # Technical indicators
        pa.field("rsi_14d", pa.float64(), nullable=True),
        pa.field("sma_20d", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field("volatility_30d", pa.float64(), nullable=True),
        pa.field("beta", pa.float64(), nullable=True),  # New field in v1.1.0
        # Metadata
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("version", pa.string(), nullable=False),
    ]
)

# Market Data Schema v1.2.0 - Added OHLC data
MARKET_DATA_V1_2_0 = pa.schema(
    [
        # Primary identifiers
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        # OHLC Price data
        pa.field("open", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field("high", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field("low", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field(
            "close", pa.decimal128(precision=10, scale=4), nullable=False
        ),  # Primary price
        pa.field(
            "price", pa.decimal128(precision=10, scale=4), nullable=False
        ),  # Current/last price
        pa.field("volume", pa.int64(), nullable=True),
        pa.field("market_cap", pa.decimal128(precision=15, scale=2), nullable=True),
        # Financial metrics
        pa.field("pe_ratio", pa.float64(), nullable=True),
        pa.field("eps", pa.decimal128(precision=8, scale=4), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        # Technical indicators
        pa.field("rsi_14d", pa.float64(), nullable=True),
        pa.field("sma_20d", pa.decimal128(precision=10, scale=4), nullable=True),
        pa.field("volatility_30d", pa.float64(), nullable=True),
        pa.field("beta", pa.float64(), nullable=True),
        # Metadata
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("version", pa.string(), nullable=False),
    ]
)

# Schema version mapping
MARKET_DATA_SCHEMAS: dict[str, pa.Schema] = {
    "v1.0.0": MARKET_DATA_V1_0_0,
    "v1.1.0": MARKET_DATA_V1_1_0,
    "v1.2.0": MARKET_DATA_V1_2_0,
}

# Default to latest version
MARKET_DATA_SCHEMA_LATEST = MARKET_DATA_V1_2_0
LATEST_VERSION = "v1.2.0"


def get_market_data_schema(version: str = "latest") -> pa.Schema:
    """Get market data schema by version"""
    if version == "latest":
        return MARKET_DATA_SCHEMA_LATEST

    if version not in MARKET_DATA_SCHEMAS:
        available_versions = list(MARKET_DATA_SCHEMAS.keys())
        raise ValueError(
            f"Schema version '{version}' not found. Available versions: {available_versions}"
        )

    return MARKET_DATA_SCHEMAS[version]


def get_available_versions() -> list:
    """Get list of available schema versions"""
    return list(MARKET_DATA_SCHEMAS.keys())
