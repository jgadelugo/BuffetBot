#!/usr/bin/env python3
"""
Advanced Query Builder for Market Analysis

Generates optimized SQL queries for complex market analysis using BigQuery.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import sqlparse
except ImportError:
    sqlparse = None

try:
    from jinja2 import Template
except ImportError:
    # Fallback to string formatting if Jinja2 not available
    Template = None


class AggregationType(Enum):
    """Supported aggregation types."""

    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    STDDEV = "STDDEV"


@dataclass
class DateRange:
    """Date range specification."""

    start_date: datetime
    end_date: datetime

    def to_sql_filter(self, date_column: str = "timestamp") -> str:
        """Convert to SQL WHERE clause."""
        return f"{date_column} BETWEEN '{self.start_date.isoformat()}' AND '{self.end_date.isoformat()}'"


@dataclass
class TimeSeriesQuery:
    """Time-series query configuration."""

    symbol: str
    date_range: DateRange
    metrics: list[str]
    interval: str = "1h"
    moving_averages: Optional[list[int]] = None


class AdvancedQueryBuilder:
    """
    Intelligent SQL generation for complex market analysis.
    """

    def __init__(self, dataset_id: str = "buffetbot_analytics"):
        """Initialize Advanced Query Builder."""
        self.dataset_id = dataset_id
        self.logger = logging.getLogger(__name__)
        self.market_data_table = f"{dataset_id}.market_data"

    def build_time_series_query(self, config: TimeSeriesQuery) -> str:
        """Generate optimized time-series analysis queries."""

        # Build metrics selection
        metrics_sql = ", ".join(config.metrics)

        # Build moving averages if specified
        ma_sql = ""
        if config.moving_averages:
            ma_parts = []
            for window in config.moving_averages:
                ma_parts.append(
                    f"""
                AVG(close) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                ) as ma_{window}"""
                )
            ma_sql = "," + ",".join(ma_parts)

        # Build the complete query
        query = f"""
        WITH time_series_data AS (
            SELECT
                symbol,
                timestamp,
                {metrics_sql},
                ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                ) as row_num
            FROM {self.market_data_table}
            WHERE symbol = '{config.symbol}'
                AND {config.date_range.to_sql_filter()}
        )

        SELECT
            symbol,
            timestamp,
            {metrics_sql}{ma_sql},
            close - LAG(close, 1) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
            ) as price_change,
            (close - LAG(close, 1) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
            )) / LAG(close, 1) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
            ) * 100 as price_change_pct
        FROM time_series_data
        ORDER BY timestamp
        """

        return self._optimize_query(query)

    def build_multi_symbol_comparison_query(
        self, symbols: list[str], date_range: DateRange, metrics: list[str] = None
    ) -> str:
        """Build query for comparing multiple symbols."""

        if metrics is None:
            metrics = ["close", "volume"]

        symbols_list = "', '".join(symbols)
        metrics_sql = ", ".join(metrics)

        query = f"""
        WITH symbol_data AS (
            SELECT
                symbol,
                timestamp,
                {metrics_sql},
                close / FIRST_VALUE(close) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS UNBOUNDED PRECEDING
                ) * 100 as normalized_price,
                (close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) /
                LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) * 100 as daily_return
            FROM {self.market_data_table}
            WHERE symbol IN ('{symbols_list}')
                AND {date_range.to_sql_filter()}
        )

        SELECT
            symbol,
            timestamp,
            {metrics_sql},
            normalized_price,
            daily_return
        FROM symbol_data
        ORDER BY symbol, timestamp
        """

        return self._optimize_query(query)

    def build_volatility_analysis_query(
        self, symbol: str, date_range: DateRange, window_days: int = 30
    ) -> str:
        """Build query for volatility analysis."""

        query = f"""
        WITH daily_returns AS (
            SELECT
                symbol,
                timestamp,
                close,
                (close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) /
                LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as daily_return
            FROM {self.market_data_table}
            WHERE symbol = '{symbol}'
                AND {date_range.to_sql_filter()}
        ),

        volatility_calc AS (
            SELECT
                symbol,
                timestamp,
                close,
                daily_return,
                STDDEV(daily_return) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {window_days - 1} PRECEDING AND CURRENT ROW
                ) as volatility_{window_days}d,
                AVG(ABS(daily_return)) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {window_days - 1} PRECEDING AND CURRENT ROW
                ) as avg_abs_return_{window_days}d
            FROM daily_returns
            WHERE daily_return IS NOT NULL
        )

        SELECT
            symbol,
            timestamp,
            close,
            daily_return,
            volatility_{window_days}d,
            avg_abs_return_{window_days}d,
            volatility_{window_days}d * SQRT(252) as annualized_volatility
        FROM volatility_calc
        ORDER BY timestamp
        """

        return self._optimize_query(query)

    def _optimize_query(self, query: str) -> str:
        """Optimize SQL query for BigQuery performance."""
        try:
            if sqlparse:
                # Parse and format the query
                formatted = sqlparse.format(
                    query, reindent=True, keyword_case="upper", strip_comments=True
                )
                return formatted
            else:
                # Basic cleanup if sqlparse not available
                lines = query.split("\n")
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                return "\n".join(cleaned_lines)

        except Exception as e:
            self.logger.warning(f"Query optimization failed: {e}")
            return query

    def estimate_query_cost(self, query: str) -> dict[str, Any]:
        """Estimate query cost and complexity."""
        query_lower = query.lower()

        # Count complexity factors
        join_count = query_lower.count("join")
        window_count = query_lower.count("over (")
        subquery_count = query_lower.count("with ") + query_lower.count("select") - 1

        # Estimate relative cost (1-10 scale)
        complexity_score = min(
            10, 1 + join_count + window_count + (subquery_count * 0.5)
        )

        return {
            "complexity_score": complexity_score,
            "estimated_cost_tier": "LOW"
            if complexity_score < 3
            else "MEDIUM"
            if complexity_score < 7
            else "HIGH",
            "join_count": join_count,
            "window_functions": window_count,
            "subqueries": subquery_count,
        }
