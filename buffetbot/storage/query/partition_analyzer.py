"""
Partition Analysis Module

Analyzes partition structure and provides metadata for query optimization.
Does not implement its own caching - relies on the BuffetBot cache system.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...utils.cache import Cache

logger = logging.getLogger(__name__)


@dataclass
class PartitionInfo:
    """Information about a storage partition"""

    path: str
    date: datetime
    data_type: str
    symbols: set[str]
    file_count: int
    total_size_bytes: int
    last_modified: datetime


@dataclass
class PartitionStats:
    """Statistics about partition usage"""

    total_partitions: int
    date_range: dict[str, datetime]
    avg_files_per_partition: float
    avg_size_per_partition: int
    most_active_symbols: list[str]
    partition_distribution: dict[str, int]


class PartitionAnalyzer:
    """
    Analyzes storage partitions for query optimization.

    Uses the existing BuffetBot cache system for storing analysis results.
    """

    def __init__(self, cache_ttl_seconds: int = 3600):
        self.logger = logging.getLogger(__name__)
        self.cache = Cache(cache_type="memory")
        self.cache_ttl = cache_ttl_seconds

    def analyze_partitions(
        self, data_type: str, date_range: dict[str, str] = None
    ) -> list[PartitionInfo]:
        """
        Analyze partitions for a given data type and date range.

        Args:
            data_type: Type of data (e.g., 'market_data', 'options_data')
            date_range: Optional date range with 'start' and 'end' keys

        Returns:
            List of PartitionInfo objects
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                "partition_analysis", data_type, date_range
            )

            # Check cache first
            cached_result = self.cache.get(cache_key, expiration=self.cache_ttl)
            if cached_result:
                self.logger.debug(f"Cache hit for partition analysis: {data_type}")
                return cached_result

            # Perform analysis
            partitions = self._analyze_partition_structure(data_type, date_range)

            # Cache the results
            self.cache.set(cache_key, partitions, expiration=self.cache_ttl)

            self.logger.info(f"Analyzed {len(partitions)} partitions for {data_type}")
            return partitions

        except Exception as e:
            self.logger.error(f"Failed to analyze partitions for {data_type}: {str(e)}")
            raise

    def get_partition_paths(
        self, data_type: str, date_range: dict[str, str], symbols: list[str] = None
    ) -> list[str]:
        """
        Get optimized list of partition paths for a query.

        Args:
            data_type: Type of data
            date_range: Date range with 'start' and 'end' keys
            symbols: Optional list of symbols to filter by

        Returns:
            List of partition paths to scan
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                "partition_paths", data_type, date_range, symbols
            )

            # Check cache first
            cached_paths = self.cache.get(cache_key, expiration=self.cache_ttl)
            if cached_paths:
                self.logger.debug(f"Cache hit for partition paths: {data_type}")
                return cached_paths

            # Generate partition paths
            paths = self._generate_partition_paths(data_type, date_range, symbols)

            # Cache the results
            self.cache.set(cache_key, paths, expiration=self.cache_ttl)

            self.logger.debug(f"Generated {len(paths)} partition paths for {data_type}")
            return paths

        except Exception as e:
            self.logger.error(f"Failed to generate partition paths: {str(e)}")
            raise

    def get_partition_statistics(self, data_type: str) -> PartitionStats:
        """
        Get comprehensive statistics about partitions.

        Args:
            data_type: Type of data to analyze

        Returns:
            PartitionStats object with comprehensive statistics
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key("partition_stats", data_type)

            # Check cache first
            cached_stats = self.cache.get(cache_key, expiration=self.cache_ttl)
            if cached_stats:
                self.logger.debug(f"Cache hit for partition statistics: {data_type}")
                return cached_stats

            # Calculate statistics
            stats = self._calculate_partition_statistics(data_type)

            # Cache the results
            self.cache.set(cache_key, stats, expiration=self.cache_ttl)

            self.logger.info(f"Generated partition statistics for {data_type}")
            return stats

        except Exception as e:
            self.logger.error(f"Failed to get partition statistics: {str(e)}")
            raise

    def estimate_query_cost(self, partition_paths: list[str]) -> dict[str, float]:
        """
        Estimate the cost of scanning given partitions.

        Args:
            partition_paths: List of partition paths to scan

        Returns:
            Dictionary with cost estimates
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key("query_cost", sorted(partition_paths))

            # Check cache first
            cached_cost = self.cache.get(cache_key, expiration=self.cache_ttl)
            if cached_cost:
                self.logger.debug("Cache hit for query cost estimation")
                return cached_cost

            # Calculate cost estimate
            cost_estimate = self._calculate_query_cost(partition_paths)

            # Cache the results
            self.cache.set(cache_key, cost_estimate, expiration=self.cache_ttl)

            self.logger.debug(
                f"Estimated query cost for {len(partition_paths)} partitions"
            )
            return cost_estimate

        except Exception as e:
            self.logger.error(f"Failed to estimate query cost: {str(e)}")
            raise

    def _analyze_partition_structure(
        self, data_type: str, date_range: dict[str, str] = None
    ) -> list[PartitionInfo]:
        """Analyze the actual partition structure"""
        partitions = []

        # In a real implementation, this would scan GCS buckets
        # For now, we'll simulate partition analysis

        if date_range:
            start_date = datetime.fromisoformat(date_range["start"])
            end_date = datetime.fromisoformat(date_range["end"])
        else:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

        current_date = start_date
        while current_date <= end_date:
            partition_path = f"raw/{data_type}/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"

            # Simulate partition metadata
            partition = PartitionInfo(
                path=partition_path,
                date=current_date,
                data_type=data_type,
                symbols={"AAPL", "MSFT", "GOOGL", "TSLA"},  # Simulated
                file_count=50,  # Simulated
                total_size_bytes=1024 * 1024 * 100,  # 100MB simulated
                last_modified=current_date,
            )

            partitions.append(partition)
            current_date += timedelta(days=1)

        return partitions

    def _generate_partition_paths(
        self, data_type: str, date_range: dict[str, str], symbols: list[str] = None
    ) -> list[str]:
        """Generate optimized partition paths for query"""
        paths = []

        start_date = datetime.fromisoformat(date_range["start"])
        end_date = datetime.fromisoformat(date_range["end"])

        current_date = start_date
        while current_date <= end_date:
            partition_path = f"raw/{data_type}/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
            paths.append(partition_path)
            current_date += timedelta(days=1)

        return paths

    def _calculate_partition_statistics(self, data_type: str) -> PartitionStats:
        """Calculate comprehensive partition statistics"""
        # In a real implementation, this would analyze actual GCS data
        # For now, we'll return simulated statistics

        return PartitionStats(
            total_partitions=365,  # Simulated: 1 year of daily partitions
            date_range={
                "earliest": datetime.now() - timedelta(days=365),
                "latest": datetime.now(),
            },
            avg_files_per_partition=45.5,
            avg_size_per_partition=1024 * 1024 * 95,  # ~95MB average
            most_active_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"],
            partition_distribution={
                "market_data": 300,
                "options_data": 50,
                "forecasts": 15,
            },
        )

    def _calculate_query_cost(self, partition_paths: list[str]) -> dict[str, float]:
        """Calculate estimated cost for scanning partitions"""
        # Simulate cost calculation based on partition count and estimated size
        estimated_size_gb = len(partition_paths) * 0.1  # ~100MB per partition
        estimated_scan_cost = estimated_size_gb * 0.005  # $0.005 per GB scanned
        estimated_duration_ms = len(partition_paths) * 50  # 50ms per partition

        return {
            "estimated_scan_cost_usd": round(estimated_scan_cost, 4),
            "estimated_duration_ms": estimated_duration_ms,
            "estimated_data_size_gb": round(estimated_size_gb, 2),
            "partition_count": len(partition_paths),
            "confidence": 0.8,  # 80% confidence in estimate
        }

    def _generate_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for partition analysis operations"""
        import hashlib
        import json

        key_data = {
            "operation": operation,
            "args": [str(arg) for arg in args if arg is not None],
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return f"partition_analyzer_{hashlib.md5(key_str.encode()).hexdigest()}"

    def invalidate_cache(self, data_type: str = None):
        """
        Invalidate cached partition analysis results.

        Args:
            data_type: If specified, only invalidate cache for this data type
        """
        try:
            if data_type:
                # In a more sophisticated implementation, we would selectively
                # invalidate cache entries. For now, we'll clear all.
                self.logger.info(f"Invalidating partition cache for {data_type}")
            else:
                self.logger.info("Invalidating all partition cache")

            # Clear the entire cache for simplicity
            self.cache.clear()

        except Exception as e:
            self.logger.error(f"Failed to invalidate cache: {str(e)}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cache usage"""
        try:
            cache_stats = self.cache.get_stats()

            return {
                "cache_type": cache_stats.get("type", "unknown"),
                "cache_size": cache_stats.get("size", 0),
                "cache_items": cache_stats.get("items", 0),
                "cache_expired_items": cache_stats.get("expired", 0),
                "cache_ttl_seconds": self.cache_ttl,
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
