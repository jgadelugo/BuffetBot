"""
Query Optimizer

Optimizes queries for performance and cost efficiency by implementing
partition pruning, filter optimization, and caching strategies.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataQuery:
    """Represents a data query with filters and options"""

    data_type: str
    filters: dict[str, Any]
    date_range: Optional[dict[str, str]] = None
    symbols: Optional[list[str]] = None
    limit: Optional[int] = None
    order_by: Optional[str] = None


@dataclass
class OptimizedQuery:
    """Represents an optimized query with execution plan"""

    original_query: DataQuery
    partition_paths: list[str]
    optimized_filters: dict[str, Any]
    cache_key: str
    estimated_cost: float
    estimated_duration_ms: int


class QueryOptimizer:
    """Optimize queries for performance and cost efficiency"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.partition_cache = {}

        # Cost estimation constants
        self.base_scan_cost = 0.01  # Base cost per partition
        self.per_record_cost = 0.000001  # Cost per record processed
        self.network_cost_factor = 0.1  # Network transfer cost factor

    def optimize(self, query: DataQuery) -> OptimizedQuery:
        """Optimize query for execution"""
        try:
            self.logger.debug(f"Optimizing query for {query.data_type}")

            # Analyze required partitions
            partition_paths = self._analyze_partitions(query)

            # Apply partition pruning
            pruned_partitions = self._prune_partitions(partition_paths, query.filters)

            # Optimize filters
            optimized_filters = self._optimize_filters(query.filters)

            # Generate cache key
            cache_key = self._generate_cache_key(query, optimized_filters)

            # Estimate cost and duration
            estimated_cost = self._estimate_cost(pruned_partitions)
            estimated_duration = self._estimate_duration(
                pruned_partitions, optimized_filters
            )

            optimized_query = OptimizedQuery(
                original_query=query,
                partition_paths=pruned_partitions,
                optimized_filters=optimized_filters,
                cache_key=cache_key,
                estimated_cost=estimated_cost,
                estimated_duration_ms=estimated_duration,
            )

            self.logger.info(
                f"Query optimized: {len(partition_paths)} -> {len(pruned_partitions)} partitions, "
                f"estimated cost: ${estimated_cost:.4f}, duration: {estimated_duration}ms"
            )

            return optimized_query

        except Exception as e:
            self.logger.error(f"Query optimization failed: {str(e)}")
            raise

    def _analyze_partitions(self, query: DataQuery) -> list[str]:
        """Determine which partitions need to be scanned"""
        partitions = []

        try:
            if query.date_range:
                start_date = datetime.fromisoformat(query.date_range["start"])
                end_date = datetime.fromisoformat(query.date_range["end"])

                current_date = start_date
                while current_date <= end_date:
                    partition_path = (
                        f"raw/{query.data_type}/"
                        f"year={current_date.year}/"
                        f"month={current_date.month:02d}/"
                        f"day={current_date.day:02d}/"
                    )
                    partitions.append(partition_path)
                    current_date += timedelta(days=1)
            else:
                # No date range specified, scan recent data (last 7 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)

                current_date = start_date
                while current_date <= end_date:
                    partition_path = (
                        f"raw/{query.data_type}/"
                        f"year={current_date.year}/"
                        f"month={current_date.month:02d}/"
                        f"day={current_date.day:02d}/"
                    )
                    partitions.append(partition_path)
                    current_date += timedelta(days=1)

            self.logger.debug(f"Identified {len(partitions)} partitions for scanning")
            return partitions

        except Exception as e:
            self.logger.error(f"Partition analysis failed: {str(e)}")
            return []

    def _prune_partitions(self, partitions: list[str], filters: dict) -> list[str]:
        """Apply partition pruning based on filters"""
        pruned = []

        try:
            for partition in partitions:
                if self._partition_matches_filters(partition, filters):
                    pruned.append(partition)

            pruning_ratio = (
                (len(partitions) - len(pruned)) / len(partitions) if partitions else 0
            )

            self.logger.info(
                f"Partition pruning: {len(partitions)} -> {len(pruned)} partitions "
                f"({pruning_ratio:.1%} pruned)"
            )

            return pruned

        except Exception as e:
            self.logger.warning(f"Partition pruning failed: {str(e)}")
            return partitions  # Return original partitions if pruning fails

    def _partition_matches_filters(self, partition: str, filters: dict) -> bool:
        """Check if partition matches the given filters"""
        try:
            # For now, assume all partitions match
            # In a full implementation, this would check partition metadata
            # against filters to determine if the partition could contain matching data
            return True

        except Exception as e:
            self.logger.warning(
                f"Partition filter check failed for {partition}: {str(e)}"
            )
            return True  # Conservative approach - include partition if unsure

    def _optimize_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Optimize filters for better performance"""
        try:
            optimized = {}

            for field, value in filters.items():
                # Convert single values to list for IN operations when beneficial
                if isinstance(value, (str, int, float)) and field in ["symbol"]:
                    optimized[field] = [value]
                # Optimize range filters
                elif isinstance(value, dict) and "start" in value and "end" in value:
                    optimized[field] = {"gte": value["start"], "lte": value["end"]}
                else:
                    optimized[field] = value

            self.logger.debug(f"Optimized {len(filters)} filters")
            return optimized

        except Exception as e:
            self.logger.warning(f"Filter optimization failed: {str(e)}")
            return filters  # Return original filters if optimization fails

    def _generate_cache_key(self, query: DataQuery, optimized_filters: dict) -> str:
        """Generate unique cache key for query"""
        try:
            key_data = {
                "data_type": query.data_type,
                "filters": optimized_filters,
                "date_range": query.date_range,
                "symbols": sorted(query.symbols) if query.symbols else None,
                "limit": query.limit,
                "order_by": query.order_by,
            }

            key_str = json.dumps(key_data, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            self.logger.debug(f"Generated cache key: {cache_key}")
            return cache_key

        except Exception as e:
            self.logger.warning(f"Cache key generation failed: {str(e)}")
            return f"query_{hash(str(query))}"

    def _estimate_cost(self, partitions: list[str]) -> float:
        """Estimate the cost of executing the query"""
        try:
            # Base cost for scanning partitions
            base_cost = len(partitions) * self.base_scan_cost

            # Estimate based on partition count (simplified)
            # In reality, this would consider file sizes, record counts, etc.
            estimated_records = (
                len(partitions) * 10000
            )  # Assume 10k records per partition
            processing_cost = estimated_records * self.per_record_cost

            # Network transfer cost (simplified)
            estimated_data_size_mb = len(partitions) * 10  # Assume 10MB per partition
            network_cost = estimated_data_size_mb * self.network_cost_factor

            total_cost = base_cost + processing_cost + network_cost

            self.logger.debug(f"Estimated query cost: ${total_cost:.4f}")
            return total_cost

        except Exception as e:
            self.logger.warning(f"Cost estimation failed: {str(e)}")
            return 1.0  # Default cost estimate

    def _estimate_duration(self, partitions: list[str], filters: dict) -> int:
        """Estimate query execution duration in milliseconds"""
        try:
            # Base duration for partition scanning
            base_duration = len(partitions) * 50  # 50ms per partition

            # Additional time for filtering
            filter_complexity = len(filters)
            filter_duration = filter_complexity * 10  # 10ms per filter

            # Network latency
            network_duration = 100  # Base network latency

            total_duration = base_duration + filter_duration + network_duration

            self.logger.debug(f"Estimated query duration: {total_duration}ms")
            return total_duration

        except Exception as e:
            self.logger.warning(f"Duration estimation failed: {str(e)}")
            return 1000  # Default 1 second estimate

    def get_query_plan(self, optimized_query: OptimizedQuery) -> dict[str, Any]:
        """Get detailed query execution plan"""
        try:
            plan = {
                "query_id": optimized_query.cache_key,
                "data_type": optimized_query.original_query.data_type,
                "partitions": {
                    "total_partitions": len(optimized_query.partition_paths),
                    "partition_paths": optimized_query.partition_paths,
                },
                "filters": {
                    "original_filters": optimized_query.original_query.filters,
                    "optimized_filters": optimized_query.optimized_filters,
                },
                "estimates": {
                    "cost_usd": optimized_query.estimated_cost,
                    "duration_ms": optimized_query.estimated_duration_ms,
                },
                "optimizations": {
                    "partition_pruning_enabled": True,
                    "filter_optimization_enabled": True,
                    "caching_enabled": True,
                },
            }

            return plan

        except Exception as e:
            self.logger.error(f"Failed to generate query plan: {str(e)}")
            return {"error": str(e)}

    def analyze_query_performance(
        self, query: DataQuery, actual_duration_ms: int, actual_records: int
    ) -> dict[str, Any]:
        """Analyze query performance and provide optimization suggestions"""
        try:
            optimized_query = self.optimize(query)

            # Compare estimates with actual performance
            duration_accuracy = (
                optimized_query.estimated_duration_ms / actual_duration_ms
                if actual_duration_ms > 0
                else 0
            )

            suggestions = []

            # Duration analysis
            if actual_duration_ms > 2000:  # > 2 seconds
                suggestions.append(
                    "Consider adding more specific filters to reduce data scanning"
                )

            if actual_duration_ms > optimized_query.estimated_duration_ms * 2:
                suggestions.append(
                    "Query took longer than expected - check network connectivity"
                )

            # Partition analysis
            if len(optimized_query.partition_paths) > 10:
                suggestions.append(
                    "Consider narrowing date range to scan fewer partitions"
                )

            # Record count analysis
            if actual_records > 100000:
                suggestions.append(
                    "Large result set - consider using LIMIT or more selective filters"
                )

            analysis = {
                "performance": {
                    "actual_duration_ms": actual_duration_ms,
                    "estimated_duration_ms": optimized_query.estimated_duration_ms,
                    "duration_accuracy": duration_accuracy,
                    "actual_records": actual_records,
                },
                "optimization_suggestions": suggestions,
                "query_efficiency": "good"
                if actual_duration_ms < 1000
                else "needs_improvement",
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            return {"error": str(e)}
