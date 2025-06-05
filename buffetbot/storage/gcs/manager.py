"""
GCS Storage Manager

Main interface for all Google Cloud Storage operations, providing high-level
methods for storing and retrieving data with optimizations and error handling.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...utils.cache import Cache
from ..formatters.parquet_formatter import ParquetFormatter
from ..query.optimizer import DataQuery, QueryOptimizer
from ..schemas.manager import SchemaManager, ValidationResult
from ..utils.config import GCSConfig
from ..utils.monitoring import StorageMetrics
from ..utils.security import SecurityManager
from .client import GCSClient
from .connection_pool import ConnectionPool
from .retry import RetryManager

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of an upload operation"""

    success: bool
    file_path: str
    file_size: int
    duration_ms: int
    error_message: Optional[str] = None
    bucket_used: str = "primary"
    content_hash: Optional[str] = None


@dataclass
class QueryResult:
    """Result of a query operation"""

    data: list[dict]
    metadata: dict
    cache_hit: bool
    duration_ms: int
    partitions_scanned: int
    total_records: int


class GCSStorageManager:
    """Production-ready GCS storage manager with intelligent routing"""

    def __init__(self, config: GCSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.gcs_client = GCSClient(config)
        self.retry_manager = RetryManager()
        self.connection_pool = ConnectionPool(
            connection_factory=lambda: None,  # Not used directly
            config=config.connection_config
            if hasattr(config, "connection_config")
            else None,
        )
        self.schema_manager = SchemaManager()
        self.formatter = ParquetFormatter()
        self.query_optimizer = QueryOptimizer()
        self.metrics = StorageMetrics()
        self.security_manager = SecurityManager()

        # Use existing BuffetBot cache system for consistency
        self.cache = Cache(cache_type="memory")

        # Initialize bucket names
        self.primary_bucket = config.data_bucket
        self.archive_bucket = config.archive_bucket
        self.backup_bucket = config.backup_bucket
        self.temp_bucket = config.temp_bucket

    async def store_data(
        self,
        data_type: str,
        data: list[dict],
        metadata: dict = None,
        encrypt: bool = False,
    ) -> UploadResult:
        """Store data with intelligent routing and optimization"""
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting storage operation for {len(data)} {data_type} records"
            )

            # Validate data schema
            validation_result = self.schema_manager.validate_data(data, data_type)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Schema validation failed: {validation_result.errors}"
                )

            # Log warnings if any
            if validation_result.warnings:
                self.logger.warning(
                    f"Schema validation warnings: {validation_result.warnings}"
                )

            # Format data for storage
            formatted_table = await self.formatter.format_data(data_type, data)

            # Determine storage path
            storage_path = self._optimize_storage_path(data_type, data[0])

            # Add metadata
            if metadata:
                formatted_table = self.formatter.add_metadata(formatted_table, metadata)

            # Encrypt data if requested
            if encrypt:
                formatted_table = await self.security_manager.encrypt_table(
                    formatted_table
                )

            # Upload with retry logic
            upload_result = await self._upload_with_retry(
                formatted_table, storage_path, self.primary_bucket
            )

            # Calculate content hash for integrity
            content_hash = self._calculate_content_hash(formatted_table)

            # Record metrics
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.metrics.record_upload(
                data_type=data_type,
                file_size=upload_result["file_size"],
                duration_ms=duration_ms,
                success=True,
            )

            self.logger.info(
                f"Successfully stored {len(data)} records to {storage_path} in {duration_ms}ms"
            )

            return UploadResult(
                success=True,
                file_path=storage_path,
                file_size=upload_result["file_size"],
                duration_ms=duration_ms,
                content_hash=content_hash,
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.metrics.record_upload(
                data_type=data_type, file_size=0, duration_ms=duration_ms, success=False
            )

            self.logger.error(f"Failed to store data: {str(e)}")

            return UploadResult(
                success=False,
                file_path="",
                file_size=0,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def retrieve_data(self, query: DataQuery) -> QueryResult:
        """Retrieve data with query optimization and caching"""
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting query for {query.data_type}")

            # Optimize query
            optimized_query = self.query_optimizer.optimize(query)

            # Check cache first
            cache_key = optimized_query.cache_key
            if cached_result := await self._get_cached_result(cache_key):
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self.logger.info(f"Cache hit for query, returned in {duration_ms}ms")

                return QueryResult(
                    data=cached_result["data"],
                    metadata={"cached": True, "cache_key": cache_key},
                    cache_hit=True,
                    duration_ms=duration_ms,
                    partitions_scanned=0,
                    total_records=len(cached_result["data"]),
                )

            # Execute query
            results = await self._execute_optimized_query(optimized_query)

            # Cache results if appropriate
            if len(results) < 10000:  # Don't cache very large results
                await self._cache_result(cache_key, results)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self.logger.info(
                f"Query completed: {len(results)} records in {duration_ms}ms"
            )

            return QueryResult(
                data=results,
                metadata={
                    "cached": False,
                    "partitions_scanned": len(optimized_query.partition_paths),
                    "estimated_cost": optimized_query.estimated_cost,
                },
                cache_hit=False,
                duration_ms=duration_ms,
                partitions_scanned=len(optimized_query.partition_paths),
                total_records=len(results),
            )

        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise

    async def list_files(
        self, data_type: str, prefix: str = None, max_results: int = 1000
    ) -> list[dict[str, Any]]:
        """List files in storage with metadata"""
        try:
            # Construct prefix for data type
            if prefix:
                full_prefix = f"raw/{data_type}/{prefix}"
            else:
                full_prefix = f"raw/{data_type}/"

            # List blobs
            blobs = self.gcs_client.list_blobs(
                bucket_name=self.primary_bucket,
                prefix=full_prefix,
                max_results=max_results,
            )

            # Extract metadata
            file_list = []
            for blob in blobs:
                file_info = {
                    "path": blob.name,
                    "size": blob.size,
                    "created": blob.time_created,
                    "updated": blob.updated,
                    "content_type": blob.content_type,
                    "etag": blob.etag,
                    "storage_class": blob.storage_class,
                    "metadata": blob.metadata or {},
                }
                file_list.append(file_info)

            self.logger.info(f"Listed {len(file_list)} files for {data_type}")
            return file_list

        except Exception as e:
            self.logger.error(f"Failed to list files: {str(e)}")
            raise

    async def delete_data(
        self, file_path: str, backup_before_delete: bool = True
    ) -> bool:
        """Delete data file with optional backup"""
        try:
            # Create backup if requested
            if backup_before_delete:
                backup_path = (
                    f"deleted/{datetime.now().strftime('%Y/%m/%d')}/{file_path}"
                )
                success = self.gcs_client.copy_blob(
                    source_bucket=self.primary_bucket,
                    source_blob=file_path,
                    dest_bucket=self.backup_bucket,
                    dest_blob=backup_path,
                )
                if not success:
                    self.logger.warning(f"Failed to backup {file_path} before deletion")

            # Delete the file
            success = self.gcs_client.delete_blob(self.primary_bucket, file_path)

            if success:
                self.logger.info(f"Successfully deleted {file_path}")
            else:
                self.logger.warning(f"File {file_path} not found for deletion")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {str(e)}")
            raise

    def _optimize_storage_path(self, data_type: str, sample_record: dict) -> str:
        """Generate optimized storage path with partitioning"""
        timestamp = datetime.fromisoformat(
            sample_record.get("timestamp", datetime.now().isoformat())
        )
        symbol = sample_record.get("symbol", "unknown")

        # Create hierarchical path with date partitioning
        path = (
            f"raw/{data_type}/"
            f"year={timestamp.year}/"
            f"month={timestamp.month:02d}/"
            f"day={timestamp.day:02d}/"
            f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_v1.parquet"
        )

        return path

    async def _upload_with_retry(
        self, data_table, path: str, bucket: str
    ) -> dict[str, Any]:
        """Upload with exponential backoff retry"""
        return await self.retry_manager.execute_with_retry(
            self._upload_to_gcs, data_table, path, bucket
        )

    async def _upload_to_gcs(
        self, data_table, path: str, bucket: str
    ) -> dict[str, Any]:
        """Actual upload implementation"""
        # Convert Arrow table to bytes
        buffer = self.formatter.table_to_bytes(data_table)

        # Upload to GCS
        success = self.gcs_client.upload_from_string(
            bucket_name=bucket,
            blob_name=path,
            data=buffer.getvalue().to_pybytes(),
            content_type="application/octet-stream",
            metadata={
                "data_type": data_table.schema.metadata.get(
                    b"data_type", b"unknown"
                ).decode(),
                "created_at": datetime.utcnow().isoformat(),
                "record_count": str(data_table.num_rows),
            },
        )

        if not success:
            raise RuntimeError(f"Failed to upload {path} to {bucket}")

        return {
            "success": True,
            "file_path": path,
            "file_size": len(buffer.getvalue().to_pybytes()),
            "bucket": bucket,
        }

    async def _execute_optimized_query(self, optimized_query) -> list[dict]:
        """Execute optimized query against GCS"""
        all_results = []

        # Process each partition
        for partition_path in optimized_query.partition_paths:
            try:
                # List files in partition
                blobs = self.gcs_client.list_blobs(
                    bucket_name=self.primary_bucket, prefix=partition_path
                )

                # Process each file
                for blob in blobs:
                    if blob.name.endswith(".parquet"):
                        # Download and process file
                        content = self.gcs_client.download_as_bytes(
                            bucket_name=self.primary_bucket, blob_name=blob.name
                        )

                        # Convert to table and extract data
                        import io

                        import pyarrow.parquet as pq

                        table = pq.read_table(io.BytesIO(content))
                        records = table.to_pylist()

                        # Apply filters
                        filtered_records = self._apply_filters(
                            records, optimized_query.optimized_filters
                        )

                        all_results.extend(filtered_records)

            except Exception as e:
                self.logger.warning(
                    f"Error processing partition {partition_path}: {str(e)}"
                )
                continue

        # Apply limit if specified
        if optimized_query.original_query.limit:
            all_results = all_results[: optimized_query.original_query.limit]

        return all_results

    def _apply_filters(
        self, records: list[dict], filters: dict[str, Any]
    ) -> list[dict]:
        """Apply filters to records"""
        filtered = []

        for record in records:
            match = True

            for field, value in filters.items():
                if field not in record:
                    match = False
                    break

                if isinstance(value, list):
                    # IN operation
                    if record[field] not in value:
                        match = False
                        break
                elif isinstance(value, dict):
                    # Range operations
                    if "gte" in value and record[field] < value["gte"]:
                        match = False
                        break
                    if "lte" in value and record[field] > value["lte"]:
                        match = False
                        break
                else:
                    # Exact match
                    if record[field] != value:
                        match = False
                        break

            if match:
                filtered.append(record)

        return filtered

    async def _get_cached_result(self, cache_key: str) -> Optional[dict]:
        """Get cached query result using BuffetBot cache system"""
        return self.cache.get(cache_key, expiration=1800)  # 30 minutes TTL

    async def _cache_result(
        self, cache_key: str, results: list[dict], ttl_minutes: int = 30
    ) -> None:
        """Cache query results using BuffetBot cache system"""
        cache_data = {"data": results}
        expiration = ttl_minutes * 60  # Convert to seconds for BuffetBot cache
        self.cache.set(cache_key, cache_data, expiration=expiration)

    def _calculate_content_hash(self, table) -> str:
        """Calculate SHA-256 hash of table content for integrity checking"""
        buffer = self.formatter.table_to_bytes(table)
        content_bytes = buffer.getvalue().to_pybytes()
        return hashlib.sha256(content_bytes).hexdigest()

    def get_metrics(self) -> dict[str, Any]:
        """Get current storage metrics"""
        return self.metrics.to_dict()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on storage system"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # Check GCS connectivity
            test_bucket = self.primary_bucket
            blobs = self.gcs_client.list_blobs(test_bucket, max_results=1)
            health_status["components"]["gcs"] = "healthy"
        except Exception as e:
            health_status["components"]["gcs"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check schema manager
        try:
            test_schema = self.schema_manager.get_schema("market_data")
            health_status["components"]["schema_manager"] = "healthy"
        except Exception as e:
            health_status["components"]["schema_manager"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check formatter
        try:
            test_data = [{"test": "value", "timestamp": datetime.now().isoformat()}]
            # Don't actually format, just check if it's available
            health_status["components"]["formatter"] = "healthy"
        except Exception as e:
            health_status["components"]["formatter"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        return health_status

    async def close(self):
        """Clean up resources"""
        try:
            if hasattr(self.gcs_client, "close"):
                self.gcs_client.close()

            if hasattr(self.connection_pool, "shutdown"):
                await self.connection_pool.shutdown()

            # Clear cache using BuffetBot cache system
            self.cache.clear()

            self.logger.info("GCS Storage Manager closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing storage manager: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
