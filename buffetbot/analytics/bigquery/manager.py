#!/usr/bin/env python3
"""
BigQuery Analytics Manager

Primary interface for BigQuery analytics operations that integrates seamlessly
with Phase 1 GCS storage for complete data pipeline functionality.

This manager provides:
- Automated data loading from GCS to BigQuery
- Query execution with optimization
- Table management with partitioning
- Cost monitoring and estimation
- Integration with Phase 1 cache system
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, QueryJobConfig, Table
from google.cloud.exceptions import Conflict, NotFound

# Import Phase 1 components - DO NOT MODIFY
from buffetbot.storage.gcs.manager import GCSStorageManager
from buffetbot.storage.schemas.manager import SchemaManager
from buffetbot.utils.cache import Cache


@dataclass
class LoadConfig:
    """Configuration for GCS to BigQuery data loading."""

    source_format: str = "PARQUET"
    write_disposition: str = "WRITE_TRUNCATE"  # WRITE_APPEND, WRITE_EMPTY
    create_disposition: str = "CREATE_IF_NEEDED"
    skip_leading_rows: int = 0
    max_bad_records: int = 0
    allow_jagged_rows: bool = False
    allow_quoted_newlines: bool = False
    field_delimiter: str = ","
    quote_character: str = '"'
    partition_field: Optional[str] = None
    clustering_fields: Optional[list[str]] = None


@dataclass
class QueryResult:
    """Result from BigQuery query execution."""

    query_id: str
    rows: list[dict[str, Any]]
    total_rows: int
    bytes_processed: int
    cache_hit: bool
    execution_time_ms: int
    cost_estimate: float
    schema: list[dict[str, str]]


@dataclass
class LoadResult:
    """Result from GCS to BigQuery load operation."""

    job_id: str
    table_id: str
    rows_loaded: int
    bytes_processed: int
    load_time_ms: int
    errors: list[str]
    warnings: list[str]


@dataclass
class TableSchema:
    """BigQuery table schema definition."""

    table_id: str
    dataset_id: str
    fields: list[dict[str, Any]]
    description: Optional[str] = None
    labels: Optional[dict[str, str]] = None


@dataclass
class PartitionConfig:
    """BigQuery table partitioning configuration."""

    type: str  # "TIME", "RANGE", "INTEGER"
    field: str
    require_partition_filter: bool = True
    expiration_days: Optional[int] = None


class BigQueryAnalyticsManager:
    """
    Primary interface for BigQuery analytics operations.
    Integrates with Phase 1 GCS storage for seamless data flow.
    """

    def __init__(self, project_id: str, dataset_id: str = "buffetbot_analytics"):
        """
        Initialize BigQuery Analytics Manager.

        Args:
            project_id: GCP project ID
            dataset_id: Default BigQuery dataset for analytics
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)

        # Initialize Phase 1 integrations
        self.gcs_manager = GCSStorageManager()
        self.schema_manager = SchemaManager()
        self.cache = Cache(cache_type="memory")  # Use Phase 1 unified cache

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize dataset
        asyncio.create_task(self._ensure_dataset_exists())

    async def _ensure_dataset_exists(self) -> None:
        """Ensure the analytics dataset exists."""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            self.client.get_dataset(dataset_ref)
            self.logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.description = "BuffetBot Analytics Data Warehouse"
            dataset.location = "US"

            # Set default table expiration (optional)
            # dataset.default_table_expiration_ms = 365 * 24 * 60 * 60 * 1000  # 1 year

            self.client.create_dataset(dataset, timeout=30)
            self.logger.info(f"Created dataset {self.dataset_id}")

    async def load_from_gcs(
        self, gcs_path: str, table_id: str, load_config: Optional[LoadConfig] = None
    ) -> LoadResult:
        """
        Load data from GCS to BigQuery with optimization.

        Args:
            gcs_path: GCS path (gs://bucket/path/to/file)
            table_id: Target BigQuery table ID
            load_config: Optional load configuration

        Returns:
            LoadResult with job details and metrics
        """
        start_time = datetime.now()

        if load_config is None:
            load_config = LoadConfig()

        # Construct full table reference
        table_ref = self.client.dataset(self.dataset_id).table(table_id)

        # Configure load job
        job_config = LoadJobConfig(
            source_format=getattr(bigquery.SourceFormat, load_config.source_format),
            write_disposition=getattr(
                bigquery.WriteDisposition, load_config.write_disposition
            ),
            create_disposition=getattr(
                bigquery.CreateDisposition, load_config.create_disposition
            ),
            skip_leading_rows=load_config.skip_leading_rows,
            max_bad_records=load_config.max_bad_records,
            allow_jagged_rows=load_config.allow_jagged_rows,
            allow_quoted_newlines=load_config.allow_quoted_newlines,
        )

        # Set partitioning if specified
        if load_config.partition_field:
            job_config.time_partitioning = bigquery.TimePartitioning(
                field=load_config.partition_field
            )

        # Set clustering if specified
        if load_config.clustering_fields:
            job_config.clustering_fields = load_config.clustering_fields

        try:
            # Start load job
            load_job = self.client.load_table_from_uri(
                gcs_path, table_ref, job_config=job_config
            )

            # Wait for job completion
            load_job.result()  # Raises exception on failure

            # Calculate metrics
            end_time = datetime.now()
            load_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Get job statistics
            destination_table = self.client.get_table(table_ref)

            return LoadResult(
                job_id=load_job.job_id,
                table_id=table_id,
                rows_loaded=load_job.output_rows or 0,
                bytes_processed=load_job.input_file_bytes or 0,
                load_time_ms=load_time_ms,
                errors=[],
                warnings=[],
            )

        except Exception as e:
            self.logger.error(f"Load job failed: {e}")
            return LoadResult(
                job_id="",
                table_id=table_id,
                rows_loaded=0,
                bytes_processed=0,
                load_time_ms=0,
                errors=[str(e)],
                warnings=[],
            )

    async def execute_query(
        self, query: str, optimization_level: str = "auto", use_cache: bool = True
    ) -> QueryResult:
        """
        Execute optimized BigQuery with result caching.

        Args:
            query: SQL query to execute
            optimization_level: "auto", "cost_effective", "performance"
            use_cache: Whether to use query result caching

        Returns:
            QueryResult with data and metrics
        """
        start_time = datetime.now()

        # Generate cache key from query
        cache_key = f"bq_query_{hash(query)}"

        # Check cache first
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.logger.info("Query result served from cache")
                cached_result["cache_hit"] = True
                return QueryResult(**cached_result)

        # Configure query job
        job_config = QueryJobConfig()

        # Apply optimization based on level
        if optimization_level == "cost_effective":
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 1000000000  # 1GB limit
        elif optimization_level == "performance":
            job_config.use_query_cache = False  # Always fresh data
            job_config.priority = bigquery.QueryPriority.INTERACTIVE
        else:  # auto
            job_config.use_query_cache = True
            job_config.dry_run = False

        try:
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()

            # Convert results to list of dictionaries
            rows = []
            schema = []

            if results.schema:
                schema = [
                    {"name": field.name, "type": field.field_type, "mode": field.mode}
                    for field in results.schema
                ]

            for row in results:
                rows.append(dict(row))

            # Calculate metrics
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Estimate cost (approximate: $5 per TB)
            bytes_processed = query_job.total_bytes_processed or 0
            cost_estimate = (bytes_processed / (1024**4)) * 5.0  # TB to cost

            result = QueryResult(
                query_id=query_job.job_id,
                rows=rows,
                total_rows=query_job.num_dml_affected_rows or len(rows),
                bytes_processed=bytes_processed,
                cache_hit=False,
                execution_time_ms=execution_time_ms,
                cost_estimate=cost_estimate,
                schema=schema,
            )

            # Cache result if enabled
            if use_cache and len(rows) < 10000:  # Don't cache huge results
                await self.cache.set(
                    cache_key, result.__dict__, ttl_seconds=3600  # 1 hour cache
                )

            return result

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return QueryResult(
                query_id="",
                rows=[],
                total_rows=0,
                bytes_processed=0,
                cache_hit=False,
                execution_time_ms=0,
                cost_estimate=0.0,
                schema=[],
            )

    async def create_partitioned_table(
        self, table_schema: TableSchema, partition_config: PartitionConfig
    ) -> bool:
        """
        Create optimally partitioned tables.

        Args:
            table_schema: Table schema definition
            partition_config: Partitioning configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table reference
            table_ref = self.client.dataset(self.dataset_id).table(
                table_schema.table_id
            )

            # Build schema fields
            schema_fields = []
            for field in table_schema.fields:
                schema_fields.append(
                    bigquery.SchemaField(
                        field["name"],
                        field["type"],
                        mode=field.get("mode", "NULLABLE"),
                        description=field.get("description"),
                    )
                )

            # Create table object
            table = bigquery.Table(table_ref, schema=schema_fields)
            table.description = table_schema.description

            # Configure partitioning
            if partition_config.type == "TIME":
                table.time_partitioning = bigquery.TimePartitioning(
                    field=partition_config.field,
                    require_partition_filter=partition_config.require_partition_filter,
                )
                if partition_config.expiration_days:
                    table.time_partitioning.expiration_ms = (
                        partition_config.expiration_days * 24 * 60 * 60 * 1000
                    )

            # Set labels if provided
            if table_schema.labels:
                table.labels = table_schema.labels

            # Create table
            table = self.client.create_table(table)
            self.logger.info(f"Created partitioned table {table_schema.table_id}")
            return True

        except Conflict:
            self.logger.warning(f"Table {table_schema.table_id} already exists")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create table {table_schema.table_id}: {e}")
            return False

    async def get_table_info(self, table_id: str) -> Optional[dict[str, Any]]:
        """Get information about a BigQuery table."""
        try:
            table_ref = self.client.dataset(self.dataset_id).table(table_id)
            table = self.client.get_table(table_ref)

            return {
                "table_id": table.table_id,
                "dataset_id": table.dataset_id,
                "project_id": table.project,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created,
                "modified": table.modified,
                "schema": [
                    {
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description,
                    }
                    for field in table.schema
                ],
                "partitioning": {
                    "type": table.time_partitioning.type_
                    if table.time_partitioning
                    else None,
                    "field": table.time_partitioning.field
                    if table.time_partitioning
                    else None,
                }
                if table.time_partitioning
                else None,
                "clustering_fields": table.clustering_fields,
                "labels": table.labels,
            }
        except NotFound:
            return None
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_id}: {e}")
            return None

    async def list_tables(self) -> list[str]:
        """List all tables in the analytics dataset."""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            tables = self.client.list_tables(dataset_ref)
            return [table.table_id for table in tables]
        except Exception as e:
            self.logger.error(f"Failed to list tables: {e}")
            return []

    def get_client(self) -> bigquery.Client:
        """Get the underlying BigQuery client for advanced operations."""
        return self.client
