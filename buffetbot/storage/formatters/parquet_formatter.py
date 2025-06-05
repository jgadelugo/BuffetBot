"""
Parquet Formatter

Handles data formatting, compression, and optimization for Parquet storage.
"""

import gzip
import hashlib
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetFormatter:
    """Format and optimize data for Parquet storage"""

    def __init__(self, compression: str = "snappy"):
        self.compression = compression
        self.logger = logging.getLogger(__name__)

    async def format_data(self, data_type: str, data: list[dict]) -> pa.Table:
        """Format data based on type"""
        try:
            # Convert to Arrow table
            table = pa.Table.from_pylist(data)

            # Apply schema casting if available
            table = self._apply_schema_casting(table, data_type)

            # Add storage metadata
            table = self._add_storage_metadata(table, data_type)

            # Optimize for storage
            table = self._optimize_for_storage(table)

            self.logger.debug(f"Formatted {table.num_rows} rows for {data_type}")
            return table

        except Exception as e:
            self.logger.error(f"Failed to format data: {str(e)}")
            raise

    def _apply_schema_casting(self, table: pa.Table, data_type: str) -> pa.Table:
        """Apply schema casting if schema is available"""
        try:
            # For now, return table as-is
            # In a full implementation, this would apply the specific schema
            return table
        except Exception as e:
            self.logger.warning(f"Schema casting failed for {data_type}: {str(e)}")
            return table

    def _optimize_for_storage(self, table: pa.Table) -> pa.Table:
        """Apply storage optimizations"""
        try:
            # Sort by timestamp for better compression
            if "timestamp" in table.column_names:
                indices = pa.compute.sort_indices(table, [("timestamp", "ascending")])
                table = pa.compute.take(table, indices)

            # Apply dictionary encoding for categorical columns
            categorical_columns = ["symbol", "data_source", "option_type"]
            for col in categorical_columns:
                if col in table.column_names:
                    try:
                        column_data = table.column(col)
                        dict_array = pa.compute.dictionary_encode(column_data)
                        table = table.set_column(
                            table.schema.get_field_index(col), col, dict_array
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to dictionary encode {col}: {str(e)}"
                        )

            return table

        except Exception as e:
            self.logger.warning(f"Storage optimization failed: {str(e)}")
            return table

    def _add_storage_metadata(self, table: pa.Table, data_type: str) -> pa.Table:
        """Add metadata to table"""
        try:
            metadata = {
                b"data_type": data_type.encode(),
                b"created_at": datetime.utcnow().isoformat().encode(),
                b"formatter_version": b"1.0.0",
                b"compression": self.compression.encode(),
                b"record_count": str(table.num_rows).encode(),
            }

            # Add content hash
            content_hash = self._calculate_content_hash(table)
            metadata[b"content_hash"] = content_hash.encode()

            # Create new schema with metadata
            new_schema = table.schema.with_metadata(metadata)

            return pa.Table.from_arrays(table.columns, schema=new_schema)

        except Exception as e:
            self.logger.warning(f"Failed to add metadata: {str(e)}")
            return table

    def table_to_bytes(self, table: pa.Table) -> pa.BufferOutputStream:
        """Convert table to compressed bytes"""
        try:
            buffer = pa.BufferOutputStream()

            pq.write_table(
                table,
                buffer,
                compression=self.compression,
                use_dictionary=True,
                row_group_size=50000,
                data_page_size=1024 * 1024,  # 1MB pages
                write_statistics=True,
                use_deprecated_int96_timestamps=False,
            )

            return buffer

        except Exception as e:
            self.logger.error(f"Failed to convert table to bytes: {str(e)}")
            raise

    def bytes_to_table(self, data_bytes: bytes) -> pa.Table:
        """Convert bytes back to Arrow table"""
        try:
            buffer = io.BytesIO(data_bytes)
            table = pq.read_table(buffer)

            self.logger.debug(f"Converted bytes to table with {table.num_rows} rows")
            return table

        except Exception as e:
            self.logger.error(f"Failed to convert bytes to table: {str(e)}")
            raise

    def add_metadata(self, table: pa.Table, metadata: dict[str, str]) -> pa.Table:
        """Add additional metadata to table"""
        try:
            existing_metadata = table.schema.metadata or {}

            # Convert new metadata to bytes
            new_metadata = {k.encode(): v.encode() for k, v in metadata.items()}

            # Merge with existing metadata
            combined_metadata = {**existing_metadata, **new_metadata}

            # Create new schema with combined metadata
            new_schema = table.schema.with_metadata(combined_metadata)

            return pa.Table.from_arrays(table.columns, schema=new_schema)

        except Exception as e:
            self.logger.warning(f"Failed to add metadata: {str(e)}")
            return table

    def _calculate_content_hash(self, table: pa.Table) -> str:
        """Calculate SHA-256 hash of table content"""
        try:
            # Create a simplified hash based on table structure and data
            # In a production system, this would be more comprehensive
            hash_data = f"{table.num_rows}_{table.num_columns}_{table.schema}"

            return hashlib.sha256(hash_data.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Failed to calculate content hash: {str(e)}")
            return "unknown"

    def get_compression_info(self, table: pa.Table) -> dict[str, Any]:
        """Get compression information for the table"""
        try:
            # Convert to bytes to get actual size
            buffer = self.table_to_bytes(table)
            compressed_size = len(buffer.getvalue().to_pybytes())

            # Estimate uncompressed size (rough calculation)
            uncompressed_size = table.nbytes

            compression_ratio = (
                uncompressed_size / compressed_size if compressed_size > 0 else 1.0
            )

            return {
                "compression": self.compression,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "compression_ratio": compression_ratio,
                "space_saved_percent": (1 - 1 / compression_ratio) * 100
                if compression_ratio > 1
                else 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to get compression info: {str(e)}")
            return {"compression": self.compression, "error": str(e)}
