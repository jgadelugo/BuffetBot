"""
GCS Client Wrapper

Provides a simplified interface to Google Cloud Storage with error handling,
authentication, and connection management.
"""

import io
import logging
from typing import Any, BinaryIO, Dict, List, Optional, Union

from google.cloud import storage
from google.cloud.storage import Blob, Bucket

from ..utils.config import GCSConfig
from .connection_pool import ConnectionConfig, ConnectionPool
from .retry import RetryManager, retry_on_failure

logger = logging.getLogger(__name__)


class GCSClient:
    """Simplified GCS client with error handling and retry logic"""

    def __init__(self, config: GCSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize retry manager
        self.retry_manager = RetryManager()

        # Initialize the client (this will be created lazily)
        self._client: Optional[storage.Client] = None
        self._buckets: dict[str, Bucket] = {}

        # Connection pool for managing multiple operations
        self.connection_pool = ConnectionPool(
            connection_factory=self._create_client,
            config=ConnectionConfig(
                max_connections=config.max_connections,
                connection_timeout_seconds=config.timeout_seconds,
            ),
        )

    def _create_client(self) -> storage.Client:
        """Create a new GCS client instance"""
        try:
            # Create client with proper authentication
            if self.config.service_account_key_path:
                client = storage.Client.from_service_account_json(
                    self.config.service_account_key_path, project=self.config.project_id
                )
            else:
                # Use default credentials (ADC)
                client = storage.Client(project=self.config.project_id)

            self.logger.debug(
                f"Created GCS client for project: {self.config.project_id}"
            )
            return client

        except Exception as e:
            self.logger.error(f"Failed to create GCS client: {str(e)}")
            raise

    @property
    def client(self) -> storage.Client:
        """Get the GCS client (lazy initialization)"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def get_bucket(self, bucket_name: str) -> Bucket:
        """Get a bucket object with caching"""
        if bucket_name not in self._buckets:
            try:
                bucket = self.client.bucket(bucket_name)
                self._buckets[bucket_name] = bucket
                self.logger.debug(f"Retrieved bucket: {bucket_name}")
            except Exception as e:
                self.logger.error(f"Failed to get bucket {bucket_name}: {str(e)}")
                raise

        return self._buckets[bucket_name]

    @retry_on_failure()
    def upload_from_string(
        self,
        bucket_name: str,
        blob_name: str,
        data: Union[str, bytes],
        content_type: str = None,
        metadata: dict[str, str] = None,
    ) -> bool:
        """Upload data from string/bytes to GCS"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Set metadata if provided
            if metadata:
                blob.metadata = metadata

            # Set content type if provided
            if content_type:
                blob.content_type = content_type

            # Upload the data
            blob.upload_from_string(data)

            self.logger.debug(
                f"Uploaded {len(data)} bytes to {bucket_name}/{blob_name}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to upload to {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    @retry_on_failure()
    def upload_from_file(
        self,
        bucket_name: str,
        blob_name: str,
        file_obj: BinaryIO,
        content_type: str = None,
        metadata: dict[str, str] = None,
    ) -> bool:
        """Upload data from file object to GCS"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Set metadata if provided
            if metadata:
                blob.metadata = metadata

            # Set content type if provided
            if content_type:
                blob.content_type = content_type

            # Upload the file
            blob.upload_from_file(file_obj)

            self.logger.debug(f"Uploaded file to {bucket_name}/{blob_name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to upload file to {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    @retry_on_failure()
    def download_as_bytes(self, bucket_name: str, blob_name: str) -> bytes:
        """Download blob content as bytes"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Check if blob exists
            if not blob.exists():
                raise FileNotFoundError(f"Blob {bucket_name}/{blob_name} not found")

            content = blob.download_as_bytes()

            self.logger.debug(
                f"Downloaded {len(content)} bytes from {bucket_name}/{blob_name}"
            )
            return content

        except Exception as e:
            self.logger.error(f"Failed to download {bucket_name}/{blob_name}: {str(e)}")
            raise

    @retry_on_failure()
    def download_as_text(
        self, bucket_name: str, blob_name: str, encoding: str = "utf-8"
    ) -> str:
        """Download blob content as text"""
        try:
            content_bytes = self.download_as_bytes(bucket_name, blob_name)
            return content_bytes.decode(encoding)

        except Exception as e:
            self.logger.error(
                f"Failed to download text from {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    @retry_on_failure()
    def download_to_file(
        self, bucket_name: str, blob_name: str, file_obj: BinaryIO
    ) -> None:
        """Download blob content to file object"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Check if blob exists
            if not blob.exists():
                raise FileNotFoundError(f"Blob {bucket_name}/{blob_name} not found")

            blob.download_to_file(file_obj)

            self.logger.debug(f"Downloaded {bucket_name}/{blob_name} to file")

        except Exception as e:
            self.logger.error(
                f"Failed to download {bucket_name}/{blob_name} to file: {str(e)}"
            )
            raise

    @retry_on_failure()
    def list_blobs(
        self, bucket_name: str, prefix: str = None, max_results: int = None
    ) -> list[storage.Blob]:
        """List blobs in bucket with optional prefix filter"""
        try:
            bucket = self.get_bucket(bucket_name)

            blobs = list(bucket.list_blobs(prefix=prefix, max_results=max_results))

            self.logger.debug(
                f"Listed {len(blobs)} blobs from {bucket_name} with prefix '{prefix}'"
            )
            return blobs

        except Exception as e:
            self.logger.error(f"Failed to list blobs in {bucket_name}: {str(e)}")
            raise

    @retry_on_failure()
    def blob_exists(self, bucket_name: str, blob_name: str) -> bool:
        """Check if blob exists"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            exists = blob.exists()
            self.logger.debug(f"Blob {bucket_name}/{blob_name} exists: {exists}")
            return exists

        except Exception as e:
            self.logger.error(
                f"Failed to check existence of {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    @retry_on_failure()
    def delete_blob(self, bucket_name: str, blob_name: str) -> bool:
        """Delete a blob"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if blob.exists():
                blob.delete()
                self.logger.debug(f"Deleted blob {bucket_name}/{blob_name}")
                return True
            else:
                self.logger.warning(f"Blob {bucket_name}/{blob_name} does not exist")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete {bucket_name}/{blob_name}: {str(e)}")
            raise

    @retry_on_failure()
    def get_blob_metadata(
        self, bucket_name: str, blob_name: str
    ) -> Optional[dict[str, Any]]:
        """Get blob metadata"""
        try:
            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return None

            # Reload to get latest metadata
            blob.reload()

            metadata = {
                "name": blob.name,
                "bucket": blob.bucket.name,
                "size": blob.size,
                "created": blob.time_created,
                "updated": blob.updated,
                "content_type": blob.content_type,
                "etag": blob.etag,
                "generation": blob.generation,
                "metadata": blob.metadata or {},
                "storage_class": blob.storage_class,
            }

            self.logger.debug(f"Retrieved metadata for {bucket_name}/{blob_name}")
            return metadata

        except Exception as e:
            self.logger.error(
                f"Failed to get metadata for {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    @retry_on_failure()
    def copy_blob(
        self, source_bucket: str, source_blob: str, dest_bucket: str, dest_blob: str
    ) -> bool:
        """Copy blob from source to destination"""
        try:
            source_bucket_obj = self.get_bucket(source_bucket)
            dest_bucket_obj = self.get_bucket(dest_bucket)

            source_blob_obj = source_bucket_obj.blob(source_blob)

            if not source_blob_obj.exists():
                raise FileNotFoundError(
                    f"Source blob {source_bucket}/{source_blob} not found"
                )

            # Copy the blob
            dest_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)

            self.logger.debug(
                f"Copied {source_bucket}/{source_blob} to {dest_bucket}/{dest_blob}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy blob: {str(e)}")
            raise

    @retry_on_failure()
    def generate_signed_url(
        self,
        bucket_name: str,
        blob_name: str,
        expiration_hours: int = 1,
        method: str = "GET",
    ) -> str:
        """Generate a signed URL for blob access"""
        try:
            from datetime import timedelta

            bucket = self.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            url = blob.generate_signed_url(
                expiration=timedelta(hours=expiration_hours), method=method
            )

            self.logger.debug(f"Generated signed URL for {bucket_name}/{blob_name}")
            return url

        except Exception as e:
            self.logger.error(
                f"Failed to generate signed URL for {bucket_name}/{blob_name}: {str(e)}"
            )
            raise

    def close(self):
        """Close the client and clean up resources"""
        try:
            # Close connection pool
            if hasattr(self, "connection_pool"):
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.connection_pool.shutdown())
                except RuntimeError:
                    # No event loop running
                    pass

            # Clear cached buckets
            self._buckets.clear()

            # Close client if it exists
            if self._client:
                if hasattr(self._client, "close"):
                    self._client.close()
                self._client = None

            self.logger.debug("GCS client closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing GCS client: {str(e)}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
