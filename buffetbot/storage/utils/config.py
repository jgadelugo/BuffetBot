"""
GCS Configuration Management

Handles configuration loading from environment variables and Terraform outputs.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GCSConfig:
    """Configuration for Google Cloud Storage integration"""

    project_id: str
    data_bucket: str
    archive_bucket: str
    backup_bucket: str
    temp_bucket: str
    service_account_email: str
    service_account_key_path: Optional[str] = None
    kms_key_id: Optional[str] = None
    region: str = "us-central1"
    max_connections: int = 50
    retry_attempts: int = 3
    timeout_seconds: int = 60
    compression: str = "snappy"

    @classmethod
    def from_environment(cls) -> "GCSConfig":
        """Load configuration from environment variables"""
        logger.info("Loading GCS configuration from environment variables")

        try:
            return cls(
                project_id=cls._get_required_env("GOOGLE_CLOUD_PROJECT"),
                data_bucket=cls._get_required_env("GCS_DATA_BUCKET"),
                archive_bucket=cls._get_required_env("GCS_ARCHIVE_BUCKET"),
                backup_bucket=cls._get_required_env("GCS_BACKUP_BUCKET"),
                temp_bucket=cls._get_required_env("GCS_TEMP_BUCKET"),
                service_account_email=cls._get_required_env(
                    "GCS_SERVICE_ACCOUNT_EMAIL"
                ),
                service_account_key_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                kms_key_id=os.getenv("GCS_KMS_KEY_ID"),
                region=os.getenv("GCS_REGION", "us-central1"),
                max_connections=int(os.getenv("GCS_MAX_CONNECTIONS", "50")),
                retry_attempts=int(os.getenv("GCS_RETRY_ATTEMPTS", "3")),
                timeout_seconds=int(os.getenv("GCS_TIMEOUT_SECONDS", "60")),
                compression=os.getenv("GCS_COMPRESSION", "snappy"),
            )
        except Exception as e:
            logger.error(f"Failed to load configuration from environment: {e}")
            raise

    @classmethod
    def from_terraform_output(cls, config_file: str) -> "GCSConfig":
        """Load configuration from Terraform output file"""
        logger.info(f"Loading GCS configuration from Terraform output: {config_file}")

        try:
            with open(config_file) as f:
                config = json.load(f)

            storage_config = config.get("storage_config", {}).get("value", {})
            deployment_vars = config.get("deployment_variables", {}).get("value", {})

            return cls(
                project_id=deployment_vars.get("GOOGLE_CLOUD_PROJECT"),
                data_bucket=storage_config.get("data_bucket_name"),
                archive_bucket=storage_config.get("archive_bucket_name"),
                backup_bucket=storage_config.get("backup_bucket_name"),
                temp_bucket=storage_config.get("temp_bucket_name"),
                service_account_email=storage_config.get(
                    "storage_service_account_email"
                ),
                kms_key_id=storage_config.get("storage_kms_key_id"),
                region=storage_config.get("region", "us-central1"),
            )
        except Exception as e:
            logger.error(f"Failed to load configuration from Terraform output: {e}")
            raise

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "GCSConfig":
        """Create configuration from dictionary"""
        return cls(**config_dict)

    @staticmethod
    def _get_required_env(key: str) -> str:
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value

    def validate(self) -> bool:
        """Validate configuration values"""
        required_fields = [
            "project_id",
            "data_bucket",
            "archive_bucket",
            "backup_bucket",
            "temp_bucket",
            "service_account_email",
        ]

        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(
                    f"Required configuration field '{field}' is missing or empty"
                )

        if self.max_connections <= 0:
            raise ValueError("max_connections must be positive")

        if self.retry_attempts < 0:
            raise ValueError("retry_attempts cannot be negative")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "project_id": self.project_id,
            "data_bucket": self.data_bucket,
            "archive_bucket": self.archive_bucket,
            "backup_bucket": self.backup_bucket,
            "temp_bucket": self.temp_bucket,
            "service_account_email": self.service_account_email,
            "service_account_key_path": self.service_account_key_path,
            "kms_key_id": self.kms_key_id,
            "region": self.region,
            "max_connections": self.max_connections,
            "retry_attempts": self.retry_attempts,
            "timeout_seconds": self.timeout_seconds,
            "compression": self.compression,
        }

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
