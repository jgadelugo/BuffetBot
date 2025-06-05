"""
GCS Configuration for BuffetBot

Example configuration setup for Google Cloud Storage integration.
"""

import os

from buffetbot.storage.utils.config import GCSConfig


# Example configuration for development environment
def get_development_config() -> GCSConfig:
    """Get development configuration"""
    return GCSConfig(
        project_id="buffetbot-dev",
        data_bucket="buffetbot-data-dev",
        archive_bucket="buffetbot-archive-dev",
        backup_bucket="buffetbot-backup-dev",
        temp_bucket="buffetbot-temp-dev",
        service_account_email="buffetbot-storage@buffetbot-dev.iam.gserviceaccount.com",
        service_account_key_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        kms_key_id="projects/buffetbot-dev/locations/us-central1/keyRings/buffetbot/cryptoKeys/storage",
        region="us-central1",
        max_connections=25,
        retry_attempts=3,
        timeout_seconds=30,
        compression="snappy",
    )


# Example configuration for production environment
def get_production_config() -> GCSConfig:
    """Get production configuration"""
    return GCSConfig(
        project_id="buffetbot-prod",
        data_bucket="buffetbot-data-prod",
        archive_bucket="buffetbot-archive-prod",
        backup_bucket="buffetbot-backup-prod",
        temp_bucket="buffetbot-temp-prod",
        service_account_email="buffetbot-storage@buffetbot-prod.iam.gserviceaccount.com",
        service_account_key_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        kms_key_id="projects/buffetbot-prod/locations/us-central1/keyRings/buffetbot/cryptoKeys/storage",
        region="us-central1",
        max_connections=50,
        retry_attempts=5,
        timeout_seconds=60,
        compression="snappy",
    )


def get_config() -> GCSConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return get_production_config()
    elif env == "development":
        return get_development_config()
    else:
        # Try to load from environment variables
        try:
            return GCSConfig.from_environment()
        except ValueError:
            # Fall back to development config
            return get_development_config()
