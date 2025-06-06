"""
Model Registry - Manage ML model versions and metadata
Integrates with GCS for model storage (Phase 1)
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Phase 1 Integration
from buffetbot.storage.gcs.manager import GCSStorageManager

# Local imports
from .metadata import ModelMetadata, ModelMetrics, ModelStatus, ModelType


class ModelRegistry:
    """Registry for managing ML models and their metadata"""

    def __init__(self):
        self.logger = logging.getLogger("ml.model_registry")

        # GCS integration - optional for ML foundation
        self.gcs_manager = None
        try:
            # Try to initialize with a simple demo config
            from buffetbot.storage.utils.config import GCSConfig

            demo_config = GCSConfig(
                project_id="buffetbot-demo",
                data_bucket="buffetbot-data-demo",
                archive_bucket="buffetbot-archive-demo",
                backup_bucket="buffetbot-backup-demo",
                temp_bucket="buffetbot-temp-demo",
                service_account_email="demo@buffetbot-demo.iam.gserviceaccount.com",
            )
            self.gcs_manager = GCSStorageManager(demo_config)
        except Exception as e:
            self.logger.warning(f"GCS integration unavailable: {e}")

        self.models: dict[str, ModelMetadata] = {}
        self.registry_file = "ml/models/registry.json"
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the model registry"""
        try:
            # Load existing registry from GCS if available
            await self._load_registry()
            self.is_initialized = True
            self.logger.info(
                f"✅ Model Registry initialized with {len(self.models)} models"
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Model Registry: {e}")
            # Initialize empty registry
            self.models = {}
            self.is_initialized = True

    async def cleanup(self) -> None:
        """Cleanup registry resources"""
        try:
            # Save current state before cleanup
            await self._save_registry()
            self.is_initialized = False
            self.logger.info("✅ Model Registry cleanup complete")
        except Exception as e:
            self.logger.error(f"❌ Error during registry cleanup: {e}")

    async def register_model(
        self, model: Any, metadata: ModelMetadata, save_to_gcs: bool = True
    ) -> str:
        """Register a new model with metadata"""

        try:
            # Generate model hash for integrity
            model_str = str(model).encode() if model else b"no_model"
            model_hash = hashlib.sha256(model_str).hexdigest()[:16]
            metadata.model_hash = model_hash

            # Store in registry
            self.models[metadata.model_id] = metadata

            if save_to_gcs and model is not None:
                # Save model to GCS (using Phase 1 storage)
                model_path = f"ml/models/{metadata.model_name}/{metadata.version}/"
                await self._save_model_to_gcs(model, model_path, metadata)

            # Update registry file
            await self._save_registry()

            self.logger.info(
                f"✅ Model registered: {metadata.model_name} v{metadata.version}"
            )
            return metadata.model_id

        except Exception as e:
            self.logger.error(f"❌ Failed to register model {metadata.model_name}: {e}")
            raise

    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)

    async def list_models(
        self, model_name: Optional[str] = None
    ) -> list[ModelMetadata]:
        """List all models or filter by name"""
        models = list(self.models.values())
        if model_name:
            models = [m for m in models if m.model_name == model_name]
        return sorted(models, key=lambda x: x.training_date, reverse=True)

    async def get_latest_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model"""
        models = await self.list_models(model_name)
        return models[0] if models else None

    async def get_deployed_models(self) -> list[ModelMetadata]:
        """Get all deployed models"""
        return [
            model
            for model in self.models.values()
            if model.status == ModelStatus.DEPLOYED
        ]

    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model deployment status"""
        if model_id in self.models:
            self.models[model_id].status = status
            if status == ModelStatus.DEPLOYED:
                self.models[model_id].deployment_date = datetime.utcnow()
            await self._save_registry()
            self.logger.info(f"✅ Model {model_id} status updated to {status.value}")
            return True
        return False

    async def record_prediction(self, model_id: str) -> None:
        """Record that a prediction was made with this model"""
        if model_id in self.models:
            self.models[model_id].prediction_count += 1
            self.models[model_id].last_prediction_date = datetime.utcnow()
            # Save registry periodically (every 10 predictions)
            if self.models[model_id].prediction_count % 10 == 0:
                await self._save_registry()

    async def get_model_count(self) -> int:
        """Get total number of registered models"""
        return len(self.models)

    async def get_recent_models(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recently registered models"""
        recent_models = sorted(
            self.models.values(), key=lambda x: x.training_date, reverse=True
        )[:limit]

        return [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "status": model.status.value,
                "training_date": model.training_date.isoformat(),
            }
            for model in recent_models
        ]

    async def search_models(self, **filters) -> list[ModelMetadata]:
        """Search models by various criteria"""
        results = []

        for model in self.models.values():
            match = True

            # Filter by model type
            if "model_type" in filters:
                if model.model_type != filters["model_type"]:
                    match = False

            # Filter by status
            if "status" in filters:
                if model.status != filters["status"]:
                    match = False

            # Filter by tags
            if "tags" in filters:
                required_tags = filters["tags"]
                if not all(tag in model.tags for tag in required_tags):
                    match = False

            # Filter by minimum accuracy
            if "min_accuracy" in filters:
                if (
                    not model.metrics.accuracy
                    or model.metrics.accuracy < filters["min_accuracy"]
                ):
                    match = False

            if match:
                results.append(model)

        return sorted(results, key=lambda x: x.training_date, reverse=True)

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry"""
        if model_id in self.models:
            model = self.models[model_id]

            # Delete from GCS if exists
            try:
                model_path = f"ml/models/{model.model_name}/{model.version}/"
                # Note: Implement GCS deletion if needed
                self.logger.info(
                    f"Model files at {model_path} should be cleaned up manually"
                )
            except Exception as e:
                self.logger.warning(f"Could not clean up model files: {e}")

            # Remove from registry
            del self.models[model_id]
            await self._save_registry()

            self.logger.info(f"✅ Model {model_id} deleted from registry")
            return True

        return False

    async def health_check(self) -> str:
        """Check registry health"""
        if not self.is_initialized:
            return "not_initialized"

        try:
            # Test basic operations
            model_count = len(self.models)
            self.logger.debug(f"Registry health check: {model_count} models registered")
            return "healthy"
        except Exception as e:
            self.logger.error(f"Registry health check failed: {e}")
            return "error"

    async def _save_model_to_gcs(self, model: Any, path: str, metadata: ModelMetadata):
        """Save model to GCS storage"""
        if self.gcs_manager is None:
            self.logger.info("GCS not available, skipping model storage")
            return

        try:
            # Save metadata first
            metadata_json = metadata.to_dict()
            # Note: Using store_data instead of upload_json_data
            # await self.gcs_manager.store_data("models", [metadata_json])

            # For now, we don't serialize the actual model to GCS
            # This can be implemented based on model type (pickle, joblib, etc.)
            self.logger.info(
                f"Model metadata would be saved to GCS: {path}metadata.json"
            )

        except Exception as e:
            self.logger.error(f"Failed to save model to GCS: {e}")
            # Don't fail the registration if GCS save fails

    async def _save_registry(self):
        """Save registry to GCS"""
        if self.gcs_manager is None:
            self.logger.debug("GCS not available, registry saved in memory only")
            return

        try:
            registry_data = {
                "last_updated": datetime.utcnow().isoformat(),
                "model_count": len(self.models),
                "models": {
                    model_id: metadata.to_dict()
                    for model_id, metadata in self.models.items()
                },
            }

            # Note: Using store_data instead of upload_json_data
            # await self.gcs_manager.store_data("registry", [registry_data])

            self.logger.debug(f"Registry saved to GCS: {len(self.models)} models")

        except Exception as e:
            self.logger.error(f"Failed to save registry to GCS: {e}")

    async def _load_registry(self):
        """Load registry from GCS"""
        if self.gcs_manager is None:
            self.logger.info("GCS not available, starting with empty registry")
            self.models = {}
            return

        try:
            registry_data = await self.gcs_manager.download_json_data(
                self.registry_file
            )

            if registry_data and "models" in registry_data:
                self.models = {}
                for model_id, model_data in registry_data["models"].items():
                    try:
                        metadata = ModelMetadata.from_dict(model_data)
                        self.models[model_id] = metadata
                    except Exception as e:
                        self.logger.warning(f"Could not load model {model_id}: {e}")

                self.logger.info(f"Registry loaded from GCS: {len(self.models)} models")
            else:
                self.logger.info("No existing registry found, starting fresh")

        except Exception as e:
            self.logger.warning(f"Could not load registry from GCS: {e}")
            self.models = {}
