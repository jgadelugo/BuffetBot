"""
Metadata Management System

Provides comprehensive metadata handling for storage operations with validation,
serialization, and extensible schema support following enterprise best practices.
"""

import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MetadataType(Enum):
    """Types of metadata supported by the system"""

    FILE = "file"
    SCHEMA = "schema"
    STORAGE = "storage"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    USER = "user"
    SYSTEM = "system"


class MetadataFormat(Enum):
    """Supported metadata serialization formats"""

    JSON = "json"
    YAML = "yaml"
    PARQUET = "parquet"
    AVRO = "avro"


class ValidationSeverity(Enum):
    """Validation severity levels"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of metadata validation"""

    is_valid: bool
    severity: ValidationSeverity
    field: Optional[str] = None
    message: Optional[str] = None
    code: Optional[str] = None
    context: Optional[dict[str, Any]] = None


class MetadataValidator(Protocol):
    """Protocol for metadata validators"""

    def validate(self, metadata: dict[str, Any]) -> list[ValidationResult]:
        """Validate metadata and return validation results"""
        ...


@dataclass
class BaseMetadata:
    """Base metadata class with common fields"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MetadataType = MetadataType.SYSTEM
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamps are timezone-aware"""
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)


@dataclass
class FileMetadata(BaseMetadata):
    """Metadata for files and storage objects"""

    type: MetadataType = field(default=MetadataType.FILE, init=False)

    # File identification
    path: str = ""
    filename: str = ""
    extension: str = ""

    # File properties
    size_bytes: int = 0
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None

    # Storage properties
    bucket: Optional[str] = None
    storage_class: Optional[str] = None
    compression: Optional[str] = None
    encryption: Optional[str] = None

    # Data properties
    record_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_version: Optional[str] = None
    data_format: Optional[str] = None

    # Lifecycle
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

    def calculate_checksums(self, content: bytes) -> None:
        """Calculate and set file checksums"""
        self.checksum_md5 = hashlib.md5(content).hexdigest()
        self.checksum_sha256 = hashlib.sha256(content).hexdigest()


@dataclass
class SchemaMetadata(BaseMetadata):
    """Metadata for data schemas"""

    type: MetadataType = field(default=MetadataType.SCHEMA, init=False)

    # Schema identification
    name: str = ""
    namespace: Optional[str] = None
    data_type: str = ""

    # Schema definition
    fields: list[dict[str, Any]] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    indexes: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)

    # Evolution tracking
    parent_version: Optional[str] = None
    breaking_changes: list[str] = field(default_factory=list)
    migration_script: Optional[str] = None
    compatibility_mode: str = "forward"

    # Documentation
    description: Optional[str] = None
    examples: list[dict[str, Any]] = field(default_factory=list)
    documentation_url: Optional[str] = None


@dataclass
class PerformanceMetadata(BaseMetadata):
    """Metadata for performance metrics"""

    type: MetadataType = field(default=MetadataType.PERFORMANCE, init=False)

    # Operation details
    operation_type: str = ""
    operation_id: Optional[str] = None

    # Timing metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Throughput metrics
    records_processed: Optional[int] = None
    bytes_processed: Optional[int] = None
    records_per_second: Optional[float] = None
    bytes_per_second: Optional[float] = None

    # Resource usage
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_io_mb: Optional[float] = None
    network_io_mb: Optional[float] = None

    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    retry_count: int = 0

    # Cost tracking
    estimated_cost_usd: Optional[float] = None
    cost_center: Optional[str] = None


@dataclass
class AuditMetadata(BaseMetadata):
    """Metadata for audit and compliance tracking"""

    type: MetadataType = field(default=MetadataType.AUDIT, init=False)

    # Event details
    event_type: str = ""
    event_id: Optional[str] = None
    session_id: Optional[str] = None

    # Actor information
    user_id: Optional[str] = None
    service_account: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Resource information
    resource_type: str = ""
    resource_id: Optional[str] = None
    resource_path: Optional[str] = None

    # Operation details
    action: str = ""
    outcome: str = ""  # success, failure, partial
    risk_level: str = "low"  # low, medium, high, critical

    # Compliance
    retention_period_days: Optional[int] = None
    classification: str = "internal"  # public, internal, confidential, restricted
    regulatory_tags: list[str] = field(default_factory=list)


class MetadataSerializer:
    """Handles metadata serialization and deserialization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def serialize(
        self, metadata: BaseMetadata, format_type: MetadataFormat = MetadataFormat.JSON
    ) -> str:
        """Serialize metadata to specified format"""
        try:
            if format_type == MetadataFormat.JSON:
                return self._serialize_json(metadata)
            elif format_type == MetadataFormat.YAML:
                return self._serialize_yaml(metadata)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
        except Exception as e:
            self.logger.error(f"Metadata serialization failed: {str(e)}")
            raise

    def deserialize(
        self,
        data: str,
        metadata_class: type,
        format_type: MetadataFormat = MetadataFormat.JSON,
    ) -> BaseMetadata:
        """Deserialize metadata from specified format"""
        try:
            if format_type == MetadataFormat.JSON:
                return self._deserialize_json(data, metadata_class)
            elif format_type == MetadataFormat.YAML:
                return self._deserialize_yaml(data, metadata_class)
            else:
                raise ValueError(f"Unsupported deserialization format: {format_type}")
        except Exception as e:
            self.logger.error(f"Metadata deserialization failed: {str(e)}")
            raise

    def _serialize_json(self, metadata: BaseMetadata) -> str:
        """Serialize to JSON with datetime handling"""

        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data = asdict(metadata)
        return json.dumps(data, default=datetime_handler, indent=2)

    def _deserialize_json(self, data: str, metadata_class: type) -> BaseMetadata:
        """Deserialize from JSON with datetime parsing"""
        parsed_data = json.loads(data)

        # Convert ISO format timestamps back to datetime objects
        for field_name in [
            "created_at",
            "updated_at",
            "start_time",
            "end_time",
            "expires_at",
            "last_accessed",
        ]:
            if field_name in parsed_data and parsed_data[field_name]:
                parsed_data[field_name] = datetime.fromisoformat(
                    parsed_data[field_name]
                )

        return metadata_class(**parsed_data)

    def _serialize_yaml(self, metadata: BaseMetadata) -> str:
        """Serialize to YAML format"""
        try:
            import yaml

            data = asdict(metadata)
            return yaml.dump(data, default_flow_style=False)
        except ImportError:
            self.logger.warning("PyYAML not available, falling back to JSON")
            return self._serialize_json(metadata)

    def _deserialize_yaml(self, data: str, metadata_class: type) -> BaseMetadata:
        """Deserialize from YAML format"""
        try:
            import yaml

            parsed_data = yaml.safe_load(data)
            return metadata_class(**parsed_data)
        except ImportError:
            self.logger.warning("PyYAML not available, trying JSON")
            return self._deserialize_json(data, metadata_class)


class StandardValidator:
    """Standard metadata validator with common validation rules"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(self, metadata: BaseMetadata) -> list[ValidationResult]:
        """Validate metadata using standard rules"""
        results = []

        # Validate required fields
        results.extend(self._validate_required_fields(metadata))

        # Validate field formats
        results.extend(self._validate_field_formats(metadata))

        # Validate business rules
        results.extend(self._validate_business_rules(metadata))

        return results

    def _validate_required_fields(
        self, metadata: BaseMetadata
    ) -> list[ValidationResult]:
        """Validate that required fields are present and not empty"""
        results = []

        required_fields = ["id", "type", "version", "created_at"]

        for field_name in required_fields:
            value = getattr(metadata, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field=field_name,
                        message=f"Required field '{field_name}' is missing or empty",
                        code="REQUIRED_FIELD_MISSING",
                    )
                )

        return results

    def _validate_field_formats(self, metadata: BaseMetadata) -> list[ValidationResult]:
        """Validate field formats and types"""
        results = []

        # Validate UUID format for ID
        try:
            uuid.UUID(metadata.id)
        except (ValueError, AttributeError):
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="id",
                    message="ID should be a valid UUID",
                    code="INVALID_UUID_FORMAT",
                )
            )

        # Validate version format (semantic versioning)
        if not self._is_valid_semver(metadata.version):
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field="version",
                    message="Version should follow semantic versioning (x.y.z)",
                    code="INVALID_VERSION_FORMAT",
                )
            )

        return results

    def _validate_business_rules(
        self, metadata: BaseMetadata
    ) -> list[ValidationResult]:
        """Validate business-specific rules"""
        results = []

        # Validate timestamp order
        if metadata.updated_at < metadata.created_at:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field="updated_at",
                    message="Updated timestamp cannot be before created timestamp",
                    code="INVALID_TIMESTAMP_ORDER",
                )
            )

        # File-specific validations
        if isinstance(metadata, FileMetadata):
            if metadata.size_bytes < 0:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field="size_bytes",
                        message="File size cannot be negative",
                        code="INVALID_FILE_SIZE",
                    )
                )

        return results

    def _is_valid_semver(self, version: str) -> bool:
        """Check if version follows semantic versioning"""
        try:
            parts = version.split(".")
            if len(parts) != 3:
                return False
            for part in parts:
                int(part)  # Check if each part is a number
            return True
        except (ValueError, AttributeError):
            return False


class MetadataManager:
    """Central manager for all metadata operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Metadata registry
        self._metadata_registry: dict[str, BaseMetadata] = {}

        # Type mappings
        self._type_mappings = {
            MetadataType.FILE: FileMetadata,
            MetadataType.PERFORMANCE: PerformanceMetadata,
        }

    def create_metadata(self, metadata_type: MetadataType, **kwargs) -> BaseMetadata:
        """Create new metadata instance of specified type"""
        try:
            metadata_class = self._type_mappings.get(metadata_type, BaseMetadata)
            metadata = metadata_class(**kwargs)

            # Register metadata
            self._metadata_registry[metadata.id] = metadata

            self.logger.info(
                f"Created {metadata_type.value} metadata with ID: {metadata.id}"
            )
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to create metadata: {str(e)}")
            raise

    def get_metadata(self, metadata_id: str) -> Optional[BaseMetadata]:
        """Retrieve metadata by ID"""
        return self._metadata_registry.get(metadata_id)

    def update_metadata(self, metadata_id: str, **updates) -> BaseMetadata:
        """Update existing metadata"""
        metadata = self.get_metadata(metadata_id)
        if not metadata:
            raise ValueError(f"Metadata with ID {metadata_id} not found")

        # Update fields
        for field, value in updates.items():
            if hasattr(metadata, field):
                setattr(metadata, field, value)

        # Update timestamp
        metadata.updated_at = datetime.now(timezone.utc)

        self.logger.info(f"Updated metadata with ID: {metadata_id}")
        return metadata

    def to_dict(self, metadata: BaseMetadata) -> dict[str, Any]:
        """Convert metadata to dictionary"""

        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        data = asdict(metadata)

        # Handle datetime objects
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()

        return data
