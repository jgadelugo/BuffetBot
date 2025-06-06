"""
ML Model Metadata Classes
Defines data structures for model information
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ModelStatus(Enum):
    """Model deployment status"""

    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelType(Enum):
    """Supported model types"""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    CUSTOM = "custom"


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    # Time series specific metrics
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    directional_accuracy: Optional[float] = None  # % of correct direction predictions

    # Custom metrics
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                result[field_name] = field_value
        return result


@dataclass
class ModelMetadata:
    """Complete model metadata structure"""

    # Required fields (no defaults)
    model_id: str
    model_name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    training_date: datetime
    training_duration_minutes: int
    training_data_size: int
    feature_columns: list[str]
    target_column: str
    metrics: ModelMetrics
    file_path: str
    model_hash: str

    # Optional fields (with defaults)
    validation_score: Optional[float] = None
    model_size_mb: float = 0.0
    cost_to_train: float = 0.0
    cost_per_prediction: float = 0.0
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: list[str] = field(default_factory=list)
    deployment_date: Optional[datetime] = None
    last_prediction_date: Optional[datetime] = None
    prediction_count: int = 0
    drift_detected: bool = False
    last_drift_check: Optional[datetime] = None
    performance_degradation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, ModelMetrics):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from dictionary"""
        # Handle datetime fields
        datetime_fields = [
            "training_date",
            "deployment_date",
            "last_prediction_date",
            "last_drift_check",
        ]
        for field in datetime_fields:
            if data.get(field) and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError:
                    data[field] = None

        # Handle enum fields
        if "model_type" in data and isinstance(data["model_type"], str):
            data["model_type"] = ModelType(data["model_type"])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = ModelStatus(data["status"])

        # Handle metrics
        if "metrics" in data and isinstance(data["metrics"], dict):
            data["metrics"] = ModelMetrics(**data["metrics"])

        return cls(**data)


@dataclass
class PredictionMetadata:
    """Metadata for individual predictions"""

    prediction_id: str
    model_id: str
    timestamp: datetime
    input_features: dict[str, Any]
    prediction: Any
    confidence: Optional[float] = None
    prediction_time_ms: Optional[float] = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        result = self.__dict__.copy()
        result["timestamp"] = self.timestamp.isoformat()
        return result
