"""
BuffetBot ML & AI Integration Package

This package provides machine learning capabilities for BuffetBot including:
- Model training and management
- Feature engineering
- Prediction services
- Cost monitoring
- Cloud upgrade interfaces

Phase 3 Implementation - Completely FREE with local ML services
"""

from .managers.ml_manager import MLManager
from .models.registry import ModelMetadata, ModelRegistry
from .monitoring.cost_monitor import MLCostMonitor

__version__ = "3.0.0"
__author__ = "BuffetBot Team"

__all__ = [
    "MLManager",
    "ModelRegistry",
    "ModelMetadata",
    "MLCostMonitor",
]
