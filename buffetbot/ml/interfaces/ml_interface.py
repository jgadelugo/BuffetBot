"""
Abstract ML Interface - Common interface for all ML services
Enables seamless switching between local and cloud implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class MLServiceCapabilities:
    """Capabilities of an ML service"""

    supports_training: bool = True
    supports_prediction: bool = True
    supports_batch_prediction: bool = True
    supports_hyperparameter_optimization: bool = False
    supports_feature_importance: bool = False
    supports_model_versioning: bool = False
    max_features: int = 1000
    max_training_samples: int = 1000000
    cost_per_training_hour: float = 0.0
    cost_per_prediction: float = 0.0


class MLInterface(ABC):
    """Abstract interface for ML services (local and cloud)"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the ML service"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup ML service resources"""
        pass

    @abstractmethod
    def get_capabilities(self) -> MLServiceCapabilities:
        """Get service capabilities"""
        pass

    @abstractmethod
    async def train_model(
        self,
        training_data: pd.DataFrame,
        target_column: str,
        model_type: str,
        hyperparameters: dict[str, Any] = None,
    ) -> str:
        """
        Train a model and return model ID

        Args:
            training_data: Training dataset
            target_column: Target column name
            model_type: Type of model to train
            hyperparameters: Model hyperparameters

        Returns:
            Model ID for the trained model
        """
        pass

    @abstractmethod
    async def predict_single(
        self, model_id: str, features: dict[str, Union[float, int, str]]
    ) -> dict[str, Any]:
        """
        Make a single prediction

        Args:
            model_id: ID of trained model
            features: Feature values

        Returns:
            Prediction result with metadata
        """
        pass

    @abstractmethod
    async def predict_batch(
        self, model_id: str, features_list: list[dict[str, Union[float, int, str]]]
    ) -> list[dict[str, Any]]:
        """
        Make batch predictions

        Args:
            model_id: ID of trained model
            features_list: List of feature dictionaries

        Returns:
            List of prediction results
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Get information about a trained model

        Args:
            model_id: ID of the model

        Returns:
            Model metadata and performance information
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models

        Returns:
            List of model information dictionaries
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete a trained model

        Args:
            model_id: ID of model to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_cost_estimate(self, operation: str, **kwargs) -> dict[str, float]:
        """
        Get cost estimate for an operation

        Args:
            operation: Operation type ('train', 'predict', 'batch_predict')
            **kwargs: Operation-specific parameters

        Returns:
            Cost breakdown dictionary
        """
        pass

    @abstractmethod
    def get_service_health(self) -> dict[str, Any]:
        """
        Get service health status

        Returns:
            Health status dictionary
        """
        pass
