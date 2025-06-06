"""
Enhanced Local ML Models - XGBoost, LightGBM, and more
All models run locally with zero cloud costs and superior performance
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Enhanced ML Libraries
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler


class ModelType(Enum):
    """Supported model types for local ML"""

    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    r2_score: float
    mse: float
    mae: float
    rmse: float
    mape: float = 0.0  # Mean Absolute Percentage Error
    training_time_seconds: float = 0.0
    prediction_time_ms: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for ML models"""

    model_type: ModelType
    hyperparameters: dict[str, Any]
    use_feature_scaling: bool = True
    cross_validation_folds: int = 5
    random_state: int = 42


class LocalMLModel(ABC):
    """Abstract base class for all local ML models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.use_feature_scaling else None
        self.feature_names: list[str] = []
        self.target_name: str = ""
        self.is_trained = False
        self.performance: Optional[ModelPerformance] = None
        self.logger = logging.getLogger(f"ml.{config.model_type.value}")

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying ML model"""
        pass

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_split: float = 0.2,
    ) -> ModelPerformance:
        """Train the model with performance evaluation"""
        start_time = datetime.utcnow()

        # Convert inputs to DataFrame/Series if they're numpy arrays
        if isinstance(X, np.ndarray):
            if not hasattr(self, "feature_names") or self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)

        if isinstance(y, np.ndarray):
            if not hasattr(self, "target_name") or self.target_name is None:
                self.target_name = "target"
            y = pd.Series(y, name=self.target_name)

        # Store feature information
        if not hasattr(self, "feature_names") or self.feature_names is None:
            self.feature_names = list(X.columns)
        if not hasattr(self, "target_name") or self.target_name is None:
            self.target_name = y.name or "target"

        # Split data for validation
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = X, X.iloc[:0]  # Empty validation set
            y_train, y_val = y, y.iloc[:0]  # Empty validation set

        # Feature scaling if enabled
        if self.scaler:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val), columns=X_val.columns, index=X_val.index
            )
        else:
            X_train_scaled, X_val_scaled = X_train, X_val

        # Create and train model
        self.model = self._create_model()
        self._fit_model(X_train_scaled, y_train)

        # Mark as trained before making predictions
        self.is_trained = True

        # Calculate training time
        training_time = (datetime.utcnow() - start_time).total_seconds()

        # Evaluate performance
        if validation_split > 0 and len(X_val) > 0:
            y_pred = self.predict(X_val)
            self.performance = self._calculate_performance(y_val, y_pred, training_time)
        else:
            # Use training data for performance calculation when no validation split
            y_pred = self.predict(X_train)
            self.performance = self._calculate_performance(
                y_train, y_pred, training_time
            )
        self.logger.info(
            f"Model trained - R²: {self.performance.r2_score:.4f}, "
            f"RMSE: {self.performance.rmse:.6f}, "
            f"Time: {training_time:.2f}s"
        )

        return self.performance

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        start_time = datetime.utcnow()

        # Feature scaling if enabled
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), columns=X.columns, index=X.index
            )
        else:
            X_scaled = X

        predictions = self.model.predict(X_scaled)

        # Update prediction timing
        if self.performance:
            pred_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.performance.prediction_time_ms = pred_time_ms

        return predictions

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Perform cross-validation"""
        if self.scaler:
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        else:
            X_scaled = X

        model = self._create_model()

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)

        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="r2")

        return {
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
            "min_r2": scores.min(),
            "max_r2": scores.max(),
        }

    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the specific model implementation"""
        pass

    def _calculate_performance(
        self, y_true: pd.Series, y_pred: np.ndarray, training_time: float
    ) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return ModelPerformance(
            r2_score=r2,
            mse=mse,
            mae=mae,
            rmse=rmse,
            mape=mape,
            training_time_seconds=training_time,
        )

    def save_model(self, file_path: str) -> None:
        """Save model to file"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "performance": self.performance,
        }
        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, file_path: str) -> None:
        """Load model from file"""
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.config = model_data["config"]
        self.feature_names = model_data["feature_names"]
        self.target_name = model_data["target_name"]
        self.performance = model_data["performance"]
        self.is_trained = True


class LinearRegressionModel(LocalMLModel):
    """Enhanced Linear Regression with regularization options"""

    def _create_model(self) -> LinearRegression:
        params = self.config.hyperparameters

        if self.config.model_type == ModelType.RIDGE_REGRESSION:
            return Ridge(**params)
        elif self.config.model_type == ModelType.LASSO_REGRESSION:
            return Lasso(**params)
        elif self.config.model_type == ModelType.ELASTIC_NET:
            return ElasticNet(**params)
        else:
            return LinearRegression(**params)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)


class RandomForestModel(LocalMLModel):
    """Enhanced Random Forest with optimized parameters"""

    def _create_model(self) -> RandomForestRegressor:
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": self.config.random_state,
            "n_jobs": -1,  # Use all cores
        }
        default_params.update(self.config.hyperparameters)
        return RandomForestRegressor(**default_params)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        return importance


class XGBoostModel(LocalMLModel):
    """XGBoost implementation with hyperparameter optimization"""

    def _create_model(self) -> xgb.XGBRegressor:
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }
        default_params.update(self.config.hyperparameters)
        return xgb.XGBRegressor(**default_params)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)

    def get_feature_importance(self) -> pd.Series:
        """Get XGBoost feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        return importance


class LightGBMModel(LocalMLModel):
    """LightGBM implementation optimized for speed and accuracy"""

    def _create_model(self) -> lgb.LGBMRegressor:
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.config.random_state,
            "n_jobs": -1,
            "verbose": -1,  # Suppress output
        }
        default_params.update(self.config.hyperparameters)
        return lgb.LGBMRegressor(**default_params)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(
            X,
            y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )

    def get_feature_importance(self) -> pd.Series:
        """Get LightGBM feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        return importance


def create_model(
    model_type: ModelType,
    hyperparameters: dict[str, Any] = None,
    feature_columns: list[str] = None,
    target_column: str = "target",
) -> LocalMLModel:
    """Factory function to create ML models"""
    hyperparameters = hyperparameters or {}

    config = ModelConfig(model_type=model_type, hyperparameters=hyperparameters)

    model_map = {
        ModelType.LINEAR_REGRESSION: LinearRegressionModel,
        ModelType.RIDGE_REGRESSION: LinearRegressionModel,
        ModelType.LASSO_REGRESSION: LinearRegressionModel,
        ModelType.ELASTIC_NET: LinearRegressionModel,
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.XGBOOST: XGBoostModel,
        ModelType.LIGHTGBM: LightGBMModel,
    }

    model_class = model_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_class(config)

    # Set feature columns and target if provided
    if feature_columns:
        model.feature_names = feature_columns
    if target_column:
        model.target_name = target_column

    return model
