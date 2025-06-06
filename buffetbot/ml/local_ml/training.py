"""
Advanced Training Pipeline - Hyperparameter optimization and model comparison
Uses Optuna for efficient hyperparameter search with zero cloud costs
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from .models import LocalMLModel, ModelPerformance, ModelType, create_model


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""

    model_types: list[ModelType]
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    hyperparameter_trials: int = 50
    optimization_timeout_minutes: int = 30
    random_state: int = 42
    parallel_training: bool = True
    early_stopping_patience: int = 10


@dataclass
class TrainingResult:
    """Results from model training"""

    model: LocalMLModel
    performance: ModelPerformance
    cross_validation_scores: dict[str, float]
    hyperparameters: dict[str, Any]
    training_config: TrainingConfig
    feature_importance: Optional[pd.Series] = None


class ModelTrainer:
    """Individual model trainer with hyperparameter optimization"""

    def __init__(self, model_type: ModelType, config: TrainingConfig):
        self.model_type = model_type
        self.config = config
        self.logger = logging.getLogger(f"ml.trainer.{model_type.value}")

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            """Optuna objective function"""
            # Get feature columns from training data
            feature_columns = (
                list(X_train.columns)
                if hasattr(X_train, "columns")
                else [f"feature_{i}" for i in range(X_train.shape[1])]
            )

            # Define hyperparameter search spaces for each model type
            if self.model_type == ModelType.XGBOOST:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                }

            elif self.model_type == ModelType.LIGHTGBM:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                }

            elif self.model_type == ModelType.RANDOM_FOREST:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                }

            elif self.model_type in [
                ModelType.RIDGE_REGRESSION,
                ModelType.LASSO_REGRESSION,
            ]:
                params = {"alpha": trial.suggest_float("alpha", 0.01, 100, log=True)}

            elif self.model_type == ModelType.ELASTIC_NET:
                params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 100, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
                }

            else:  # LINEAR_REGRESSION
                params = {}

            # Create and train model with suggested parameters
            model = create_model(
                self.model_type, params, feature_columns, target_column="target"
            )
            try:
                # Convert to DataFrame/Series if needed for train method
                if isinstance(X_train, np.ndarray):
                    X_train_df = pd.DataFrame(X_train, columns=feature_columns)
                    y_train_series = pd.Series(y_train, name="target")
                    X_val_df = pd.DataFrame(X_val, columns=feature_columns)
                    y_val_series = pd.Series(y_val, name="target")
                else:
                    X_train_df, y_train_series = X_train, y_train
                    X_val_df, y_val_series = X_val, y_val

                # Train without internal validation since we're doing our own
                performance = model.train(
                    X_train_df, y_train_series, validation_split=0
                )
                predictions = model.predict(X_val_df)

                # Calculate validation R² score
                from sklearn.metrics import r2_score

                val_r2 = r2_score(y_val_series, predictions)

                return val_r2
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return -999  # Return very low score for failed trials

        # Create and run Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        # Set timeout
        timeout = self.config.optimization_timeout_minutes * 60

        try:
            study.optimize(
                objective,
                n_trials=self.config.hyperparameter_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(
                f"Hyperparameter optimization complete - "
                f"Best R²: {best_score:.4f}, "
                f"Trials: {len(study.trials)}"
            )

            return best_params

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {}  # Return empty params for default model

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_hyperparameters: bool = True,
    ) -> TrainingResult:
        """Train a single model with optional hyperparameter optimization"""

        start_time = datetime.utcnow()

        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            self.logger.info(
                f"Optimizing hyperparameters for {self.model_type.value}..."
            )
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = {}

        # Create model with optimized parameters
        feature_columns = (
            list(X_train.columns)
            if hasattr(X_train, "columns")
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )
        model = create_model(self.model_type, best_params, feature_columns, "target")

        # Train model on full training data
        full_X = (
            pd.concat([X_train, X_val])
            if hasattr(X_train, "columns")
            else pd.DataFrame(np.vstack([X_train, X_val]), columns=feature_columns)
        )
        full_y = (
            pd.concat([y_train, y_val])
            if hasattr(y_train, "name")
            else pd.Series(np.concatenate([y_train, y_val]), name="target")
        )

        performance = model.train(
            full_X, full_y, validation_split=self.config.validation_split
        )

        # Perform cross-validation
        cv_scores = model.cross_validate(X_train, y_train)

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
            except:
                pass

        training_time = (datetime.utcnow() - start_time).total_seconds()

        self.logger.info(
            f"Model {self.model_type.value} trained - "
            f"R²: {performance.r2_score:.4f}, "
            f"CV R²: {cv_scores['mean_r2']:.4f} ± {cv_scores['std_r2']:.4f}, "
            f"Time: {training_time:.1f}s"
        )

        return TrainingResult(
            model=model,
            performance=performance,
            cross_validation_scores=cv_scores,
            hyperparameters=best_params,
            training_config=self.config,
            feature_importance=feature_importance,
        )


class TrainingPipeline:
    """Advanced training pipeline with model comparison and selection"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("ml.training_pipeline")
        self.results: list[TrainingResult] = []

    def train_all_models(
        self, X: pd.DataFrame, y: pd.Series, optimize_hyperparameters: bool = True
    ) -> list[TrainingResult]:
        """Train all configured models and compare performance"""

        self.logger.info(
            f"Starting training pipeline with {len(self.config.model_types)} models"
        )

        # Ensure minimum data size for proper splits
        if len(X) < 50:
            raise ValueError(
                f"Dataset too small ({len(X)} samples). Minimum 50 samples required."
            )

        # Split data into train/validation/test sets with minimum sizes
        total_samples = len(X)
        test_size = max(int(total_samples * self.config.test_split), 10)
        val_size = max(int(total_samples * self.config.validation_split), 10)
        train_size = total_samples - test_size - val_size

        # Ensure train size is reasonable
        if train_size < 20:
            # Adjust splits to ensure minimum training size
            train_size = max(20, int(total_samples * 0.6))
            val_size = max(10, int((total_samples - train_size) * 0.5))
            test_size = total_samples - train_size - val_size

        train_end = train_size
        val_end = train_size + val_size

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end].copy()
        X_val = X.iloc[train_end:val_end].copy()
        y_val = y.iloc[train_end:val_end].copy()
        X_test = X.iloc[val_end:].copy()
        y_test = y.iloc[val_end:].copy()

        self.logger.info(
            f"Data split - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Validate that all splits have data
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"Invalid data split - Train: {len(X_train)}, Val: {len(X_val)}"
            )

        # Ensure features are properly formatted
        feature_columns = list(X.columns)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Train models (parallel or sequential)
        if self.config.parallel_training:
            results = self._train_models_parallel(
                X_train, y_train, X_val, y_val, optimize_hyperparameters
            )
        else:
            results = self._train_models_sequential(
                X_train, y_train, X_val, y_val, optimize_hyperparameters
            )

        # Evaluate on test set
        for result in results:
            test_predictions = result.model.predict(X_test)
            from sklearn.metrics import mean_squared_error, r2_score

            test_r2 = r2_score(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

            self.logger.info(
                f"Test performance {result.model.config.model_type.value}: "
                f"R² = {test_r2:.4f}, RMSE = {test_rmse:.6f}"
            )

        self.results = results
        return results

    def _train_models_sequential(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_hyperparameters: bool,
    ) -> list[TrainingResult]:
        """Train models sequentially"""
        results = []

        for model_type in self.config.model_types:
            trainer = ModelTrainer(model_type, self.config)
            result = trainer.train_model(
                X_train, y_train, X_val, y_val, optimize_hyperparameters
            )
            results.append(result)

        return results

    def _train_models_parallel(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_hyperparameters: bool,
    ) -> list[TrainingResult]:
        """Train models in parallel using ThreadPoolExecutor"""

        def train_single_model(model_type):
            trainer = ModelTrainer(model_type, self.config)
            return trainer.train_model(
                X_train, y_train, X_val, y_val, optimize_hyperparameters
            )

        with ThreadPoolExecutor(
            max_workers=min(len(self.config.model_types), 4)
        ) as executor:
            futures = [
                executor.submit(train_single_model, model_type)
                for model_type in self.config.model_types
            ]

            results = []
            for future in futures:
                try:
                    result = future.result(
                        timeout=self.config.optimization_timeout_minutes * 60
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Model training failed: {e}")

        return results

    def get_best_model(self, metric: str = "r2_score") -> Optional[TrainingResult]:
        """Get the best performing model based on specified metric"""
        if not self.results:
            return None

        if metric == "r2_score":
            return max(self.results, key=lambda x: x.performance.r2_score)
        elif metric == "rmse":
            return min(self.results, key=lambda x: x.performance.rmse)
        elif metric == "cv_r2":
            return max(self.results, key=lambda x: x.cross_validation_scores["mean_r2"])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison table of all trained models"""
        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for result in self.results:
            comparison_data.append(
                {
                    "Model": result.model.config.model_type.value,
                    "R²_Score": result.performance.r2_score,
                    "RMSE": result.performance.rmse,
                    "MAE": result.performance.mae,
                    "MAPE_%": result.performance.mape,
                    "CV_R²_Mean": result.cross_validation_scores["mean_r2"],
                    "CV_R²_Std": result.cross_validation_scores["std_r2"],
                    "Training_Time_s": result.performance.training_time_seconds,
                    "Prediction_Time_ms": result.performance.prediction_time_ms,
                }
            )

        df = pd.DataFrame(comparison_data)
        return df.sort_values("R²_Score", ascending=False)

    def save_results(self, file_path: str) -> None:
        """Save training results to file"""
        results_data = {
            "config": {
                "model_types": [mt.value for mt in self.config.model_types],
                "validation_split": self.config.validation_split,
                "test_split": self.config.test_split,
                "hyperparameter_trials": self.config.hyperparameter_trials,
                "training_date": datetime.utcnow().isoformat(),
            },
            "results": [],
        }

        for result in self.results:
            result_data = {
                "model_type": result.model.config.model_type.value,
                "performance": {
                    "r2_score": result.performance.r2_score,
                    "mse": result.performance.mse,
                    "mae": result.performance.mae,
                    "rmse": result.performance.rmse,
                    "mape": result.performance.mape,
                    "training_time_seconds": result.performance.training_time_seconds,
                },
                "cross_validation": result.cross_validation_scores,
                "hyperparameters": result.hyperparameters,
            }
            results_data["results"].append(result_data)

        with open(file_path, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Training results saved to {file_path}")


def create_default_training_config(
    model_types: list[ModelType] = None, quick_mode: bool = False
) -> TrainingConfig:
    """Create a default training configuration"""

    if model_types is None:
        model_types = [
            ModelType.LINEAR_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
        ]

    if quick_mode:
        # Faster configuration for development/testing
        return TrainingConfig(
            model_types=model_types,
            validation_split=0.2,
            test_split=0.1,
            hyperparameter_trials=10,
            optimization_timeout_minutes=5,
            parallel_training=True,
        )
    else:
        # Full configuration for production
        return TrainingConfig(
            model_types=model_types,
            validation_split=0.2,
            test_split=0.1,
            hyperparameter_trials=50,
            optimization_timeout_minutes=30,
            parallel_training=True,
        )
