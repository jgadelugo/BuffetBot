"""
ML Validation Utilities
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MLValidator:
    """Utilities for validating ML data and models"""

    def __init__(self):
        self.logger = logging.getLogger("ml.validator")

    def validate_dataframe(
        self, df: pd.DataFrame, required_columns: list[str] = None
    ) -> dict[str, Any]:
        """Validate DataFrame for ML readiness"""
        validation_results = {"valid": True, "issues": [], "warnings": [], "stats": {}}

        # Basic checks
        if df.empty:
            validation_results["valid"] = False
            validation_results["issues"].append("DataFrame is empty")
            return validation_results

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_results["valid"] = False
                validation_results["issues"].append(
                    f"Missing required columns: {list(missing_cols)}"
                )

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            validation_results["warnings"].append(
                f"Found {missing_count} missing values"
            )

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["warnings"].append(
                f"Found {duplicate_count} duplicate rows"
            )

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            validation_results["warnings"].append(f"Found {inf_count} infinite values")

        # Collect statistics
        validation_results["stats"] = {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
            "infinite_values": inf_count,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.select_dtypes(include=[object]).columns),
        }

        return validation_results

    def validate_time_series_data(
        self, df: pd.DataFrame, datetime_col: str, target_col: str
    ) -> dict[str, Any]:
        """Validate time series data specifically"""
        validation_results = self.validate_dataframe(df, [datetime_col, target_col])

        # Time series specific checks
        try:
            dt_series = pd.to_datetime(df[datetime_col])

            # Check for proper sorting
            if not dt_series.is_monotonic_increasing:
                validation_results["warnings"].append("DateTime column is not sorted")

            # Check for gaps in time series
            time_diffs = dt_series.diff().dropna()
            if len(time_diffs.unique()) > 10:  # Too many different intervals
                validation_results["warnings"].append(
                    "Irregular time intervals detected"
                )

            # Check date range
            date_range = dt_series.max() - dt_series.min()
            validation_results["stats"]["date_range_days"] = date_range.days
            validation_results["stats"]["start_date"] = dt_series.min()
            validation_results["stats"]["end_date"] = dt_series.max()

        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Invalid datetime column: {e}")

        # Check target column
        try:
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                validation_results["valid"] = False
                validation_results["issues"].append("Target column must be numeric")

            # Check for sufficient variance
            target_std = df[target_col].std()
            if target_std == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("Target column has no variance")

            validation_results["stats"]["target_mean"] = df[target_col].mean()
            validation_results["stats"]["target_std"] = target_std
            validation_results["stats"]["target_range"] = (
                df[target_col].max() - df[target_col].min()
            )

        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Invalid target column: {e}")

        return validation_results

    def validate_feature_target_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any]:
        """Validate feature and target data for ML training"""
        validation_results = {"valid": True, "issues": [], "warnings": [], "stats": {}}

        # Check shape compatibility
        if len(X) != len(y):
            validation_results["valid"] = False
            validation_results["issues"].append(
                f"Feature and target lengths don't match: {len(X)} vs {len(y)}"
            )

        # Check for minimum sample size
        min_samples = max(50, X.shape[1] * 5)  # At least 5 samples per feature
        if len(X) < min_samples:
            validation_results["warnings"].append(
                f"Small dataset: {len(X)} samples, recommended: {min_samples}"
            )

        # Check feature matrix
        feature_validation = self.validate_dataframe(X)
        if not feature_validation["valid"]:
            validation_results["valid"] = False
            validation_results["issues"].extend(feature_validation["issues"])

        # Check for high correlation features
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            # Remove diagonal
            corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
            high_corr_pairs = (corr_matrix > 0.95).sum().sum()
            if high_corr_pairs > 0:
                validation_results["warnings"].append(
                    f"Found {high_corr_pairs} highly correlated feature pairs"
                )

        # Check target distribution
        if pd.api.types.is_numeric_dtype(y):
            # Continuous target
            validation_results["stats"]["target_type"] = "continuous"
            validation_results["stats"]["target_skewness"] = y.skew()
            validation_results["stats"]["target_kurtosis"] = y.kurtosis()
        else:
            # Categorical target
            validation_results["stats"]["target_type"] = "categorical"
            class_counts = y.value_counts()
            validation_results["stats"]["class_distribution"] = class_counts.to_dict()

            # Check for class imbalance
            min_class_ratio = class_counts.min() / class_counts.max()
            if min_class_ratio < 0.1:
                validation_results["warnings"].append(
                    f"Severe class imbalance detected: {min_class_ratio:.3f}"
                )

        validation_results["stats"]["feature_count"] = X.shape[1]
        validation_results["stats"]["sample_count"] = len(X)

        return validation_results

    def validate_model_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Validate model predictions"""
        validation_results = {"valid": True, "issues": [], "warnings": [], "stats": {}}

        # Check shape compatibility
        if len(y_true) != len(y_pred):
            validation_results["valid"] = False
            validation_results["issues"].append(
                f"True and predicted lengths don't match: {len(y_true)} vs {len(y_pred)}"
            )
            return validation_results

        # Check for missing values
        if np.isnan(y_pred).any():
            nan_count = np.isnan(y_pred).sum()
            validation_results["warnings"].append(
                f"Predictions contain {nan_count} NaN values"
            )

        # Check for infinite values
        if np.isinf(y_pred).any():
            inf_count = np.isinf(y_pred).sum()
            validation_results["warnings"].append(
                f"Predictions contain {inf_count} infinite values"
            )

        # Check prediction range vs true values
        if np.isfinite(y_true).all() and np.isfinite(y_pred).all():
            true_range = y_true.max() - y_true.min()
            pred_range = y_pred.max() - y_pred.min()

            if pred_range > true_range * 2:
                validation_results["warnings"].append(
                    "Predictions have much larger range than true values"
                )
            elif pred_range < true_range * 0.5:
                validation_results["warnings"].append(
                    "Predictions have much smaller range than true values"
                )

        validation_results["stats"]["prediction_count"] = len(y_pred)
        validation_results["stats"]["true_mean"] = np.nanmean(y_true)
        validation_results["stats"]["pred_mean"] = np.nanmean(y_pred)
        validation_results["stats"]["true_std"] = np.nanstd(y_true)
        validation_results["stats"]["pred_std"] = np.nanstd(y_pred)

        return validation_results

    def check_data_leakage(
        self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None
    ) -> dict[str, Any]:
        """Check for potential data leakage in features"""
        leakage_results = {"potential_leakage": False, "issues": [], "warnings": []}

        # Check for future information in feature names
        future_keywords = ["future", "next", "tomorrow", "lead", "forward"]
        future_features = [
            col
            for col in X.columns
            if any(keyword in col.lower() for keyword in future_keywords)
        ]

        if future_features:
            leakage_results["potential_leakage"] = True
            leakage_results["issues"].append(
                f"Features with future information: {future_features}"
            )

        # Check for perfect correlations with target
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            for col in numeric_features.columns:
                if not numeric_features[col].isnull().all():
                    correlation = np.corrcoef(numeric_features[col].fillna(0), y)[0, 1]
                    if abs(correlation) > 0.99:
                        leakage_results["potential_leakage"] = True
                        leakage_results["issues"].append(
                            f"Perfect correlation with target: {col} (r={correlation:.4f})"
                        )

        # Time-based leakage check
        if datetime_col and datetime_col in X.columns:
            # Check if any features are derived from future dates
            dt_series = pd.to_datetime(X[datetime_col])
            max_date = dt_series.max()

            # This is a simplified check - more sophisticated checks could be added
            leakage_results["warnings"].append(
                f"Manual review recommended for time-based features (data until {max_date})"
            )

        return leakage_results
