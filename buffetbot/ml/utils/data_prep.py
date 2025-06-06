"""
Data Preprocessing Utilities for ML
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Utilities for preparing data for ML models"""

    def __init__(self):
        self.logger = logging.getLogger("ml.data_preprocessor")
        self.scalers = {}
        self.encoders = {}

    def create_time_features(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Create time-based features from datetime column"""
        df = df.copy()
        dt_col = pd.to_datetime(df[datetime_col])

        # Basic time features
        df[f"{datetime_col}_hour"] = dt_col.dt.hour
        df[f"{datetime_col}_day"] = dt_col.dt.day
        df[f"{datetime_col}_month"] = dt_col.dt.month
        df[f"{datetime_col}_year"] = dt_col.dt.year
        df[f"{datetime_col}_weekday"] = dt_col.dt.weekday
        df[f"{datetime_col}_quarter"] = dt_col.dt.quarter

        # Market-specific features
        df[f"{datetime_col}_is_weekend"] = dt_col.dt.weekday >= 5
        df[f"{datetime_col}_is_month_start"] = dt_col.dt.is_month_start
        df[f"{datetime_col}_is_month_end"] = dt_col.dt.is_month_end
        df[f"{datetime_col}_is_quarter_start"] = dt_col.dt.is_quarter_start
        df[f"{datetime_col}_is_quarter_end"] = dt_col.dt.is_quarter_end

        self.logger.info(f"Created time features from {datetime_col}")
        return df

    def create_lag_features(
        self, df: pd.DataFrame, target_col: str, lags: list[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = df.copy()

        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        self.logger.info(f"Created lag features for {target_col}: {lags}")
        return df

    def create_rolling_features(
        self, df: pd.DataFrame, target_col: str, windows: list[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()

        for window in windows:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].rolling(window).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].rolling(window).std()
            )
            df[f"{target_col}_rolling_min_{window}"] = (
                df[target_col].rolling(window).min()
            )
            df[f"{target_col}_rolling_max_{window}"] = (
                df[target_col].rolling(window).max()
            )

        self.logger.info(f"Created rolling features for {target_col}: {windows}")
        return df

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical indicators (simplified versions)"""
        df = df.copy()

        # Assumes OHLCV data structure
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(
                "OHLCV columns not found, skipping technical indicators"
            )
            return df

        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()

        # Price ratios
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["open_close_ratio"] = df["open"] / df["close"]
        df["high_close_ratio"] = df["high"] / df["close"]
        df["low_close_ratio"] = df["low"] / df["close"]

        # Volume indicators
        df["volume_sma_10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]

        # Volatility
        df["price_volatility_5"] = df["close"].pct_change().rolling(5).std()
        df["price_volatility_20"] = df["close"].pct_change().rolling(20).std()

        self.logger.info("Created technical indicators")
        return df

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "forward_fill"
    ) -> pd.DataFrame:
        """Handle missing values with various strategies"""
        df = df.copy()

        if strategy == "forward_fill":
            df = df.fillna(method="ffill")
        elif strategy == "backward_fill":
            df = df.fillna(method="bfill")
        elif strategy == "interpolate":
            df = df.interpolate()
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "mean":
            df = df.fillna(df.mean())
        else:
            self.logger.warning(f"Unknown strategy {strategy}, using forward fill")
            df = df.fillna(method="ffill")

        self.logger.info(f"Handled missing values with strategy: {strategy}")
        return df

    def normalize_features(
        self, df: pd.DataFrame, columns: list[str], method: str = "minmax"
    ) -> pd.DataFrame:
        """Normalize numerical features"""
        df = df.copy()

        if method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        elif method == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
        else:
            self.logger.error(f"Unknown normalization method: {method}")
            return df

        # Fit and transform
        df[columns] = scaler.fit_transform(df[columns])

        # Store scaler for inverse transformation
        scaler_key = f"{method}_{'_'.join(columns)}"
        self.scalers[scaler_key] = scaler

        self.logger.info(f"Normalized {len(columns)} features using {method}")
        return df

    def encode_categorical(
        self, df: pd.DataFrame, columns: list[str], method: str = "onehot"
    ) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()

        for col in columns:
            if method == "onehot":
                from sklearn.preprocessing import OneHotEncoder

                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(df[[col]])

                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

                # Store encoder
                self.encoders[col] = encoder

                # Replace original column
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)

            elif method == "label":
                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder

            else:
                self.logger.error(f"Unknown encoding method: {method}")

        self.logger.info(f"Encoded {len(columns)} categorical features using {method}")
        return df

    def create_target_features(
        self, df: pd.DataFrame, target_col: str, prediction_horizon: int = 1
    ) -> pd.DataFrame:
        """Create target variables for prediction"""
        df = df.copy()

        # Future price
        df[f"{target_col}_future_{prediction_horizon}"] = df[target_col].shift(
            -prediction_horizon
        )

        # Price change
        df[f"{target_col}_change_{prediction_horizon}"] = (
            df[f"{target_col}_future_{prediction_horizon}"] - df[target_col]
        )

        # Price change percentage
        df[f"{target_col}_pct_change_{prediction_horizon}"] = (
            df[f"{target_col}_change_{prediction_horizon}"] / df[target_col]
        )

        # Direction (up/down)
        df[f"{target_col}_direction_{prediction_horizon}"] = (
            df[f"{target_col}_change_{prediction_horizon}"] > 0
        ).astype(int)

        self.logger.info(
            f"Created target features for {target_col} with horizon {prediction_horizon}"
        )
        return df

    def prepare_time_series_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
        prediction_horizon: int = 1,
    ) -> pd.DataFrame:
        """Complete preprocessing pipeline for time series data"""
        self.logger.info("Starting complete time series preprocessing")

        # Sort by datetime
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # Create time features
        df = self.create_time_features(df, datetime_col)

        # Create lag features
        df = self.create_lag_features(df, target_col)

        # Create rolling features
        df = self.create_rolling_features(df, target_col)

        # Create technical indicators if possible
        df = self.create_technical_indicators(df)

        # Create target features
        df = self.create_target_features(df, target_col, prediction_horizon)

        # Handle missing values
        df = self.handle_missing_values(df)

        self.logger.info("Completed time series preprocessing")
        return df

    def get_feature_importance_data(
        self, df: pd.DataFrame, target_col: str
    ) -> dict[str, Any]:
        """Prepare data for feature importance analysis"""
        # Separate features and target
        feature_cols = [
            col
            for col in df.columns
            if col != target_col and not col.startswith(target_col + "_future")
        ]

        X = df[feature_cols].dropna()
        y = df.loc[X.index, target_col]

        return {
            "features": X,
            "target": y,
            "feature_names": feature_cols,
            "target_name": target_col,
        }
