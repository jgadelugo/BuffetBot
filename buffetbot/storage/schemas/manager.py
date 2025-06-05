"""
Schema Management System

Comprehensive schema management with validation, versioning, and evolution capabilities.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa

from .forecasts import FORECAST_SCHEMAS, get_forecast_schema
from .market_data import MARKET_DATA_SCHEMAS, get_market_data_schema
from .options_data import OPTIONS_DATA_SCHEMAS, get_options_data_schema

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Schema validation strictness levels"""

    STRICT = "strict"  # Fail on any validation error
    WARNING = "warning"  # Log warnings but continue
    DISABLED = "disabled"  # Skip validation entirely


@dataclass
class ValidationError:
    """Represents a schema validation error"""

    field: str
    error_type: str
    message: str
    record_index: Optional[int] = None
    severity: str = "error"


@dataclass
class ValidationResult:
    """Result of schema validation"""

    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]
    schema_version: str
    validation_duration_ms: int


class SchemaManager:
    """Manage data schemas and validation across all data types"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)

        # Registry of all available schemas
        self.schema_registry = {
            "market_data": MARKET_DATA_SCHEMAS,
            "forecasts": FORECAST_SCHEMAS,
            "options_data": OPTIONS_DATA_SCHEMAS,
            "ecosystem_metrics": {},  # To be implemented
        }

        # Cache for frequently used schemas
        self._schema_cache: dict[str, pa.Schema] = {}

    def get_schema(self, data_type: str, version: str = "latest") -> pa.Schema:
        """Get schema for data type and version"""
        cache_key = f"{data_type}_{version}"

        # Check cache first
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        if data_type not in self.schema_registry:
            available_types = list(self.schema_registry.keys())
            raise ValueError(
                f"Unknown data type: {data_type}. Available types: {available_types}"
            )

        type_schemas = self.schema_registry[data_type]

        if version == "latest":
            if not type_schemas:
                raise ValueError(f"No schemas available for data type: {data_type}")
            version = max(type_schemas.keys())

        if version not in type_schemas:
            available_versions = list(type_schemas.keys())
            raise ValueError(
                f"Schema version {version} not found for {data_type}. Available versions: {available_versions}"
            )

        schema = type_schemas[version]

        # Cache for future use
        self._schema_cache[cache_key] = schema

        self.logger.debug(f"Retrieved schema for {data_type} v{version}")
        return schema

    def validate_data(
        self, data: list[dict], data_type: str, version: str = "latest"
    ) -> ValidationResult:
        """Validate data against schema"""
        start_time = datetime.now()

        try:
            schema = self.get_schema(data_type, version)
            errors = []
            warnings = []

            self.logger.debug(
                f"Validating {len(data)} records against {data_type} schema v{version}"
            )

            # Convert to Arrow table for validation
            try:
                table = pa.Table.from_pylist(data)
            except Exception as e:
                errors.append(
                    ValidationError(
                        field="table_conversion",
                        error_type="conversion_error",
                        message=f"Failed to convert data to Arrow table: {str(e)}",
                    )
                )

                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    schema_version=version,
                    validation_duration_ms=duration_ms,
                )

            # Schema compatibility validation
            schema_errors = self._validate_schema_compatibility(table.schema, schema)
            errors.extend(schema_errors)

            # Field-level validation
            field_errors = self._validate_fields(table, schema)
            errors.extend(field_errors)

            # Business rule validation
            business_errors = self._validate_business_rules(table, data_type)
            errors.extend(business_errors)

            # Determine validation result based on level
            is_valid = (
                len(errors) == 0 or self.validation_level == ValidationLevel.DISABLED
            )

            if errors and self.validation_level == ValidationLevel.WARNING:
                warnings.extend(errors)
                errors = []
                is_valid = True

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if errors:
                self.logger.warning(
                    f"Schema validation failed for {data_type}: {len(errors)} errors"
                )
            else:
                self.logger.debug(
                    f"Schema validation passed for {data_type} in {duration_ms}ms"
                )

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                schema_version=version,
                validation_duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.error(f"Schema validation error: {str(e)}")

            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        field="validation_system",
                        error_type="system_error",
                        message=f"Validation system error: {str(e)}",
                    )
                ],
                warnings=[],
                schema_version=version,
                validation_duration_ms=duration_ms,
            )

    def _validate_schema_compatibility(
        self, data_schema: pa.Schema, target_schema: pa.Schema
    ) -> list[ValidationError]:
        """Validate schema compatibility"""
        errors = []

        # Create field maps for easy lookup
        target_fields = {field.name: field for field in target_schema}
        data_fields = {field.name: field for field in data_schema}

        # Check for missing required fields
        for field_name, field in target_fields.items():
            if field_name not in data_fields:
                if not field.nullable:
                    errors.append(
                        ValidationError(
                            field=field_name,
                            error_type="missing_required_field",
                            message=f"Required field '{field_name}' is missing",
                        )
                    )

        # Check data types
        for field_name, data_field in data_fields.items():
            if field_name in target_fields:
                target_field = target_fields[field_name]
                if not self._types_compatible(data_field.type, target_field.type):
                    errors.append(
                        ValidationError(
                            field=field_name,
                            error_type="type_mismatch",
                            message=f"Field '{field_name}' type mismatch: expected {target_field.type}, got {data_field.type}",
                        )
                    )

        return errors

    def _types_compatible(
        self, data_type: pa.DataType, target_type: pa.DataType
    ) -> bool:
        """Check if data type is compatible with target type"""
        # Exact match
        if data_type.equals(target_type):
            return True

        # Compatible numeric types
        numeric_types = [
            pa.int8(),
            pa.int16(),
            pa.int32(),
            pa.int64(),
            pa.uint8(),
            pa.uint16(),
            pa.uint32(),
            pa.uint64(),
            pa.float32(),
            pa.float64(),
        ]

        if str(data_type) in [str(t) for t in numeric_types] and str(target_type) in [
            str(t) for t in numeric_types
        ]:
            return True

        # String types
        if pa.types.is_string(data_type) and pa.types.is_string(target_type):
            return True

        # Timestamp types (allow different precisions)
        if pa.types.is_timestamp(data_type) and pa.types.is_timestamp(target_type):
            return True

        return False

    def _validate_fields(
        self, table: pa.Table, schema: pa.Schema
    ) -> list[ValidationError]:
        """Validate individual field constraints"""
        errors = []

        for field in schema:
            field_name = field.name

            if field_name not in table.column_names:
                continue

            column = table.column(field_name)

            # Check for nulls in non-nullable fields
            if not field.nullable and column.null_count > 0:
                errors.append(
                    ValidationError(
                        field=field_name,
                        error_type="null_constraint_violation",
                        message=f"Non-nullable field '{field_name}' contains {column.null_count} null values",
                    )
                )

        return errors

    def _validate_business_rules(
        self, table: pa.Table, data_type: str
    ) -> list[ValidationError]:
        """Apply business-specific validation rules"""
        errors = []

        try:
            if data_type == "market_data":
                errors.extend(self._validate_market_data_rules(table))
            elif data_type == "options_data":
                errors.extend(self._validate_options_data_rules(table))
            elif data_type == "forecasts":
                errors.extend(self._validate_forecast_rules(table))
        except Exception as e:
            errors.append(
                ValidationError(
                    field="business_rules",
                    error_type="validation_error",
                    message=f"Business rule validation failed: {str(e)}",
                )
            )

        return errors

    def _validate_market_data_rules(self, table: pa.Table) -> list[ValidationError]:
        """Validate market data business rules"""
        errors = []

        try:
            # Price must be positive
            if "price" in table.column_names:
                prices = table.column("price").to_pandas().dropna()
                negative_price_indices = prices[prices <= 0].index.tolist()
                for idx in negative_price_indices:
                    errors.append(
                        ValidationError(
                            field="price",
                            error_type="invalid_value",
                            message="Price must be positive",
                            record_index=int(idx),
                        )
                    )

            # Volume must be non-negative
            if "volume" in table.column_names:
                volumes = table.column("volume").to_pandas().dropna()
                negative_volume_indices = volumes[volumes < 0].index.tolist()
                for idx in negative_volume_indices:
                    errors.append(
                        ValidationError(
                            field="volume",
                            error_type="invalid_value",
                            message="Volume cannot be negative",
                            record_index=int(idx),
                        )
                    )

            # Market cap must be positive if present
            if "market_cap" in table.column_names:
                market_caps = table.column("market_cap").to_pandas().dropna()
                invalid_market_cap_indices = market_caps[
                    market_caps <= 0
                ].index.tolist()
                for idx in invalid_market_cap_indices:
                    errors.append(
                        ValidationError(
                            field="market_cap",
                            error_type="invalid_value",
                            message="Market cap must be positive",
                            record_index=int(idx),
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    field="market_data_validation",
                    error_type="validation_error",
                    message=f"Market data validation error: {str(e)}",
                )
            )

        return errors

    def _validate_options_data_rules(self, table: pa.Table) -> list[ValidationError]:
        """Validate options data business rules"""
        errors = []

        try:
            # Strike price must be positive
            if "strike_price" in table.column_names:
                strikes = table.column("strike_price").to_pandas().dropna()
                invalid_strike_indices = strikes[strikes <= 0].index.tolist()
                for idx in invalid_strike_indices:
                    errors.append(
                        ValidationError(
                            field="strike_price",
                            error_type="invalid_value",
                            message="Strike price must be positive",
                            record_index=int(idx),
                        )
                    )

            # Option type must be 'call' or 'put'
            if "option_type" in table.column_names:
                option_types = table.column("option_type").to_pandas().dropna()
                valid_types = {"call", "put", "CALL", "PUT"}
                invalid_type_indices = option_types[
                    ~option_types.isin(valid_types)
                ].index.tolist()
                for idx in invalid_type_indices:
                    errors.append(
                        ValidationError(
                            field="option_type",
                            error_type="invalid_value",
                            message=f"Option type must be 'call' or 'put', got: {option_types.iloc[idx]}",
                            record_index=int(idx),
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    field="options_data_validation",
                    error_type="validation_error",
                    message=f"Options data validation error: {str(e)}",
                )
            )

        return errors

    def _validate_forecast_rules(self, table: pa.Table) -> list[ValidationError]:
        """Validate forecast data business rules"""
        errors = []

        try:
            # Predicted price must be positive
            if "predicted_price" in table.column_names:
                predicted_prices = table.column("predicted_price").to_pandas().dropna()
                invalid_price_indices = predicted_prices[
                    predicted_prices <= 0
                ].index.tolist()
                for idx in invalid_price_indices:
                    errors.append(
                        ValidationError(
                            field="predicted_price",
                            error_type="invalid_value",
                            message="Predicted price must be positive",
                            record_index=int(idx),
                        )
                    )

            # Confidence score must be between 0 and 1
            if "confidence_score" in table.column_names:
                confidence_scores = (
                    table.column("confidence_score").to_pandas().dropna()
                )
                invalid_confidence_indices = confidence_scores[
                    (confidence_scores < 0) | (confidence_scores > 1)
                ].index.tolist()
                for idx in invalid_confidence_indices:
                    errors.append(
                        ValidationError(
                            field="confidence_score",
                            error_type="invalid_value",
                            message="Confidence score must be between 0 and 1",
                            record_index=int(idx),
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    field="forecast_validation",
                    error_type="validation_error",
                    message=f"Forecast data validation error: {str(e)}",
                )
            )

        return errors

    def get_available_data_types(self) -> list[str]:
        """Get list of available data types"""
        return list(self.schema_registry.keys())

    def get_available_versions(self, data_type: str) -> list[str]:
        """Get available versions for a data type"""
        if data_type not in self.schema_registry:
            raise ValueError(f"Unknown data type: {data_type}")
        return list(self.schema_registry[data_type].keys())

    def evolve_schema(
        self,
        data_type: str,
        current_version: str,
        new_schema: pa.Schema,
        new_version: str,
    ) -> bool:
        """Add a new schema version with compatibility validation"""
        # Validate that the new schema is backward compatible
        try:
            current_schema = self.get_schema(data_type, current_version)
            compatibility_errors = self._validate_schema_compatibility(
                current_schema, new_schema
            )

            if compatibility_errors:
                self.logger.error(
                    f"Schema evolution failed: {len(compatibility_errors)} compatibility errors"
                )
                return False

            # Add new schema version
            self.schema_registry[data_type][new_version] = new_schema

            # Clear cache for this data type
            cache_keys_to_remove = [
                key
                for key in self._schema_cache.keys()
                if key.startswith(f"{data_type}_")
            ]
            for key in cache_keys_to_remove:
                del self._schema_cache[key]

            self.logger.info(
                f"Schema evolved for {data_type}: {current_version} -> {new_version}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Schema evolution error: {str(e)}")
            return False
