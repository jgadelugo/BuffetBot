"""
API Models Package

Pydantic models for request/response serialization and validation.
Provides type-safe API contracts and automatic OpenAPI documentation.
"""

from .domain import *
from .requests import *
from .responses import *

__all__ = [
    # Request models
    "AnalysisRequest",
    "PortfolioCreateRequest",
    "PortfolioUpdateRequest",
    "PositionRequest",
    "MarketDataRequest",
    # Response models
    "AnalysisResponse",
    "TaskStatusResponse",
    "PortfolioResponse",
    "PositionResponse",
    "MarketDataResponse",
    "ErrorResponse",
    "PaginatedResponse",
    # Domain models
    "UserModel",
    "PortfolioModel",
    "PositionModel",
    "AnalysisResultModel",
    "MarketDataModel",
]
