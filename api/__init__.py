"""
BuffetBot API Layer

Enterprise-grade FastAPI service layer providing RESTful endpoints
for financial analysis, portfolio management, and market data access.

Features:
- Async request processing with Celery integration
- Multi-layer caching strategy
- Comprehensive error handling and logging
- OpenAPI documentation
- Rate limiting and authentication
- Dependency injection pattern
"""

__version__ = "2.0.0"
__author__ = "BuffetBot Team"

from .main import create_app

__all__ = ["create_app"]
