"""
FastAPI Application Factory

Creates and configures the main FastAPI application with all necessary
middleware, error handlers, and route registrations.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from infrastructure.cache.cache_factory import CacheFactory
from infrastructure.monitoring.logging import setup_logging
from infrastructure.monitoring.metrics import setup_metrics
from starlette.exceptions import HTTPException as StarletteHTTPException

from database.connection import Database

from .middleware.auth import AuthenticationMiddleware
from .middleware.monitoring import MonitoringMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routers import analysis, health, market, portfolio

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting BuffetBot API server")

    # Initialize database connection
    db = Database()
    await db.initialize()
    app.state.db = db

    # Initialize cache factory
    cache_factory = CacheFactory()
    await cache_factory.initialize()
    app.state.cache_factory = cache_factory

    # Setup monitoring
    setup_metrics()
    setup_logging()

    logger.info("BuffetBot API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down BuffetBot API server")

    # Close database connections
    if hasattr(app.state, "db"):
        await app.state.db.close()

    # Close cache connections
    if hasattr(app.state, "cache_factory"):
        await app.state.cache_factory.close()

    logger.info("BuffetBot API server shutdown complete")


def create_app(config: dict[str, Any] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured FastAPI application instance
    """
    # Default configuration
    default_config = {
        "title": "BuffetBot API",
        "description": "Enterprise Financial Analysis Platform",
        "version": "2.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "allowed_hosts": os.getenv("ALLOWED_HOSTS", "*").split(","),
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    }

    if config:
        default_config.update(config)

    # Create FastAPI app
    app = FastAPI(
        title=default_config["title"],
        description=default_config["description"],
        version=default_config["version"],
        docs_url=default_config["docs_url"],
        redoc_url=default_config["redoc_url"],
        openapi_url=default_config["openapi_url"],
        debug=default_config["debug"],
        lifespan=lifespan,
    )

    # Add middleware (order matters!)

    # Trusted host middleware (security)
    if default_config["allowed_hosts"] != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=default_config["allowed_hosts"]
        )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=default_config["cors_origins"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(MonitoringMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthenticationMiddleware)

    # Exception handlers

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with structured logging."""
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_error",
                    "timestamp": time.time(),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle request validation errors."""
        logger.warning(
            "Request validation error",
            errors=exc.errors(),
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Request validation failed",
                    "type": "validation_error",
                    "details": exc.errors(),
                    "timestamp": time.time(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected error occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            method=request.method,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "type": "internal_error",
                    "timestamp": time.time(),
                }
            },
        )

    # Register routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
    app.include_router(
        portfolio.router, prefix="/api/v1/portfolios", tags=["Portfolios"]
    )
    app.include_router(market.router, prefix="/api/v1/market", tags=["Market Data"])

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": default_config["title"],
            "version": default_config["version"],
            "description": default_config["description"],
            "docs_url": default_config["docs_url"],
            "redoc_url": default_config["redoc_url"],
            "status": "healthy",
            "timestamp": time.time(),
        }

    return app


# Create default app instance
app = create_app()
