import os
from pathlib import Path
from typing import Any, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
API_CONFIG = {
    "yfinance": {"timeout": 10, "retries": 3, "cache_duration": 3600}  # 1 hour
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "value": {
        "default_growth_rate": 0.05,
        "default_discount_rate": 0.10,
        "forecast_years": 5,
        "terminal_growth_rate": 0.02,
    },
    "health": {
        "current_ratio_threshold": 1.5,
        "debt_to_equity_threshold": 2.0,
        "interest_coverage_threshold": 3.0,
    },
    "growth": {
        "revenue_growth_threshold": 0.10,
        "eps_growth_threshold": 0.08,
        "roe_threshold": 0.15,
    },
    "risk": {
        "volatility_threshold": 0.30,
        "beta_threshold": 1.5,
        "var_threshold": -0.05,
    },
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "default_ticker": "AAPL",
    "default_period": "5y",
    "refresh_interval": 300,  # 5 minutes
    "max_tickers": 5,
    "chart_height": 400,
    "theme": "light",
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "%(asctime)s [%(levelname)s] %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "file": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "file",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}

# Cache Configuration
CACHE_CONFIG = {
    "enabled": True,
    "type": "file",
    "expiration": 3600,  # 1 hour
    "max_size": 1000,  # Maximum number of cached items
}


def get_config() -> dict[str, Any]:
    """
    Get the complete configuration dictionary.

    Returns:
        Dict containing all configuration settings
    """
    return {
        "project_paths": {
            "root": str(PROJECT_ROOT),
            "data": str(DATA_DIR),
            "cache": str(CACHE_DIR),
            "logs": str(LOG_DIR),
        },
        "api": API_CONFIG,
        "analysis": ANALYSIS_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "logging": LOGGING_CONFIG,
        "cache": CACHE_CONFIG,
    }


def get_env_config() -> dict[str, Any]:
    """
    Get configuration from environment variables.
    Overrides default configuration with environment variables if present.

    Returns:
        Dict containing environment-based configuration
    """
    env_config = {}

    # API Configuration
    if "YFINANCE_TIMEOUT" in os.environ:
        env_config["yfinance_timeout"] = int(os.environ["YFINANCE_TIMEOUT"])
    if "YFINANCE_RETRIES" in os.environ:
        env_config["yfinance_retries"] = int(os.environ["YFINANCE_RETRIES"])

    # Analysis Configuration
    if "DEFAULT_GROWTH_RATE" in os.environ:
        env_config["default_growth_rate"] = float(os.environ["DEFAULT_GROWTH_RATE"])
    if "DEFAULT_DISCOUNT_RATE" in os.environ:
        env_config["default_discount_rate"] = float(os.environ["DEFAULT_DISCOUNT_RATE"])

    # Dashboard Configuration
    if "DEFAULT_TICKER" in os.environ:
        env_config["default_ticker"] = os.environ["DEFAULT_TICKER"]
    if "REFRESH_INTERVAL" in os.environ:
        env_config["refresh_interval"] = int(os.environ["REFRESH_INTERVAL"])

    # Cache Configuration
    if "CACHE_ENABLED" in os.environ:
        env_config["cache_enabled"] = os.environ["CACHE_ENABLED"].lower() == "true"
    if "CACHE_EXPIRATION" in os.environ:
        env_config["cache_expiration"] = int(os.environ["CACHE_EXPIRATION"])

    return env_config
