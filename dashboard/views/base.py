"""
Base View Classes and Registry Pattern

This module provides the foundation for all dashboard views, implementing:
- Base view interface for consistency
- View registry for dynamic discovery
- Standard error handling and logging
- Performance monitoring capabilities
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.logger import get_logger

logger = get_logger(__name__)


class ViewCategory(Enum):
    """Categories for organizing views."""

    CORE_ANALYSIS = "core_analysis"
    ADVANCED_TOOLS = "advanced_tools"
    REFERENCE = "reference"


@dataclass
class ViewMetadata:
    """Metadata for a dashboard view."""

    name: str
    title: str
    description: str
    icon: str
    category: ViewCategory
    requires_data: bool
    requires_ticker: bool = True
    min_data_quality: float = 0.0
    dependencies: list[str] | None = None


class BaseView(ABC):
    """Base class for all dashboard views."""

    def __init__(self, metadata: ViewMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"{__name__}.{metadata.name}")

    @abstractmethod
    def render(
        self,
        data: dict[str, Any] | None = None,
        ticker: str | None = None,
        **kwargs,
    ) -> None:
        """Render the view content.

        Args:
            data: Stock data dictionary (if required)
            ticker: Stock ticker symbol (if required)
            **kwargs: Additional view-specific parameters
        """
        pass

    def validate_inputs(self, data: dict[str, Any] | None, ticker: str | None) -> bool:
        """Validate inputs before rendering.

        Args:
            data: Stock data dictionary
            ticker: Stock ticker symbol

        Returns:
            bool: True if inputs are valid
        """
        if self.metadata.requires_ticker and not ticker:
            st.error(f"{self.metadata.title} requires a ticker symbol")
            return False

        if self.metadata.requires_data and not data:
            st.error(f"{self.metadata.title} requires stock data")
            return False

        return True

    def render_with_error_handling(
        self,
        data: dict[str, Any] | None = None,
        ticker: str | None = None,
        **kwargs,
    ) -> None:
        """Render view with comprehensive error handling and monitoring.

        Args:
            data: Stock data dictionary
            ticker: Stock ticker symbol
            **kwargs: Additional parameters
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_inputs(data, ticker):
                return

            # Log render start
            self.logger.info(
                f"Rendering {self.metadata.name} view for ticker: {ticker}"
            )

            # Render the view
            self.render(data, ticker, **kwargs)

            # Log successful completion
            render_time = time.time() - start_time
            self.logger.info(
                f"Successfully rendered {self.metadata.name} in {render_time:.2f}s"
            )

        except Exception as e:
            render_time = time.time() - start_time
            self.logger.error(
                f"Error rendering {self.metadata.name} after {render_time:.2f}s: {str(e)}",
                exc_info=True,
            )

            # Display user-friendly error
            st.error(f"Error in {self.metadata.title}: {str(e)}")
            st.info(
                "Please try refreshing the data or contact support if the issue persists."
            )


class ViewRegistry:
    """Registry for managing dashboard views."""

    def __init__(self):
        self._views: dict[str, BaseView] = {}
        self._legacy_functions: dict[str, Callable] = {}

    def register_view(self, view: BaseView) -> None:
        """Register a view in the registry.

        Args:
            view: View instance to register
        """
        self._views[view.metadata.name] = view
        logger.info(f"Registered view: {view.metadata.name}")

    def register_legacy_function(
        self, name: str, func: Callable, metadata: ViewMetadata
    ) -> None:
        """Register a legacy function as a view.

        Args:
            name: View name
            func: Legacy function
            metadata: View metadata
        """
        self._legacy_functions[name] = {"function": func, "metadata": metadata}
        logger.info(f"Registered legacy function as view: {name}")

    def get_view(self, name: str) -> BaseView | None:
        """Get a view by name.

        Args:
            name: View name

        Returns:
            View instance if found
        """
        return self._views.get(name)

    def get_legacy_function(self, name: str) -> Callable | None:
        """Get a legacy function by name.

        Args:
            name: Function name

        Returns:
            Function if found
        """
        legacy = self._legacy_functions.get(name)
        return legacy["function"] if legacy else None

    def get_all_views(self) -> dict[str, BaseView]:
        """Get all registered views."""
        return self._views.copy()

    def get_views_by_category(self, category: ViewCategory) -> list[BaseView]:
        """Get views by category.

        Args:
            category: View category

        Returns:
            List of views in the category
        """
        return [
            view for view in self._views.values() if view.metadata.category == category
        ]

    def render_view(
        self,
        name: str,
        data: dict[str, Any] | None = None,
        ticker: str | None = None,
        **kwargs,
    ) -> bool:
        """Render a view by name with error handling.

        Args:
            name: View name
            data: Stock data
            ticker: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            bool: True if rendered successfully
        """
        # Try modern view first
        view = self.get_view(name)
        if view:
            view.render_with_error_handling(data, ticker, **kwargs)
            return True

        # Fall back to legacy function
        func = self.get_legacy_function(name)
        if func:
            try:
                if data and ticker:
                    func(data, ticker, **kwargs)
                elif data:
                    func(data, **kwargs)
                elif ticker:
                    func(ticker, **kwargs)
                else:
                    func(**kwargs)
                return True
            except Exception as e:
                logger.error(
                    f"Error calling legacy function {name}: {str(e)}", exc_info=True
                )
                st.error(f"Error in {name}: {str(e)}")
                return False

        logger.warning(f"View not found: {name}")
        st.error(f"View '{name}' not found")
        return False


def performance_monitor(func: Callable) -> Callable:
    """Decorator for monitoring view performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    return wrapper


# Global view registry instance
view_registry = ViewRegistry()
