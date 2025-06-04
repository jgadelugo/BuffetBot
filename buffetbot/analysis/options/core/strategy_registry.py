"""
Strategy registry for dynamic strategy discovery and instantiation.

This module implements the strategy pattern with a registry for
managing different options strategies.
"""

import logging
from typing import Dict, List, Type

from buffetbot.utils.logger import setup_logger

from ..core.domain_models import StrategyType
from ..core.exceptions import OptionsAdvisorError

logger = setup_logger(__name__, "logs/strategy_registry.log")


class StrategyRegistry:
    """Registry for managing options strategies."""

    def __init__(self):
        self._strategies: dict[StrategyType, type] = {}

    def register_strategy(
        self, strategy_type: StrategyType, strategy_class: type
    ) -> None:
        """
        Register a strategy implementation.

        Args:
            strategy_type: Strategy type enum
            strategy_class: Strategy implementation class
        """
        logger.info(f"Registering strategy: {strategy_type.value}")
        self._strategies[strategy_type] = strategy_class

    def get_strategy_class(self, strategy_type: StrategyType) -> type:
        """
        Get strategy class for a given type.

        Args:
            strategy_type: Strategy type to get

        Returns:
            Type: Strategy class

        Raises:
            OptionsAdvisorError: If strategy not found
        """
        if strategy_type not in self._strategies:
            available = [s.value for s in self._strategies.keys()]
            raise OptionsAdvisorError(
                f"Strategy {strategy_type.value} not registered. Available: {available}"
            )

        return self._strategies[strategy_type]

    def create_strategy(self, strategy_type: StrategyType, **kwargs):
        """
        Create strategy instance.

        Args:
            strategy_type: Strategy type to create
            **kwargs: Arguments for strategy constructor

        Returns:
            Strategy instance
        """
        strategy_class = self.get_strategy_class(strategy_type)
        logger.debug(f"Creating strategy instance: {strategy_type.value}")
        return strategy_class(**kwargs)

    def get_supported_strategies(self) -> list[StrategyType]:
        """
        Get list of supported strategy types.

        Returns:
            List[StrategyType]: Supported strategies
        """
        return list(self._strategies.keys())

    def is_strategy_supported(self, strategy_type: StrategyType) -> bool:
        """
        Check if strategy is supported.

        Args:
            strategy_type: Strategy type to check

        Returns:
            bool: True if supported
        """
        return strategy_type in self._strategies


# Global registry instance
_global_registry = StrategyRegistry()


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry."""
    return _global_registry


def register_strategy(strategy_type: StrategyType, strategy_class: type) -> None:
    """Register a strategy in the global registry."""
    _global_registry.register_strategy(strategy_type, strategy_class)


def get_supported_strategies() -> list[StrategyType]:
    """Get list of supported strategies from global registry."""
    return _global_registry.get_supported_strategies()
