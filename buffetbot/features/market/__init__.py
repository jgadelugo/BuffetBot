"""
Market Structure Features Module

This module provides advanced market structure analysis features including:
- Gap analysis and detection
- Support and resistance level identification
- Market regime detection (trending, ranging, volatile)
- Price pattern recognition
- Market microstructure features

These features enhance ML models by capturing market behavior patterns
that are crucial for accurate price predictions.

Author: BuffetBot Development Team
Date: 2024
"""

from .gaps import GapAnalysis
from .regime import MarketRegimeDetector
from .support_resistance import SupportResistance

__all__ = ["GapAnalysis", "SupportResistance", "MarketRegimeDetector"]
