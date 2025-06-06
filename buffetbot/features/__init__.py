"""
BuffetBot Features Module

Advanced feature engineering for machine learning models.
Provides comprehensive technical indicators, market structure analysis,
risk metrics, and pipeline utilities for enhancing model performance.

This module enables transformation of raw market data into sophisticated
features that capture market behavior patterns crucial for accurate
price prediction and financial analysis.

Author: BuffetBot Development Team
Date: 2024
"""

# Market Structure Features
from .market.gaps import GapAnalysis, analyze_gap_patterns
from .market.regime import MarketRegimeDetector, detect_market_regime
from .market.support_resistance import SupportResistance, identify_key_levels
from .risk.correlation_metrics import CorrelationMetrics
from .risk.drawdown_analysis import DrawdownAnalysis
from .risk.risk_adjusted_returns import RiskAdjustedReturns

# Risk Metrics Features
from .risk.var_metrics import VaRMetrics

# Technical Indicators
from .technical.momentum import MomentumIndicators
from .technical.trend import TrendIndicators
from .technical.volatility import VolatilityIndicators
from .technical.volume import VolumeIndicators

__all__ = [
    # Technical Indicators
    "MomentumIndicators",
    "TrendIndicators",
    "VolumeIndicators",
    "VolatilityIndicators",
    # Market Structure
    "GapAnalysis",
    "SupportResistance",
    "MarketRegimeDetector",
    # Risk Metrics
    "VaRMetrics",
    "DrawdownAnalysis",
    "CorrelationMetrics",
    "RiskAdjustedReturns",
    # Convenience Functions
    "analyze_gap_patterns",
    "identify_key_levels",
    "detect_market_regime",
]
