"""
Risk Metrics Features Module

This module provides comprehensive risk analysis features for enhancing ML models:
- Value at Risk (VaR) and Expected Shortfall (ES)
- Maximum Drawdown and recovery analysis
- Rolling correlation and beta calculations
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
- Volatility clustering and GARCH effects

These risk-based features capture market stress, portfolio behavior,
and risk-return relationships crucial for ML model performance.

Author: BuffetBot Development Team
Date: 2024
"""

from .correlation_metrics import CorrelationMetrics
from .drawdown_analysis import DrawdownAnalysis
from .risk_adjusted_returns import RiskAdjustedReturns
from .var_metrics import VaRMetrics

__all__ = [
    "VaRMetrics",
    "DrawdownAnalysis",
    "CorrelationMetrics",
    "RiskAdjustedReturns",
]
