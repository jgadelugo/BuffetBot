# Risk Metrics Features Documentation

## Overview

The BuffetBot Risk Metrics module provides comprehensive financial risk analysis tools for institutional-grade risk management and portfolio analysis. This module implements industry-standard risk metrics with robust error handling and performance optimization.

## Features

### 1. Value at Risk (VaR) Metrics
- **Historical VaR**: Non-parametric risk estimation using empirical distributions
- **Parametric VaR**: Model-based risk estimation (Normal, t-distribution, Skewed Normal)
- **Expected Shortfall (Conditional VaR)**: Tail risk beyond VaR thresholds
- **VaR Breach Analysis**: Model validation and backtesting
- **Tail Risk Features**: Comprehensive tail risk characterization

### 2. Drawdown Analysis
- **Maximum Drawdown**: Peak-to-trough loss analysis
- **Rolling Drawdown**: Time-series drawdown tracking
- **Drawdown Clusters**: Identification of drawdown periods and patterns
- **Recovery Analysis**: Time-to-recovery and recovery factor analysis
- **Drawdown Risk Features**: ML-ready feature extraction

### 3. Correlation Metrics
- **Rolling Correlation**: Time-varying correlation analysis
- **Rolling Beta**: Market sensitivity and systematic risk
- **Correlation Stability**: Correlation regime analysis
- **Correlation Breakdown Risk**: Crisis correlation analysis
- **Multi-Asset Correlation Matrix**: Portfolio correlation analysis

### 4. Risk-Adjusted Returns
- **Sharpe Ratio**: Return per unit of total risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return per unit of maximum drawdown
- **Information Ratio**: Active return per unit of tracking error
- **Omega Ratio**: Probability-weighted return ratio
- **Sterling Ratio**: Risk-adjusted return with drawdown stability

## Quick Start

```python
import pandas as pd
from buffetbot.features.risk.var_metrics import VaRMetrics
from buffetbot.features.risk.drawdown_analysis import DrawdownAnalysis
from buffetbot.features.risk.correlation_metrics import CorrelationMetrics
from buffetbot.features.risk.risk_adjusted_returns import RiskAdjustedReturns

# Sample portfolio returns
returns = pd.Series([0.01, -0.02, 0.015, -0.008, 0.012])

# Calculate VaR
var_results = VaRMetrics.historical_var(returns)
print(f"95% VaR: {var_results['var_95'].iloc[-1]:.2%}")

# Analyze drawdowns
prices = (1 + returns).cumprod() * 100
drawdown_data = DrawdownAnalysis.calculate_drawdowns(prices)
print(f"Current Drawdown: {drawdown_data['drawdown'].iloc[-1]:.2%}")

# Risk-adjusted returns
risk_metrics = RiskAdjustedReturns.comprehensive_metrics(returns)
print(f"Sharpe Ratio: {risk_metrics['current_sharpe']:.2f}")
```

## API Reference

### VaR Metrics

#### Historical VaR
```python
VaRMetrics.historical_var(returns, confidence_levels=[0.95, 0.99], window=252)
```

**Parameters:**
- `returns` (pd.Series): Time series of returns
- `confidence_levels` (List[float]): Confidence levels (default: [0.95, 0.99])
- `window` (int): Rolling window size (default: 252)

**Returns:**
- Dictionary with VaR series for each confidence level

#### Expected Shortfall
```python
VaRMetrics.expected_shortfall(returns, confidence_levels=[0.95, 0.99], window=252)
```

### Drawdown Analysis

#### Calculate Drawdowns
```python
DrawdownAnalysis.calculate_drawdowns(prices)
```

**Parameters:**
- `prices` (pd.Series): Price series

**Returns:**
- Dictionary with drawdown metrics

#### Recovery Analysis
```python
DrawdownAnalysis.recovery_analysis(prices)
```

### Correlation Metrics

#### Rolling Correlation
```python
CorrelationMetrics.rolling_correlation(asset_returns, market_returns, window=60)
```

### Risk-Adjusted Returns

#### Comprehensive Metrics
```python
RiskAdjustedReturns.comprehensive_metrics(returns, benchmark_returns=None, risk_free_rate=0.0)
```

## Mathematical Background

### Value at Risk (VaR)
VaR quantifies the maximum expected loss over a specific time horizon at a given confidence level.

**Historical VaR:**
```
VaR_α = -Percentile(returns, α × 100)
```

### Drawdown
**Drawdown at time t:**
```
DD_t = (Peak_t - Price_t) / Peak_t
```

### Risk-Adjusted Returns
**Sharpe Ratio:**
```
Sharpe = (R_p - R_f) / σ_p
```

## Error Handling

The module implements comprehensive error handling:

- **Input Validation**: Type checking and data quality validation
- **Insufficient Data**: Graceful handling with warnings
- **NaN Values**: Robust missing data management
- **Performance Monitoring**: Timeout protection for large datasets

## Performance

For 1000 data points with 252-day rolling windows:
- VaR calculations: < 3 seconds
- Correlation analysis: < 3 seconds
- Drawdown analysis: < 1 second
- Risk-adjusted returns: < 3 seconds

## Testing

Comprehensive test coverage with:
- Unit tests for individual functions
- Integration tests for end-to-end workflows
- Performance tests for computational efficiency
- Mathematical validation tests

**Current Test Results:**
- 67 total tests
- 59 passing (88% success rate)
- Comprehensive error handling validation

## Best Practices

1. **Data Requirements**: Minimum 60 observations for reliable statistics
2. **Window Selection**: Use 252 days (1 year) for annual metrics
3. **Model Validation**: Regular backtesting of VaR models
4. **Real-Time Monitoring**: Implement streaming calculations

## Examples

### Portfolio Risk Dashboard
```python
def create_risk_dashboard(portfolio_returns, benchmark_returns):
    # VaR Analysis
    var_metrics = VaRMetrics.historical_var(portfolio_returns)

    # Drawdown Analysis
    prices = (1 + portfolio_returns).cumprod() * 100
    drawdown_data = DrawdownAnalysis.calculate_drawdowns(prices)

    # Risk-Adjusted Returns
    risk_metrics = RiskAdjustedReturns.comprehensive_metrics(
        portfolio_returns, benchmark_returns
    )

    return {
        'var_95': var_metrics['var_95'].iloc[-1],
        'current_drawdown': drawdown_data['drawdown'].iloc[-1],
        'sharpe_ratio': risk_metrics['current_sharpe'],
        'risk_efficiency_score': risk_metrics['risk_efficiency_score']
    }
```

## Version History

- **v1.0**: Initial implementation with VaR and drawdown analysis
- **v1.1**: Added correlation metrics and risk-adjusted returns
- **v1.2**: Enhanced error handling and performance optimization
- **v1.3**: Comprehensive test suite and documentation
