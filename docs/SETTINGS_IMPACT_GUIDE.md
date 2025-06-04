# Options Analysis Settings Impact Guide

## Overview

The BuffetBot Options Analysis Dashboard provides extensive customization options that allow users to fine-tune their analysis based on their trading style, risk tolerance, and market outlook. This guide explains how each setting impacts the analysis results and scoring.

## 📊 Scoring Weights

### What They Are
Scoring weights determine how much influence each technical indicator has on the final option recommendation score. The five core indicators are:

- **RSI (Relative Strength Index)**: Momentum indicator (0-100)
- **Beta**: Market correlation coefficient
- **Momentum**: Price trend strength
- **IV (Implied Volatility)**: Option pricing uncertainty
- **Forecast**: Analyst forecast confidence

### How They Impact Analysis

#### High RSI Weight (25-30%)
- **Effect**: Prioritizes options on stocks with strong momentum
- **Best For**: Momentum traders, short-term strategies
- **Results**: Higher-scoring options will be on stocks with RSI > 50

#### High Beta Weight (25-30%)
- **Effect**: Factors in market correlation heavily
- **Best For**: Market-timing strategies, sector plays
- **Results**: Preference for stocks that move with/against market trends

#### High Momentum Weight (25-30%)
- **Effect**: Emphasizes sustained price trends
- **Best For**: Trend-following strategies
- **Results**: Options on stocks with clear directional movement score higher

#### High IV Weight (25-35%)
- **Effect**: Considers option pricing and volatility levels
- **Best For**: Income strategies, volatility trades
- **Results**: Options with attractive IV levels (high for selling, low for buying) score higher

#### High Forecast Weight (25-30%)
- **Effect**: Relies heavily on analyst predictions
- **Best For**: Fundamental-based trading, long-term holds
- **Results**: Options on stocks with strong analyst backing score higher

### Strategy-Specific Default Weights

#### Long Calls (Growth-Oriented)
```
RSI: 25%        # High momentum emphasis
Beta: 15%       # Less market correlation focus
Momentum: 25%   # Strong trend following
IV: 15%         # Lower IV preference for buying
Forecast: 20%   # Moderate analyst reliance
```

#### Bull Call Spreads (Balanced)
```
RSI: 20%        # Balanced momentum
Beta: 20%       # Market awareness
Momentum: 20%   # Trend consideration
IV: 20%         # Spread pricing
Forecast: 20%   # Equal weighting
```

#### Covered Calls (Income-Focused)
```
RSI: 15%        # Lower momentum need
Beta: 25%       # Market stability important
Momentum: 15%   # Less trend dependence
IV: 30%         # High IV for premium collection
Forecast: 15%   # Less analyst dependence
```

#### Cash-Secured Puts (Value-Oriented)
```
RSI: 25%        # Entry timing important
Beta: 20%       # Market consideration
Momentum: 15%   # Counter-trend opportunities
IV: 25%         # Premium collection focus
Forecast: 15%   # Value-based approach
```

## ⚡ Risk Tolerance Settings

### Conservative
- **Min Days to Expiry**: Increased by 60+ days minimum
- **Volume Threshold**: Higher requirements (≥50)
- **Open Interest**: Higher requirements (≥100)
- **Delta Range**: More restrictive (≤0.7)
- **Bid-Ask Spread**: Stricter limits (≤$0.50)
- **Result Impact**: Fewer but higher-quality options with more time value

### Moderate
- **Min Days to Expiry**: Standard settings
- **Volume/OI**: Moderate requirements
- **Delta Range**: Balanced approach
- **Result Impact**: Good balance of opportunity and risk management

### Aggressive
- **Min Days to Expiry**: Allows shorter timeframes
- **Top N Results**: May double recommendation count
- **Volume/OI**: Lower thresholds allowed
- **Delta Range**: Wider range permitted
- **Result Impact**: More options including higher-risk, higher-reward opportunities

## 📅 Time Horizon Impact

### Short-term (1-3 months)
- **Emphasis**: Technical indicators (RSI, Momentum) weighted higher
- **Strategy Preference**: Momentum-based trades
- **Result Impact**: Options with strong near-term catalysts score higher

### Medium-term (3-6 months)
- **Emphasis**: Balanced approach across all factors
- **Strategy Preference**: Technical + fundamental blend
- **Result Impact**: Well-rounded scoring considering multiple factors

### Long-term (6+ months)
- **Emphasis**: Fundamentals and forecasts weighted higher
- **Strategy Preference**: Growth and value plays
- **Result Impact**: Options on fundamentally strong companies score higher

## 🔧 Advanced Filtering Parameters

### Delta Threshold (0.0 - 1.0)
- **Impact**: Limits option "moneyness"
- **Low Delta (0.3-0.5)**: Out-of-the-money options, higher risk/reward
- **High Delta (0.7-0.9)**: In-the-money options, lower risk/reward
- **Default**: 0.7 for balanced approach

### Volume Threshold (minimum daily volume)
- **Impact**: Ensures liquidity for entry/exit
- **Low Threshold (10-50)**: More options available, some illiquid
- **High Threshold (100+)**: Fewer but more liquid options
- **Default**: 50 contracts/day

### Bid-Ask Spread (maximum spread in dollars)
- **Impact**: Controls transaction costs
- **Tight Spread (≤$0.25)**: Lower costs but fewer options
- **Wide Spread (≥$1.00)**: More options but higher costs
- **Default**: $0.50 for balance

### Open Interest (minimum total contracts)
- **Impact**: Indicates option popularity and liquidity
- **Low OI (≤50)**: More options but potential liquidity issues
- **High OI (≥500)**: Fewer but more established options
- **Default**: 100 contracts

## 💡 Practical Examples

### Example 1: Momentum Trading Setup
```
Settings:
- Strategy: Long Calls
- Risk: Aggressive
- Custom Weights: RSI 35%, Momentum 30%, Beta 15%, IV 10%, Forecast 10%
- Delta Threshold: 0.6
- Time Horizon: Short-term

Result: Options on strongly trending stocks with high momentum scores
```

### Example 2: Income Generation Setup
```
Settings:
- Strategy: Covered Calls
- Risk: Conservative
- Custom Weights: IV 35%, Beta 25%, RSI 15%, Momentum 10%, Forecast 15%
- Volume Threshold: 100
- Time Horizon: Medium-term

Result: High-premium options on stable stocks with good liquidity
```

### Example 3: Value Play Setup
```
Settings:
- Strategy: Cash-Secured Puts
- Risk: Moderate
- Custom Weights: Forecast 30%, RSI 25%, IV 20%, Beta 15%, Momentum 10%
- Open Interest: 200
- Time Horizon: Long-term

Result: Put options on fundamentally strong stocks with analyst support
```

## 🔄 Settings Change Management

### Auto-Refresh Feature
- **Enabled**: Analysis automatically reruns when settings change
- **Disabled**: Manual "Analyze Options" button click required
- **Recommendation**: Disable for experimentation, enable for live trading

### Caching Behavior
- **Enabled**: Results cached until settings change
- **Disabled**: Fresh analysis on every run
- **Impact**: Faster response vs. real-time data

### Change Detection
The system tracks:
- All scoring weight modifications
- Risk parameter adjustments
- Strategy and tolerance changes
- Filtering criteria updates

Visual indicators show when settings have changed but haven't been applied to the analysis yet.

## 📈 Performance Optimization

### Parallel Processing
- **Enabled**: Faster analysis using multiple CPU cores
- **Disabled**: Sequential processing, slower but more predictable
- **Default**: Enabled for better performance

### Detailed Logging
- **Enabled**: Comprehensive logs for troubleshooting
- **Disabled**: Standard logging level
- **Use Case**: Enable when investigating unexpected results

## 🎯 Best Practices

1. **Start with Strategy Defaults**: Use built-in strategy-specific weights as a baseline
2. **Gradual Adjustments**: Make small weight changes (5-10%) to observe impact
3. **Test with Paper Trading**: Validate custom settings before live implementation
4. **Monitor Performance**: Track how settings changes affect recommendation quality
5. **Document Successful Configurations**: Save effective custom settings combinations
6. **Regular Review**: Reassess settings based on market conditions and performance

## 🚨 Common Pitfalls

1. **Over-Optimization**: Excessive fine-tuning can lead to overfitting
2. **Ignoring Market Conditions**: Settings should adapt to market regime changes
3. **Extreme Weights**: Avoid putting >50% weight on any single indicator
4. **Insufficient Testing**: Always validate new settings with historical analysis
5. **Neglecting Risk Management**: Don't sacrifice risk controls for higher returns

## 📊 Settings Impact Summary

| Setting Type | Primary Impact | Secondary Impact | Recommended Frequency |
|--------------|----------------|------------------|---------------------|
| Scoring Weights | Score ranking | Option selection | Monthly |
| Risk Tolerance | Filter strictness | Result quantity | Quarterly |
| Time Horizon | Indicator emphasis | Strategy bias | Per trade setup |
| Advanced Filters | Liquidity/Cost | Result quality | Weekly |
| Auto-Refresh | User experience | Analysis timing | Per session |

This comprehensive settings system allows for sophisticated customization while maintaining ease of use for beginners through intelligent defaults and clear documentation.
