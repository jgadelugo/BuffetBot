# Financial Metrics Glossary Module

## Overview

The `glossary_data.py` module provides a comprehensive dictionary of financial metrics and KPIs used throughout the financial analysis toolkit. It includes 32 carefully documented financial metrics across four categories:

- **Growth Metrics** (5 metrics): Revenue growth, earnings growth, EPS growth, etc.
- **Value Metrics** (7 metrics): P/E ratio, P/B ratio, intrinsic value, etc.
- **Health Metrics** (12 metrics): Current ratio, debt ratios, margins, Piotroski score, etc.
- **Risk Metrics** (8 metrics): Beta, volatility, VaR, risk scores, etc.

## Features

- **Type-safe**: Uses Python type hints and TypedDict for better IDE support
- **Well-documented**: Each metric includes name, category, description, and formula
- **Reusable**: Designed for both backend analysis and frontend display
- **Searchable**: Built-in search functionality to find metrics by name or description
- **Categorized**: Easy filtering by metric category

## Usage

### Basic Import

```python
from glossary_data import GLOSSARY, get_metric_info
```

### Getting Metric Information

```python
# Get info for a specific metric
pe_info = get_metric_info("pe_ratio")
print(pe_info["name"])        # "Price-to-Earnings (P/E) Ratio"
print(pe_info["category"])    # "value"
print(pe_info["description"]) # Full description
print(pe_info["formula"])     # "Stock Price / Earnings Per Share (EPS)"
```

### Filtering by Category

```python
from glossary_data import get_metrics_by_category

# Get all health metrics
health_metrics = get_metrics_by_category("health")
for key, metric in health_metrics.items():
    print(f"{metric['name']}: {metric['description']}")
```

### Searching Metrics

```python
from glossary_data import search_metrics

# Find all debt-related metrics
debt_metrics = search_metrics("debt")
# Returns: debt_to_equity, debt_to_assets, and other debt-related metrics
```

### Integration with Analysis Modules

```python
# Example: Enhancing analysis output with metric context
def analyze_company(ticker):
    results = {
        "current_ratio": 1.5,
        "debt_to_equity": 0.8
    }
    
    # Add context from glossary
    for metric_key, value in results.items():
        info = get_metric_info(metric_key)
        print(f"{info['name']}: {value:.2f}")
        print(f"What it means: {info['description']}")
        print(f"How it's calculated: {info['formula']}")
```

## Metric Categories

### Growth Metrics
- Revenue Growth Rate
- Earnings Growth Rate
- EPS Growth
- Revenue CAGR
- Free Cash Flow Growth

### Value Metrics
- P/E Ratio
- P/B Ratio
- PEG Ratio
- EV/EBITDA
- FCF Yield
- Intrinsic Value (DCF)
- Margin of Safety

### Health Metrics
- Current Ratio
- Quick Ratio
- Debt-to-Equity
- Debt-to-Assets
- Interest Coverage
- ROE, ROA
- Gross/Operating/Net Margins
- Piotroski F-Score
- Altman Z-Score

### Risk Metrics
- Beta
- Volatility
- Value at Risk (VaR)
- Maximum Drawdown
- Sharpe Ratio
- Business/Financial/Overall Risk Scores

## Example: Building a Metric Dashboard

```python
from glossary_data import GLOSSARY, get_metrics_by_category

def create_dashboard_config():
    """Create configuration for a financial dashboard."""
    dashboard = {
        "sections": []
    }
    
    for category in ["growth", "value", "health", "risk"]:
        metrics = get_metrics_by_category(category)
        section = {
            "title": f"{category.title()} Indicators",
            "metrics": []
        }
        
        for key, metric in metrics.items():
            section["metrics"].append({
                "id": key,
                "label": metric["name"],
                "tooltip": metric["description"],
                "formula": metric["formula"]
            })
        
        dashboard["sections"].append(section)
    
    return dashboard
```

## Best Practices

1. **Use metric keys consistently**: Always use the standardized keys (e.g., `pe_ratio`, not `PE` or `p_e_ratio`)

2. **Handle missing metrics gracefully**:
   ```python
   try:
       info = get_metric_info("some_metric")
   except KeyError:
       info = {"name": "Unknown Metric", "description": "N/A"}
   ```

3. **Leverage type hints**:
   ```python
   from glossary_data import MetricDefinition
   
   def format_metric(metric: MetricDefinition) -> str:
       return f"{metric['name']} ({metric['category']})"
   ```

## Contributing

To add new metrics to the glossary:

1. Add the metric to the `GLOSSARY` dictionary in `glossary_data.py`
2. Ensure it has all required fields: `name`, `category`, `description`, `formula`
3. Use clear, business-friendly descriptions
4. Place it in the appropriate category section
5. Update this README if adding a new category

## Future Enhancements

- Add metric thresholds and benchmarks
- Include industry-specific variations
- Add calculation examples with real numbers
- Create metric relationship mappings
- Add multi-language support for international users 