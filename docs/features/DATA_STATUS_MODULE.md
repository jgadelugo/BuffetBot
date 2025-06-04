# Data Source Status Module

## Overview

The `data/source_status.py` module provides centralized data availability status reporting for the BuffetBot financial options application. It allows you to quickly check the health and availability of all major data sources without running full analysis.

## Features

- ✅ **Comprehensive Status Checking**: Tests all three main data sources (forecast, options, peers)
- 🚫 **Never Crashes**: Robust error handling ensures the status checker never breaks your application
- 📊 **Rich Reporting**: Provides detailed status information with source attribution
- 🎯 **Lightweight**: Performs quick health checks without full data processing
- 📋 **Multi-Ticker Support**: Can check health across multiple tickers simultaneously
- 🎨 **Pretty Printing**: Formatted display functions for CLI and debug use

## Main Functions

### `get_data_availability_status(ticker: str) -> Dict[str, Any]`

The core function that checks data availability for a single ticker.

**Parameters:**
- `ticker` (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')

**Returns:**
Dictionary containing:
```python
{
    "ticker": "AAPL",
    "forecast": {"available": True, "source": "yahoo"},
    "options": {"available": True, "source": "yahoo"},
    "peers": {"available": True, "source": "fallback_static"},
    "overall_health": "healthy",  # "healthy", "partial", "unhealthy"
    "total_sources": 3,
    "available_sources": 3,
    "unavailable_sources": 0,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example:**
```python
from data.source_status import get_data_availability_status

status = get_data_availability_status('AAPL')
if status['forecast']['available']:
    print(f"Forecast data available from {status['forecast']['source']}")
print(f"Overall health: {status['overall_health']}")
```

### `print_data_status(status_dict: Dict[str, Any]) -> None`

Pretty prints a status dictionary in a human-readable format.

**Parameters:**
- `status_dict` (Dict): Status dictionary from `get_data_availability_status()`

**Example:**
```python
from data.source_status import get_data_availability_status, print_data_status

status = get_data_availability_status('AAPL')
print_data_status(status)
```

**Output:**
```
Data Availability Status for AAPL
==================================
Overall Health: 💚 healthy (3/3 sources available)

📊 Forecast Data: ✅ Available    (Source: yahoo)
📈 Options Data:  ✅ Available    (Source: yahoo)
👥 Peers Data:    ✅ Available    (Source: fallback_static)

Status checked at: 2024-01-15T10:30:00Z
```

### `get_source_health_summary(tickers: List[str]) -> Dict[str, Any]`

Provides aggregate health statistics across multiple tickers.

**Parameters:**
- `tickers` (List[str]): List of ticker symbols to check

**Returns:**
```python
{
    "total_tickers_checked": 5,
    "healthy_tickers": 3,
    "partial_tickers": 1,
    "unhealthy_tickers": 1,
    "source_success_rates": {
        "forecast": 0.8,  # 80% success
        "options": 0.6,   # 60% success
        "peers": 1.0      # 100% success
    },
    "most_reliable_source": "peers",
    "least_reliable_source": "options"
}
```

**Example:**
```python
from data.source_status import get_source_health_summary

summary = get_source_health_summary(['AAPL', 'MSFT', 'GOOGL'])
print(f"Most reliable source: {summary['most_reliable_source']}")
print(f"Forecast success rate: {summary['source_success_rates']['forecast']:.1%}")
```

## Data Sources Checked

The module tests the following data sources:

1. **Forecast Data** (`get_analyst_forecast`)
   - Checks analyst price targets and forecasts
   - Sources: Yahoo Finance (primary), FMP (fallback)

2. **Options Data** (`fetch_long_dated_calls`)
   - Checks long-dated call options availability
   - Sources: Yahoo Finance (primary), FMP/EOD (fallback)

3. **Peers Data** (`get_peers`)
   - Checks peer/competitor company data
   - Sources: Yahoo Finance API (primary), FMP (fallback), Static mapping (final fallback)

## Health Status Levels

- **🟢 Healthy**: All 3 data sources available
- **🟡 Partial**: 2 out of 3 data sources available
- **🔴 Unhealthy**: 1 or 0 data sources available

## Import Options

You can import the functions in several ways:

```python
# Direct import
from data.source_status import get_data_availability_status, print_data_status

# From data package (recommended)
from data import get_data_availability_status, print_data_status, get_source_health_summary

# Import module
import data.source_status as status
```

## Common Use Cases

### 1. Quick Health Check
```python
from data import get_data_availability_status, print_data_status

# Check a single ticker
status = get_data_availability_status('AAPL')
print_data_status(status)
```

### 2. Programmatic Health Monitoring
```python
from data import get_data_availability_status

def is_ticker_ready_for_analysis(ticker):
    status = get_data_availability_status(ticker)
    return status['overall_health'] in ['healthy', 'partial']

# Usage
if is_ticker_ready_for_analysis('AAPL'):
    print("AAPL is ready for analysis")
```

### 3. Batch Health Assessment
```python
from data import get_source_health_summary

portfolio_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
summary = get_source_health_summary(portfolio_tickers)

print(f"Portfolio health: {summary['healthy_tickers']}/{summary['total_tickers_checked']} healthy")
```

### 4. Data Pipeline Monitoring
```python
from data import get_data_availability_status

def monitor_data_pipeline():
    critical_tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in critical_tickers:
        status = get_data_availability_status(ticker)

        if status['overall_health'] == 'unhealthy':
            print(f"⚠️ Alert: {ticker} has unhealthy data status")
        elif status['overall_health'] == 'partial':
            print(f"⚡ Warning: {ticker} has partial data availability")
        else:
            print(f"✅ {ticker} data is healthy")
```

## Error Handling

The module is designed to never crash your application:

- **Invalid tickers**: Returns status with `unhealthy` health and clear error messages
- **Network failures**: Gracefully handles API timeouts and connection issues
- **Missing data**: Reports unavailability rather than raising exceptions
- **Malformed responses**: Validates all data and handles parsing errors

## Logging

The module provides comprehensive logging:

- **INFO**: Successful data fetches and status updates
- **WARNING**: Failed data source attempts and fallback usage
- **ERROR**: Critical errors (but still returns valid status)

Logs are written to `logs/source_status.log`.

## Performance Considerations

- **Lightweight**: Only tests data availability, doesn't fetch full datasets
- **Efficient**: Uses existing fetcher functions with minimal overhead
- **Cacheable**: Status results can be cached since they represent point-in-time availability
- **Fast**: Typical status check takes 2-5 seconds per ticker

## Demo Script

Run the demo to see the module in action:

```bash
python examples/data_status_demo.py
```

This will demonstrate:
- Single ticker status checking
- Multi-ticker health summaries
- Direct API usage examples
- Error handling with invalid tickers
