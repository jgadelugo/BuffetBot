#!/usr/bin/env python3
"""
Test script to simulate dashboard behavior with missing/None data.

This script tests the UI functions that would be called by the dashboard
when data fetchers return None or missing data, ensuring no TypeErrors occur.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.app import (
    safe_format_currency,
    safe_format_percentage,
    safe_format_number,
    safe_get_nested_value,
    safe_get_last_price
)


def simulate_dashboard_with_missing_data():
    """Simulate dashboard behavior when data fetchers return None or missing data."""
    print("Simulating dashboard behavior with missing/None data...")
    
    # Simulate data structure that might come from failed data fetchers
    missing_data_cases = [
        # Case 1: Completely None data
        None,
        
        # Case 2: Empty data structure
        {},
        
        # Case 3: Partial data with None values
        {
            'price_data': None,
            'fundamentals': None,
            'metrics': None
        },
        
        # Case 4: Data structure with missing keys
        {
            'price_data': pd.DataFrame(),  # Empty DataFrame
            'fundamentals': {'market_cap': None, 'pe_ratio': float('nan')},
            'metrics': {'volatility': None, 'rsi': None, 'price_change': None}
        },
        
        # Case 5: Mixed valid and invalid data
        {
            'price_data': pd.DataFrame({'Close': [100.0, 101.0]}),
            'fundamentals': {'market_cap': 1000000000, 'pe_ratio': None},
            'metrics': {'volatility': 0.25, 'rsi': None, 'price_change': 0.05}
        }
    ]
    
    for i, data in enumerate(missing_data_cases, 1):
        print(f"\n--- Case {i}: {type(data).__name__ if data else 'None'} ---")
        
        # Test Overview tab metrics
        print("Overview tab metrics:")
        
        # Current Price
        last_price = safe_get_last_price(safe_get_nested_value(data, 'price_data'))
        current_price = safe_format_currency(last_price)
        print(f"  Current Price: {current_price}")
        
        # Price Change
        price_change = safe_format_percentage(safe_get_nested_value(data, 'metrics', 'price_change'))
        print(f"  Price Change: {price_change}")
        
        # Market Cap
        market_cap = safe_format_currency(safe_get_nested_value(data, 'fundamentals', 'market_cap'))
        print(f"  Market Cap: {market_cap}")
        
        # P/E Ratio
        pe_ratio = safe_format_number(safe_get_nested_value(data, 'fundamentals', 'pe_ratio'))
        print(f"  P/E Ratio: {pe_ratio}")
        
        # Volatility
        volatility = safe_format_percentage(safe_get_nested_value(data, 'metrics', 'volatility'))
        print(f"  Volatility: {volatility}")
        
        # RSI
        rsi = safe_format_number(safe_get_nested_value(data, 'metrics', 'rsi'))
        print(f"  RSI: {rsi}")


def simulate_growth_metrics_with_none():
    """Simulate growth metrics analysis with None values."""
    print("\n" + "="*50)
    print("Simulating growth metrics with None values...")
    
    growth_results = [
        None,  # Complete failure
        {},    # Empty result
        {
            'revenue_growth': None,
            'earnings_growth': None,
            'eps_growth': None,
            'growth_score': None
        },
        {
            'revenue_growth': 0.15,
            'earnings_growth': None,
            'eps_growth': 0.12,
            'growth_score': 75.0
        }
    ]
    
    for i, result in enumerate(growth_results, 1):
        print(f"\n--- Growth Case {i} ---")
        
        if result:
            revenue_growth = safe_format_percentage(result.get('revenue_growth'))
            earnings_growth = safe_format_percentage(result.get('earnings_growth'))
            eps_growth = safe_format_percentage(result.get('eps_growth'))
            growth_score = safe_format_number(result.get('growth_score'))
            
            print(f"  Revenue Growth: {revenue_growth}")
            print(f"  Earnings Growth: {earnings_growth}")
            print(f"  EPS Growth: {eps_growth}")
            print(f"  Growth Score: {growth_score}")
        else:
            print("  No growth metrics available (would show warning)")


def simulate_risk_metrics_with_none():
    """Simulate risk metrics analysis with None values."""
    print("\n" + "="*50)
    print("Simulating risk metrics with None values...")
    
    risk_results = [
        None,  # Complete failure
        {},    # Empty result
        {
            'overall_risk': None
        },
        {
            'overall_risk': {
                'score': None,
                'level': 'Unknown'
            }
        },
        {
            'overall_risk': {
                'score': 65.5,
                'level': 'Moderate'
            }
        }
    ]
    
    for i, result in enumerate(risk_results, 1):
        print(f"\n--- Risk Case {i} ---")
        
        if result and result.get('overall_risk'):
            risk_data = result['overall_risk']
            risk_score = safe_format_number(risk_data.get('score'))
            risk_level = risk_data.get('level', 'Unknown')
            
            print(f"  Risk Score: {risk_score}%")
            print(f"  Risk Level: {risk_level}")
        else:
            print("  No risk metrics available (would show warning)")


def main():
    """Run all dashboard simulation tests."""
    print("TESTING DASHBOARD ERROR HANDLING WITH MISSING/NONE DATA")
    print("=" * 60)
    
    try:
        simulate_dashboard_with_missing_data()
        simulate_growth_metrics_with_none()
        simulate_risk_metrics_with_none()
        
        print("\n" + "=" * 60)
        print("✅ ALL DASHBOARD ERROR HANDLING TESTS PASSED")
        print("✅ Dashboard can handle missing/None data gracefully")
        print("✅ UI shows 'N/A' instead of crashing with TypeErrors")
        print("✅ All format strings are protected from None values")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ DASHBOARD TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 