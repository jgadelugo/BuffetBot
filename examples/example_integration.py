"""Example integration of glossary_data with analysis modules.

This script demonstrates how the glossary can enhance the output of
existing analysis functions by providing context and explanations.
"""

import sys
import os
# Add parent directory to path to import buffetbot module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from buffetbot.glossary import get_metric_info, GLOSSARY


def enhanced_analysis_output(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance analysis results with glossary information.
    
    Args:
        analysis_results: Raw results from analysis modules
        
    Returns:
        Enhanced results with metric definitions and context
    """
    enhanced = {
        "results": analysis_results,
        "metrics_info": {},
        "interpretations": {}
    }
    
    # Example: Enhance growth analysis results
    if "growth_metrics" in analysis_results:
        growth = analysis_results["growth_metrics"]
        
        for metric in ["revenue_growth", "earnings_growth", "eps_growth"]:
            if metric in growth:
                try:
                    info = get_metric_info(metric)
                    enhanced["metrics_info"][metric] = {
                        "name": info["name"],
                        "description": info["description"],
                        "formula": info["formula"],
                        "value": growth[metric]
                    }
                    
                    # Add interpretation
                    value = growth[metric]
                    if value > 0.20:
                        interpretation = "Exceptional growth"
                    elif value > 0.10:
                        interpretation = "Strong growth"
                    elif value > 0.05:
                        interpretation = "Moderate growth"
                    elif value > 0:
                        interpretation = "Slow growth"
                    else:
                        interpretation = "Negative growth"
                    
                    enhanced["interpretations"][metric] = interpretation
                except KeyError:
                    pass
    
    # Example: Enhance health analysis results
    if "financial_ratios" in analysis_results:
        ratios = analysis_results["financial_ratios"]
        
        for ratio_key, ratio_value in ratios.items():
            if ratio_key in GLOSSARY:
                try:
                    info = get_metric_info(ratio_key)
                    enhanced["metrics_info"][ratio_key] = {
                        "name": info["name"],
                        "description": info["description"],
                        "formula": info["formula"],
                        "value": ratio_value
                    }
                    
                    # Add specific interpretations based on thresholds
                    if ratio_key == "current_ratio":
                        if ratio_value < 1.0:
                            interpretation = "Poor liquidity - may struggle to meet short-term obligations"
                        elif ratio_value < 1.5:
                            interpretation = "Adequate liquidity"
                        elif ratio_value < 2.5:
                            interpretation = "Good liquidity"
                        else:
                            interpretation = "Excess liquidity - may indicate inefficient use of assets"
                        enhanced["interpretations"][ratio_key] = interpretation
                        
                    elif ratio_key == "debt_to_equity":
                        if ratio_value < 0.5:
                            interpretation = "Low leverage - conservative capital structure"
                        elif ratio_value < 1.0:
                            interpretation = "Moderate leverage"
                        elif ratio_value < 2.0:
                            interpretation = "High leverage - increased financial risk"
                        else:
                            interpretation = "Very high leverage - significant financial risk"
                        enhanced["interpretations"][ratio_key] = interpretation
                        
                except KeyError:
                    pass
    
    return enhanced


def generate_metric_report(ticker: str, analysis_results: Dict[str, Any]) -> str:
    """Generate a formatted report with metric explanations.
    
    Args:
        ticker: Stock ticker symbol
        analysis_results: Combined results from all analysis modules
        
    Returns:
        Formatted report string
    """
    report = f"Financial Analysis Report for {ticker}\n"
    report += "=" * 50 + "\n\n"
    
    # Group metrics by category
    categories = ["growth", "value", "health", "risk"]
    
    for category in categories:
        report += f"{category.upper()} METRICS\n"
        report += "-" * 30 + "\n"
        
        # Find all metrics in results that belong to this category
        for metric_key, metric_info in GLOSSARY.items():
            if metric_info["category"] == category:
                # Check if this metric exists in any part of the results
                value = None
                
                # Search in different result sections
                if metric_key in analysis_results.get("growth_metrics", {}):
                    value = analysis_results["growth_metrics"][metric_key]
                elif metric_key in analysis_results.get("financial_ratios", {}):
                    value = analysis_results["financial_ratios"][metric_key]
                elif metric_key in analysis_results.get("risk_metrics", {}):
                    value = analysis_results["risk_metrics"][metric_key]
                elif metric_key in analysis_results.get("value_metrics", {}):
                    value = analysis_results["value_metrics"][metric_key]
                
                if value is not None:
                    report += f"\n{metric_info['name']}:"
                    if isinstance(value, (int, float)):
                        if metric_key.endswith("_ratio") or metric_key.endswith("_margin"):
                            report += f" {value:.2f}"
                        elif metric_key.endswith("_growth") or metric_key.endswith("_cagr"):
                            report += f" {value:.1%}"
                        else:
                            report += f" {value:.2f}"
                    else:
                        report += f" {value}"
                    
                    # Add brief explanation
                    description = metric_info['description']
                    if len(description) > 100:
                        description = description[:97] + "..."
                    report += f"\n  → {description}\n"
        
        report += "\n"
    
    return report


# Example usage
if __name__ == "__main__":
    # Simulated analysis results
    sample_results = {
        "growth_metrics": {
            "revenue_growth": 0.15,
            "earnings_growth": 0.18,
            "eps_growth": 0.12
        },
        "financial_ratios": {
            "current_ratio": 1.8,
            "debt_to_equity": 0.6,
            "return_on_equity": 0.22,
            "gross_margin": 0.35,
            "operating_margin": 0.18
        },
        "risk_metrics": {
            "beta": 1.2,
            "volatility": 0.25
        },
        "value_metrics": {
            "pe_ratio": 18.5,
            "pb_ratio": 3.2
        }
    }
    
    # Generate enhanced output
    enhanced = enhanced_analysis_output(sample_results)
    
    print("=== Enhanced Analysis Output ===\n")
    print("Metric Information:")
    for metric, info in enhanced["metrics_info"].items():
        print(f"\n{info['name']}: {info['value']:.2f}")
        print(f"  Interpretation: {enhanced['interpretations'].get(metric, 'N/A')}")
    
    print("\n" + "="*50 + "\n")
    
    # Generate formatted report
    report = generate_metric_report("AAPL", sample_results)
    print(report) 