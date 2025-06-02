"""Financial Metrics Glossary Module

This module provides a comprehensive dictionary of financial metrics and KPIs
used throughout the financial analysis toolkit. Each metric includes detailed
information for both technical implementation and business understanding.

The glossary is designed to be reusable across backend analysis logic and
frontend display components.
"""

from typing import Dict, Literal, TypedDict


class MetricDefinition(TypedDict):
    """Type definition for a financial metric entry."""

    name: str
    category: Literal["growth", "value", "health", "risk"]
    description: str
    formula: str


GLOSSARY: dict[str, MetricDefinition] = {
    # Growth Metrics
    "revenue_growth": {
        "name": "Revenue Growth Rate",
        "category": "growth",
        "description": "The year-over-year percentage change in a company's total revenue. This metric indicates how quickly a company is expanding its top-line sales and is a key indicator of business momentum and market demand.",
        "formula": "((Current Year Revenue - Previous Year Revenue) / Previous Year Revenue) × 100%",
    },
    "earnings_growth": {
        "name": "Earnings Growth Rate",
        "category": "growth",
        "description": "The year-over-year percentage change in a company's net income or earnings. This shows how effectively a company is growing its bottom-line profitability and is crucial for long-term investor returns.",
        "formula": "((Current Year Net Income - Previous Year Net Income) / Previous Year Net Income) × 100%",
    },
    "eps_growth": {
        "name": "Earnings Per Share (EPS) Growth",
        "category": "growth",
        "description": "The rate at which a company's earnings per share increases over time. EPS growth accounts for changes in share count and is a key metric for evaluating shareholder value creation.",
        "formula": "((Current Year EPS - Previous Year EPS) / Previous Year EPS) × 100%",
    },
    "revenue_cagr": {
        "name": "Revenue Compound Annual Growth Rate (CAGR)",
        "category": "growth",
        "description": "The mean annual growth rate of revenue over a specified period longer than one year. CAGR smooths out volatility to show consistent growth trends.",
        "formula": "((Ending Revenue / Beginning Revenue) ^ (1 / Number of Years) - 1) × 100%",
    },
    "fcf_growth": {
        "name": "Free Cash Flow Growth",
        "category": "growth",
        "description": "The year-over-year change in free cash flow, which represents cash generated after accounting for capital expenditures. Strong FCF growth indicates improving cash generation capability.",
        "formula": "((Current Year FCF - Previous Year FCF) / Previous Year FCF) × 100%",
    },
    # Value Metrics
    "pe_ratio": {
        "name": "Price-to-Earnings (P/E) Ratio",
        "category": "value",
        "description": "The ratio of a company's stock price to its earnings per share. A lower P/E may indicate undervaluation, while a higher P/E suggests growth expectations or overvaluation.",
        "formula": "Stock Price / Earnings Per Share (EPS)",
    },
    "pb_ratio": {
        "name": "Price-to-Book (P/B) Ratio",
        "category": "value",
        "description": "Compares a company's market value to its book value. A P/B under 1.0 might indicate the stock is undervalued or the company is earning poor returns on assets.",
        "formula": "Market Price per Share / Book Value per Share",
    },
    "peg_ratio": {
        "name": "Price/Earnings to Growth (PEG) Ratio",
        "category": "value",
        "description": "The P/E ratio divided by the earnings growth rate. A PEG below 1.0 may indicate undervaluation relative to growth prospects.",
        "formula": "P/E Ratio / Annual EPS Growth Rate",
    },
    "ev_ebitda": {
        "name": "Enterprise Value to EBITDA",
        "category": "value",
        "description": "Compares the total value of a company to its earnings before interest, taxes, depreciation, and amortization. Lower ratios may indicate better value.",
        "formula": "Enterprise Value / EBITDA",
    },
    "fcf_yield": {
        "name": "Free Cash Flow Yield",
        "category": "value",
        "description": "The ratio of free cash flow per share to the current share price. Higher yields indicate better cash generation relative to market value.",
        "formula": "Free Cash Flow per Share / Current Share Price × 100%",
    },
    "intrinsic_value": {
        "name": "Intrinsic Value (DCF)",
        "category": "value",
        "description": "The present value of all future free cash flows, discounted at an appropriate rate. This represents the 'true' value of a business based on its cash generation ability.",
        "formula": "Sum of (Future Cash Flows / (1 + Discount Rate)^Period) + Terminal Value",
    },
    "margin_of_safety": {
        "name": "Margin of Safety",
        "category": "value",
        "description": "The difference between intrinsic value and market price, expressed as a percentage. A higher margin provides more downside protection.",
        "formula": "((Intrinsic Value - Market Price) / Intrinsic Value) × 100%",
    },
    # Health Metrics
    "current_ratio": {
        "name": "Current Ratio",
        "category": "health",
        "description": "Measures a company's ability to pay short-term obligations with current assets. A ratio above 1.0 indicates good short-term financial health.",
        "formula": "Current Assets / Current Liabilities",
    },
    "quick_ratio": {
        "name": "Quick Ratio (Acid Test)",
        "category": "health",
        "description": "A more stringent measure of liquidity that excludes inventory. Values above 1.0 indicate strong ability to meet immediate obligations.",
        "formula": "(Current Assets - Inventory) / Current Liabilities",
    },
    "debt_to_equity": {
        "name": "Debt-to-Equity Ratio",
        "category": "health",
        "description": "Compares total liabilities to shareholders' equity. Lower ratios indicate less financial leverage and potentially lower financial risk.",
        "formula": "Total Liabilities / Total Shareholders' Equity",
    },
    "debt_to_assets": {
        "name": "Debt-to-Assets Ratio",
        "category": "health",
        "description": "Shows what percentage of assets is financed by debt. Lower ratios suggest stronger financial position and less dependency on borrowing.",
        "formula": "Total Debt / Total Assets",
    },
    "interest_coverage": {
        "name": "Interest Coverage Ratio",
        "category": "health",
        "description": "Measures how many times a company can pay its interest expenses from earnings. Higher ratios indicate better ability to service debt.",
        "formula": "Earnings Before Interest and Taxes (EBIT) / Interest Expense",
    },
    "return_on_equity": {
        "name": "Return on Equity (ROE)",
        "category": "health",
        "description": "Measures profitability relative to shareholders' equity. Higher ROE indicates more efficient use of equity capital. Warren Buffett favors companies with ROE above 15%.",
        "formula": "Net Income / Average Shareholders' Equity × 100%",
    },
    "return_on_assets": {
        "name": "Return on Assets (ROA)",
        "category": "health",
        "description": "Indicates how efficiently a company uses its assets to generate profit. Higher ROA suggests better asset utilization.",
        "formula": "Net Income / Total Assets × 100%",
    },
    "gross_margin": {
        "name": "Gross Profit Margin",
        "category": "health",
        "description": "The percentage of revenue retained after direct costs of goods sold. Higher margins indicate pricing power and operational efficiency.",
        "formula": "(Revenue - Cost of Goods Sold) / Revenue × 100%",
    },
    "operating_margin": {
        "name": "Operating Margin",
        "category": "health",
        "description": "Profitability after accounting for operating expenses. This metric shows operational efficiency before interest and taxes.",
        "formula": "Operating Income / Revenue × 100%",
    },
    "net_margin": {
        "name": "Net Profit Margin",
        "category": "health",
        "description": "The percentage of revenue that translates to net profit after all expenses. Higher margins indicate better overall profitability.",
        "formula": "Net Income / Revenue × 100%",
    },
    "piotroski_score": {
        "name": "Piotroski F-Score",
        "category": "health",
        "description": "A 9-point scoring system that assesses financial strength based on profitability, leverage, liquidity, and operating efficiency. Scores of 7-9 indicate strong financial health.",
        "formula": "Sum of 9 binary tests (0 or 1) covering profitability (4 tests), leverage/liquidity (3 tests), and operating efficiency (2 tests)",
    },
    "altman_z_score": {
        "name": "Altman Z-Score",
        "category": "health",
        "description": "Predicts the probability of bankruptcy within two years. Scores above 3.0 indicate low bankruptcy risk, while scores below 1.8 suggest high risk.",
        "formula": "1.2×(Working Capital/Total Assets) + 1.4×(Retained Earnings/Total Assets) + 3.3×(EBIT/Total Assets) + 0.6×(Market Value of Equity/Total Liabilities) + 1.0×(Sales/Total Assets)",
    },
    # Risk Metrics
    "beta": {
        "name": "Beta",
        "category": "risk",
        "description": "Measures a stock's volatility relative to the overall market. Beta > 1 indicates higher volatility than market, < 1 indicates lower volatility.",
        "formula": "Covariance(Stock Returns, Market Returns) / Variance(Market Returns)",
    },
    "volatility": {
        "name": "Price Volatility",
        "category": "risk",
        "description": "The degree of variation in a stock's price over time, typically measured as annualized standard deviation of returns. Higher volatility indicates greater price uncertainty.",
        "formula": "Standard Deviation of Daily Returns × √252 (trading days per year)",
    },
    "value_at_risk": {
        "name": "Value at Risk (VaR) 95%",
        "category": "risk",
        "description": "The maximum expected loss over a given time period at a 95% confidence level. Used to quantify potential downside risk.",
        "formula": "5th percentile of historical return distribution",
    },
    "max_drawdown": {
        "name": "Maximum Drawdown",
        "category": "risk",
        "description": "The largest peak-to-trough decline in value. Measures the worst-case historical loss an investor would have experienced.",
        "formula": "(Trough Value - Peak Value) / Peak Value × 100%",
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "category": "risk",
        "description": "Risk-adjusted return metric that measures excess return per unit of risk. Higher ratios indicate better risk-adjusted performance.",
        "formula": "(Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation",
    },
    "business_risk_score": {
        "name": "Business Risk Score",
        "category": "risk",
        "description": "Composite score assessing operational risks including revenue volatility, operating leverage, and industry factors. Lower scores indicate lower business risk.",
        "formula": "Weighted average of revenue volatility (50%) and operating leverage (50%), scaled 0-100",
    },
    "financial_risk_score": {
        "name": "Financial Risk Score",
        "category": "risk",
        "description": "Composite score evaluating financial leverage and solvency risks. Based on debt ratios and interest coverage metrics.",
        "formula": "Weighted average of debt-to-equity impact (40%) and interest coverage impact (60%), scaled 0-100",
    },
    "overall_risk_score": {
        "name": "Overall Risk Score",
        "category": "risk",
        "description": "Comprehensive risk assessment combining market risk (40%), financial risk (35%), and business risk (25%). Scores range from 0-100 with higher scores indicating greater risk.",
        "formula": "0.40 × Market Risk Score + 0.35 × Financial Risk Score + 0.25 × Business Risk Score",
    },
    # Technical Indicators
    "rsi": {
        "name": "Relative Strength Index (RSI)",
        "category": "risk",
        "description": "A momentum oscillator that measures the speed and magnitude of price changes. RSI ranges from 0 to 100, with values above 70 indicating overbought conditions and below 30 indicating oversold conditions.",
        "formula": "100 - (100 / (1 + (Average Gain / Average Loss)))",
    },
    "price_change": {
        "name": "Price Change",
        "category": "value",
        "description": "The percentage change in stock price over a specific period, typically compared to the previous close or a year ago. Positive values indicate price appreciation.",
        "formula": "((Current Price - Previous Price) / Previous Price) × 100%",
    },
    "momentum": {
        "name": "Price Momentum",
        "category": "risk",
        "description": "The rate of acceleration of a security's price or volume. Momentum indicators help identify the strength of a trend and potential reversal points.",
        "formula": "(Current Price - Price N periods ago) / Price N periods ago × 100%",
    },
    "latest_price": {
        "name": "Latest Stock Price",
        "category": "value",
        "description": "The most recent trading price of the stock. This represents the last price at which the stock was bought or sold in the market.",
        "formula": "Last traded price from market data",
    },
    "growth_score": {
        "name": "Composite Growth Score",
        "category": "growth",
        "description": "A comprehensive score that combines multiple growth metrics including revenue growth, earnings growth, and EPS growth to provide an overall growth assessment. Higher scores indicate stronger growth characteristics.",
        "formula": "Weighted average of normalized growth metrics (Revenue: 40%, Earnings: 35%, EPS: 25%)",
    },
    "market_cap": {
        "name": "Market Capitalization",
        "category": "value",
        "description": "The total market value of a company's outstanding shares. Market cap is calculated by multiplying the current stock price by the total number of outstanding shares.",
        "formula": "Current Stock Price × Total Outstanding Shares",
    },
    "revenue": {
        "name": "Total Revenue",
        "category": "health",
        "description": "The total income generated by a company from its business operations before any expenses are deducted. Also known as gross sales or top line.",
        "formula": "Sum of all sales and income from business operations",
    },
    "discount_rate": {
        "name": "Discount Rate",
        "category": "value",
        "description": "The interest rate used to determine the present value of future cash flows in DCF analysis. It represents the required rate of return and accounts for the time value of money and investment risk.",
        "formula": "Risk-free rate + Beta × (Market return - Risk-free rate)",
    },
    "growth_rate": {
        "name": "Expected Growth Rate",
        "category": "growth",
        "description": "The anticipated annual rate at which a company's earnings, revenue, or dividends are expected to grow. Used in valuation models to project future performance.",
        "formula": "Average of historical growth rates or analyst consensus estimates",
    },
    # Options Advisor Metrics
    "implied_volatility": {
        "name": "Implied Volatility (IV)",
        "category": "risk",
        "description": "The market's forecast of a security's future volatility as reflected in the option's price. Higher IV generally means more expensive options premiums. Lower IV can indicate better value for option buyers.",
        "formula": "Derived from option pricing models (Black-Scholes) using current option price, strike, expiry, and underlying price",
    },
    "composite_score": {
        "name": "Options Composite Score",
        "category": "value",
        "description": "A weighted technical scoring system that combines RSI (25%), Beta (25%), Momentum (25%), and Implied Volatility (25%) to rank option contracts. Higher scores indicate more attractive options based on technical analysis.",
        "formula": "0.25 × RSI Score + 0.25 × Beta Score + 0.25 × Momentum Score + 0.25 × IV Score",
    },
    "option_strike": {
        "name": "Option Strike Price",
        "category": "value",
        "description": "The predetermined price at which the option holder can buy (call) or sell (put) the underlying security when exercising the option. Strike prices closer to current stock price have higher intrinsic value.",
        "formula": "Fixed price specified in the option contract",
    },
    "option_expiry": {
        "name": "Option Expiration Date",
        "category": "risk",
        "description": "The date when the option contract expires and becomes worthless if not exercised. Options lose time value as they approach expiration. Longer-dated options have more time value.",
        "formula": "Calendar date specified in the option contract",
    },
    "option_price": {
        "name": "Option Premium",
        "category": "value",
        "description": "The current market price of the option contract. This represents the cost to purchase the option and includes both intrinsic value and time value. Lower premiums relative to potential payoff indicate better value.",
        "formula": "Intrinsic Value + Time Value (based on volatility, time to expiry, interest rates)",
    },
    "days_to_expiry": {
        "name": "Days to Expiration",
        "category": "risk",
        "description": "The number of calendar days remaining until the option expires. Time decay accelerates as expiration approaches, particularly in the final 30 days. Longer-dated options provide more time for the underlying to move favorably.",
        "formula": "Expiration Date - Current Date",
    },
    # Analyst Forecast Metrics
    "forecast_confidence": {
        "name": "Analyst Forecast Confidence",
        "category": "value",
        "description": "A composite score indicating the reliability of analyst price targets based on the number of analysts and the dispersion of their forecasts. Higher scores indicate stronger consensus among analysts.",
        "formula": "(Analyst Count Factor × 0.6) + (Consensus Factor × 0.4), where Consensus Factor = 1 - (Std Dev / Mean Target)",
    },
    "mean_target": {
        "name": "Mean Analyst Target",
        "category": "value",
        "description": "The average price target across all analysts covering the stock. This represents the consensus view of where the stock price should trade in the next 12 months based on fundamental analysis.",
        "formula": "Sum of all analyst price targets / Number of analysts",
    },
    "median_target": {
        "name": "Median Analyst Target",
        "category": "value",
        "description": "The middle value of all analyst price targets when arranged in order. Less sensitive to outliers than the mean target and often considered more robust in representing consensus.",
        "formula": "Middle value of sorted analyst price targets",
    },
    "target_range": {
        "name": "Analyst Target Range",
        "category": "risk",
        "description": "The difference between the highest and lowest analyst price targets. A wider range indicates greater disagreement among analysts about the stock's fair value and potential future uncertainty.",
        "formula": "Highest Target - Lowest Target",
    },
    "target_std_dev": {
        "name": "Target Standard Deviation",
        "category": "risk",
        "description": "Statistical measure of how much individual analyst targets deviate from the mean target. Lower values indicate stronger consensus, while higher values suggest more disagreement among analysts.",
        "formula": "√(Σ(Target - Mean Target)² / (Number of Targets - 1))",
    },
    "coefficient_variation": {
        "name": "Coefficient of Variation",
        "category": "risk",
        "description": "The ratio of standard deviation to the mean target, expressed as a percentage. This normalized measure allows comparison of forecast dispersion across different stocks regardless of price level.",
        "formula": "(Standard Deviation / Mean Target) × 100%",
    },
    "forecast_timeframe": {
        "name": "Forecast Time Window",
        "category": "value",
        "description": "The time period filter applied to analyst forecasts, ranging from recent forecasts (last 1-3 months) to all available forecasts. More recent forecasts may reflect updated market conditions and company developments.",
        "formula": "Current Date - Forecast Publication Date ≤ Selected Window (in days)",
    },
}


def get_metrics_by_category(
    category: Literal["growth", "value", "health", "risk"]
) -> dict[str, MetricDefinition]:
    """Get all metrics for a specific category.

    Args:
        category: The category to filter by

    Returns:
        Dictionary of metrics in the specified category

    Example:
        >>> growth_metrics = get_metrics_by_category("growth")
        >>> print(growth_metrics["revenue_growth"]["name"])
        'Revenue Growth Rate'
    """
    return {
        key: metric
        for key, metric in GLOSSARY.items()
        if metric["category"] == category
    }


def get_metric_names() -> dict[str, str]:
    """Get a simple mapping of metric keys to human-readable names.

    Returns:
        Dictionary mapping metric keys to their display names

    Example:
        >>> names = get_metric_names()
        >>> print(names["pe_ratio"])
        'Price-to-Earnings (P/E) Ratio'
    """
    return {key: metric["name"] for key, metric in GLOSSARY.items()}


def search_metrics(search_term: str) -> dict[str, MetricDefinition]:
    """Search for metrics by name or description.

    Args:
        search_term: Term to search for (case-insensitive)

    Returns:
        Dictionary of matching metrics

    Example:
        >>> debt_metrics = search_metrics("debt")
        >>> print(len(debt_metrics))
        3
    """
    search_lower = search_term.lower()
    return {
        key: metric
        for key, metric in GLOSSARY.items()
        if search_lower in metric["name"].lower()
        or search_lower in metric["description"].lower()
    }


def get_metric_info(metric_key: str) -> MetricDefinition:
    """Get detailed information for a specific metric.

    Args:
        metric_key: The key of the metric to retrieve

    Returns:
        MetricDefinition dictionary for the specified metric

    Raises:
        KeyError: If the metric key is not found

    Example:
        >>> info = get_metric_info("current_ratio")
        >>> print(info["formula"])
        'Current Assets / Current Liabilities'
    """
    if metric_key not in GLOSSARY:
        raise KeyError(f"Metric '{metric_key}' not found in glossary")
    return GLOSSARY[metric_key]
