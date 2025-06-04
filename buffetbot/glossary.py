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
    category: Literal["growth", "value", "health", "risk", "options"]
    description: str
    formula: str


class OptionsStrategyDefinition(TypedDict):
    """Type definition for an options strategy entry."""

    name: str
    category: Literal["options"]
    description: str
    objective: str
    market_outlook: str
    risk_profile: str
    default_weights: dict[str, float]
    weights_rationale: str
    max_profit: str
    max_loss: str
    breakeven: str


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
    "sortino_ratio": {
        "name": "Sortino Ratio",
        "category": "risk",
        "description": "Similar to Sharpe ratio but only considers downside volatility. This provides a better measure of risk-adjusted returns by focusing on harmful volatility.",
        "formula": "(Portfolio Return - Risk-Free Rate) / Downside Deviation",
    },
    "treynor_ratio": {
        "name": "Treynor Ratio",
        "category": "risk",
        "description": "Risk-adjusted return measure that uses beta instead of standard deviation. It shows return per unit of systematic risk.",
        "formula": "(Portfolio Return - Risk-Free Rate) / Beta",
    },
    "information_ratio": {
        "name": "Information Ratio",
        "category": "risk",
        "description": "Measures risk-adjusted return relative to a benchmark. Higher ratios indicate better performance relative to the benchmark per unit of tracking error.",
        "formula": "(Portfolio Return - Benchmark Return) / Tracking Error",
    },
    "downside_deviation": {
        "name": "Downside Deviation",
        "category": "risk",
        "description": "Measures volatility of negative returns only. This focuses on the risk investors care about most - losing money.",
        "formula": "Standard Deviation of Returns Below Target Return",
    },
}

# Options Strategies Glossary
OPTIONS_STRATEGIES: dict[str, OptionsStrategyDefinition] = {
    # Single-leg strategies
    "long_calls": {
        "name": "Long Calls",
        "category": "options",
        "description": "Buying call options to profit from upward price movement. This is a bullish strategy that provides leveraged exposure to stock appreciation with limited downside risk.",
        "objective": "Profit from significant upward price movement with limited risk",
        "market_outlook": "Bullish - expecting stock price to rise significantly",
        "risk_profile": "Limited risk (premium paid), unlimited profit potential",
        "default_weights": {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        },
        "weights_rationale": "Emphasizes RSI (25%) and momentum (25%) for timing entry on oversold conditions with strong upward momentum. Moderate forecast weight (20%) for directional conviction. Lower IV weight (15%) as high volatility increases cost.",
        "max_profit": "Unlimited (Stock Price - Strike Price - Premium Paid)",
        "max_loss": "Limited to premium paid",
        "breakeven": "Strike Price + Premium Paid",
    },
    "long_puts": {
        "name": "Long Puts",
        "category": "options",
        "description": "Buying put options to profit from downward price movement. This is a bearish strategy that provides leveraged exposure to stock decline with limited downside risk.",
        "objective": "Profit from significant downward price movement with limited risk",
        "market_outlook": "Bearish - expecting stock price to fall significantly",
        "risk_profile": "Limited risk (premium paid), high profit potential",
        "default_weights": {
            "rsi": 0.30,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.15,
            "forecast": 0.15,
        },
        "weights_rationale": "Highest RSI weight (30%) to identify overbought conditions for optimal entry. Moderate beta (20%) and momentum (20%) for market correlation analysis. Lower forecast weight (15%) as puts benefit from unexpected declines.",
        "max_profit": "Strike Price - Premium Paid (when stock goes to $0)",
        "max_loss": "Limited to premium paid",
        "breakeven": "Strike Price - Premium Paid",
    },
    "covered_call": {
        "name": "Covered Call",
        "category": "options",
        "description": "Selling call options against owned stock to generate income. This income strategy reduces cost basis while limiting upside if stock rises above strike price.",
        "objective": "Generate additional income from stock holdings while providing some downside protection",
        "market_outlook": "Neutral to slightly bullish - expecting stock to stay below strike price",
        "risk_profile": "Reduced stock risk due to premium income, but limited upside potential",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.25,
            "momentum": 0.15,
            "iv": 0.30,
            "forecast": 0.15,
        },
        "weights_rationale": "Emphasizes IV (30%) for higher premium collection and beta (25%) for stable, less volatile stocks. Lower RSI (15%) and momentum (15%) as strategy works best in sideways markets.",
        "max_profit": "Premium Received + (Strike Price - Stock Purchase Price) if called away",
        "max_loss": "Stock Purchase Price - Premium Received (if stock goes to $0)",
        "breakeven": "Stock Purchase Price - Premium Received",
    },
    "cash_secured_put": {
        "name": "Cash-Secured Put",
        "category": "options",
        "description": "Selling put options while holding enough cash to buy the stock if assigned. This strategy generates income while potentially acquiring stock at a discount.",
        "objective": "Generate income while potentially acquiring stock at a target price",
        "market_outlook": "Neutral to bullish - willing to own stock at strike price",
        "risk_profile": "Moderate risk - potential to own stock at strike price, premium provides some protection",
        "default_weights": {
            "rsi": 0.25,
            "beta": 0.20,
            "momentum": 0.15,
            "iv": 0.25,
            "forecast": 0.15,
        },
        "weights_rationale": "Higher RSI (25%) and IV (25%) for selling puts on oversold stocks with high premiums. Moderate beta (20%) for stock selection, lower momentum (15%) as strategy benefits from mean reversion.",
        "max_profit": "Premium Received",
        "max_loss": "Strike Price - Premium Received (if stock goes to $0)",
        "breakeven": "Strike Price - Premium Received",
    },
    # Vertical spreads
    "bull_call_spread": {
        "name": "Bull Call Spread",
        "category": "options",
        "description": "Buying a lower strike call and selling a higher strike call with same expiration. This reduces cost and risk compared to long calls but caps profit potential.",
        "objective": "Profit from moderate upward price movement with defined risk and reward",
        "market_outlook": "Moderately bullish - expecting stock to rise to target level",
        "risk_profile": "Limited risk and reward - both maximum profit and loss are defined",
        "default_weights": {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        },
        "weights_rationale": "Equal weights (20% each) provide balanced analysis suitable for moderate directional plays. No single factor dominates as strategy has defined risk/reward parameters.",
        "max_profit": "Difference in Strike Prices - Net Premium Paid",
        "max_loss": "Net Premium Paid",
        "breakeven": "Lower Strike Price + Net Premium Paid",
    },
    "bear_put_spread": {
        "name": "Bear Put Spread",
        "category": "options",
        "description": "Buying a higher strike put and selling a lower strike put with same expiration. This reduces cost compared to long puts but limits profit potential.",
        "objective": "Profit from moderate downward price movement with defined risk and reward",
        "market_outlook": "Moderately bearish - expecting stock to decline to target level",
        "risk_profile": "Limited risk and reward - both maximum profit and loss are defined",
        "default_weights": {
            "rsi": 0.30,
            "beta": 0.15,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.15,
        },
        "weights_rationale": "Higher RSI weight (30%) for identifying overbought entry points. Lower beta (15%) and forecast (15%) as bearish spreads work well during market stress periods.",
        "max_profit": "Difference in Strike Prices - Net Premium Paid",
        "max_loss": "Net Premium Paid",
        "breakeven": "Higher Strike Price - Net Premium Paid",
    },
    "bull_put_spread": {
        "name": "Bull Put Spread",
        "category": "options",
        "description": "Selling a higher strike put and buying a lower strike put with same expiration. This generates income while limiting downside risk.",
        "objective": "Generate income while maintaining bullish exposure with limited risk",
        "market_outlook": "Bullish - expecting stock to stay above higher strike price",
        "risk_profile": "Limited risk, limited reward - net credit received upfront",
        "default_weights": {
            "rsi": 0.20,
            "beta": 0.25,
            "momentum": 0.15,
            "iv": 0.25,
            "forecast": 0.15,
        },
        "weights_rationale": "Emphasizes IV (25%) for higher credit collection and beta (25%) for stable stocks. Lower momentum (15%) as strategy benefits from stable price action.",
        "max_profit": "Net Premium Received",
        "max_loss": "Difference in Strike Prices - Net Premium Received",
        "breakeven": "Higher Strike Price - Net Premium Received",
    },
    "bear_call_spread": {
        "name": "Bear Call Spread",
        "category": "options",
        "description": "Selling a lower strike call and buying a higher strike call with same expiration. This generates income while maintaining bearish exposure.",
        "objective": "Generate income from bearish outlook with limited risk",
        "market_outlook": "Bearish - expecting stock to stay below lower strike price",
        "risk_profile": "Limited risk, limited reward - net credit received upfront",
        "default_weights": {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.20,
            "iv": 0.25,
            "forecast": 0.15,
        },
        "weights_rationale": "Higher RSI (25%) and IV (25%) for selling calls on overbought stocks with high premiums. Lower beta (15%) as strategy works well during high volatility periods.",
        "max_profit": "Net Premium Received",
        "max_loss": "Difference in Strike Prices - Net Premium Received",
        "breakeven": "Lower Strike Price + Net Premium Received",
    },
    # Income strategies
    "iron_condor": {
        "name": "Iron Condor",
        "category": "options",
        "description": "Combination of bull put spread and bear call spread. This market-neutral strategy profits from low volatility and time decay when stock stays within a range.",
        "objective": "Generate income from time decay in low volatility, range-bound markets",
        "market_outlook": "Neutral - expecting stock to trade within defined range",
        "risk_profile": "Limited risk and reward - profits from time decay and low volatility",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.30,
            "momentum": 0.10,
            "iv": 0.35,
            "forecast": 0.10,
        },
        "weights_rationale": "Highest IV weight (35%) as strategy benefits from selling high volatility and profiting from volatility decline. High beta weight (30%) for stable, predictable stocks. Low momentum (10%) as strategy requires range-bound action.",
        "max_profit": "Net Premium Received",
        "max_loss": "Width of Wider Spread - Net Premium Received",
        "breakeven": "Two breakeven points: Put Strike - Net Credit and Call Strike + Net Credit",
    },
    "iron_butterfly": {
        "name": "Iron Butterfly",
        "category": "options",
        "description": "Combination of bull put spread and bear call spread centered at same strike price. This strategy profits when stock stays very close to center strike.",
        "objective": "Generate income from minimal price movement around center strike",
        "market_outlook": "Neutral - expecting stock to stay very close to current price",
        "risk_profile": "Limited risk and reward - requires precise price prediction",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.25,
            "momentum": 0.15,
            "iv": 0.35,
            "forecast": 0.10,
        },
        "weights_rationale": "Highest IV weight (35%) for volatility selling. Moderate beta (25%) for stable stocks. Lower momentum (15%) and forecast (10%) as strategy requires minimal price movement.",
        "max_profit": "Net Premium Received",
        "max_loss": "Strike Width - Net Premium Received",
        "breakeven": "Two points: Center Strike ± Net Premium Received",
    },
    "calendar_spread": {
        "name": "Calendar Spread",
        "category": "options",
        "description": "Selling near-term option and buying longer-term option at same strike. This strategy profits from time decay acceleration and volatility differences between expirations.",
        "objective": "Profit from time decay differences and volatility changes between expirations",
        "market_outlook": "Neutral - expecting stock to stay near strike price until near-term expiration",
        "risk_profile": "Limited risk - maximum loss is net premium paid",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.20,
            "momentum": 0.10,
            "iv": 0.40,
            "forecast": 0.15,
        },
        "weights_rationale": "Highest IV weight (40%) as strategy is most sensitive to volatility changes. Lower momentum (10%) as strategy benefits from stable prices. Moderate forecast (15%) for timing expiration cycles.",
        "max_profit": "Varies based on volatility and time decay",
        "max_loss": "Net Premium Paid",
        "breakeven": "Complex - depends on volatility and time to expiration",
    },
    # Volatility strategies
    "long_straddle": {
        "name": "Long Straddle",
        "category": "options",
        "description": "Buying call and put options at same strike and expiration. This strategy profits from large price movements in either direction, requiring high volatility.",
        "objective": "Profit from large price movements in either direction",
        "market_outlook": "Neutral direction, expecting high volatility and significant price movement",
        "risk_profile": "Limited risk (premium paid), unlimited profit potential in either direction",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.15,
            "momentum": 0.20,
            "iv": 0.40,
            "forecast": 0.10,
        },
        "weights_rationale": "Highest IV weight (40%) as strategy requires buying volatility cheaply. Moderate momentum (20%) for identifying potential breakout candidates. Lower directional weights as strategy is direction-neutral.",
        "max_profit": "Unlimited in either direction",
        "max_loss": "Total Premium Paid",
        "breakeven": "Two points: Strike Price ± Total Premium Paid",
    },
    "long_strangle": {
        "name": "Long Strangle",
        "category": "options",
        "description": "Buying out-of-the-money call and put options with same expiration. Lower cost than straddle but requires larger price movements for profitability.",
        "objective": "Profit from large price movements in either direction at lower cost than straddle",
        "market_outlook": "Neutral direction, expecting very high volatility and large price movement",
        "risk_profile": "Limited risk (premium paid), unlimited profit potential with larger breakeven range",
        "default_weights": {
            "rsi": 0.15,
            "beta": 0.15,
            "momentum": 0.20,
            "iv": 0.40,
            "forecast": 0.10,
        },
        "weights_rationale": "Similar to straddle - highest IV weight (40%) for volatility analysis. Momentum (20%) helps identify potential large moves. Direction-neutral weighting for RSI, beta, and forecast.",
        "max_profit": "Unlimited in either direction",
        "max_loss": "Total Premium Paid",
        "breakeven": "Two points: Call Strike + Total Premium and Put Strike - Total Premium",
    },
    "short_straddle": {
        "name": "Short Straddle",
        "category": "options",
        "description": "Selling call and put options at same strike and expiration. This strategy profits from low volatility and minimal price movement, collecting time decay.",
        "objective": "Generate income from time decay and volatility decline",
        "market_outlook": "Neutral - expecting low volatility and price stability",
        "risk_profile": "Limited profit (premium received), unlimited risk in either direction",
        "default_weights": {
            "rsi": 0.20,
            "beta": 0.25,
            "momentum": 0.15,
            "iv": 0.30,
            "forecast": 0.10,
        },
        "weights_rationale": "High IV weight (30%) for selling overpriced volatility. Higher beta (25%) for stable, predictable stocks. Lower momentum (15%) as strategy requires minimal price movement.",
        "max_profit": "Total Premium Received",
        "max_loss": "Unlimited in either direction",
        "breakeven": "Two points: Strike Price ± Total Premium Received",
    },
    "short_strangle": {
        "name": "Short Strangle",
        "category": "options",
        "description": "Selling out-of-the-money call and put options with same expiration. Safer than short straddle with defined range for profitability but lower premium collected.",
        "objective": "Generate income with wider profit range than short straddle",
        "market_outlook": "Neutral - expecting stock to stay between strike prices",
        "risk_profile": "Limited profit (premium received), very high risk outside strikes",
        "default_weights": {
            "rsi": 0.20,
            "beta": 0.25,
            "momentum": 0.15,
            "iv": 0.30,
            "forecast": 0.10,
        },
        "weights_rationale": "Similar to short straddle - high IV (30%) for premium collection, higher beta (25%) for stability. Lower momentum (15%) and forecast (10%) as strategy benefits from range-bound markets.",
        "max_profit": "Total Premium Received",
        "max_loss": "Unlimited beyond breakeven points",
        "breakeven": "Two points: Put Strike - Premium and Call Strike + Premium",
    },
}


def get_metrics_by_category(
    category: Literal["growth", "value", "health", "risk", "options"]
) -> dict[str, MetricDefinition] | dict[str, OptionsStrategyDefinition]:
    """
    Filter metrics by category.

    Args:
        category: The category to filter by

    Returns:
        Dictionary of metrics in the specified category
    """
    if category == "options":
        return OPTIONS_STRATEGIES
    return {k: v for k, v in GLOSSARY.items() if v["category"] == category}


def get_metric_names() -> dict[str, str]:
    """
    Get a mapping of metric keys to their display names.

    Returns:
        Dictionary mapping metric keys to display names
    """
    names = {k: v["name"] for k, v in GLOSSARY.items()}
    names.update({k: v["name"] for k, v in OPTIONS_STRATEGIES.items()})
    return names


def get_all_definitions() -> dict[str, MetricDefinition | OptionsStrategyDefinition]:
    """
    Get all metric and strategy definitions.

    Returns:
        Dictionary of all definitions
    """
    all_definitions = GLOSSARY.copy()
    all_definitions.update(OPTIONS_STRATEGIES)
    return all_definitions


def get_options_strategy_info(strategy_key: str) -> OptionsStrategyDefinition:
    """
    Get detailed information about a specific options strategy.

    Args:
        strategy_key: The key of the strategy to look up

    Returns:
        OptionsStrategyDefinition: Complete strategy information

    Raises:
        KeyError: If the strategy key is not found
    """
    if strategy_key not in OPTIONS_STRATEGIES:
        available_strategies = ", ".join(OPTIONS_STRATEGIES.keys())
        raise KeyError(
            f"Strategy '{strategy_key}' not found. Available strategies: {available_strategies}"
        )
    return OPTIONS_STRATEGIES[strategy_key]


def search_options_strategies(search_term: str) -> dict[str, OptionsStrategyDefinition]:
    """
    Search for options strategies by name or description.

    Args:
        search_term: Term to search for (case-insensitive)

    Returns:
        Dictionary of matching strategies
    """
    search_term = search_term.lower()
    matches = {}

    for key, strategy in OPTIONS_STRATEGIES.items():
        if (
            search_term in strategy["name"].lower()
            or search_term in strategy["description"].lower()
            or search_term in strategy["objective"].lower()
            or search_term in strategy["market_outlook"].lower()
        ):
            matches[key] = strategy

    return matches


def get_strategies_by_outlook(
    outlook: Literal["bullish", "bearish", "neutral"]
) -> dict[str, OptionsStrategyDefinition]:
    """
    Get options strategies filtered by market outlook.

    Args:
        outlook: Market outlook to filter by

    Returns:
        Dictionary of strategies matching the outlook
    """
    outlook_keywords = {
        "bullish": ["bullish", "bull"],
        "bearish": ["bearish", "bear"],
        "neutral": ["neutral", "range-bound", "low volatility"],
    }

    keywords = outlook_keywords.get(outlook, [])
    matches = {}

    for key, strategy in OPTIONS_STRATEGIES.items():
        market_outlook = strategy["market_outlook"].lower()
        if any(keyword in market_outlook for keyword in keywords):
            matches[key] = strategy

    return matches


def get_strategies_by_risk_profile(
    risk_level: Literal["limited", "unlimited", "moderate"]
) -> dict[str, OptionsStrategyDefinition]:
    """
    Get options strategies filtered by risk profile.

    Args:
        risk_level: Risk level to filter by

    Returns:
        Dictionary of strategies matching the risk profile
    """
    matches = {}

    for key, strategy in OPTIONS_STRATEGIES.items():
        risk_profile = strategy["risk_profile"].lower()
        if risk_level.lower() in risk_profile:
            matches[key] = strategy

    return matches


def search_metrics(search_term: str) -> dict[str, MetricDefinition]:
    """
    Search for metrics by name or description.

    Args:
        search_term: Term to search for (case-insensitive)

    Returns:
        Dictionary of matching metrics
    """
    search_lower = search_term.lower()
    return {
        k: v
        for k, v in GLOSSARY.items()
        if search_lower in v["name"].lower() or search_lower in v["description"].lower()
    }


def get_metric_info(metric_key: str) -> MetricDefinition:
    """
    Get detailed information about a specific metric.

    Args:
        metric_key: The key of the metric to look up

    Returns:
        MetricDefinition for the requested metric

    Raises:
        KeyError: If the metric key is not found
    """
    if metric_key not in GLOSSARY:
        available_keys = list(GLOSSARY.keys())
        raise KeyError(
            f"Metric '{metric_key}' not found. Available metrics: {available_keys}"
        )
    return GLOSSARY[metric_key]
