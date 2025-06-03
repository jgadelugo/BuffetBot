from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def analyze_risk_metrics(data: dict) -> dict:
    """
    Analyze various risk metrics for the company.

    Args:
        data: Dictionary containing financial statements and metrics

    Returns:
        Dict containing:
            - market_risk: Market risk metrics
            - financial_risk: Financial risk metrics
            - business_risk: Business risk metrics
            - overall_risk: Overall risk assessment
    """
    try:
        logger.info("Starting risk analysis")

        # Initialize result dictionary
        result = {
            "market_risk": {},
            "financial_risk": {},
            "business_risk": {},
            "overall_risk": {
                "score": 0.0,
                "level": "Unknown",
                "factors": [],
                "warnings": [],
                "errors": [],
            },
        }

        # Validate input data
        if not isinstance(data, dict):
            error_msg = "Invalid input: data must be a dictionary"
            logger.error(error_msg)
            result["overall_risk"]["errors"].append(error_msg)
            return result

        # Get financial statements and fundamentals
        income_stmt = data.get("income_stmt")
        balance_sheet = data.get("balance_sheet")
        cash_flow = data.get("cash_flow")
        fundamentals = data.get("fundamentals", {})

        # Calculate market risk metrics
        try:
            logger.info("Calculating market risk metrics")
            market_risk = {}

            # Beta
            beta = fundamentals.get("beta")
            if beta is not None:
                market_risk["beta"] = beta
                if beta > 1.5:
                    result["overall_risk"]["factors"].append(
                        "High market sensitivity (beta > 1.5)"
                    )
                elif beta < 0.5:
                    result["overall_risk"]["factors"].append(
                        "Low market sensitivity (beta < 0.5)"
                    )
            else:
                result["overall_risk"]["warnings"].append(
                    "Beta not available in fundamentals data"
                )

            # Volatility
            if "price_data" in data and data["price_data"] is not None:
                try:
                    returns = data["price_data"]["Close"].pct_change().dropna()
                    if len(returns) < 2:
                        result["overall_risk"]["warnings"].append(
                            "Insufficient price data for volatility calculation"
                        )
                    else:
                        volatility = returns.std() * np.sqrt(
                            252
                        )  # Annualized volatility
                        market_risk["volatility"] = volatility
                        if volatility > 0.4:
                            result["overall_risk"]["factors"].append(
                                "High price volatility (> 40%)"
                            )
                        elif volatility < 0.1:
                            result["overall_risk"]["factors"].append(
                                "Low price volatility (< 10%)"
                            )
                except Exception as e:
                    error_msg = f"Error calculating volatility: {str(e)}"
                    logger.error(error_msg)
                    result["overall_risk"]["errors"].append(error_msg)
            else:
                result["overall_risk"]["warnings"].append(
                    "Price data not available for volatility calculation"
                )

            result["market_risk"] = market_risk
            logger.info("Successfully calculated market risk metrics")

        except Exception as e:
            error_msg = f"Error calculating market risk metrics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["overall_risk"]["errors"].append(error_msg)

        # Calculate financial risk metrics
        try:
            logger.info("Calculating financial risk metrics")
            financial_risk = {}

            if balance_sheet is None or balance_sheet.empty:
                result["overall_risk"]["warnings"].append(
                    "Balance sheet data not available"
                )
            else:
                # Debt to equity
                if (
                    "Total Liabilities" in balance_sheet.columns
                    and "Total Stockholder Equity" in balance_sheet.columns
                ):
                    total_debt = balance_sheet["Total Liabilities"].iloc[0]
                    total_equity = balance_sheet["Total Stockholder Equity"].iloc[0]
                    if total_equity != 0:
                        debt_to_equity = total_debt / total_equity
                        financial_risk["debt_to_equity"] = debt_to_equity
                        if debt_to_equity > 1.0:
                            result["overall_risk"]["factors"].append(
                                "High leverage (D/E > 1.0)"
                            )
                    else:
                        result["overall_risk"]["warnings"].append(
                            "Total Stockholder Equity is zero"
                        )
                else:
                    result["overall_risk"]["warnings"].append(
                        "Missing required columns for debt-to-equity calculation"
                    )

                # Interest coverage
                if income_stmt is not None and not income_stmt.empty:
                    if (
                        "Operating Income" in income_stmt.columns
                        and "Interest Expense" in income_stmt.columns
                    ):
                        ebit = income_stmt["Operating Income"].iloc[0]
                        interest_expense = income_stmt["Interest Expense"].iloc[0]
                        if interest_expense != 0:
                            interest_coverage = ebit / interest_expense
                            financial_risk["interest_coverage"] = interest_coverage
                            if interest_coverage < 1.0:
                                result["overall_risk"]["factors"].append(
                                    "Low interest coverage (< 1.0)"
                                )
                        else:
                            result["overall_risk"]["warnings"].append(
                                "Interest Expense is zero"
                            )
                    else:
                        result["overall_risk"]["warnings"].append(
                            "Missing required columns for interest coverage calculation"
                        )
                else:
                    result["overall_risk"]["warnings"].append(
                        "Income statement data not available"
                    )

            result["financial_risk"] = financial_risk
            logger.info("Successfully calculated financial risk metrics")

        except Exception as e:
            error_msg = f"Error calculating financial risk metrics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["overall_risk"]["errors"].append(error_msg)

        # Calculate business risk metrics
        try:
            logger.info("Calculating business risk metrics")
            business_risk = {}

            if income_stmt is None or income_stmt.empty:
                result["overall_risk"]["warnings"].append(
                    "Income statement data not available"
                )
            else:
                # Operating margin
                if (
                    "Operating Income" in income_stmt.columns
                    and "Total Revenue" in income_stmt.columns
                ):
                    operating_income = income_stmt["Operating Income"].iloc[0]
                    revenue = income_stmt["Total Revenue"].iloc[0]
                    if revenue != 0:
                        operating_margin = operating_income / revenue
                        business_risk["operating_margin"] = operating_margin
                        if operating_margin < 0.1:
                            result["overall_risk"]["factors"].append(
                                "Low operating margin (< 10%)"
                            )
                    else:
                        result["overall_risk"]["warnings"].append(
                            "Total Revenue is zero"
                        )
                else:
                    result["overall_risk"]["warnings"].append(
                        "Missing required columns for operating margin calculation"
                    )

                # Revenue concentration
                if "Total Revenue" in income_stmt.columns:
                    revenue = income_stmt["Total Revenue"].iloc[0]
                    if revenue > 0:
                        business_risk["revenue"] = revenue
                        if revenue < 1e9:  # Less than $1B
                            result["overall_risk"]["factors"].append(
                                "Small revenue base (< $1B)"
                            )
                    else:
                        result["overall_risk"]["warnings"].append(
                            "Total Revenue is zero or negative"
                        )
                else:
                    result["overall_risk"]["warnings"].append(
                        "Total Revenue column not found"
                    )

            result["business_risk"] = business_risk
            logger.info("Successfully calculated business risk metrics")

        except Exception as e:
            error_msg = f"Error calculating business risk metrics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["overall_risk"]["errors"].append(error_msg)

        # Calculate overall risk score
        try:
            logger.info("Calculating overall risk score")

            # Initialize weights
            weights = {"market_risk": 0.3, "financial_risk": 0.4, "business_risk": 0.3}

            # Calculate component scores
            market_score = 0.0
            if result["market_risk"]:
                beta = result["market_risk"].get("beta", 1.0)
                volatility = result["market_risk"].get("volatility", 0.2)
                # Higher beta and volatility = higher risk
                market_score = (
                    min(abs(beta - 1.0) * 2, 2.0) / 2.0 + min(volatility * 2, 1.0)
                ) / 2

                # Add detailed explanations for market risk
                if beta > 1.5:
                    result["overall_risk"]["factors"].append(
                        f"Beta of {beta:.2f} indicates high volatility relative to market"
                    )
                elif beta < 0.5:
                    result["overall_risk"]["factors"].append(
                        f"Beta of {beta:.2f} indicates low correlation with market"
                    )

                if volatility > 0.4:
                    result["overall_risk"]["factors"].append(
                        f"Annual volatility of {volatility:.1%} is very high"
                    )
                elif volatility > 0.3:
                    result["overall_risk"]["factors"].append(
                        f"Annual volatility of {volatility:.1%} is above average"
                    )
            else:
                result["overall_risk"]["warnings"].append(
                    "Market risk metrics not available for score calculation"
                )

            financial_score = 0.0
            if result["financial_risk"]:
                debt_to_equity = result["financial_risk"].get("debt_to_equity", 0.5)
                interest_coverage = result["financial_risk"].get(
                    "interest_coverage", 3.0
                )
                # Higher debt and lower coverage = higher risk
                financial_score = (
                    min(debt_to_equity, 2.0) / 2.0 + max(0, 1 - interest_coverage / 5.0)
                ) / 2

                # Add detailed explanations for financial risk
                if debt_to_equity > 2.0:
                    result["overall_risk"]["factors"].append(
                        f"Debt-to-equity ratio of {debt_to_equity:.2f} indicates very high leverage"
                    )
                elif debt_to_equity > 1.0:
                    result["overall_risk"]["factors"].append(
                        f"Debt-to-equity ratio of {debt_to_equity:.2f} indicates significant leverage"
                    )

                if interest_coverage < 1.5:
                    result["overall_risk"]["factors"].append(
                        f"Interest coverage of {interest_coverage:.2f}x is dangerously low"
                    )
                elif interest_coverage < 3.0:
                    result["overall_risk"]["factors"].append(
                        f"Interest coverage of {interest_coverage:.2f}x may be insufficient in downturns"
                    )
            else:
                result["overall_risk"]["warnings"].append(
                    "Financial risk metrics not available for score calculation"
                )

            business_score = 0.0
            if result["business_risk"]:
                operating_margin = result["business_risk"].get("operating_margin", 0.15)
                revenue = result["business_risk"].get("revenue", 1e9)
                # Lower margins and smaller revenue = higher risk
                business_score = (
                    max(0, 1 - operating_margin / 0.3) + max(0, 1 - revenue / 1e10)
                ) / 2

                # Add detailed explanations for business risk
                if operating_margin < 0.05:
                    result["overall_risk"]["factors"].append(
                        f"Operating margin of {operating_margin:.1%} is very low"
                    )
                elif operating_margin < 0.10:
                    result["overall_risk"]["factors"].append(
                        f"Operating margin of {operating_margin:.1%} is below healthy levels"
                    )

                if revenue < 1e8:  # Less than $100M
                    result["overall_risk"]["factors"].append(
                        f"Small company with revenue of ${revenue/1e6:.0f}M"
                    )
                elif revenue < 1e9:  # Less than $1B
                    result["overall_risk"]["factors"].append(
                        f"Mid-size company with revenue of ${revenue/1e6:.0f}M"
                    )
            else:
                result["overall_risk"]["warnings"].append(
                    "Business risk metrics not available for score calculation"
                )

            # Calculate weighted average
            overall_score = (
                market_score * weights["market_risk"]
                + financial_score * weights["financial_risk"]
                + business_score * weights["business_risk"]
            ) * 100  # Convert to percentage

            result["overall_risk"]["score"] = overall_score

            # Determine risk level with detailed explanations
            if overall_score >= 70:
                risk_level = "High"
                result["overall_risk"]["factors"].insert(
                    0,
                    "High risk due to combination of market volatility, financial leverage, and business challenges",
                )
            elif overall_score >= 40:
                risk_level = "Moderate"
                result["overall_risk"]["factors"].insert(
                    0,
                    "Moderate risk with some concerns in financial health or market exposure",
                )
            else:
                risk_level = "Low"
                result["overall_risk"]["factors"].insert(
                    0, "Low risk with stable financials and manageable market exposure"
                )

            result["overall_risk"]["level"] = risk_level
            logger.info(f"Calculated risk score: {overall_score:.2f}")
            logger.info(f"Determined risk level: {risk_level}")

        except Exception as e:
            error_msg = f"Error calculating overall risk score: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["overall_risk"]["errors"].append(error_msg)
            result["overall_risk"].update(
                {
                    "score": 0.0,
                    "level": "Unknown",
                    "factors": ["Error calculating risk score"],
                }
            )

        logger.info("Successfully completed risk analysis")
        return result

    except Exception as e:
        error_msg = f"Error in risk analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "market_risk": {},
            "financial_risk": {},
            "business_risk": {},
            "overall_risk": {
                "score": 0.0,
                "level": "Unknown",
                "factors": ["Error in risk analysis"],
                "warnings": [],
                "errors": [error_msg],
            },
        }


def _calculate_market_risk(
    price_data: pd.DataFrame, market_data: pd.DataFrame | None = None
) -> dict[str, float]:
    """
    Calculate market risk metrics.

    Args:
        price_data: Historical price data
        market_data: Market index data

    Returns:
        Dict containing market risk metrics
    """
    try:
        logger.info("Calculating market risk metrics")

        # Calculate returns
        returns = price_data["Close"].pct_change().dropna()

        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Calculate Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Calculate beta if market data is available
        beta = None
        if market_data is not None and not market_data.empty:
            market_returns = market_data["Close"].pct_change().dropna()
            beta = returns.cov(market_returns) / market_returns.var()

        metrics = {
            "volatility": volatility,
            "var_95": var_95,
            "max_drawdown": max_drawdown,
            "beta": beta,
        }

        logger.info("Successfully calculated market risk metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating market risk metrics: {str(e)}")
        raise


def _calculate_financial_risk(
    financials: dict[str, pd.DataFrame | dict]
) -> dict[str, float]:
    """
    Calculate financial risk metrics.

    Args:
        financials: Dictionary containing financial statements and metrics

    Returns:
        Dict containing financial risk metrics
    """
    try:
        logger.info("Calculating financial risk metrics")

        balance_sheet = financials.get("balance_sheet")
        income_stmt = financials.get("income_stmt")

        if balance_sheet is None or income_stmt is None:
            raise ValueError("Missing required financial statements")

        # Calculate debt ratios
        total_debt = (
            balance_sheet.loc["Total Debt"].iloc[0]
            if "Total Debt" in balance_sheet.index
            else 0
        )
        total_assets = (
            balance_sheet.loc["Total Assets"].iloc[0]
            if "Total Assets" in balance_sheet.index
            else 0
        )
        total_equity = (
            balance_sheet.loc["Total Equity"].iloc[0]
            if "Total Equity" in balance_sheet.index
            else 0
        )

        debt_to_equity = (
            total_debt / total_equity if total_equity != 0 else float("inf")
        )
        debt_to_assets = (
            total_debt / total_assets if total_assets != 0 else float("inf")
        )

        # Calculate interest coverage
        ebit = income_stmt.loc["EBIT"].iloc[0] if "EBIT" in income_stmt.index else 0
        interest_expense = (
            income_stmt.loc["Interest Expense"].iloc[0]
            if "Interest Expense" in income_stmt.index
            else 0
        )
        interest_coverage = (
            ebit / interest_expense if interest_expense != 0 else float("inf")
        )

        metrics = {
            "debt_to_equity": debt_to_equity,
            "debt_to_assets": debt_to_assets,
            "interest_coverage": interest_coverage,
        }

        logger.info("Successfully calculated financial risk metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating financial risk metrics: {str(e)}")
        raise


def _calculate_business_risk(
    financials: dict[str, pd.DataFrame | dict]
) -> dict[str, float]:
    """
    Calculate business risk metrics.

    Args:
        financials: Dictionary containing financial statements and metrics

    Returns:
        Dict containing business risk metrics
    """
    try:
        logger.info("Calculating business risk metrics")

        income_stmt = financials.get("income_stmt")

        if income_stmt is None:
            raise ValueError("Missing income statement")

        # Calculate revenue volatility
        revenue = (
            income_stmt.loc["Total Revenue"]
            if "Total Revenue" in income_stmt.index
            else None
        )
        if revenue is not None and len(revenue) > 1:
            revenue_volatility = revenue.pct_change().std()
        else:
            revenue_volatility = None

        # Calculate operating leverage
        ebit = income_stmt.loc["EBIT"] if "EBIT" in income_stmt.index else None
        if ebit is not None and revenue is not None and len(ebit) > 1:
            operating_leverage = (ebit.pct_change() / revenue.pct_change()).mean()
        else:
            operating_leverage = None

        metrics = {
            "revenue_volatility": revenue_volatility,
            "operating_leverage": operating_leverage,
        }

        logger.info("Successfully calculated business risk metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating business risk metrics: {str(e)}")
        raise


def _calculate_risk_score(
    market_risk: dict[str, float],
    financial_risk: dict[str, float],
    business_risk: dict[str, float],
) -> float:
    """
    Calculate overall risk score.

    Args:
        market_risk: Market risk metrics
        financial_risk: Financial risk metrics
        business_risk: Business risk metrics

    Returns:
        float: Overall risk score (0-100)
    """
    try:
        logger.info("Calculating overall risk score")

        # Market risk score (40% weight)
        market_score = 0
        if market_risk["volatility"] is not None:
            market_score += min(market_risk["volatility"] * 100, 100) * 0.4
        if market_risk["beta"] is not None:
            market_score += min(abs(market_risk["beta"]) * 20, 100) * 0.3
        if market_risk["max_drawdown"] is not None:
            market_score += min(abs(market_risk["max_drawdown"]) * 200, 100) * 0.3

        # Financial risk score (35% weight)
        financial_score = 0
        if financial_risk["debt_to_equity"] is not None:
            financial_score += min(financial_risk["debt_to_equity"] * 20, 100) * 0.4
        if financial_risk["interest_coverage"] is not None:
            financial_score += min(100 / financial_risk["interest_coverage"], 100) * 0.6

        # Business risk score (25% weight)
        business_score = 0
        if business_risk["revenue_volatility"] is not None:
            business_score += min(business_risk["revenue_volatility"] * 200, 100) * 0.5
        if business_risk["operating_leverage"] is not None:
            business_score += (
                min(abs(business_risk["operating_leverage"]) * 20, 100) * 0.5
            )

        # Calculate weighted average
        risk_score = market_score * 0.4 + financial_score * 0.35 + business_score * 0.25

        logger.info(f"Calculated risk score: {risk_score:.2f}")
        return risk_score

    except Exception as e:
        logger.error(f"Error calculating risk score: {str(e)}")
        raise


def _determine_risk_level(risk_score: float) -> str:
    """
    Determine risk level based on risk score.

    Args:
        risk_score: Overall risk score (0-100)

    Returns:
        str: Risk level classification
    """
    try:
        logger.info("Determining risk level")

        if risk_score < 20:
            level = "Very Low"
        elif risk_score < 40:
            level = "Low"
        elif risk_score < 60:
            level = "Moderate"
        elif risk_score < 80:
            level = "High"
        else:
            level = "Very High"

        logger.info(f"Determined risk level: {level}")
        return level

    except Exception as e:
        logger.error(f"Error determining risk level: {str(e)}")
        raise
