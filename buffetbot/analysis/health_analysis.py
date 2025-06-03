from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from buffetbot.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Financial ratio thresholds
RATIO_THRESHOLDS = {
    "current_ratio": {"low": 1.0, "high": 2.0},
    "debt_to_equity": {"low": 0.5, "high": 1.0},
    "debt_to_assets": {"low": 0.3, "high": 0.5},
    "interest_coverage": {"low": 1.0, "high": 3.0},
    "return_on_equity": {"low": 0.1, "high": 0.2},
    "return_on_assets": {"low": 0.05, "high": 0.1},
    "gross_margin": {"low": 0.2, "high": 0.4},
    "operating_margin": {"low": 0.1, "high": 0.2},
    "net_margin": {"low": 0.05, "high": 0.15},
}


def _validate_financial_data(
    financials: pd.DataFrame, required_columns: list[str]
) -> bool:
    """
    Validate that required financial data columns exist.

    Args:
        financials: DataFrame containing financial data
        required_columns: List of required column names

    Returns:
        bool: True if all required columns exist, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in financials.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True


def _calculate_financial_ratio(
    financials: pd.DataFrame, numerator: str, denominator: str, default: float = 0.0
) -> tuple[float, str | None]:
    """
    Calculate a financial ratio with error handling.

    Args:
        financials: DataFrame with financial data
        numerator: Column name for numerator
        denominator: Column name for denominator
        default: Default value if calculation fails

    Returns:
        Tuple[float, Optional[str]]: (ratio value, warning message if any)
    """
    try:
        if numerator not in financials.columns or denominator not in financials.columns:
            return (
                default,
                f"Missing columns for ratio calculation: {numerator}/{denominator}",
            )

        num_value = financials[numerator].iloc[-1]
        den_value = financials[denominator].iloc[-1]

        if den_value == 0:
            return default, f"Zero denominator in ratio calculation: {denominator}"

        return num_value / den_value, None

    except Exception as e:
        logger.warning(f"Error calculating ratio {numerator}/{denominator}: {str(e)}")
        return default, str(e)


def calculate_piotroski_score(
    income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame
) -> int:
    """
    Calculate the Piotroski F-Score for financial health assessment.

    Args:
        income_stmt: Income statement DataFrame
        balance_sheet: Balance sheet DataFrame
        cash_flow: Cash flow statement DataFrame

    Returns:
        int: Piotroski F-Score (0-9)
    """
    try:
        logger.info("Calculating Piotroski F-Score")
        score = 0

        # Profitability criteria
        if income_stmt is not None and not income_stmt.empty:
            # Net Income > 0
            if (
                "Net Income" in income_stmt.columns
                and income_stmt["Net Income"].iloc[0] > 0
            ):
                score += 1

            # Operating Cash Flow > 0
            if (
                cash_flow is not None
                and "Operating Cash Flow" in cash_flow.columns
                and cash_flow["Operating Cash Flow"].iloc[0] > 0
            ):
                score += 1

            # Return on Assets > 0
            if (
                "Net Income" in income_stmt.columns
                and "Total Assets" in balance_sheet.columns
            ):
                total_assets = balance_sheet["Total Assets"].iloc[0]
                if total_assets != 0:  # Check for division by zero
                    roa = income_stmt["Net Income"].iloc[0] / total_assets
                    if roa > 0:
                        score += 1
                else:
                    logger.warning("Total Assets is zero, skipping ROA calculation")

        # Leverage, Liquidity, and Source of Funds criteria
        if balance_sheet is not None and not balance_sheet.empty:
            # Long-term debt to assets ratio
            if (
                "Long Term Debt" in balance_sheet.columns
                and "Total Assets" in balance_sheet.columns
            ):
                total_assets = balance_sheet["Total Assets"].iloc[0]
                if total_assets != 0:  # Check for division by zero
                    debt_ratio = balance_sheet["Long Term Debt"].iloc[0] / total_assets
                    if debt_ratio < 0.4:  # Conservative threshold
                        score += 1
                else:
                    logger.warning(
                        "Total Assets is zero, skipping debt ratio calculation"
                    )

            # Current ratio
            if (
                "Total Current Assets" in balance_sheet.columns
                and "Total Current Liabilities" in balance_sheet.columns
            ):
                current_liabilities = balance_sheet["Total Current Liabilities"].iloc[0]
                if current_liabilities != 0:  # Check for division by zero
                    current_ratio = (
                        balance_sheet["Total Current Assets"].iloc[0]
                        / current_liabilities
                    )
                    if current_ratio > 1:
                        score += 1
                else:
                    logger.warning(
                        "Current Liabilities is zero, skipping current ratio calculation"
                    )

        # Operating Efficiency criteria
        if income_stmt is not None and not income_stmt.empty:
            # Gross margin
            if (
                "Gross Profit" in income_stmt.columns
                and "Total Revenue" in income_stmt.columns
            ):
                revenue = income_stmt["Total Revenue"].iloc[0]
                if revenue != 0:  # Check for division by zero
                    gross_margin = income_stmt["Gross Profit"].iloc[0] / revenue
                    if gross_margin > 0.2:  # Conservative threshold
                        score += 1
                else:
                    logger.warning(
                        "Total Revenue is zero, skipping gross margin calculation"
                    )

            # Asset turnover
            if (
                "Total Revenue" in income_stmt.columns
                and "Total Assets" in balance_sheet.columns
            ):
                total_assets = balance_sheet["Total Assets"].iloc[0]
                if total_assets != 0:  # Check for division by zero
                    asset_turnover = income_stmt["Total Revenue"].iloc[0] / total_assets
                    if asset_turnover > 0.1:  # Conservative threshold
                        score += 1
                else:
                    logger.warning(
                        "Total Assets is zero, skipping asset turnover calculation"
                    )

        logger.info(f"Calculated Piotroski F-Score: {score}/9")
        return score

    except Exception as e:
        logger.error(f"Error calculating Piotroski F-Score: {str(e)}", exc_info=True)
        return 0


def calculate_altman_z_score(
    balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame
) -> float:
    """
    Calculate the Altman Z-Score for bankruptcy risk assessment.

    Args:
        balance_sheet: Balance sheet DataFrame
        income_stmt: Income statement DataFrame

    Returns:
        float: Altman Z-Score
    """
    try:
        logger.info("Calculating Altman Z-Score")

        # Required columns
        required_cols = {
            "balance_sheet": [
                "Total Assets",
                "Total Current Assets",
                "Total Current Liabilities",
                "Total Liabilities",
                "Retained Earnings",
            ],
            "income_stmt": ["EBIT", "Total Revenue"],
        }

        # Check for required columns
        missing_cols = []
        for statement, cols in required_cols.items():
            df = balance_sheet if statement == "balance_sheet" else income_stmt
            if df is None or df.empty:
                missing_cols.extend(cols)
            else:
                # Special handling for EBIT - use Operating Income as fallback
                if (
                    statement == "income_stmt"
                    and "EBIT" not in df.columns
                    and "Operating Income" in df.columns
                ):
                    # Use Operating Income as EBIT
                    income_stmt = income_stmt.copy()
                    income_stmt["EBIT"] = income_stmt["Operating Income"]
                    logger.info(
                        "Using Operating Income as EBIT for Altman Z-Score calculation"
                    )
                else:
                    missing_cols.extend([col for col in cols if col not in df.columns])

        if missing_cols:
            logger.warning(
                f"Missing required columns for Altman Z-Score: {', '.join(missing_cols)}"
            )
            return 0.0

        # Get total assets once
        total_assets = balance_sheet["Total Assets"].iloc[0]
        if total_assets == 0:
            logger.warning("Total Assets is zero, cannot calculate Altman Z-Score")
            return 0.0

        # Calculate components with division by zero checks
        try:
            X1 = (
                balance_sheet["Total Current Assets"].iloc[0]
                - balance_sheet["Total Current Liabilities"].iloc[0]
            ) / total_assets
            X2 = balance_sheet["Retained Earnings"].iloc[0] / total_assets
            X3 = income_stmt["EBIT"].iloc[0] / total_assets
            X4 = balance_sheet["Total Liabilities"].iloc[0] / total_assets
            X5 = income_stmt["Total Revenue"].iloc[0] / total_assets

            # Calculate Z-Score
            z_score = (1.2 * X1) + (1.4 * X2) + (3.3 * X3) + (0.6 * X4) + (1.0 * X5)

            logger.info(f"Calculated Altman Z-Score: {z_score:.2f}")
            return z_score

        except ZeroDivisionError:
            logger.warning("Division by zero encountered in Altman Z-Score calculation")
            return 0.0

    except Exception as e:
        logger.error(f"Error calculating Altman Z-Score: {str(e)}", exc_info=True)
        return 0.0


def calculate_financial_ratios(
    income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate key financial ratios.

    Args:
        income_stmt: Income statement DataFrame
        balance_sheet: Balance sheet DataFrame
        cash_flow: Cash flow statement DataFrame

    Returns:
        Dict containing calculated financial ratios
    """
    try:
        logger.info("Calculating financial ratios")
        ratios = {}

        # Current ratio
        if balance_sheet is not None and not balance_sheet.empty:
            if (
                "Total Current Assets" in balance_sheet.columns
                and "Total Current Liabilities" in balance_sheet.columns
            ):
                current_liabilities = balance_sheet["Total Current Liabilities"].iloc[0]
                if current_liabilities != 0:
                    ratios["current_ratio"] = (
                        balance_sheet["Total Current Assets"].iloc[0]
                        / current_liabilities
                    )
                else:
                    logger.warning(
                        "Current Liabilities is zero, skipping current ratio calculation"
                    )

        # Debt to equity
        if balance_sheet is not None and not balance_sheet.empty:
            if (
                "Total Liabilities" in balance_sheet.columns
                and "Total Stockholder Equity" in balance_sheet.columns
            ):
                total_equity = balance_sheet["Total Stockholder Equity"].iloc[0]
                if total_equity != 0:
                    ratios["debt_to_equity"] = (
                        balance_sheet["Total Liabilities"].iloc[0] / total_equity
                    )
                else:
                    logger.warning(
                        "Total Stockholder Equity is zero, skipping debt to equity calculation"
                    )

        # Debt to assets
        if balance_sheet is not None and not balance_sheet.empty:
            if (
                "Total Liabilities" in balance_sheet.columns
                and "Total Assets" in balance_sheet.columns
            ):
                total_assets = balance_sheet["Total Assets"].iloc[0]
                if total_assets != 0:
                    ratios["debt_to_assets"] = (
                        balance_sheet["Total Liabilities"].iloc[0] / total_assets
                    )
                else:
                    logger.warning(
                        "Total Assets is zero, skipping debt to assets calculation"
                    )

        # Interest coverage
        if income_stmt is not None and not income_stmt.empty:
            if (
                "Operating Income" in income_stmt.columns
                and "Interest Expense" in income_stmt.columns
            ):
                operating_income = income_stmt["Operating Income"].iloc[0]
                interest_expense = income_stmt["Interest Expense"].iloc[0]

                # Handle NaN values - try to use a previous year's value if current is NaN
                if pd.isna(interest_expense) and len(income_stmt) > 1:
                    for i in range(
                        1, min(len(income_stmt), 3)
                    ):  # Check up to 2 previous years
                        alt_interest = income_stmt["Interest Expense"].iloc[i]
                        if not pd.isna(alt_interest) and alt_interest != 0:
                            interest_expense = alt_interest
                            logger.info(
                                f"Using interest expense from year {i} back: {interest_expense}"
                            )
                            break

                if not pd.isna(interest_expense) and interest_expense != 0:
                    ratios["interest_coverage"] = operating_income / interest_expense
                elif interest_expense == 0:
                    logger.warning(
                        "Interest Expense is zero, skipping interest coverage calculation"
                    )
                else:
                    logger.warning(
                        "Interest Expense is NaN, skipping interest coverage calculation"
                    )

        # Return on equity
        if (
            income_stmt is not None
            and not income_stmt.empty
            and balance_sheet is not None
            and not balance_sheet.empty
        ):
            if (
                "Net Income" in income_stmt.columns
                and "Total Stockholder Equity" in balance_sheet.columns
            ):
                total_equity = balance_sheet["Total Stockholder Equity"].iloc[0]
                if total_equity != 0:
                    ratios["return_on_equity"] = (
                        income_stmt["Net Income"].iloc[0] / total_equity
                    )
                else:
                    logger.warning(
                        "Total Stockholder Equity is zero, skipping return on equity calculation"
                    )

        # Return on assets
        if (
            income_stmt is not None
            and not income_stmt.empty
            and balance_sheet is not None
            and not balance_sheet.empty
        ):
            if (
                "Net Income" in income_stmt.columns
                and "Total Assets" in balance_sheet.columns
            ):
                total_assets = balance_sheet["Total Assets"].iloc[0]
                if total_assets != 0:
                    ratios["return_on_assets"] = (
                        income_stmt["Net Income"].iloc[0] / total_assets
                    )
                else:
                    logger.warning(
                        "Total Assets is zero, skipping return on assets calculation"
                    )

        # Gross margin
        if income_stmt is not None and not income_stmt.empty:
            if (
                "Gross Profit" in income_stmt.columns
                and "Total Revenue" in income_stmt.columns
            ):
                revenue = income_stmt["Total Revenue"].iloc[0]
                if revenue != 0:
                    ratios["gross_margin"] = (
                        income_stmt["Gross Profit"].iloc[0] / revenue
                    )
                else:
                    logger.warning(
                        "Total Revenue is zero, skipping gross margin calculation"
                    )

        # Operating margin
        if income_stmt is not None and not income_stmt.empty:
            if (
                "Operating Income" in income_stmt.columns
                and "Total Revenue" in income_stmt.columns
            ):
                revenue = income_stmt["Total Revenue"].iloc[0]
                if revenue != 0:
                    ratios["operating_margin"] = (
                        income_stmt["Operating Income"].iloc[0] / revenue
                    )
                else:
                    logger.warning(
                        "Total Revenue is zero, skipping operating margin calculation"
                    )

        # Net margin
        if income_stmt is not None and not income_stmt.empty:
            if (
                "Net Income" in income_stmt.columns
                and "Total Revenue" in income_stmt.columns
            ):
                revenue = income_stmt["Total Revenue"].iloc[0]
                if revenue != 0:
                    ratios["net_margin"] = income_stmt["Net Income"].iloc[0] / revenue
                else:
                    logger.warning(
                        "Total Revenue is zero, skipping net margin calculation"
                    )

        logger.info(f"Calculated {len(ratios)} financial ratios")
        return ratios

    except Exception as e:
        logger.error(f"Error calculating financial ratios: {str(e)}", exc_info=True)
        return {}


def generate_health_flags(result: dict) -> list[str]:
    """
    Generate health flags based on calculated metrics.

    Args:
        result: Dictionary containing calculated metrics and ratios

    Returns:
        List of health flags
    """
    try:
        logger.info("Generating health flags")
        flags = []

        # Check Piotroski score
        piotroski_score = result.get("piotroski_score", 0)
        if piotroski_score >= 7:
            flags.append("Strong financial health (Piotroski score >= 7)")
        elif piotroski_score <= 3:
            flags.append("Weak financial health (Piotroski score <= 3)")

        # Check Altman Z-score
        altman_z_score = result.get("altman_z_score", 0.0)
        if altman_z_score > 2.99:
            flags.append("Low bankruptcy risk (Altman Z-score > 2.99)")
        elif altman_z_score < 1.81:
            flags.append("High bankruptcy risk (Altman Z-score < 1.81)")

        # Check financial ratios
        ratios = result.get("financial_ratios", {})

        # Current ratio
        current_ratio = ratios.get("current_ratio", 0)
        if current_ratio < RATIO_THRESHOLDS["current_ratio"]["low"]:
            flags.append("Low liquidity (Current ratio < 1.0)")
        elif current_ratio > RATIO_THRESHOLDS["current_ratio"]["high"]:
            flags.append("Excess liquidity (Current ratio > 2.0)")

        # Debt to equity
        debt_to_equity = ratios.get("debt_to_equity", 0)
        if debt_to_equity > RATIO_THRESHOLDS["debt_to_equity"]["high"]:
            flags.append("High leverage (Debt/Equity > 1.0)")

        # Return on equity
        roe = ratios.get("return_on_equity", 0)
        if roe < RATIO_THRESHOLDS["return_on_equity"]["low"]:
            flags.append("Low return on equity (< 10%)")
        elif roe > RATIO_THRESHOLDS["return_on_equity"]["high"]:
            flags.append("Strong return on equity (> 20%)")

        # Operating margin
        operating_margin = ratios.get("operating_margin", 0)
        if operating_margin < RATIO_THRESHOLDS["operating_margin"]["low"]:
            flags.append("Low operating margin (< 10%)")
        elif operating_margin > RATIO_THRESHOLDS["operating_margin"]["high"]:
            flags.append("Strong operating margin (> 20%)")

        logger.info(f"Generated {len(flags)} health flags")
        return flags

    except Exception as e:
        logger.error(f"Error generating health flags: {str(e)}", exc_info=True)
        return []


def analyze_financial_health(data: dict) -> dict:
    """
    Analyze the financial health of a company using various metrics.

    Args:
        data: Dictionary containing financial statements and metrics

    Returns:
        Dict containing:
            - piotroski_score: Piotroski F-Score (0-9)
            - altman_z_score: Altman Z-Score
            - financial_ratios: Dictionary of key financial ratios
            - health_flags: List of health indicators
    """
    try:
        logger.info("Starting financial health analysis")

        # Initialize result dictionary
        result = {
            "piotroski_score": 0,
            "altman_z_score": 0.0,
            "financial_ratios": {},
            "health_flags": [],
        }

        # Get financial statements
        income_stmt = data.get("income_stmt")
        balance_sheet = data.get("balance_sheet")
        cash_flow = data.get("cash_flow")

        # Calculate Piotroski F-Score
        try:
            result["piotroski_score"] = calculate_piotroski_score(
                income_stmt, balance_sheet, cash_flow
            )
        except Exception as e:
            logger.error(f"Error calculating Piotroski score: {str(e)}", exc_info=True)

        # Calculate Altman Z-Score
        try:
            result["altman_z_score"] = calculate_altman_z_score(
                balance_sheet, income_stmt
            )
        except Exception as e:
            logger.error(f"Error calculating Altman Z-score: {str(e)}", exc_info=True)

        # Calculate financial ratios
        try:
            result["financial_ratios"] = calculate_financial_ratios(
                income_stmt, balance_sheet, cash_flow
            )
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}", exc_info=True)
            result["financial_ratios"] = {}

        # Generate health flags
        try:
            result["health_flags"] = generate_health_flags(result)
        except Exception as e:
            logger.error(f"Error generating health flags: {str(e)}", exc_info=True)
            result["health_flags"] = []

        logger.info("Successfully completed financial health analysis")
        return result

    except Exception as e:
        logger.error(f"Error in financial health analysis: {str(e)}", exc_info=True)
        return {
            "piotroski_score": 0,
            "altman_z_score": 0.0,
            "financial_ratios": {},
            "health_flags": ["Error in financial health analysis"],
        }
