from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from buffetbot.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Recommendation thresholds and scoring rules
RECOMMENDATION_THRESHOLDS = {"strong_buy": 8, "buy": 6, "hold": 4}

SCORING_RULES = {
    # Value metrics
    "margin_of_safety": {
        "threshold": 0.20,  # 20%
        "points": 2,
        "description": "High margin of safety",
        "category": "value",
    },
    "pe_ratio": {
        "threshold": 15.0,
        "points": 1,
        "description": "Reasonable P/E ratio",
        "category": "value",
    },
    "pb_ratio": {
        "threshold": 1.5,
        "points": 1,
        "description": "Attractive price-to-book ratio",
        "category": "value",
    },
    # Health metrics
    "piotroski_score": {
        "threshold": 7,
        "points": 2,
        "description": "Strong financial health",
        "category": "health",
    },
    "altman_z_score": {
        "threshold": 3.0,
        "points": 1,
        "description": "Low bankruptcy risk",
        "category": "health",
    },
    "current_ratio": {
        "threshold": 1.5,
        "points": 1,
        "description": "Strong liquidity position",
        "category": "health",
    },
    "debt_to_equity": {
        "threshold": 0.5,
        "points": 1,
        "description": "Conservative debt levels",
        "category": "health",
    },
    # Growth metrics
    "growth_score": {
        "threshold": 4,
        "points": 2,
        "description": "Strong growth profile",
        "category": "growth",
    },
    "revenue_cagr": {
        "threshold": 0.15,  # 15%
        "points": 1,
        "description": "Strong revenue growth",
        "category": "growth",
    },
    "eps_cagr": {
        "threshold": 0.10,  # 10%
        "points": 1,
        "description": "Consistent earnings growth",
        "category": "growth",
    },
    # Profitability metrics
    "roe": {
        "threshold": 0.15,  # 15%
        "points": 1,
        "description": "Strong return on equity",
        "category": "profitability",
    },
    "operating_margin": {
        "threshold": 0.15,  # 15%
        "points": 1,
        "description": "Healthy operating margins",
        "category": "profitability",
    },
    "fcf_margin": {
        "threshold": 0.10,  # 10%
        "points": 1,
        "description": "Strong free cash flow generation",
        "category": "profitability",
    },
}


def _calculate_score(
    value_metrics: dict[str, float],
    health_metrics: dict[str, int | float | str | list[str]],
    growth_metrics: dict[str, float | dict[str, float]],
) -> dict[str, Any]:
    """
    Calculate recommendation score based on analysis metrics.

    Args:
        value_metrics: Value analysis results
        health_metrics: Health analysis results
        growth_metrics: Growth analysis results

    Returns:
        Dict containing:
            - total_score: Overall recommendation score
            - score_breakdown: Points from each category
            - reasoning: List of reasons for points
    """
    try:
        score = 0
        score_breakdown = {
            category: 0 for category in ["value", "health", "growth", "profitability"]
        }
        reasoning = []

        # Value metrics scoring
        for metric, rule in {
            k: v for k, v in SCORING_RULES.items() if v["category"] == "value"
        }.items():
            if metric in value_metrics:
                value = value_metrics[metric]
                if value >= rule["threshold"]:
                    score += rule["points"]
                    score_breakdown["value"] += rule["points"]
                    reasoning.append(rule["description"])

        # Health metrics scoring
        for metric, rule in {
            k: v for k, v in SCORING_RULES.items() if v["category"] == "health"
        }.items():
            if metric in health_metrics:
                value = health_metrics[metric]
                if value >= rule["threshold"]:
                    score += rule["points"]
                    score_breakdown["health"] += rule["points"]
                    reasoning.append(rule["description"])

        # Growth metrics scoring
        for metric, rule in {
            k: v for k, v in SCORING_RULES.items() if v["category"] == "growth"
        }.items():
            if metric in growth_metrics:
                value = growth_metrics[metric]
                if value >= rule["threshold"]:
                    score += rule["points"]
                    score_breakdown["growth"] += rule["points"]
                    reasoning.append(rule["description"])

        # Profitability metrics scoring
        for metric, rule in {
            k: v for k, v in SCORING_RULES.items() if v["category"] == "profitability"
        }.items():
            if (
                metric in value_metrics
            ):  # Profitability metrics are stored in value_metrics
                value = value_metrics[metric]
                if value >= rule["threshold"]:
                    score += rule["points"]
                    score_breakdown["profitability"] += rule["points"]
                    reasoning.append(rule["description"])

        return {
            "total_score": score,
            "score_breakdown": score_breakdown,
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.error(f"Error calculating recommendation score: {str(e)}")
        return {
            "total_score": 0,
            "score_breakdown": {},
            "reasoning": ["Error in score calculation"],
        }


def _get_recommendation_label(score: int) -> str:
    """
    Convert numerical score to recommendation label.

    Args:
        score: Total recommendation score

    Returns:
        str: Recommendation label
    """
    if score >= RECOMMENDATION_THRESHOLDS["strong_buy"]:
        return "Strong Buy"
    elif score >= RECOMMENDATION_THRESHOLDS["buy"]:
        return "Buy"
    elif score >= RECOMMENDATION_THRESHOLDS["hold"]:
        return "Hold"
    else:
        return "Sell"


def _extract_key_metrics(
    value_metrics: dict[str, float],
    health_metrics: dict[str, int | float | str | list[str]],
    growth_metrics: dict[str, float | dict[str, float]],
) -> dict[str, Any]:
    """
    Extract key metrics used in recommendation.

    Args:
        value_metrics: Value analysis results
        health_metrics: Health analysis results
        growth_metrics: Growth analysis results

    Returns:
        Dict: Key metrics used in recommendation
    """
    return {
        "intrinsic_value": value_metrics.get("intrinsic_value", 0.0),
        "current_price": value_metrics.get("current_price", 0.0),
        "margin_of_safety": value_metrics.get("margin_of_safety", 0.0),
        "piotroski_score": health_metrics.get("piotroski_score", 0),
        "altman_z_score": health_metrics.get("altman_z_score", 0.0),
        "growth_classification": growth_metrics.get("growth_classification", "Unknown"),
        "growth_score": growth_metrics.get("growth_score", 0.0),
    }


def generate_recommendation(
    value_metrics: dict[str, float],
    health_metrics: dict[str, int | float | str | list[str]],
    growth_metrics: dict[str, float | dict[str, float]],
) -> dict[str, Any]:
    """
    Generate stock recommendation based on analysis metrics.

    Args:
        value_metrics: Value analysis results
        health_metrics: Health analysis results
        growth_metrics: Growth analysis results

    Returns:
        Dict containing:
            - recommendation: Buy/Hold/Sell recommendation
            - score: Numerical score (0-10)
            - reasoning: List of reasons for recommendation
            - metrics_used: Key metrics used in analysis
            - analysis_date: Timestamp
            - status: Success/error message
    """
    try:
        logger.info("Generating stock recommendation")

        # Calculate recommendation score
        score_results = _calculate_score(value_metrics, health_metrics, growth_metrics)

        # Get recommendation label
        recommendation = _get_recommendation_label(score_results["total_score"])

        # Extract key metrics
        metrics_used = _extract_key_metrics(
            value_metrics, health_metrics, growth_metrics
        )

        # Log recommendation
        logger.info(
            f"Generated recommendation: {recommendation} (Score: {score_results['total_score']})"
        )
        logger.info(f"Reasoning: {', '.join(score_results['reasoning'])}")

        return {
            "recommendation": recommendation,
            "score": score_results["total_score"],
            "reasoning": score_results["reasoning"],
            "metrics_used": metrics_used,
            "analysis_date": datetime.now().isoformat(),
            "status": "Success",
        }

    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        return {
            "recommendation": "Error",
            "score": 0,
            "reasoning": ["Error in recommendation generation"],
            "metrics_used": {},
            "analysis_date": datetime.now().isoformat(),
            "status": f"Error: {str(e)}",
        }


def explain_recommendation(recommendation: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a detailed explanation of the stock recommendation.

    Args:
        recommendation: Dictionary containing recommendation details

    Returns:
        Dict containing:
            - summary: Brief summary of the recommendation
            - detailed_analysis: Category-by-category analysis
            - key_strengths: List of key positive factors
            - key_concerns: List of potential concerns
            - investment_thesis: Suggested investment approach
    """
    try:
        logger.info("Generating detailed recommendation explanation")

        # Extract metrics and scores
        metrics = recommendation["metrics_used"]
        score_breakdown = recommendation.get("score_breakdown", {})

        # Generate category analysis
        detailed_analysis = {
            "value": {
                "score": score_breakdown.get("value", 0),
                "max_score": sum(
                    rule["points"]
                    for rule in SCORING_RULES.values()
                    if rule["category"] == "value"
                ),
                "metrics": {
                    "margin_of_safety": f"{metrics.get('margin_of_safety', 0):.1%}",
                    "pe_ratio": f"{metrics.get('pe_ratio', 0):.1f}",
                    "pb_ratio": f"{metrics.get('pb_ratio', 0):.1f}",
                },
            },
            "health": {
                "score": score_breakdown.get("health", 0),
                "max_score": sum(
                    rule["points"]
                    for rule in SCORING_RULES.values()
                    if rule["category"] == "health"
                ),
                "metrics": {
                    "piotroski_score": metrics.get("piotroski_score", 0),
                    "altman_z_score": f"{metrics.get('altman_z_score', 0):.1f}",
                    "current_ratio": f"{metrics.get('current_ratio', 0):.1f}",
                },
            },
            "growth": {
                "score": score_breakdown.get("growth", 0),
                "max_score": sum(
                    rule["points"]
                    for rule in SCORING_RULES.values()
                    if rule["category"] == "growth"
                ),
                "metrics": {
                    "growth_score": f"{metrics.get('growth_score', 0):.1f}",
                    "revenue_cagr": f"{metrics.get('revenue_cagr', 0):.1%}",
                    "eps_cagr": f"{metrics.get('eps_cagr', 0):.1%}",
                },
            },
            "profitability": {
                "score": score_breakdown.get("profitability", 0),
                "max_score": sum(
                    rule["points"]
                    for rule in SCORING_RULES.values()
                    if rule["category"] == "profitability"
                ),
                "metrics": {
                    "roe": f"{metrics.get('roe', 0):.1%}",
                    "operating_margin": f"{metrics.get('operating_margin', 0):.1%}",
                    "fcf_margin": f"{metrics.get('fcf_margin', 0):.1%}",
                },
            },
        }

        # Identify key strengths and concerns
        key_strengths = []
        key_concerns = []

        for category, analysis in detailed_analysis.items():
            score_ratio = analysis["score"] / analysis["max_score"]
            if score_ratio >= 0.7:
                key_strengths.append(f"Strong {category} metrics")
            elif score_ratio <= 0.3:
                key_concerns.append(f"Weak {category} metrics")

        # Generate investment thesis
        if recommendation["recommendation"] in ["Strong Buy", "Buy"]:
            investment_thesis = (
                "This stock appears to be a good value investment opportunity. "
                "Consider building a position gradually, starting with a small initial "
                "investment and adding more on price weakness."
            )
        elif recommendation["recommendation"] == "Hold":
            investment_thesis = (
                "The stock shows mixed signals. Consider waiting for better entry points "
                "or more clarity on key metrics before making a significant investment."
            )
        else:
            investment_thesis = (
                "The current metrics suggest avoiding this stock. If already holding, "
                "consider reducing or exiting the position unless there are strong "
                "contrarian reasons to maintain exposure."
            )

        explanation = {
            "summary": f"{recommendation['recommendation']} recommendation based on a score of {recommendation['score']}/10",
            "detailed_analysis": detailed_analysis,
            "key_strengths": key_strengths,
            "key_concerns": key_concerns,
            "investment_thesis": investment_thesis,
            "analysis_date": recommendation["analysis_date"],
        }

        logger.info("Successfully generated detailed explanation")
        return explanation

    except Exception as e:
        logger.error(f"Error generating recommendation explanation: {str(e)}")
        return {
            "summary": "Error generating explanation",
            "detailed_analysis": {},
            "key_strengths": [],
            "key_concerns": [],
            "investment_thesis": "Unable to generate investment thesis due to error",
            "analysis_date": datetime.now().isoformat(),
        }
