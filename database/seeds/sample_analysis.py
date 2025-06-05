"""
Sample analysis results for BuffetBot development.
"""

import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.analysis import AnalysisResult
from ..models.portfolio import Portfolio


async def create_sample_analysis_results(session: AsyncSession) -> None:
    """Create sample analysis results for development and testing."""

    # Get existing portfolios to create analysis results for
    result = await session.execute(select(Portfolio))
    portfolios = result.scalars().all()

    if not portfolios:
        return  # No portfolios to analyze

    # Sample analysis types and strategies
    analysis_configs = [
        {
            "analysis_type": "value_investing",
            "strategy": "dcf_analysis",
            "tickers": ["AAPL", "MSFT", "JNJ"],
            "score_range": (65, 85),
        },
        {
            "analysis_type": "growth_analysis",
            "strategy": "growth_metrics",
            "tickers": ["TSLA", "NVDA", "GOOGL"],
            "score_range": (70, 90),
        },
        {
            "analysis_type": "risk_assessment",
            "strategy": "volatility_analysis",
            "tickers": ["VTI", "BND"],
            "score_range": (80, 95),
        },
        {
            "analysis_type": "momentum_analysis",
            "strategy": "technical_indicators",
            "tickers": ["AAPL", "TSLA", "NVDA"],
            "score_range": (60, 80),
        },
        {
            "analysis_type": "dividend_analysis",
            "strategy": "dividend_growth",
            "tickers": ["JNJ", "PG", "KO"],
            "score_range": (75, 90),
        },
    ]

    # Create analysis results for each portfolio and ticker combination
    for portfolio in portfolios:
        for config in analysis_configs:
            for ticker in config["tickers"]:
                # Generate realistic score within range
                min_score, max_score = config["score_range"]
                score = (
                    min_score + (max_score - min_score) * 0.7
                )  # Bias towards upper range

                # Create comprehensive metadata based on analysis type
                metadata = _generate_analysis_metadata(
                    config["analysis_type"], config["strategy"], ticker, score
                )

                analysis_result = AnalysisResult(
                    id=uuid.uuid4(),
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    analysis_type=config["analysis_type"],
                    strategy=config["strategy"],
                    score=Decimal(str(round(score, 2))),
                    analysis_metadata=metadata,
                    calculated_at=datetime.utcnow() - timedelta(hours=2),
                    expires_at=datetime.utcnow()
                    + timedelta(hours=22),  # 24 hour validity
                )
                session.add(analysis_result)

    # Add some expired analysis results for testing cleanup
    if portfolios:
        expired_analysis = AnalysisResult(
            id=uuid.uuid4(),
            portfolio_id=portfolios[0].id,
            ticker="AMD",
            analysis_type="value_investing",
            strategy="dcf_analysis",
            score=Decimal("72.50"),
            analysis_metadata={
                "status": "expired",
                "note": "Old analysis for cleanup testing",
            },
            calculated_at=datetime.utcnow() - timedelta(days=2),
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired 1 day ago
        )
        session.add(expired_analysis)


def _generate_analysis_metadata(
    analysis_type: str, strategy: str, ticker: str, score: float
) -> dict:
    """Generate realistic metadata for analysis results."""

    base_metadata = {
        "analysis_version": "2.1.0",
        "calculation_time_ms": 1250,
        "data_sources": ["yahoo_finance", "market_data_cache"],
        "confidence_level": "high" if score > 75 else "medium" if score > 60 else "low",
    }

    if analysis_type == "value_investing":
        return {
            **base_metadata,
            "dcf_valuation": {
                "intrinsic_value": round(score * 2.5, 2),
                "current_price": round(score * 2.3, 2),
                "upside_potential": f"{round((score * 2.5 / (score * 2.3) - 1) * 100, 1)}%",
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.025,
                "years_projected": 5,
            },
            "fundamental_ratios": {
                "pe_ratio": round(25 + (score - 70) * 0.5, 1),
                "pb_ratio": round(3.2 + (score - 70) * 0.1, 1),
                "debt_to_equity": round(0.8 - (score - 70) * 0.01, 2),
                "current_ratio": round(1.2 + (score - 70) * 0.02, 2),
            },
            "recommendation": "BUY" if score > 80 else "HOLD" if score > 65 else "SELL",
        }

    elif analysis_type == "growth_analysis":
        return {
            **base_metadata,
            "growth_metrics": {
                "revenue_growth_5y": f"{round(5 + (score - 70) * 0.8, 1)}%",
                "eps_growth_5y": f"{round(8 + (score - 70) * 1.0, 1)}%",
                "roe": f"{round(15 + (score - 70) * 0.5, 1)}%",
                "roic": f"{round(12 + (score - 70) * 0.4, 1)}%",
            },
            "future_projections": {
                "expected_growth_rate": f"{round(10 + (score - 70) * 0.6, 1)}%",
                "target_price_1y": round(score * 3.2, 2),
                "growth_sustainability": "high" if score > 80 else "medium",
            },
            "risk_factors": [
                "Market competition",
                "Regulatory changes" if score < 75 else None,
                "Economic downturn" if score < 70 else None,
            ],
        }

    elif analysis_type == "risk_assessment":
        return {
            **base_metadata,
            "risk_metrics": {
                "beta": round(1.2 - (score - 70) * 0.02, 2),
                "volatility_1y": f"{round(25 - (score - 70) * 0.3, 1)}%",
                "sharpe_ratio": round(0.8 + (score - 70) * 0.02, 2),
                "max_drawdown": f"{round(15 - (score - 70) * 0.2, 1)}%",
            },
            "risk_grade": "A" if score > 90 else "B" if score > 80 else "C",
            "portfolio_impact": {
                "correlation_with_market": round(0.75 + (score - 70) * 0.005, 3),
                "diversification_benefit": "high" if score > 85 else "medium",
            },
        }

    elif analysis_type == "momentum_analysis":
        return {
            **base_metadata,
            "technical_indicators": {
                "rsi_14": round(45 + (score - 70) * 0.8, 1),
                "macd_signal": "bullish" if score > 70 else "bearish",
                "moving_avg_50": round(score * 2.8, 2),
                "moving_avg_200": round(score * 2.6, 2),
                "bollinger_position": "upper" if score > 75 else "middle",
            },
            "momentum_score": round(score * 1.2, 1),
            "trend_strength": "strong" if score > 75 else "moderate",
            "support_resistance": {
                "support_level": round(score * 2.4, 2),
                "resistance_level": round(score * 3.0, 2),
            },
        }

    elif analysis_type == "dividend_analysis":
        return {
            **base_metadata,
            "dividend_metrics": {
                "current_yield": f"{round(2.5 + (score - 70) * 0.05, 2)}%",
                "payout_ratio": f"{round(60 - (score - 70) * 0.5, 1)}%",
                "dividend_growth_5y": f"{round(8 + (score - 70) * 0.3, 1)}%",
                "years_of_growth": int(15 + (score - 70) * 0.5),
            },
            "dividend_sustainability": "excellent" if score > 85 else "good",
            "projected_yield_1y": f"{round(2.8 + (score - 70) * 0.06, 2)}%",
            "dividend_grade": "A" if score > 85 else "B+" if score > 75 else "B",
        }

    else:
        return base_metadata
