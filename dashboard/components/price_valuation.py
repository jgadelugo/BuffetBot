"""Price valuation components with enhanced visual indicators."""

import logging
from typing import Dict, Optional, Tuple

import streamlit as st

# Import disclaimer components
from dashboard.components.disclaimers import render_investment_disclaimer

logger = logging.getLogger(__name__)


class PriceValuationCard:
    """Creates a visually appealing card showing price vs intrinsic value comparison."""

    def __init__(self, current_price: float, intrinsic_value: float, ticker: str):
        """Initialize the valuation card.

        Args:
            current_price: Current market price
            intrinsic_value: Calculated intrinsic value
            ticker: Stock ticker symbol
        """
        self.current_price = current_price
        self.intrinsic_value = intrinsic_value
        self.ticker = ticker
        self.margin_of_safety = self._calculate_margin_of_safety()
        self.valuation_status = self._determine_valuation_status()

    def _calculate_margin_of_safety(self) -> float | None:
        """Calculate margin of safety percentage."""
        try:
            if self.intrinsic_value > 0:
                return (
                    self.intrinsic_value - self.current_price
                ) / self.intrinsic_value
            return None
        except Exception as e:
            logger.error(f"Error calculating margin of safety: {str(e)}")
            return None

    def _determine_valuation_status(self) -> dict[str, any]:
        """Determine valuation status with color coding and messages."""
        if self.margin_of_safety is None:
            return {
                "status": "unknown",
                "color": "#gray",
                "bg_color": "#f0f0f0",
                "icon": "❓",
                "message": "Unable to determine valuation",
                "recommendation": "Insufficient data for valuation analysis",
            }

        # Calculate percentage difference
        price_diff_pct = (
            (self.current_price - self.intrinsic_value) / self.intrinsic_value
        ) * 100

        if self.margin_of_safety >= 0.25:  # 25% or more undervalued
            return {
                "status": "deeply_undervalued",
                "color": "#006400",  # Dark green
                "bg_color": "#90EE90",  # Light green
                "icon": "🟢🟢",
                "message": f"DEEPLY UNDERVALUED ({abs(price_diff_pct):.1f}% below intrinsic value)",
                "recommendation": "Strong Buy - Significant margin of safety",
            }
        elif self.margin_of_safety >= 0.10:  # 10-25% undervalued
            return {
                "status": "undervalued",
                "color": "#228B22",  # Forest green
                "bg_color": "#98FB98",  # Pale green
                "icon": "🟢",
                "message": f"UNDERVALUED ({abs(price_diff_pct):.1f}% below intrinsic value)",
                "recommendation": "Buy - Good margin of safety",
            }
        elif self.margin_of_safety >= 0:  # 0-10% undervalued
            return {
                "status": "fairly_valued",
                "color": "#FF8C00",  # Dark orange
                "bg_color": "#FFE4B5",  # Moccasin
                "icon": "🟡",
                "message": f"FAIRLY VALUED ({abs(price_diff_pct):.1f}% below intrinsic value)",
                "recommendation": "Hold - Limited margin of safety",
            }
        elif self.margin_of_safety >= -0.15:  # 0-15% overvalued
            return {
                "status": "slightly_overvalued",
                "color": "#FF6347",  # Tomato
                "bg_color": "#FFA07A",  # Light salmon
                "icon": "🟠",
                "message": f"SLIGHTLY OVERVALUED ({abs(price_diff_pct):.1f}% above intrinsic value)",
                "recommendation": "Caution - Consider waiting for better entry",
            }
        else:  # More than 15% overvalued
            return {
                "status": "overvalued",
                "color": "#DC143C",  # Crimson
                "bg_color": "#FFB6C1",  # Light pink
                "icon": "🔴",
                "message": f"OVERVALUED ({abs(price_diff_pct):.1f}% above intrinsic value)",
                "recommendation": "Avoid - No margin of safety",
            }

    def render(self):
        """Render the valuation card with visual indicators."""
        status = self.valuation_status

        # Create a clean, professional layout using Streamlit components
        st.markdown("---")

        # Main metrics in columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric(
                label="📊 Current Price", value=f"${self.current_price:.2f}", delta=None
            )

        with col2:
            st.metric(
                label="💎 Intrinsic Value",
                value=f"${self.intrinsic_value:.2f}",
                delta=None,
            )

        with col3:
            margin_text = (
                f"{self.margin_of_safety:.1%}"
                if self.margin_of_safety is not None
                else "N/A"
            )
            # Color the delta based on margin of safety
            delta_color = (
                "normal"
                if self.margin_of_safety and self.margin_of_safety > 0
                else "inverse"
            )
            st.metric(label="🛡️ Margin of Safety", value=margin_text, delta=None)

        # Valuation status box
        st.markdown("---")

        # Create colored info/warning/error boxes based on status
        if status["status"] in ["deeply_undervalued", "undervalued"]:
            st.success(f"{status['icon']} **{status['message']}**")
            st.info(f"💡 **Recommendation:** {status['recommendation']}")
        elif status["status"] == "fairly_valued":
            st.warning(f"{status['icon']} **{status['message']}**")
            st.info(f"💡 **Recommendation:** {status['recommendation']}")
        elif status["status"] in ["slightly_overvalued", "overvalued"]:
            st.error(f"{status['icon']} **{status['message']}**")
            st.warning(f"⚠️ **Recommendation:** {status['recommendation']}")
        else:
            st.info(f"{status['icon']} **{status['message']}**")
            st.info(f"💡 **Recommendation:** {status['recommendation']}")

        # Add valuation disclaimer after recommendations
        render_investment_disclaimer("price_valuation")

        st.markdown("---")


def create_valuation_summary(
    current_price: float,
    intrinsic_value: float,
    pe_ratio: float | None = None,
    peg_ratio: float | None = None,
    price_to_book: float | None = None,
) -> dict[str, any]:
    """Create a comprehensive valuation summary with multiple metrics.

    Args:
        current_price: Current market price
        intrinsic_value: Calculated intrinsic value
        pe_ratio: Price to Earnings ratio
        peg_ratio: PEG ratio
        price_to_book: Price to Book ratio

    Returns:
        Dictionary containing valuation summary
    """
    try:
        # Calculate primary valuation metric
        margin_of_safety = (
            (intrinsic_value - current_price) / intrinsic_value
            if intrinsic_value > 0
            else None
        )

        # Analyze additional valuation metrics
        valuation_signals = []

        if pe_ratio is not None:
            if pe_ratio < 15:
                valuation_signals.append(("P/E Ratio", "Attractive", "green"))
            elif pe_ratio > 25:
                valuation_signals.append(("P/E Ratio", "Expensive", "red"))
            else:
                valuation_signals.append(("P/E Ratio", "Fair", "orange"))

        if peg_ratio is not None:
            if peg_ratio < 1:
                valuation_signals.append(("PEG Ratio", "Undervalued", "green"))
            elif peg_ratio > 2:
                valuation_signals.append(("PEG Ratio", "Overvalued", "red"))
            else:
                valuation_signals.append(("PEG Ratio", "Fair", "orange"))

        if price_to_book is not None:
            if price_to_book < 1:
                valuation_signals.append(("P/B Ratio", "Below Book Value", "green"))
            elif price_to_book > 3:
                valuation_signals.append(("P/B Ratio", "Premium Valuation", "orange"))
            else:
                valuation_signals.append(("P/B Ratio", "Normal", "blue"))

        # Overall valuation score (0-100)
        score_components = []

        if margin_of_safety is not None:
            if margin_of_safety >= 0.25:
                score_components.append(100)
            elif margin_of_safety >= 0.10:
                score_components.append(80)
            elif margin_of_safety >= 0:
                score_components.append(60)
            elif margin_of_safety >= -0.15:
                score_components.append(40)
            else:
                score_components.append(20)

        if pe_ratio is not None:
            if pe_ratio < 15:
                score_components.append(80)
            elif pe_ratio < 25:
                score_components.append(60)
            else:
                score_components.append(40)

        overall_score = (
            sum(score_components) / len(score_components) if score_components else 50
        )

        return {
            "margin_of_safety": margin_of_safety,
            "valuation_signals": valuation_signals,
            "overall_score": overall_score,
            "metrics": {
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "price_to_book": price_to_book,
            },
        }

    except Exception as e:
        logger.error(f"Error creating valuation summary: {str(e)}")
        return {
            "margin_of_safety": None,
            "valuation_signals": [],
            "overall_score": 50,
            "metrics": {},
        }
