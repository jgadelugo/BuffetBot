"""Investment Disclaimer Components

This module provides standardized investment advice disclaimers and notices
following industry best practices and regulatory requirements.
"""

from typing import Optional

import streamlit as st


def render_investment_disclaimer(
    placement: str = "general", custom_text: str | None = None
) -> None:
    """Render appropriate investment disclaimer based on context.

    Args:
        placement: Context where disclaimer is shown (general, options, analysis, etc.)
        custom_text: Optional custom disclaimer text
    """
    if custom_text:
        st.warning(custom_text)
        return

    disclaimer_text = get_disclaimer_text(placement)

    if placement == "header":
        st.info(disclaimer_text)
    elif placement == "options":
        st.error(disclaimer_text)
    elif placement == "analysis":
        st.warning(disclaimer_text)
    else:
        st.info(disclaimer_text)


def get_disclaimer_text(placement: str = "general") -> str:
    """Get appropriate disclaimer text based on context.

    Args:
        placement: Context where disclaimer will be shown

    Returns:
        Appropriate disclaimer text
    """
    disclaimers = {
        "general": """
        ⚠️ **Important Disclaimer**: This application is for educational and informational purposes only.
        Nothing contained in this analysis should be construed as investment advice. All data and calculations
        are provided 'as-is' without warranty. Past performance does not guarantee future results.
        Always consult with a qualified financial advisor before making investment decisions.
        """,
        "header": """
        📚 **Educational Tool**: This dashboard provides financial analysis for educational purposes only.
        Not investment advice. Consult a qualified financial advisor before making investment decisions.
        """,
        "options": """
        🚨 **High-Risk Investment Warning**: Options trading involves substantial risk and is not suitable for all investors.
        Options can expire worthless, and you may lose your entire investment. This analysis is for educational purposes only
        and should not be considered as investment advice. Options trading requires sophisticated knowledge of financial markets.
        Please consult with a qualified financial advisor and understand all risks before engaging in options trading.
        """,
        "analysis": """
        ⚠️ **Analysis Disclaimer**: This financial analysis is based on historical data and mathematical models.
        Market conditions, company fundamentals, and economic factors can change rapidly. This analysis should not
        be considered as investment advice. All investment decisions should be made after thorough research and
        consultation with qualified financial professionals.
        """,
        "price_valuation": """
        💰 **Valuation Disclaimer**: Stock valuations are estimates based on available financial data and assumptions.
        Actual market prices may differ significantly from calculated intrinsic values due to market sentiment,
        timing, and unforeseen factors. This valuation is for educational purposes and should not be used as
        the sole basis for investment decisions.
        """,
        "technical_analysis": """
        📈 **Technical Analysis Disclaimer**: Technical indicators and chart patterns are based on historical price data
        and may not predict future price movements. Technical analysis should be used in conjunction with fundamental
        analysis and is not a guarantee of future performance. Market conditions can change rapidly.
        """,
        "footer": """
        ---
        **Legal Disclaimer**: BuffetBot is an educational financial analysis tool. We are not registered investment advisors,
        brokers, or financial planners. This software does not provide personalized investment advice and should not be
        relied upon for investment decisions. Users are responsible for their own investment research and decisions.
        Investing involves risk, including potential loss of principal. Please consult qualified professionals before
        making financial decisions.
        """,
        "data_quality": """
        📊 **Data Reliability Notice**: Analysis quality depends on the availability and accuracy of financial data.
        Some metrics may be unavailable or estimated due to data limitations. Users should verify important
        financial information from official company sources and SEC filings before making investment decisions.
        """,
    }

    return disclaimers.get(placement, disclaimers["general"])


def render_risk_warning_box(
    title: str = "⚠️ Risk Warning", content: str = None, risk_level: str = "medium"
) -> None:
    """Render a prominent risk warning box.

    Args:
        title: Warning box title
        content: Warning content
        risk_level: Risk level (low, medium, high)
    """
    if not content:
        content = """
        **Investment Risk Notice:**
        - All investments carry risk of loss
        - Past performance does not guarantee future results
        - Market values can fluctuate significantly
        - This analysis is for educational purposes only
        """

    if risk_level == "high":
        st.error(f"**{title}**\n\n{content}")
    elif risk_level == "medium":
        st.warning(f"**{title}**\n\n{content}")
    else:
        st.info(f"**{title}**\n\n{content}")


def render_educational_notice() -> None:
    """Render educational purpose notice."""
    st.info(
        """
    🎓 **Educational Purpose**: This tool is designed for learning about financial analysis and investment concepts.
    It should be used alongside proper financial education and professional guidance.
    """
    )


def render_compliance_footer() -> None:
    """Render comprehensive compliance footer."""
    st.markdown(
        """
    ---
    ### Important Legal Information

    **Not Investment Advice**: This application provides financial analysis tools for educational purposes only.
    Nothing on this platform constitutes investment advice, financial advice, trading advice, or any other sort of advice.

    **Risk Disclosure**: All investments involve risk, including the potential loss of principal. Past performance
    does not guarantee future results. The value of investments can go down as well as up.

    **No Professional Relationship**: Use of this software does not create an advisor-client relationship.
    We are not registered investment advisors and do not provide personalized investment advice.

    **Data Accuracy**: While we strive for accuracy, we cannot guarantee the completeness or accuracy of
    financial data and calculations. Users should verify information independently.

    **Your Responsibility**: You are solely responsible for your investment decisions. Please consult with
    qualified financial professionals before making investment decisions.

    ---
    *BuffetBot v2024 - Educational Financial Analysis Tool*
    """
    )
