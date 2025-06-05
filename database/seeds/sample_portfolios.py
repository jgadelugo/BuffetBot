"""
Sample portfolio data for BuffetBot development.
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.portfolio import Portfolio, Position
from ..models.user import User


async def create_sample_portfolios(session: AsyncSession) -> None:
    """Create sample portfolios with positions for development and testing."""

    # Create sample users first
    sample_users = [
        User(
            id=uuid.uuid4(),
            username="demo_user",
            email="demo@buffetbot.com",
            first_name="Demo",
            last_name="User",
            is_active=True,
            created_at=datetime.utcnow(),
        ),
        User(
            id=uuid.uuid4(),
            username="test_investor",
            email="investor@buffetbot.com",
            first_name="Test",
            last_name="Investor",
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=30),
        ),
    ]

    for user in sample_users:
        session.add(user)

    await session.flush()  # Get user IDs

    # Create sample portfolios with different risk profiles
    portfolios_data = [
        {
            "user_id": sample_users[0].id,
            "name": "Conservative Growth",
            "description": "Low-risk, dividend-focused portfolio for steady income",
            "risk_tolerance": "conservative",
            "positions": [
                {
                    "ticker": "JNJ",
                    "shares": 150,
                    "average_cost": 165.50,
                    "allocation": 25.0,
                },
                {
                    "ticker": "PG",
                    "shares": 100,
                    "average_cost": 140.00,
                    "allocation": 20.0,
                },
                {
                    "ticker": "KO",
                    "shares": 200,
                    "average_cost": 58.75,
                    "allocation": 18.0,
                },
                {
                    "ticker": "VTI",
                    "shares": 75,
                    "average_cost": 220.00,
                    "allocation": 22.0,
                },
                {
                    "ticker": "BND",
                    "shares": 100,
                    "average_cost": 85.50,
                    "allocation": 15.0,
                },
            ],
        },
        {
            "user_id": sample_users[0].id,
            "name": "Balanced Tech Portfolio",
            "description": "Balanced technology and growth stocks with moderate risk",
            "risk_tolerance": "moderate",
            "positions": [
                {
                    "ticker": "AAPL",
                    "shares": 50,
                    "average_cost": 180.25,
                    "allocation": 20.0,
                },
                {
                    "ticker": "MSFT",
                    "shares": 40,
                    "average_cost": 310.00,
                    "allocation": 22.0,
                },
                {
                    "ticker": "GOOGL",
                    "shares": 15,
                    "average_cost": 140.50,
                    "allocation": 18.0,
                },
                {
                    "ticker": "TSLA",
                    "shares": 25,
                    "average_cost": 220.00,
                    "allocation": 15.0,
                },
                {
                    "ticker": "NVDA",
                    "shares": 20,
                    "average_cost": 450.00,
                    "allocation": 15.0,
                },
                {
                    "ticker": "VTI",
                    "shares": 25,
                    "average_cost": 225.00,
                    "allocation": 10.0,
                },
            ],
        },
        {
            "user_id": sample_users[1].id,
            "name": "Aggressive Growth",
            "description": "High-growth, high-risk investments for maximum potential returns",
            "risk_tolerance": "aggressive",
            "positions": [
                {
                    "ticker": "TSLA",
                    "shares": 100,
                    "average_cost": 180.00,
                    "allocation": 30.0,
                },
                {
                    "ticker": "NVDA",
                    "shares": 50,
                    "average_cost": 300.00,
                    "allocation": 25.0,
                },
                {
                    "ticker": "AMD",
                    "shares": 150,
                    "average_cost": 95.00,
                    "allocation": 20.0,
                },
                {
                    "ticker": "PLTR",
                    "shares": 200,
                    "average_cost": 15.50,
                    "allocation": 10.0,
                },
                {
                    "ticker": "ARKK",
                    "shares": 75,
                    "average_cost": 65.00,
                    "allocation": 15.0,
                },
            ],
        },
        {
            "user_id": sample_users[1].id,
            "name": "Value Investing Portfolio",
            "description": "Undervalued stocks with strong fundamentals",
            "risk_tolerance": "moderate",
            "positions": [
                {
                    "ticker": "BRK.B",
                    "shares": 25,
                    "average_cost": 330.00,
                    "allocation": 22.0,
                },
                {
                    "ticker": "WMT",
                    "shares": 50,
                    "average_cost": 155.00,
                    "allocation": 18.0,
                },
                {
                    "ticker": "JPM",
                    "shares": 40,
                    "average_cost": 145.00,
                    "allocation": 20.0,
                },
                {
                    "ticker": "BAC",
                    "shares": 100,
                    "average_cost": 32.50,
                    "allocation": 15.0,
                },
                {
                    "ticker": "XOM",
                    "shares": 75,
                    "average_cost": 95.00,
                    "allocation": 15.0,
                },
                {
                    "ticker": "HD",
                    "shares": 20,
                    "average_cost": 320.00,
                    "allocation": 10.0,
                },
            ],
        },
    ]

    # Create portfolios and positions
    for portfolio_data in portfolios_data:
        portfolio = Portfolio(
            id=uuid.uuid4(),
            user_id=portfolio_data["user_id"],
            name=portfolio_data["name"],
            description=portfolio_data["description"],
            risk_tolerance=portfolio_data["risk_tolerance"],
            created_at=datetime.utcnow() - timedelta(days=30),
        )
        session.add(portfolio)
        await session.flush()  # Get portfolio ID

        # Add positions to portfolio
        for pos_data in portfolio_data["positions"]:
            position = Position(
                id=uuid.uuid4(),
                portfolio_id=portfolio.id,
                ticker=pos_data["ticker"],
                shares=pos_data["shares"],
                average_cost=Decimal(str(pos_data["average_cost"])),
                allocation_percentage=Decimal(str(pos_data["allocation"])),
                created_at=datetime.utcnow() - timedelta(days=25),
            )
            session.add(position)
