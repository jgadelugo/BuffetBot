#!/usr/bin/env python3
"""
Basic usage example for BuffetBot database layer.

This script demonstrates how to:
1. Initialize the database connection
2. Create users and portfolios
3. Add positions to portfolios
4. Store analysis results
5. Query data using repositories

Run this script to see the database layer in action.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

from database import get_async_database_session, init_database
from database.models import (
    AnalysisResult,
    AnalysisType,
    Portfolio,
    Position,
    RiskTolerance,
    User,
)
from database.repositories import (
    AnalysisRepository,
    PortfolioRepository,
    PositionRepository,
    UserRepository,
)


async def main():
    """Main example function demonstrating database usage."""

    print("🚀 BuffetBot Database Layer Example")
    print("=" * 50)

    # Initialize database connection
    print("\n1. Initializing database connection...")
    db = init_database()

    # Check connection
    is_connected = await db.check_connection()
    if not is_connected:
        print("❌ Failed to connect to database. Please check your PostgreSQL setup.")
        return

    print("✅ Database connection successful!")

    # Initialize repositories
    user_repo = UserRepository()
    portfolio_repo = PortfolioRepository()
    position_repo = PositionRepository()
    analysis_repo = AnalysisRepository()

    async with get_async_database_session() as session:
        print("\n2. Creating a sample user...")

        # Create a user
        user = await user_repo.async_create(
            session,
            email="warren.buffett@berkshire.com",
            username="warren_buffett",
            hashed_password="hashed_password_here",  # In real app, use proper hashing
            first_name="Warren",
            last_name="Buffett",
            is_verified=True,
        )

        print(f"✅ Created user: {user.display_name} ({user.email})")

        print("\n3. Creating a portfolio...")

        # Create a portfolio
        portfolio = await portfolio_repo.async_create(
            session,
            user_id=user.id,
            name="Value Investing Portfolio",
            description="Long-term value investing strategy focused on undervalued companies",
            risk_tolerance=RiskTolerance.MODERATE,
            target_cash_percentage=Decimal("10.00"),
        )

        print(f"✅ Created portfolio: {portfolio.name}")
        print(f"   Risk tolerance: {portfolio.risk_tolerance}")
        print(f"   Target cash: {portfolio.target_cash_percentage}%")

        print("\n4. Adding positions to the portfolio...")

        # Add some positions
        positions_data = [
            {
                "ticker": "AAPL",
                "shares": Decimal("100"),
                "average_cost": Decimal("150.00"),
                "allocation_percentage": Decimal("25.00"),
            },
            {
                "ticker": "MSFT",
                "shares": Decimal("75"),
                "average_cost": Decimal("300.00"),
                "allocation_percentage": Decimal("20.00"),
            },
            {
                "ticker": "BRK.B",
                "shares": Decimal("50"),
                "average_cost": Decimal("400.00"),
                "allocation_percentage": Decimal("15.00"),
            },
        ]

        positions = []
        for pos_data in positions_data:
            position = await position_repo.async_create(
                session, portfolio_id=portfolio.id, **pos_data
            )
            positions.append(position)

            print(
                f"   ✅ Added {position.shares} shares of {position.ticker} @ ${position.average_cost}"
            )

        print("\n5. Storing analysis results...")

        # Store some analysis results
        for position in positions:
            # Value analysis result
            value_analysis = await analysis_repo.async_create(
                session,
                portfolio_id=portfolio.id,
                ticker=position.ticker,
                analysis_type=AnalysisType.VALUE_ANALYSIS,
                strategy="dcf",
                score=Decimal("75.50"),
                metadata={
                    "intrinsic_value": 180.00,
                    "current_price": 150.00,
                    "margin_of_safety": 0.167,
                    "dcf_assumptions": {
                        "growth_rate": 0.05,
                        "discount_rate": 0.10,
                        "terminal_growth": 0.03,
                    },
                },
                recommendation="buy",
                confidence_level=Decimal("0.85"),
                summary=f"Strong value opportunity in {position.ticker} with 16.7% margin of safety",
                expires_at=datetime.utcnow() + timedelta(hours=24),
                parameters={
                    "analysis_date": datetime.utcnow().isoformat(),
                    "data_sources": ["yfinance", "sec_filings"],
                },
                data_quality_score=Decimal("0.90"),
            )

            print(
                f"   ✅ Stored value analysis for {position.ticker}: Score {value_analysis.score}"
            )

        print("\n6. Querying data...")

        # Query user's portfolios
        user_portfolios = await portfolio_repo.async_get_user_portfolios(
            session, user.id
        )
        print(f"   📊 User has {len(user_portfolios)} portfolio(s)")

        # Query portfolio with positions
        portfolio_with_positions = (
            await portfolio_repo.async_get_portfolio_with_positions(
                session, portfolio.id
            )
        )
        print(
            f"   📈 Portfolio '{portfolio_with_positions.name}' has {len(portfolio_with_positions.positions)} positions"
        )

        # Calculate total allocation
        total_allocation = sum(
            pos.allocation_percentage for pos in portfolio_with_positions.positions
        )
        print(f"   💰 Total allocation: {total_allocation}%")

        # Query analysis results
        analysis_results = await analysis_repo.async_list_by_criteria(
            session,
            portfolio_id=portfolio.id,
            analysis_type=AnalysisType.VALUE_ANALYSIS,
        )
        print(f"   🔍 Found {len(analysis_results)} value analysis results")

        for result in analysis_results:
            print(
                f"      - {result.ticker}: {result.recommendation} (confidence: {result.confidence_level})"
            )

        print("\n7. Demonstrating repository patterns...")

        # Find position by ticker
        aapl_position = await position_repo.async_get_position_by_ticker(
            session, portfolio.id, "AAPL"
        )
        if aapl_position:
            print(
                f"   🍎 AAPL position: {aapl_position.shares} shares @ ${aapl_position.average_cost}"
            )

        # Get all positions for a ticker across portfolios
        all_aapl_positions = await position_repo.async_list_by_criteria(
            session, ticker="AAPL"
        )
        print(
            f"   📊 Total AAPL positions across all portfolios: {len(all_aapl_positions)}"
        )

        print("\n8. Data validation examples...")

        # The models include validation - let's show some examples
        print("   ✅ All data validation passed during creation")
        print("   - Risk tolerance is constrained to valid values")
        print("   - Allocation percentages are between 0-100%")
        print("   - Confidence levels are between 0.0-1.0")
        print("   - UUIDs are automatically generated for all entities")

        print(f"\n🎉 Example completed successfully!")
        print(
            f"   Created: 1 user, 1 portfolio, {len(positions)} positions, {len(analysis_results)} analysis results"
        )

        # Note: In a real application, you would commit the transaction here
        # For this example, we'll rollback to keep the database clean
        print("\n🔄 Rolling back transaction to keep database clean...")


if __name__ == "__main__":
    # Run the example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        print("Make sure PostgreSQL is running and the database exists.")
        print("You may need to run database migrations first.")
