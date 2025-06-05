"""
Comprehensive integration tests for database layer.

Tests complex workflows that span multiple repositories and validate:
- End-to-end user workflows
- Cross-repository data consistency
- Transaction management across repositories
- Real-world usage scenarios
- Data integrity in complex operations
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database.exceptions import RepositoryError, ValidationError
from database.models import (
    AnalysisResult,
    AnalysisType,
    MarketDataCache,
    OptionsData,
    Portfolio,
    Position,
    PriceHistory,
    RiskTolerance,
    User,
)
from database.repositories import RepositoryRegistry
from database.repositories.analysis_repo import AnalysisRepository
from database.repositories.market_data_repo import MarketDataRepository
from database.repositories.portfolio_repo import PortfolioRepository, PositionRepository

from .conftest import (
    generate_test_portfolios,
    generate_test_positions,
    verify_database_state,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestUserPortfolioWorkflow:
    """Test complete user portfolio management workflows."""

    async def test_create_user_with_multiple_portfolios(
        self, db_session: AsyncSession, repository_registry: RepositoryRegistry
    ):
        """Test creating a user with multiple portfolios and positions."""

        # Create user
        user = User(
            username="integration_user",
            email="integration@example.com",
            first_name="Integration",
            last_name="Test",
        )

        # Add user directly to session for this test
        db_session.add(user)
        await db_session.flush()
        await db_session.refresh(user)

        # Create multiple portfolios
        portfolio_repo = PortfolioRepository(db_session)
        portfolios_data = [
            {
                "name": "Growth Portfolio",
                "risk_tolerance": RiskTolerance.HIGH,
                "cash_balance": Decimal("100000.00"),
            },
            {
                "name": "Income Portfolio",
                "risk_tolerance": RiskTolerance.LOW,
                "cash_balance": Decimal("50000.00"),
            },
            {
                "name": "Balanced Portfolio",
                "risk_tolerance": RiskTolerance.MODERATE,
                "cash_balance": Decimal("75000.00"),
            },
        ]

        created_portfolios = []
        for portfolio_data in portfolios_data:
            portfolio = Portfolio(user_id=user.id, **portfolio_data)
            created_portfolio = await portfolio_repo.create(portfolio)
            created_portfolios.append(created_portfolio)

        # Add positions to each portfolio
        position_repo = PositionRepository(db_session)
        all_positions = []

        for portfolio in created_portfolios:
            positions = [
                Position(
                    portfolio_id=portfolio.id,
                    symbol="AAPL",
                    quantity=100,
                    average_cost=Decimal("150.00"),
                    current_price=Decimal("155.00"),
                    position_type="STOCK",
                ),
                Position(
                    portfolio_id=portfolio.id,
                    symbol="GOOGL",
                    quantity=50,
                    average_cost=Decimal("2500.00"),
                    current_price=Decimal("2600.00"),
                    position_type="STOCK",
                ),
            ]
            created_positions = await position_repo.bulk_create(positions)
            all_positions.extend(created_positions)

        # Verify the complete workflow
        assert len(created_portfolios) == 3
        assert len(all_positions) == 6  # 2 positions per portfolio

        # Verify user has all portfolios
        user_portfolios = await portfolio_repo.list_by_criteria(user_id=user.id)
        assert len(user_portfolios) == 3

        # Verify each portfolio has positions
        for portfolio in created_portfolios:
            portfolio_positions = await position_repo.list_by_criteria(
                portfolio_id=portfolio.id
            )
            assert len(portfolio_positions) == 2

    async def test_portfolio_rebalancing_workflow(
        self, db_session: AsyncSession, sample_user: User, sample_portfolio: Portfolio
    ):
        """Test a complete portfolio rebalancing workflow."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Create initial positions
        initial_positions = [
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="AAPL",
                quantity=200,
                average_cost=Decimal("150.00"),
                current_price=Decimal("160.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="GOOGL",
                quantity=50,
                average_cost=Decimal("2500.00"),
                current_price=Decimal("2400.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="BONDS",
                quantity=1000,
                average_cost=Decimal("100.00"),
                current_price=Decimal("98.00"),
                position_type="BOND",
            ),
        ]

        await position_repo.bulk_create(initial_positions)

        # Simulate rebalancing: reduce AAPL, increase BONDS
        # Get current positions
        current_positions = await position_repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )

        for position in current_positions:
            if position.symbol == "AAPL":
                # Reduce AAPL position
                position.quantity = 150
                await position_repo.update(position)
            elif position.symbol == "BONDS":
                # Increase BONDS position
                position.quantity = 1200
                await position_repo.update(position)

        # Update portfolio cash balance (from selling AAPL)
        sample_portfolio.cash_balance += Decimal("8000.00")  # 50 shares * $160
        sample_portfolio.cash_balance -= Decimal("19600.00")  # 200 shares * $98
        updated_portfolio = await portfolio_repo.update(sample_portfolio)

        # Verify rebalancing results
        final_positions = await position_repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )

        aapl_position = next(p for p in final_positions if p.symbol == "AAPL")
        bonds_position = next(p for p in final_positions if p.symbol == "BONDS")

        assert aapl_position.quantity == 150
        assert bonds_position.quantity == 1200
        assert (
            updated_portfolio.cash_balance < sample_portfolio.cash_balance
        )  # Cash was used


@pytest.mark.integration
@pytest.mark.asyncio
class TestAnalysisWorkflow:
    """Test analysis workflow integration."""

    async def test_complete_analysis_workflow(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test complete analysis workflow from data to insights."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)
        analysis_repo = AnalysisRepository(db_session)
        market_data_repo = MarketDataRepository(db_session)

        # Step 1: Create portfolio positions
        positions = [
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="AAPL",
                quantity=100,
                average_cost=Decimal("150.00"),
                current_price=Decimal("155.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="GOOGL",
                quantity=25,
                average_cost=Decimal("2500.00"),
                current_price=Decimal("2600.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=sample_portfolio.id,
                symbol="MSFT",
                quantity=75,
                average_cost=Decimal("300.00"),
                current_price=Decimal("320.00"),
                position_type="STOCK",
            ),
        ]
        await position_repo.bulk_create(positions)

        # Step 2: Add market data for analysis
        market_data_entries = []
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            market_data = MarketDataCache(
                symbol=symbol,
                data_type="QUOTE",
                data={
                    "price": 155.0
                    if symbol == "AAPL"
                    else (2600.0 if symbol == "GOOGL" else 320.0),
                    "volume": 1000000,
                    "volatility": 0.25,
                    "beta": 1.2
                    if symbol == "AAPL"
                    else (0.9 if symbol == "GOOGL" else 1.1),
                },
                expires_at=datetime.utcnow() + timedelta(hours=1),
            )
            market_data_entries.append(market_data)

        await market_data_repo.bulk_create(market_data_entries)

        # Step 3: Perform risk analysis
        portfolio_positions = await position_repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )

        # Calculate portfolio metrics
        total_value = sum(p.quantity * p.current_price for p in portfolio_positions)
        weights = {
            p.symbol: (p.quantity * p.current_price) / total_value
            for p in portfolio_positions
        }

        # Get market data for risk calculation
        portfolio_beta = 0.0
        for symbol, weight in weights.items():
            market_data = await market_data_repo.find_cached_data(symbol, "QUOTE")
            if market_data:
                portfolio_beta += weight * market_data.data.get("beta", 1.0)

        # Create risk analysis
        risk_analysis = AnalysisResult(
            portfolio_id=sample_portfolio.id,
            analysis_type=AnalysisType.RISK_ASSESSMENT,
            result_data={
                "portfolio_beta": portfolio_beta,
                "total_value": float(total_value),
                "weights": weights,
                "risk_level": "MODERATE" if 0.8 <= portfolio_beta <= 1.2 else "HIGH",
                "diversification_score": len(portfolio_positions)
                / 10.0,  # Simple metric
                "recommendations": [
                    "Consider adding bonds for diversification",
                    "Monitor sector concentration",
                ],
            },
            confidence_score=0.85,
            metadata={
                "calculation_method": "beta_weighted",
                "data_freshness": "1_hour",
            },
        )

        created_analysis = await analysis_repo.create(risk_analysis)

        # Step 4: Verify analysis results
        assert created_analysis.id is not None
        assert created_analysis.result_data["total_value"] > 0
        assert "portfolio_beta" in created_analysis.result_data
        assert len(created_analysis.result_data["recommendations"]) > 0

        # Step 5: Create performance analysis
        performance_analysis = AnalysisResult(
            portfolio_id=sample_portfolio.id,
            analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
            result_data={
                "total_return": 0.067,  # 6.7% return
                "unrealized_gains": sum(
                    (p.current_price - p.average_cost) * p.quantity
                    for p in portfolio_positions
                ),
                "best_performer": "GOOGL",
                "worst_performer": "AAPL",
                "time_period": "1_month",
            },
            confidence_score=0.9,
        )

        await analysis_repo.create(performance_analysis)

        # Verify multiple analyses exist
        all_analyses = await analysis_repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )
        assert len(all_analyses) >= 2

        risk_analyses = [
            a for a in all_analyses if a.analysis_type == AnalysisType.RISK_ASSESSMENT
        ]
        performance_analyses = [
            a
            for a in all_analyses
            if a.analysis_type == AnalysisType.PERFORMANCE_ANALYSIS
        ]

        assert len(risk_analyses) >= 1
        assert len(performance_analyses) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestMarketDataIntegration:
    """Test market data integration workflows."""

    async def test_market_data_price_history_workflow(self, db_session: AsyncSession):
        """Test complete market data and price history workflow."""
        market_data_repo = MarketDataRepository(db_session)

        # Step 1: Create current market data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        current_data = []

        for symbol in symbols:
            market_data = MarketDataCache(
                symbol=symbol,
                data_type="QUOTE",
                data={
                    "price": 155.0,
                    "volume": 1000000,
                    "bid": 154.95,
                    "ask": 155.05,
                    "day_high": 157.0,
                    "day_low": 153.0,
                },
                expires_at=datetime.utcnow() + timedelta(minutes=15),
                source="test_provider",
            )
            current_data.append(market_data)

        await market_data_repo.bulk_create(current_data)

        # Step 2: Create historical price data
        base_date = datetime.utcnow().date()
        price_history = []

        for symbol in symbols:
            for i in range(30):  # 30 days of history
                price = PriceHistory(
                    symbol=symbol,
                    date=base_date - timedelta(days=i),
                    open_price=Decimal("150.00") + Decimal(str(i * 0.1)),
                    high_price=Decimal("155.00") + Decimal(str(i * 0.1)),
                    low_price=Decimal("148.00") + Decimal(str(i * 0.1)),
                    close_price=Decimal("152.00") + Decimal(str(i * 0.1)),
                    volume=1000000 + i * 1000,
                    adjusted_close=Decimal("152.00") + Decimal(str(i * 0.1)),
                )
                price_history.append(price)

        await market_data_repo.bulk_create(price_history)

        # Step 3: Create options data
        expiry_date = datetime.utcnow() + timedelta(days=30)
        options_data = []

        for symbol in symbols:
            for strike in [140, 150, 160, 170]:
                for option_type in ["CALL", "PUT"]:
                    option = OptionsData(
                        symbol=symbol,
                        expiry_date=expiry_date,
                        strike_price=Decimal(str(strike)),
                        option_type=option_type,
                        last_price=Decimal("5.50"),
                        bid=Decimal("5.40"),
                        ask=Decimal("5.60"),
                        volume=100,
                        open_interest=500,
                        implied_volatility=0.25,
                    )
                    options_data.append(option)

        await market_data_repo.bulk_create(options_data)

        # Step 4: Verify data retrieval workflows

        # Test current data retrieval
        for symbol in symbols:
            current_quote = await market_data_repo.find_cached_data(symbol, "QUOTE")
            assert current_quote is not None
            assert current_quote.symbol == symbol
            assert not market_data_repo.is_data_expired(current_quote)

        # Test historical data retrieval
        start_date = base_date - timedelta(days=7)
        end_date = base_date

        for symbol in symbols:
            history = await market_data_repo.get_price_history(
                symbol, start_date, end_date
            )
            assert len(history) >= 7
            assert all(h.symbol == symbol for h in history)

        # Test options data retrieval
        for symbol in symbols:
            calls = await market_data_repo.get_options_chain(
                symbol, expiry_date, "CALL"
            )
            puts = await market_data_repo.get_options_chain(symbol, expiry_date, "PUT")

            assert len(calls) >= 4  # 4 strike prices
            assert len(puts) >= 4
            assert all(opt.option_type == "CALL" for opt in calls)
            assert all(opt.option_type == "PUT" for opt in puts)


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataConsistencyWorkflows:
    """Test data consistency across repository operations."""

    async def test_transaction_consistency_across_repositories(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test transaction consistency when operations span multiple repositories."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)
        analysis_repo = AnalysisRepository(db_session)

        # Create portfolio
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Transaction Test Portfolio",
            cash_balance=Decimal("50000.00"),
        )
        created_portfolio = await portfolio_repo.create(portfolio)

        # Create positions
        positions = [
            Position(
                portfolio_id=created_portfolio.id,
                symbol="AAPL",
                quantity=100,
                average_cost=Decimal("150.00"),
                current_price=Decimal("155.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=created_portfolio.id,
                symbol="GOOGL",
                quantity=50,
                average_cost=Decimal("2500.00"),
                current_price=Decimal("2550.00"),
                position_type="STOCK",
            ),
        ]
        created_positions = await position_repo.bulk_create(positions)

        # Create analysis
        analysis = AnalysisResult(
            portfolio_id=created_portfolio.id,
            analysis_type=AnalysisType.RISK_ASSESSMENT,
            result_data={"test": "consistency_check"},
            confidence_score=0.8,
        )
        created_analysis = await analysis_repo.create(analysis)

        # Verify all data exists and is consistent
        # Check portfolio exists
        retrieved_portfolio = await portfolio_repo.get_by_id(created_portfolio.id)
        assert retrieved_portfolio is not None

        # Check positions exist and reference correct portfolio
        portfolio_positions = await position_repo.list_by_criteria(
            portfolio_id=created_portfolio.id
        )
        assert len(portfolio_positions) == 2
        assert all(p.portfolio_id == created_portfolio.id for p in portfolio_positions)

        # Check analysis exists and references correct portfolio
        portfolio_analyses = await analysis_repo.list_by_criteria(
            portfolio_id=created_portfolio.id
        )
        assert len(portfolio_analyses) >= 1
        assert all(a.portfolio_id == created_portfolio.id for a in portfolio_analyses)

        # Verify cross-repository data integrity
        # All positions should sum to meaningful portfolio value
        total_position_value = sum(
            p.quantity * p.current_price for p in portfolio_positions
        )
        assert total_position_value > 0

        # Analysis should be related to actual portfolio data
        assert created_analysis.portfolio_id == created_portfolio.id

    async def test_cascading_operations_consistency(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test consistency when operations cascade across repositories."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Create portfolio with positions
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Cascading Test Portfolio",
            cash_balance=Decimal("100000.00"),
        )
        created_portfolio = await portfolio_repo.create(portfolio)

        # Create multiple positions
        positions = generate_test_positions(10, created_portfolio.id)
        created_positions = await position_repo.bulk_create(positions)

        # Update portfolio based on positions
        total_invested = sum(p.quantity * p.average_cost for p in created_positions)

        # Update portfolio cash balance (simulate purchases)
        created_portfolio.cash_balance -= total_invested
        updated_portfolio = await portfolio_repo.update(created_portfolio)

        # Verify consistency
        assert updated_portfolio.cash_balance < Decimal("100000.00")

        # Verify all positions still exist and are valid
        final_positions = await position_repo.list_by_criteria(
            portfolio_id=created_portfolio.id
        )
        assert len(final_positions) == 10
        assert all(p.portfolio_id == created_portfolio.id for p in final_positions)

        # Test bulk position updates
        for position in final_positions:
            position.current_price += Decimal("5.00")  # Market moved up

        # Update all positions
        for position in final_positions:
            await position_repo.update(position)

        # Verify updates were applied consistently
        updated_positions = await position_repo.list_by_criteria(
            portfolio_id=created_portfolio.id
        )

        for orig, updated in zip(created_positions, updated_positions):
            assert updated.current_price == orig.current_price + Decimal("5.00")


@pytest.mark.integration
@pytest.mark.asyncio
class TestComplexQueryIntegration:
    """Test complex queries that span multiple repositories and tables."""

    async def test_portfolio_summary_complex_query(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test complex query that aggregates data across multiple tables."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)
        analysis_repo = AnalysisRepository(db_session)

        # Create multiple portfolios with positions and analyses
        portfolios_data = []
        for i in range(3):
            portfolio = Portfolio(
                user_id=sample_user.id,
                name=f"Complex Query Portfolio {i+1}",
                cash_balance=Decimal(f"{(i+1) * 25000}.00"),
            )
            created_portfolio = await portfolio_repo.create(portfolio)
            portfolios_data.append(created_portfolio)

            # Add positions to each portfolio
            positions = generate_test_positions(5, created_portfolio.id)
            await position_repo.bulk_create(positions)

            # Add analysis to each portfolio
            analysis = AnalysisResult(
                portfolio_id=created_portfolio.id,
                analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                result_data={"return": (i + 1) * 0.05},  # Different returns
                confidence_score=0.8 + (i * 0.05),
            )
            await analysis_repo.create(analysis)

        # Execute complex aggregation query
        complex_query = text(
            """
            SELECT
                u.username,
                p.name as portfolio_name,
                p.cash_balance,
                COUNT(DISTINCT pos.id) as position_count,
                COUNT(DISTINCT a.id) as analysis_count,
                COALESCE(SUM(pos.quantity * pos.current_price), 0) as total_position_value,
                p.cash_balance + COALESCE(SUM(pos.quantity * pos.current_price), 0) as total_portfolio_value,
                AVG(a.confidence_score) as avg_analysis_confidence
            FROM users u
            JOIN portfolios p ON u.id = p.user_id
            LEFT JOIN positions pos ON p.id = pos.portfolio_id
            LEFT JOIN analysis_results a ON p.id = a.portfolio_id
            WHERE u.id = :user_id
            GROUP BY u.id, u.username, p.id, p.name, p.cash_balance
            ORDER BY total_portfolio_value DESC
        """
        )

        result = await db_session.execute(complex_query, {"user_id": sample_user.id})
        rows = result.fetchall()

        # Verify query results
        assert len(rows) >= 3  # At least 3 portfolios

        for row in rows:
            assert row.username == sample_user.username
            assert row.position_count >= 5  # Each portfolio has 5 positions
            assert row.analysis_count >= 1  # Each portfolio has 1 analysis
            assert row.total_position_value > 0
            assert row.total_portfolio_value > row.cash_balance
            assert 0.8 <= row.avg_analysis_confidence <= 1.0

    async def test_cross_repository_performance_query(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test performance of complex cross-repository queries."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Create larger dataset
        portfolios = generate_test_portfolios(20, sample_user.id)
        created_portfolios = await portfolio_repo.bulk_create(portfolios)

        # Add positions to portfolios
        all_positions = []
        for portfolio in created_portfolios:
            positions = generate_test_positions(10, portfolio.id)
            all_positions.extend(positions)

        await position_repo.bulk_create(all_positions)

        # Execute performance-sensitive query
        start_time = asyncio.get_event_loop().time()

        performance_query = text(
            """
            SELECT
                p.id,
                p.name,
                COUNT(pos.id) as position_count,
                SUM(pos.quantity * pos.current_price) as total_value,
                AVG(pos.current_price / pos.average_cost) as avg_return_ratio
            FROM portfolios p
            LEFT JOIN positions pos ON p.id = pos.portfolio_id
            WHERE p.user_id = :user_id
            GROUP BY p.id, p.name
            HAVING COUNT(pos.id) > 5
            ORDER BY total_value DESC
            LIMIT 10
        """
        )

        result = await db_session.execute(
            performance_query, {"user_id": sample_user.id}
        )
        rows = result.fetchall()

        end_time = asyncio.get_event_loop().time()
        query_time = end_time - start_time

        # Verify performance and results
        assert len(rows) >= 10  # Should return top 10 portfolios
        assert query_time < 1.0  # Query should complete within 1 second

        # Verify data quality
        for row in rows:
            assert row.position_count > 5  # HAVING clause
            assert row.total_value > 0
            assert row.avg_return_ratio > 0
