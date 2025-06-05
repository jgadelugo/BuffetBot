"""
Comprehensive repository unit tests.

Enhanced testing suite that provides thorough coverage of:
- All repository CRUD operations with edge cases
- Advanced query scenarios and filtering
- Bulk operations and batch processing
- Data validation and business rules
- Repository-specific functionality for each domain
- Integration between repositories
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database.exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    RepositoryError,
    ValidationError,
)
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
from database.repositories.analysis_repo import AnalysisRepository
from database.repositories.market_data_repo import MarketDataRepository
from database.repositories.portfolio_repo import PortfolioRepository, PositionRepository
from database.repositories.user_repo import UserRepository

from .conftest import (
    generate_test_portfolios,
    generate_test_positions,
    verify_database_state,
)


@pytest.mark.asyncio
class TestUserRepository:
    """Comprehensive tests for UserRepository."""

    async def test_create_user_with_all_fields(self, db_session: AsyncSession):
        """Test creating user with all optional fields."""
        repo = UserRepository(db_session)

        user = User(
            username="comprehensive_user",
            email="comprehensive@example.com",
            first_name="John",
            last_name="Doe",
            is_active=True,
            created_at=datetime.utcnow(),
        )

        created_user = await repo.create(user)

        assert created_user.id is not None
        assert created_user.username == "comprehensive_user"
        assert created_user.email == "comprehensive@example.com"
        assert created_user.first_name == "John"
        assert created_user.last_name == "Doe"
        assert created_user.is_active is True
        assert created_user.created_at is not None

    async def test_find_user_by_username(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test finding user by username."""
        repo = UserRepository(db_session)

        found_user = await repo.find_by_username(sample_user.username)

        assert found_user is not None
        assert found_user.id == sample_user.id
        assert found_user.username == sample_user.username

    async def test_find_user_by_email(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test finding user by email."""
        repo = UserRepository(db_session)

        found_user = await repo.find_by_email(sample_user.email)

        assert found_user is not None
        assert found_user.id == sample_user.id
        assert found_user.email == sample_user.email

    async def test_user_existence_checks(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test user existence checking methods."""
        repo = UserRepository(db_session)

        # Test username exists
        exists = await repo.username_exists(sample_user.username)
        assert exists is True

        # Test email exists
        exists = await repo.email_exists(sample_user.email)
        assert exists is True

        # Test non-existent username
        exists = await repo.username_exists("nonexistent_user")
        assert exists is False

        # Test non-existent email
        exists = await repo.email_exists("nonexistent@example.com")
        assert exists is False

    async def test_activate_deactivate_user(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test user activation and deactivation."""
        repo = UserRepository(db_session)

        # Deactivate user
        await repo.deactivate_user(sample_user.id)
        updated_user = await repo.get_by_id(sample_user.id)
        assert updated_user.is_active is False

        # Activate user
        await repo.activate_user(sample_user.id)
        updated_user = await repo.get_by_id(sample_user.id)
        assert updated_user.is_active is True


@pytest.mark.asyncio
class TestPortfolioRepositoryComprehensive:
    """Comprehensive tests for PortfolioRepository."""

    async def test_create_portfolio_with_validation(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test portfolio creation with comprehensive validation."""
        repo = PortfolioRepository(db_session)

        # Test valid portfolio
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Comprehensive Test Portfolio",
            description="A portfolio for comprehensive testing",
            risk_tolerance=RiskTolerance.MODERATE,
            cash_balance=Decimal("50000.00"),
        )

        created = await repo.create(portfolio)

        assert created.id is not None
        assert created.name == "Comprehensive Test Portfolio"
        assert created.risk_tolerance == RiskTolerance.MODERATE
        assert created.cash_balance == Decimal("50000.00")

    async def test_find_portfolios_by_user(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test finding all portfolios for a user."""
        repo = PortfolioRepository(db_session)

        # Create multiple portfolios
        portfolios = generate_test_portfolios(5, sample_user.id)
        await repo.bulk_create(portfolios)

        # Find all user portfolios
        user_portfolios = await repo.list_by_criteria(user_id=sample_user.id)

        assert len(user_portfolios) >= 5
        assert all(p.user_id == sample_user.id for p in user_portfolios)

    async def test_portfolio_filtering_and_sorting(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test advanced portfolio filtering and sorting."""
        repo = PortfolioRepository(db_session)

        # Create portfolios with different characteristics
        portfolios = [
            Portfolio(
                user_id=sample_user.id,
                name="Conservative Portfolio",
                risk_tolerance=RiskTolerance.LOW,
                cash_balance=Decimal("10000.00"),
            ),
            Portfolio(
                user_id=sample_user.id,
                name="Aggressive Portfolio",
                risk_tolerance=RiskTolerance.HIGH,
                cash_balance=Decimal("100000.00"),
            ),
            Portfolio(
                user_id=sample_user.id,
                name="Moderate Portfolio",
                risk_tolerance=RiskTolerance.MODERATE,
                cash_balance=Decimal("50000.00"),
            ),
        ]

        await repo.bulk_create(portfolios)

        # Test filtering by risk tolerance
        conservative = await repo.list_by_criteria(
            user_id=sample_user.id, risk_tolerance=RiskTolerance.LOW
        )
        assert len(conservative) >= 1
        assert all(p.risk_tolerance == RiskTolerance.LOW for p in conservative)


@pytest.mark.asyncio
class TestPositionRepositoryComprehensive:
    """Comprehensive tests for PositionRepository."""

    async def test_create_position_with_validation(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test position creation with comprehensive validation."""
        repo = PositionRepository(db_session)

        position = Position(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            quantity=100,
            average_cost=Decimal("150.25"),
            current_price=Decimal("155.75"),
            position_type="STOCK",
        )

        created = await repo.create(position)

        assert created.id is not None
        assert created.symbol == "AAPL"
        assert created.quantity == 100
        assert created.average_cost == Decimal("150.25")
        assert created.current_price == Decimal("155.75")

    async def test_find_positions_by_portfolio(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test finding all positions for a portfolio."""
        repo = PositionRepository(db_session)

        # Create multiple positions
        positions = generate_test_positions(10, sample_portfolio.id)
        await repo.bulk_create(positions)

        # Find all portfolio positions
        portfolio_positions = await repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )

        assert len(portfolio_positions) >= 10
        assert all(p.portfolio_id == sample_portfolio.id for p in portfolio_positions)

    async def test_position_filtering_by_symbol(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test filtering positions by symbol."""
        repo = PositionRepository(db_session)

        # Create positions with specific symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "AAPL", "TSLA"]
        positions = []

        for i, symbol in enumerate(symbols):
            position = Position(
                portfolio_id=sample_portfolio.id,
                symbol=symbol,
                quantity=100 + i * 10,
                average_cost=Decimal("150.00"),
                current_price=Decimal("155.00"),
                position_type="STOCK",
            )
            positions.append(position)

        await repo.bulk_create(positions)

        # Find AAPL positions
        aapl_positions = await repo.list_by_criteria(symbol="AAPL")
        assert len(aapl_positions) >= 2
        assert all(p.symbol == "AAPL" for p in aapl_positions)


@pytest.mark.asyncio
class TestAnalysisRepository:
    """Comprehensive tests for AnalysisRepository."""

    async def test_create_analysis_result(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test creating comprehensive analysis results."""
        repo = AnalysisRepository(db_session)

        analysis = AnalysisResult(
            portfolio_id=sample_portfolio.id,
            analysis_type=AnalysisType.RISK_ASSESSMENT,
            result_data={
                "overall_risk_score": 7.2,
                "risk_level": "MODERATE",
                "factors": {
                    "market_risk": 6.5,
                    "sector_concentration": 8.0,
                    "volatility": 7.5,
                },
                "recommendations": [
                    "Consider diversifying into bonds",
                    "Reduce concentration in technology sector",
                ],
            },
            confidence_score=0.85,
            metadata={
                "analysis_version": "2.1",
                "data_sources": ["yahoo_finance", "alpha_vantage"],
                "processing_time_ms": 1250,
            },
        )

        created = await repo.create(analysis)

        assert created.id is not None
        assert created.analysis_type == AnalysisType.RISK_ASSESSMENT
        assert created.result_data["overall_risk_score"] == 7.2
        assert created.confidence_score == 0.85
        assert "analysis_version" in created.metadata

    async def test_find_analysis_by_type(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test finding analysis results by type."""
        repo = AnalysisRepository(db_session)

        # Create multiple analysis types
        analysis_types = [
            AnalysisType.RISK_ASSESSMENT,
            AnalysisType.PERFORMANCE_ANALYSIS,
            AnalysisType.OPTIMIZATION,
            AnalysisType.RISK_ASSESSMENT,  # Duplicate type
        ]

        analyses = []
        for analysis_type in analysis_types:
            analysis = AnalysisResult(
                portfolio_id=sample_portfolio.id,
                analysis_type=analysis_type,
                result_data={"test": True},
                confidence_score=0.8,
            )
            analyses.append(analysis)

        await repo.bulk_create(analyses)

        # Find risk assessments
        risk_analyses = await repo.find_by_analysis_type(
            sample_portfolio.id, AnalysisType.RISK_ASSESSMENT
        )
        assert len(risk_analyses) >= 2
        assert all(
            a.analysis_type == AnalysisType.RISK_ASSESSMENT for a in risk_analyses
        )

    async def test_latest_analysis_retrieval(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test retrieving latest analysis results."""
        repo = AnalysisRepository(db_session)

        # Create analyses with different timestamps
        base_time = datetime.utcnow()
        analyses = []

        for i in range(3):
            analysis = AnalysisResult(
                portfolio_id=sample_portfolio.id,
                analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                result_data={"iteration": i},
                confidence_score=0.8,
                created_at=base_time + timedelta(minutes=i),
            )
            analyses.append(analysis)

        await repo.bulk_create(analyses)

        # Get latest analysis
        latest = await repo.get_latest_analysis(
            sample_portfolio.id, AnalysisType.PERFORMANCE_ANALYSIS
        )

        assert latest is not None
        assert latest.result_data["iteration"] == 2  # Latest one

    async def test_analysis_filtering_by_confidence(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test filtering analyses by confidence score."""
        repo = AnalysisRepository(db_session)

        # Create analyses with different confidence scores
        confidence_scores = [0.95, 0.75, 0.85, 0.65, 0.90]
        analyses = []

        for score in confidence_scores:
            analysis = AnalysisResult(
                portfolio_id=sample_portfolio.id,
                analysis_type=AnalysisType.OPTIMIZATION,
                result_data={"confidence": score},
                confidence_score=score,
            )
            analyses.append(analysis)

        await repo.bulk_create(analyses)

        # Find high-confidence analyses
        high_confidence = await repo.list_by_criteria(
            portfolio_id=sample_portfolio.id, min_confidence_score=0.8
        )

        assert len(high_confidence) >= 3  # 0.95, 0.85, 0.90
        assert all(a.confidence_score >= 0.8 for a in high_confidence)


@pytest.mark.asyncio
class TestMarketDataRepository:
    """Comprehensive tests for MarketDataRepository."""

    async def test_create_market_data_cache(self, db_session: AsyncSession):
        """Test creating market data cache entries."""
        repo = MarketDataRepository(db_session)

        market_data = MarketDataCache(
            symbol="AAPL",
            data_type="QUOTE",
            data={
                "price": 155.75,
                "volume": 85000000,
                "bid": 155.70,
                "ask": 155.80,
                "day_high": 157.25,
                "day_low": 154.10,
                "market_cap": 2500000000000,
            },
            expires_at=datetime.utcnow() + timedelta(minutes=15),
            source="yahoo_finance",
        )

        created = await repo.create(market_data)

        assert created.id is not None
        assert created.symbol == "AAPL"
        assert created.data_type == "QUOTE"
        assert created.data["price"] == 155.75
        assert created.source == "yahoo_finance"

    async def test_find_cached_data(self, db_session: AsyncSession):
        """Test finding cached market data."""
        repo = MarketDataRepository(db_session)

        # Create cache entries
        symbols = ["AAPL", "GOOGL", "MSFT"]
        cache_entries = []

        for symbol in symbols:
            entry = MarketDataCache(
                symbol=symbol,
                data_type="QUOTE",
                data={"price": 100.0},
                expires_at=datetime.utcnow() + timedelta(hours=1),
            )
            cache_entries.append(entry)

        await repo.bulk_create(cache_entries)

        # Find by symbol and type
        aapl_quote = await repo.find_cached_data("AAPL", "QUOTE")
        assert aapl_quote is not None
        assert aapl_quote.symbol == "AAPL"
        assert not repo.is_data_expired(aapl_quote)

    async def test_price_history_operations(self, db_session: AsyncSession):
        """Test price history operations."""
        repo = MarketDataRepository(db_session)

        # Create price history
        base_date = datetime.utcnow().date()
        price_history = []

        for i in range(30):  # 30 days of data
            price = PriceHistory(
                symbol="AAPL",
                date=base_date - timedelta(days=i),
                open_price=Decimal("150.00") + Decimal(str(i * 0.5)),
                high_price=Decimal("155.00") + Decimal(str(i * 0.5)),
                low_price=Decimal("148.00") + Decimal(str(i * 0.5)),
                close_price=Decimal("152.00") + Decimal(str(i * 0.5)),
                volume=1000000 + i * 10000,
                adjusted_close=Decimal("152.00") + Decimal(str(i * 0.5)),
            )
            price_history.append(price)

        await repo.bulk_create(price_history)

        # Test date range queries
        start_date = base_date - timedelta(days=7)
        end_date = base_date

        recent_prices = await repo.get_price_history("AAPL", start_date, end_date)

        assert len(recent_prices) >= 7
        assert all(p.symbol == "AAPL" for p in recent_prices)

    async def test_options_data_operations(self, db_session: AsyncSession):
        """Test options data operations."""
        repo = MarketDataRepository(db_session)

        # Create options chain
        expiry_date = datetime.utcnow() + timedelta(days=30)
        strikes = [140, 145, 150, 155, 160]
        option_types = ["CALL", "PUT"]

        options = []
        for strike in strikes:
            for option_type in option_types:
                option = OptionsData(
                    symbol="AAPL",
                    expiry_date=expiry_date,
                    strike_price=Decimal(str(strike)),
                    option_type=option_type,
                    last_price=Decimal("5.50"),
                    bid=Decimal("5.40"),
                    ask=Decimal("5.60"),
                    volume=500,
                    open_interest=1000,
                    implied_volatility=0.25,
                )
                options.append(option)

        await repo.bulk_create(options)

        # Test options chain retrieval
        calls = await repo.get_options_chain("AAPL", expiry_date, "CALL")
        puts = await repo.get_options_chain("AAPL", expiry_date, "PUT")

        assert len(calls) >= 5
        assert len(puts) >= 5
        assert all(opt.option_type == "CALL" for opt in calls)
        assert all(opt.option_type == "PUT" for opt in puts)


@pytest.mark.asyncio
class TestRepositoryIntegration:
    """Test integration between different repositories."""

    async def test_portfolio_position_integration(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test integration between portfolio and position repositories."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Create positions for portfolio
        positions = generate_test_positions(5, sample_portfolio.id)
        await position_repo.bulk_create(positions)

        # Verify relationships
        portfolio = await portfolio_repo.get_by_id(sample_portfolio.id)
        portfolio_positions = await position_repo.list_by_criteria(
            portfolio_id=sample_portfolio.id
        )

        assert portfolio is not None
        assert len(portfolio_positions) >= 5
        assert all(p.portfolio_id == sample_portfolio.id for p in portfolio_positions)

    async def test_cross_repository_data_consistency(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test data consistency across repositories."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Create portfolio with positions
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Consistency Test Portfolio",
            cash_balance=Decimal("10000.00"),
        )
        created_portfolio = await portfolio_repo.create(portfolio)

        # Create positions
        positions = [
            Position(
                portfolio_id=created_portfolio.id,
                symbol="STOCK1",
                quantity=100,
                average_cost=Decimal("50.00"),
                current_price=Decimal("55.00"),
                position_type="STOCK",
            ),
            Position(
                portfolio_id=created_portfolio.id,
                symbol="STOCK2",
                quantity=200,
                average_cost=Decimal("25.00"),
                current_price=Decimal("30.00"),
                position_type="STOCK",
            ),
        ]
        await position_repo.bulk_create(positions)

        # Verify relationships exist
        portfolio_positions = await position_repo.list_by_criteria(
            portfolio_id=created_portfolio.id
        )
        assert len(portfolio_positions) == 2


@pytest.mark.asyncio
class TestRepositoryEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_empty_result_handling(self, db_session: AsyncSession):
        """Test handling of empty query results."""
        repo = PortfolioRepository(db_session)

        # Search for non-existent user's portfolios
        fake_user_id = str(uuid4())
        portfolios = await repo.list_by_criteria(user_id=fake_user_id)

        assert portfolios == []
        assert isinstance(portfolios, list)

    async def test_large_dataset_handling(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test handling of large datasets."""
        repo = PortfolioRepository(db_session)

        # Create large number of portfolios
        large_batch = generate_test_portfolios(100, sample_user.id)
        created = await repo.bulk_create(large_batch)

        assert len(created) == 100

        # Test pagination
        page_1 = await repo.list_by_criteria(user_id=sample_user.id, offset=0, limit=50)
        page_2 = await repo.list_by_criteria(
            user_id=sample_user.id, offset=50, limit=50
        )

        assert len(page_1) == 50
        assert len(page_2) >= 50

        # Verify no overlap
        page_1_ids = {p.id for p in page_1}
        page_2_ids = {p.id for p in page_2}
        assert page_1_ids.isdisjoint(page_2_ids)


@pytest.mark.performance
@pytest.mark.asyncio
class TestRepositoryPerformanceEdgeCases:
    """Test repository performance under various conditions."""

    async def test_bulk_operation_performance(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test bulk operation performance."""
        repo = PortfolioRepository(db_session)

        # Test bulk create performance
        start_time = asyncio.get_event_loop().time()
        large_batch = generate_test_portfolios(500, sample_user.id)
        created = await repo.bulk_create(large_batch)
        end_time = asyncio.get_event_loop().time()

        assert len(created) == 500
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds

    async def test_complex_query_performance(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test complex query performance."""
        portfolio_repo = PortfolioRepository(db_session)
        position_repo = PositionRepository(db_session)

        # Setup dataset
        portfolios = generate_test_portfolios(50, sample_user.id)
        created_portfolios = await portfolio_repo.bulk_create(portfolios)

        all_positions = []
        for portfolio in created_portfolios:
            positions = generate_test_positions(5, portfolio.id)
            all_positions.extend(positions)

        await position_repo.bulk_create(all_positions)

        # Test complex aggregation query performance
        start_time = asyncio.get_event_loop().time()

        # Complex query: portfolios with position counts
        result = await db_session.execute(
            text(
                """
            SELECT
                p.id,
                p.name,
                p.cash_balance,
                COUNT(pos.id) as position_count
            FROM portfolios p
            LEFT JOIN positions pos ON p.id = pos.portfolio_id
            WHERE p.user_id = :user_id
            GROUP BY p.id, p.name, p.cash_balance
            ORDER BY position_count DESC
        """
            ),
            {"user_id": sample_user.id},
        )

        results = result.fetchall()
        end_time = asyncio.get_event_loop().time()

        assert len(results) > 0
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
