"""
Comprehensive test configuration for database testing infrastructure.

Provides enterprise-grade testing fixtures for:
- Test database management with automatic isolation
- Repository testing with realistic data
- Performance benchmarking infrastructure
- Concurrent access testing support
- Migration testing utilities
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import database components
from database.config import (
    DatabaseConfig,
    DatabaseEnvironment,
    get_test_database_config,
)
from database.connection import Base
from database.initialization import DatabaseInitializer
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
from database.repositories.session_manager import DatabaseSessionManager

# Test constants
TEST_DATABASE_PREFIX = "buffetbot_test"
DEFAULT_TEST_TIMEOUT = 30  # seconds


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database_config():
    """Create test database configuration with unique database name."""
    config = get_test_database_config()
    # Ensure unique database name per test session
    config.database = f"{TEST_DATABASE_PREFIX}_{uuid.uuid4().hex[:8]}"
    return config


@pytest.fixture(scope="session")
async def test_engine(
    test_database_config: DatabaseConfig,
) -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine with proper cleanup."""
    # Create engine for administrative tasks (connecting to default database)
    admin_config = test_database_config.model_copy()
    admin_config.database = "postgres"  # Connect to default database
    admin_engine = create_async_engine(
        admin_config.get_database_url(),
        isolation_level="AUTOCOMMIT",
        echo=test_database_config.echo_sql,
    )

    try:
        # Create test database
        async with admin_engine.connect() as conn:
            await conn.execute(
                text(f"DROP DATABASE IF EXISTS {test_database_config.database}")
            )
            await conn.execute(text(f"CREATE DATABASE {test_database_config.database}"))

        # Create engine for test database
        engine = create_async_engine(
            test_database_config.get_database_url(),
            **test_database_config.engine_kwargs,
            echo=test_database_config.echo_sql,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine

    finally:
        # Cleanup: drop test database
        await engine.dispose()
        async with admin_engine.connect() as conn:
            # Terminate connections to test database
            await conn.execute(
                text(
                    f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{test_database_config.database}'
                AND pid <> pg_backend_pid()
            """
                )
            )
            await conn.execute(
                text(f"DROP DATABASE IF EXISTS {test_database_config.database}")
            )
        await admin_engine.dispose()


@pytest.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with automatic rollback for test isolation."""
    SessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with SessionLocal() as session:
        # Start transaction
        transaction = await session.begin()

        # Create savepoint for nested rollback
        savepoint = await session.begin_nested()

        try:
            yield session
        finally:
            # Rollback to savepoint (isolates test changes)
            await savepoint.rollback()
            # Rollback main transaction
            await transaction.rollback()


@pytest.fixture
async def db_session_manager(test_engine: AsyncEngine) -> DatabaseSessionManager:
    """Create test database session manager."""
    session_manager = DatabaseSessionManager()
    session_manager.engine = test_engine
    session_manager.session_factory = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    return session_manager


@pytest.fixture
async def repository_registry(
    db_session_manager: DatabaseSessionManager,
) -> RepositoryRegistry:
    """Create repository registry for testing."""
    return RepositoryRegistry(db_session_manager)


@pytest.fixture
async def sample_user(db_session: AsyncSession) -> User:
    """Create a sample user for testing."""
    user = User(
        username=f"testuser_{uuid.uuid4().hex[:8]}",
        email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        created_at=datetime.utcnow(),
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def sample_portfolio(db_session: AsyncSession, sample_user: User) -> Portfolio:
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        user_id=sample_user.id,
        name="Test Portfolio",
        description="Portfolio for testing",
        risk_tolerance=RiskTolerance.MODERATE,
        cash_balance=10000.0,
        created_at=datetime.utcnow(),
    )
    db_session.add(portfolio)
    await db_session.flush()
    await db_session.refresh(portfolio)
    return portfolio


@pytest.fixture
async def sample_position(
    db_session: AsyncSession, sample_portfolio: Portfolio
) -> Position:
    """Create a sample position for testing."""
    position = Position(
        portfolio_id=sample_portfolio.id,
        symbol="AAPL",
        quantity=100,
        average_cost=150.0,
        current_price=155.0,
        position_type="STOCK",
        created_at=datetime.utcnow(),
    )
    db_session.add(position)
    await db_session.flush()
    await db_session.refresh(position)
    return position


@pytest.fixture
async def sample_analysis_result(
    db_session: AsyncSession, sample_portfolio: Portfolio
) -> AnalysisResult:
    """Create a sample analysis result for testing."""
    analysis = AnalysisResult(
        portfolio_id=sample_portfolio.id,
        analysis_type=AnalysisType.RISK_ASSESSMENT,
        result_data={
            "risk_score": 0.65,
            "risk_level": "MODERATE",
            "recommendations": ["Diversify holdings", "Consider defensive positions"],
        },
        confidence_score=0.85,
        created_at=datetime.utcnow(),
    )
    db_session.add(analysis)
    await db_session.flush()
    await db_session.refresh(analysis)
    return analysis


@pytest.fixture
async def sample_market_data(db_session: AsyncSession) -> MarketDataCache:
    """Create sample market data for testing."""
    market_data = MarketDataCache(
        symbol="AAPL",
        data_type="QUOTE",
        data={"price": 155.0, "volume": 1000000, "bid": 154.95, "ask": 155.05},
        expires_at=datetime.utcnow() + timedelta(minutes=15),
        created_at=datetime.utcnow(),
    )
    db_session.add(market_data)
    await db_session.flush()
    await db_session.refresh(market_data)
    return market_data


@pytest.fixture
async def sample_price_history(db_session: AsyncSession) -> list[PriceHistory]:
    """Create sample price history for testing."""
    base_date = datetime.utcnow() - timedelta(days=30)
    price_history = []

    for i in range(30):
        price = PriceHistory(
            symbol="AAPL",
            date=base_date + timedelta(days=i),
            open_price=150.0 + i * 0.5,
            high_price=155.0 + i * 0.5,
            low_price=148.0 + i * 0.5,
            close_price=152.0 + i * 0.5,
            volume=1000000 + i * 10000,
            adjusted_close=152.0 + i * 0.5,
        )
        price_history.append(price)
        db_session.add(price)

    await db_session.flush()
    for price in price_history:
        await db_session.refresh(price)

    return price_history


@pytest.fixture
async def sample_options_data(db_session: AsyncSession) -> OptionsData:
    """Create sample options data for testing."""
    options = OptionsData(
        symbol="AAPL",
        expiry_date=datetime.utcnow() + timedelta(days=30),
        strike_price=160.0,
        option_type="CALL",
        last_price=5.50,
        bid=5.40,
        ask=5.60,
        volume=500,
        open_interest=1000,
        implied_volatility=0.25,
        created_at=datetime.utcnow(),
    )
    db_session.add(options)
    await db_session.flush()
    await db_session.refresh(options)
    return options


@pytest.fixture
def performance_metrics():
    """Fixture for tracking performance metrics during tests."""

    class PerformanceTracker:
        def __init__(self):
            self.metrics: dict[str, list[float]] = {}

        def record(self, operation: str, duration: float):
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)

        def get_average(self, operation: str) -> float:
            if operation not in self.metrics:
                return 0.0
            return sum(self.metrics[operation]) / len(self.metrics[operation])

        def get_max(self, operation: str) -> float:
            if operation not in self.metrics:
                return 0.0
            return max(self.metrics[operation])

        def reset(self):
            self.metrics.clear()

    return PerformanceTracker()


@pytest.fixture
async def concurrent_sessions(
    test_engine: AsyncEngine, request
) -> AsyncGenerator[list[AsyncSession], None]:
    """Create multiple concurrent database sessions for concurrency testing."""
    session_count = getattr(request, "param", 5)  # Default to 5 sessions

    SessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    sessions = []
    transactions = []

    try:
        for _ in range(session_count):
            session = SessionLocal()
            transaction = await session.begin()
            sessions.append(session)
            transactions.append(transaction)

        yield sessions

    finally:
        # Cleanup all sessions
        for i, session in enumerate(sessions):
            try:
                await transactions[i].rollback()
                await session.close()
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture
def mock_external_api():
    """Mock external API calls for testing."""
    return AsyncMock()


@pytest.fixture
async def database_initializer(
    test_database_config: DatabaseConfig,
) -> DatabaseInitializer:
    """Create database initializer for testing migrations and initialization."""
    return DatabaseInitializer(test_database_config)


# Utility functions for test data generation
def generate_test_portfolios(count: int, user_id: str) -> list[Portfolio]:
    """Generate test portfolios for bulk operations testing."""
    portfolios = []
    for i in range(count):
        portfolio = Portfolio(
            user_id=user_id,
            name=f"Test Portfolio {i+1}",
            description=f"Generated portfolio {i+1} for testing",
            risk_tolerance=RiskTolerance.MODERATE,
            cash_balance=10000.0 + i * 1000,
            created_at=datetime.utcnow(),
        )
        portfolios.append(portfolio)
    return portfolios


def generate_test_positions(count: int, portfolio_id: str) -> list[Position]:
    """Generate test positions for bulk operations testing."""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
    positions = []

    for i in range(count):
        position = Position(
            portfolio_id=portfolio_id,
            symbol=symbols[i % len(symbols)],
            quantity=100 + i * 10,
            average_cost=100.0 + i * 5,
            current_price=105.0 + i * 5,
            position_type="STOCK",
            created_at=datetime.utcnow(),
        )
        positions.append(position)
    return positions


# Performance testing utilities
@asynccontextmanager
async def performance_timer(tracker, operation: str):
    """Context manager for timing operations."""
    import time

    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        tracker.record(operation, duration)


# Test database verification utilities
async def verify_database_state(session: AsyncSession) -> dict[str, int]:
    """Verify database state by counting records in each table."""
    tables = [
        ("users", User),
        ("portfolios", Portfolio),
        ("positions", Position),
        ("analysis_results", AnalysisResult),
        ("market_data_cache", MarketDataCache),
        ("price_history", PriceHistory),
        ("options_data", OptionsData),
    ]

    counts = {}
    for table_name, model in tables:
        result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        counts[table_name] = result.scalar()

    return counts


# pytest configuration
def pytest_configure(config):
    """Configure pytest for database testing."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "concurrent: mark test as concurrency test")
    config.addinivalue_line("markers", "migration: mark test as migration test")


# Test utilities class
class DatabaseTestUtils:
    """Utility methods for database testing."""

    @staticmethod
    async def wait_for_condition(
        condition_func, timeout: int = DEFAULT_TEST_TIMEOUT, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true with timeout."""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)

        return False

    @staticmethod
    async def execute_concurrent_operations(
        operations: list, max_concurrent: int = 10
    ) -> list:
        """Execute multiple operations concurrently with limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_operation(op):
            async with semaphore:
                return await op

        return await asyncio.gather(*[bounded_operation(op) for op in operations])


@pytest.fixture
def db_test_utils():
    """Provide database test utilities."""
    return DatabaseTestUtils()
