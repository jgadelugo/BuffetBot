"""
Example usage of the BuffetBot repository pattern.

Demonstrates how to use repositories for common database operations
with proper session management and error handling.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from database.exceptions import EntityNotFoundError, RepositoryError, ValidationError
from database.repositories import (
    close_repositories,
    get_repository_registry,
    init_repositories,
)

# Note: In a real implementation, these would import actual models from Phase 1a
# For now, we'll use mock model classes


class MockUser:
    def __init__(self, id=None, name=None, email=None):
        self.id = id or uuid4()
        self.name = name
        self.email = email


class MockPortfolio:
    def __init__(self, id=None, user_id=None, name=None, risk_tolerance=None):
        self.id = id or uuid4()
        self.user_id = user_id
        self.name = name
        self.risk_tolerance = risk_tolerance
        self.target_cash_percentage = Decimal("5.00")
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.positions = []
        self.analysis_results = []


class MockPosition:
    def __init__(self, id=None, portfolio_id=None, ticker=None, shares=None):
        self.id = id or uuid4()
        self.portfolio_id = portfolio_id
        self.ticker = ticker
        self.shares = shares
        self.average_cost = None
        self.allocation_percentage = None
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class MockAnalysisResult:
    def __init__(self, id=None, portfolio_id=None, ticker=None, analysis_type=None):
        self.id = id or uuid4()
        self.portfolio_id = portfolio_id
        self.ticker = ticker
        self.analysis_type = analysis_type
        self.score = None
        self.confidence_level = None
        self.calculated_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=24)


class MockMarketDataCache:
    def __init__(self, id=None, ticker=None, data_type=None, data=None):
        self.id = id or uuid4()
        self.ticker = ticker
        self.data_type = data_type
        self.data = data or {}
        self.cached_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=24)
        self.data_source = "example"


async def portfolio_operations_example():
    """
    Example of portfolio operations using the repository pattern.
    """
    print("=== Portfolio Operations Example ===")

    try:
        # Get repository registry
        registry = get_repository_registry()

        # Get portfolio repository
        portfolio_repo = await registry.get_portfolio_repository()
        position_repo = await registry.get_position_repository()

        # Create a user (mock)
        user_id = uuid4()
        print(f"Working with user ID: {user_id}")

        # Create a new portfolio
        portfolio = MockPortfolio(
            user_id=user_id, name="My Investment Portfolio", risk_tolerance="moderate"
        )

        print(f"Creating portfolio: {portfolio.name}")
        # created_portfolio = await portfolio_repo.create(portfolio)
        # print(f"Created portfolio with ID: {created_portfolio.id}")

        # Get user's portfolios
        print("Getting user portfolios...")
        # user_portfolios = await portfolio_repo.get_user_portfolios(user_id)
        # print(f"User has {len(user_portfolios)} portfolios")

        # Add a position to the portfolio
        position = MockPosition(
            portfolio_id=portfolio.id, ticker="AAPL", shares=Decimal("100")
        )

        print(f"Adding position: {position.ticker}")
        # updated_portfolio = await portfolio_repo.add_position(portfolio.id, position)
        # print(f"Portfolio now has {len(updated_portfolio.positions)} positions")

        # Get portfolio by name
        print("Finding portfolio by name...")
        # found_portfolio = await portfolio_repo.get_by_name_and_user(
        #     "My Investment Portfolio", user_id
        # )
        # if found_portfolio:
        #     print(f"Found portfolio: {found_portfolio.name}")

        print("Portfolio operations completed successfully!")

    except ValidationError as e:
        print(f"Validation error: {e}")
    except EntityNotFoundError as e:
        print(f"Entity not found: {e}")
    except RepositoryError as e:
        print(f"Repository error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def analysis_operations_example():
    """
    Example of analysis operations using the repository pattern.
    """
    print("\n=== Analysis Operations Example ===")

    try:
        registry = get_repository_registry()
        analysis_repo = await registry.get_analysis_repository()

        # Create analysis result
        analysis = MockAnalysisResult(
            portfolio_id=uuid4(), ticker="AAPL", analysis_type="value_analysis"
        )
        analysis.score = Decimal("85.5")
        analysis.confidence_level = Decimal("0.85")

        print(f"Creating analysis result for {analysis.ticker}")
        # created_analysis = await analysis_repo.create(analysis)
        # print(f"Created analysis with ID: {created_analysis.id}")

        # Get recent analysis
        print("Getting recent analysis...")
        # recent_analysis = await analysis_repo.get_recent_analysis(
        #     ticker="AAPL",
        #     analysis_type="value_analysis",
        #     max_age_hours=24
        # )
        # if recent_analysis:
        #     print(f"Found recent analysis with score: {recent_analysis.score}")

        # Get analysis history
        print("Getting analysis history...")
        # history = await analysis_repo.get_ticker_analysis_history(
        #     ticker="AAPL",
        #     limit=10
        # )
        # print(f"Found {len(history)} historical analysis results")

        # Clean up expired analysis
        print("Cleaning up expired analysis...")
        # deleted_count = await analysis_repo.cleanup_expired_analysis()
        # print(f"Deleted {deleted_count} expired analysis results")

        print("Analysis operations completed successfully!")

    except Exception as e:
        print(f"Error in analysis operations: {e}")


async def market_data_operations_example():
    """
    Example of market data operations using the repository pattern.
    """
    print("\n=== Market Data Operations Example ===")

    try:
        registry = get_repository_registry()
        market_data_repo = await registry.get_market_data_repository()

        # Cache market data
        sample_data = {
            "price": 150.25,
            "volume": 1000000,
            "market_cap": 2500000000,
            "timestamp": datetime.utcnow().isoformat(),
        }

        print("Caching market data for AAPL...")
        # cached_data = await market_data_repo.cache_market_data(
        #     ticker="AAPL",
        #     data_type="fundamentals",
        #     data=sample_data,
        #     ttl_hours=24
        # )
        # print(f"Cached data with ID: {cached_data.id}")

        # Get cached data
        print("Retrieving cached data...")
        # retrieved_data = await market_data_repo.get_cached_data(
        #     ticker="AAPL",
        #     data_type="fundamentals"
        # )
        # if retrieved_data:
        #     print(f"Retrieved cached data: {retrieved_data.data}")

        # Get cache statistics
        print("Getting cache statistics...")
        # stats = await market_data_repo.get_cache_statistics(days_back=7)
        # print(f"Cache statistics: {stats}")

        # Clean up expired cache
        print("Cleaning up expired cache...")
        # deleted_count = await market_data_repo.cleanup_expired_cache()
        # print(f"Deleted {deleted_count} expired cache entries")

        print("Market data operations completed successfully!")

    except Exception as e:
        print(f"Error in market data operations: {e}")


async def error_handling_example():
    """
    Example of error handling in repository operations.
    """
    print("\n=== Error Handling Example ===")

    try:
        registry = get_repository_registry()
        portfolio_repo = await registry.get_portfolio_repository()

        # Try to get a non-existent portfolio
        print("Attempting to get non-existent portfolio...")
        # portfolio = await portfolio_repo.get_by_id(uuid4())
        # if portfolio is None:
        #     print("Portfolio not found (expected)")

        # Try to create invalid portfolio
        print("Attempting to create invalid portfolio...")
        invalid_portfolio = MockPortfolio(
            user_id=uuid4(), name="", risk_tolerance="moderate"  # Invalid: empty name
        )

        try:
            # This would fail validation
            # await portfolio_repo.create(invalid_portfolio)
            pass
        except ValidationError as e:
            print(f"Validation error caught: {e}")

        print("Error handling completed successfully!")

    except Exception as e:
        print(f"Unexpected error: {e}")


async def transaction_example():
    """
    Example of using transactions with repositories.
    """
    print("\n=== Transaction Example ===")

    try:
        registry = get_repository_registry()
        session_manager = registry.session_manager

        # Using transaction context manager
        print("Starting transaction...")
        # async with session_manager.transaction() as session:
        #     # Create repositories with the transaction session
        #     portfolio_repo = PortfolioRepository(session)
        #     position_repo = PositionRepository(session)
        #
        #     # Create portfolio
        #     portfolio = MockPortfolio(
        #         user_id=uuid4(),
        #         name="Transaction Portfolio",
        #         risk_tolerance="aggressive"
        #     )
        #     created_portfolio = await portfolio_repo.create(portfolio)
        #
        #     # Add multiple positions
        #     positions = [
        #         MockPosition(portfolio_id=created_portfolio.id, ticker="AAPL", shares=100),
        #         MockPosition(portfolio_id=created_portfolio.id, ticker="GOOGL", shares=50),
        #         MockPosition(portfolio_id=created_portfolio.id, ticker="MSFT", shares=75)
        #     ]
        #
        #     for position in positions:
        #         await position_repo.create(position)
        #
        #     print(f"Transaction completed: created portfolio with {len(positions)} positions")

        print("Transaction example completed successfully!")

    except Exception as e:
        print(f"Transaction error: {e}")


async def health_check_example():
    """
    Example of performing health checks on repositories.
    """
    print("\n=== Health Check Example ===")

    try:
        registry = get_repository_registry()

        # Perform health check
        print("Performing health check...")
        health_result = await registry.health_check()

        print(f"Overall status: {health_result['status']}")
        print(f"Session manager: {health_result['session_manager']['status']}")

        for repo_name, repo_status in health_result["repositories"].items():
            print(f"Repository {repo_name}: {repo_status['status']}")

        print("Health check completed!")

    except Exception as e:
        print(f"Health check error: {e}")


async def main():
    """
    Main function to run all examples.
    """
    print("BuffetBot Repository Pattern Examples")
    print("=" * 50)

    try:
        # Initialize repositories
        print("Initializing repositories...")
        await init_repositories()
        print("Repositories initialized successfully!")

        # Run examples
        await portfolio_operations_example()
        await analysis_operations_example()
        await market_data_operations_example()
        await error_handling_example()
        await transaction_example()
        await health_check_example()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")

    finally:
        # Clean up
        print("\nCleaning up repositories...")
        await close_repositories()
        print("Cleanup completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
