"""
Portfolio repository for BuffetBot database layer.

Repository for managing portfolios and positions with domain-specific operations.
"""

from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from buffetbot.utils.logger import setup_logger

from ..exceptions import EntityNotFoundError, RepositoryError, ValidationError
from .base import BaseRepository

# Import models - will be available when Phase 1a models are integrated
try:
    from ..models.portfolio import Portfolio, Position, RiskTolerance
    from ..models.user import User
except ImportError:
    # Placeholder classes for development
    class Portfolio:
        pass

    class Position:
        pass

    class RiskTolerance:
        pass

    class User:
        pass


# Initialize logger
logger = setup_logger(__name__)


class PortfolioRepository(BaseRepository[Portfolio]):
    """
    Repository for portfolio management operations.

    Provides domain-specific methods for portfolio CRUD operations,
    position management, and portfolio analysis support.
    """

    def __init__(self, session):
        super().__init__(session, Portfolio)

    async def _validate_entity(
        self, entity: Portfolio, is_update: bool = False
    ) -> None:
        """Validate portfolio entity before database operations."""
        if not entity.name or not entity.name.strip():
            raise ValidationError("Portfolio name is required", field="name")

        if len(entity.name) > 255:
            raise ValidationError(
                "Portfolio name too long", field="name", value=entity.name
            )

        if not entity.risk_tolerance:
            raise ValidationError("Risk tolerance is required", field="risk_tolerance")

        if entity.target_cash_percentage is not None:
            if entity.target_cash_percentage < 0 or entity.target_cash_percentage > 100:
                raise ValidationError(
                    "Target cash percentage must be between 0 and 100",
                    field="target_cash_percentage",
                    value=entity.target_cash_percentage,
                )

        if not is_update and not entity.user_id:
            raise ValidationError(
                "User ID is required for new portfolios", field="user_id"
            )

    async def _apply_eager_loading(self, query: Select) -> Select:
        """Apply eager loading for portfolio relationships."""
        return query.options(
            selectinload(Portfolio.positions),
            selectinload(Portfolio.owner),
            selectinload(Portfolio.analysis_results),
        )

    async def get_user_portfolios(self, user_id: UUID) -> list[Portfolio]:
        """
        Get all portfolios for a specific user.

        Args:
            user_id: ID of the user

        Returns:
            List[Portfolio]: List of user's portfolios

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Getting portfolios for user: {user_id}")

            return await self.list_by_criteria(
                user_id=user_id, is_active=True, order_by="-created_at"
            )

        except Exception as e:
            self.logger.error(f"Failed to get portfolios for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to get user portfolios",
                repository=self.__class__.__name__,
                operation="get_user_portfolios",
                details={"error": str(e), "user_id": str(user_id)},
            )

    async def get_by_name_and_user(
        self, name: str, user_id: UUID
    ) -> Optional[Portfolio]:
        """
        Get a portfolio by name and user ID.

        Args:
            name: Portfolio name
            user_id: ID of the user

        Returns:
            Optional[Portfolio]: Portfolio if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Getting portfolio '{name}' for user: {user_id}")

            query = select(Portfolio).where(
                Portfolio.name == name, Portfolio.user_id == user_id
            )
            query = await self._apply_eager_loading(query)

            result = await self.session.execute(query)
            portfolio = result.scalar_one_or_none()

            if portfolio:
                self.logger.debug(f"Found portfolio '{name}' for user: {user_id}")
            else:
                self.logger.debug(f"No portfolio '{name}' found for user: {user_id}")

            return portfolio

        except Exception as e:
            self.logger.error(
                f"Failed to get portfolio '{name}' for user {user_id}: {e}"
            )
            raise RepositoryError(
                "Failed to get portfolio by name and user",
                repository=self.__class__.__name__,
                operation="get_by_name_and_user",
                details={"error": str(e), "name": name, "user_id": str(user_id)},
            )

    async def add_position(self, portfolio_id: UUID, position: Position) -> Portfolio:
        """
        Add a position to a portfolio.

        Args:
            portfolio_id: ID of the portfolio
            position: Position to add

        Returns:
            Portfolio: Updated portfolio with new position

        Raises:
            EntityNotFoundError: If portfolio not found
            RepositoryError: If operation fails
        """
        try:
            self.logger.debug(f"Adding position to portfolio: {portfolio_id}")

            # Get the portfolio
            portfolio = await self.get_by_id(portfolio_id)
            if not portfolio:
                raise EntityNotFoundError("Portfolio", str(portfolio_id))

            # Set the portfolio ID on the position
            position.portfolio_id = portfolio_id

            # Add position to session
            self.session.add(position)
            await self.session.flush()
            await self.session.refresh(position)

            # Refresh portfolio to get updated positions
            await self.session.refresh(portfolio)

            self.logger.info(
                f"Added position {position.ticker} to portfolio: {portfolio_id}"
            )
            return portfolio

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to add position to portfolio {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to add position to portfolio",
                repository=self.__class__.__name__,
                operation="add_position",
                details={"error": str(e), "portfolio_id": str(portfolio_id)},
            )

    async def remove_position(self, portfolio_id: UUID, position_id: UUID) -> Portfolio:
        """
        Remove a position from a portfolio.

        Args:
            portfolio_id: ID of the portfolio
            position_id: ID of the position to remove

        Returns:
            Portfolio: Updated portfolio

        Raises:
            EntityNotFoundError: If portfolio or position not found
            RepositoryError: If operation fails
        """
        try:
            self.logger.debug(
                f"Removing position {position_id} from portfolio: {portfolio_id}"
            )

            # Get the portfolio
            portfolio = await self.get_by_id(portfolio_id)
            if not portfolio:
                raise EntityNotFoundError("Portfolio", str(portfolio_id))

            # Find and remove the position
            position = None
            for pos in portfolio.positions:
                if pos.id == position_id:
                    position = pos
                    break

            if not position:
                raise EntityNotFoundError("Position", str(position_id))

            # Remove position
            await self.session.delete(position)
            await self.session.flush()

            # Refresh portfolio
            await self.session.refresh(portfolio)

            self.logger.info(
                f"Removed position {position_id} from portfolio: {portfolio_id}"
            )
            return portfolio

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to remove position {position_id} from portfolio {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to remove position from portfolio",
                repository=self.__class__.__name__,
                operation="remove_position",
                details={
                    "error": str(e),
                    "portfolio_id": str(portfolio_id),
                    "position_id": str(position_id),
                },
            )

    async def update_position(
        self, portfolio_id: UUID, position: Position
    ) -> Portfolio:
        """
        Update a position in a portfolio.

        Args:
            portfolio_id: ID of the portfolio
            position: Updated position

        Returns:
            Portfolio: Updated portfolio

        Raises:
            EntityNotFoundError: If portfolio not found
            RepositoryError: If operation fails
        """
        try:
            self.logger.debug(f"Updating position in portfolio: {portfolio_id}")

            # Ensure position belongs to the portfolio
            position.portfolio_id = portfolio_id

            # Update position
            merged_position = await self.session.merge(position)
            await self.session.flush()
            await self.session.refresh(merged_position)

            # Get updated portfolio
            portfolio = await self.get_by_id(portfolio_id)
            if not portfolio:
                raise EntityNotFoundError("Portfolio", str(portfolio_id))

            self.logger.info(
                f"Updated position {position.ticker} in portfolio: {portfolio_id}"
            )
            return portfolio

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to update position in portfolio {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to update position in portfolio",
                repository=self.__class__.__name__,
                operation="update_position",
                details={"error": str(e), "portfolio_id": str(portfolio_id)},
            )

    async def get_portfolios_by_risk_tolerance(
        self, risk_tolerance: str
    ) -> list[Portfolio]:
        """
        Get portfolios by risk tolerance level.

        Args:
            risk_tolerance: Risk tolerance level

        Returns:
            List[Portfolio]: List of portfolios with specified risk tolerance
        """
        try:
            self.logger.debug(
                f"Getting portfolios with risk tolerance: {risk_tolerance}"
            )

            return await self.list_by_criteria(
                risk_tolerance=risk_tolerance, is_active=True, order_by="-created_at"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to get portfolios by risk tolerance {risk_tolerance}: {e}"
            )
            raise RepositoryError(
                "Failed to get portfolios by risk tolerance",
                repository=self.__class__.__name__,
                operation="get_portfolios_by_risk_tolerance",
                details={"error": str(e), "risk_tolerance": risk_tolerance},
            )

    async def get_portfolio_summary(self, portfolio_id: UUID) -> dict:
        """
        Get a summary of portfolio statistics.

        Args:
            portfolio_id: ID of the portfolio

        Returns:
            dict: Portfolio summary statistics

        Raises:
            EntityNotFoundError: If portfolio not found
            RepositoryError: If operation fails
        """
        try:
            self.logger.debug(f"Getting portfolio summary for: {portfolio_id}")

            portfolio = await self.get_by_id(portfolio_id)
            if not portfolio:
                raise EntityNotFoundError("Portfolio", str(portfolio_id))

            # Calculate summary statistics
            total_positions = len(portfolio.positions)
            total_allocation = sum(
                pos.allocation_percentage or Decimal("0") for pos in portfolio.positions
            )

            active_positions = [pos for pos in portfolio.positions if pos.is_active]

            summary = {
                "portfolio_id": str(portfolio_id),
                "name": portfolio.name,
                "risk_tolerance": portfolio.risk_tolerance,
                "total_positions": total_positions,
                "active_positions": len(active_positions),
                "total_allocation_percentage": float(total_allocation),
                "target_cash_percentage": float(portfolio.target_cash_percentage or 0),
                "created_at": portfolio.created_at.isoformat(),
                "updated_at": portfolio.updated_at.isoformat(),
            }

            self.logger.debug(f"Generated summary for portfolio: {portfolio_id}")
            return summary

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to get portfolio summary for {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to get portfolio summary",
                repository=self.__class__.__name__,
                operation="get_portfolio_summary",
                details={"error": str(e), "portfolio_id": str(portfolio_id)},
            )


class PositionRepository(BaseRepository[Position]):
    """
    Repository for position management operations.

    Provides operations for managing individual positions within portfolios.
    """

    def __init__(self, session):
        super().__init__(session, Position)

    async def _validate_entity(self, entity: Position, is_update: bool = False) -> None:
        """Validate position entity before database operations."""
        if not entity.ticker or not entity.ticker.strip():
            raise ValidationError("Ticker is required", field="ticker")

        if len(entity.ticker) > 10:
            raise ValidationError(
                "Ticker too long", field="ticker", value=entity.ticker
            )

        if entity.shares is not None and entity.shares < 0:
            raise ValidationError(
                "Shares cannot be negative", field="shares", value=entity.shares
            )

        if entity.average_cost is not None and entity.average_cost < 0:
            raise ValidationError(
                "Average cost cannot be negative",
                field="average_cost",
                value=entity.average_cost,
            )

        if entity.allocation_percentage is not None:
            if entity.allocation_percentage < 0 or entity.allocation_percentage > 100:
                raise ValidationError(
                    "Allocation percentage must be between 0 and 100",
                    field="allocation_percentage",
                    value=entity.allocation_percentage,
                )

        if not is_update and not entity.portfolio_id:
            raise ValidationError(
                "Portfolio ID is required for new positions", field="portfolio_id"
            )

    async def _apply_eager_loading(self, query: Select) -> Select:
        """Apply eager loading for position relationships."""
        return query.options(selectinload(Position.portfolio))

    async def get_by_ticker_and_portfolio(
        self, ticker: str, portfolio_id: UUID
    ) -> Optional[Position]:
        """
        Get a position by ticker and portfolio.

        Args:
            ticker: Stock ticker
            portfolio_id: ID of the portfolio

        Returns:
            Optional[Position]: Position if found, None otherwise
        """
        try:
            self.logger.debug(
                f"Getting position for ticker {ticker} in portfolio: {portfolio_id}"
            )

            query = select(Position).where(
                Position.ticker == ticker.upper(), Position.portfolio_id == portfolio_id
            )
            query = await self._apply_eager_loading(query)

            result = await self.session.execute(query)
            position = result.scalar_one_or_none()

            return position

        except Exception as e:
            self.logger.error(
                f"Failed to get position for ticker {ticker} in portfolio {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to get position by ticker and portfolio",
                repository=self.__class__.__name__,
                operation="get_by_ticker_and_portfolio",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "portfolio_id": str(portfolio_id),
                },
            )

    async def get_portfolio_positions(
        self, portfolio_id: UUID, active_only: bool = True
    ) -> list[Position]:
        """
        Get all positions for a portfolio.

        Args:
            portfolio_id: ID of the portfolio
            active_only: Whether to return only active positions

        Returns:
            List[Position]: List of positions
        """
        try:
            filters = {"portfolio_id": portfolio_id}
            if active_only:
                filters["is_active"] = True

            return await self.list_by_criteria(order_by="-created_at", **filters)

        except Exception as e:
            self.logger.error(
                f"Failed to get positions for portfolio {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to get portfolio positions",
                repository=self.__class__.__name__,
                operation="get_portfolio_positions",
                details={"error": str(e), "portfolio_id": str(portfolio_id)},
            )
