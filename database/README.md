# 🗄️ BuffetBot Database Layer

## 📋 Overview

The BuffetBot database layer provides enterprise-grade data persistence using PostgreSQL and SQLAlchemy. This layer implements the repository pattern for clean architecture and supports both synchronous and asynchronous operations.

## 🏗️ Architecture

### **Key Components**

- **Models**: SQLAlchemy ORM models organized by domain
- **Repositories**: Data access layer implementing repository pattern
- **Migrations**: Alembic-based database schema versioning
- **Connection Management**: Async-first connection pooling

### **Design Principles**

- **Domain-Driven Design**: Models organized by business domains
- **Repository Pattern**: Clean separation of data access logic
- **Async-First**: Built for high-performance async operations
- **Type Safety**: Full type hints with SQLAlchemy 2.0
- **Data Integrity**: Comprehensive validation and constraints

## 🚀 Quick Start

### **1. Prerequisites**

```bash
# Install PostgreSQL
brew install postgresql  # macOS
sudo apt-get install postgresql  # Ubuntu

# Start PostgreSQL service
brew services start postgresql  # macOS
sudo systemctl start postgresql  # Ubuntu

# Create database
createdb buffetbot
```

### **2. Install Dependencies**

```bash
# Install database dependencies (already in pyproject.toml)
pip install sqlalchemy[asyncio] asyncpg psycopg2-binary alembic
```

### **3. Environment Setup**

```bash
# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=buffetbot
export DB_USER=postgres
export DB_PASSWORD=postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/buffetbot
```

### **4. Initialize Database**

```python
from database import init_database

# Initialize database connection
db = init_database()

# Create tables (in production, use migrations)
await db.create_tables()
```

### **5. Basic Usage**

```python
import asyncio
from database import get_async_database_session
from database.models import User, Portfolio, RiskTolerance
from database.repositories import UserRepository, PortfolioRepository

async def example():
    user_repo = UserRepository()
    portfolio_repo = PortfolioRepository()

    async with get_async_database_session() as session:
        # Create a user
        user = await user_repo.async_create(
            session,
            email="investor@example.com",
            username="investor",
            hashed_password="hashed_password",
            first_name="John",
            last_name="Doe"
        )

        # Create a portfolio
        portfolio = await portfolio_repo.async_create(
            session,
            user_id=user.id,
            name="My Portfolio",
            risk_tolerance=RiskTolerance.MODERATE
        )

        print(f"Created portfolio: {portfolio.name}")

asyncio.run(example())
```

## 📊 Database Schema

### **Core Tables**

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | User accounts | UUID keys, authentication, preferences |
| `portfolios` | Investment portfolios | Risk tolerance, allocation targets |
| `positions` | Stock positions | Shares, cost basis, allocation % |
| `analysis_results` | Analysis cache | JSONB metadata, expiration |
| `market_data_cache` | Market data cache | Unified caching, TTL |
| `price_history` | Historical prices | Time-series optimized |
| `options_data` | Options chains | Greeks, expiration tracking |

### **Relationships**

```
users (1) ──→ (many) portfolios
portfolios (1) ──→ (many) positions
portfolios (1) ──→ (many) analysis_results
```

## 🔧 Repository Pattern

### **Base Repository**

All repositories inherit from `BaseRepository` which provides:

- **CRUD Operations**: Create, read, update, delete
- **Query Methods**: Flexible filtering and pagination
- **Async Support**: Both sync and async variants
- **Type Safety**: Generic type parameters

```python
from database.repositories import BaseRepository
from database.models import MyModel

class MyRepository(BaseRepository[MyModel]):
    def __init__(self):
        super().__init__(MyModel)

    # Add domain-specific methods here
    async def get_by_custom_field(self, session, value):
        return await self.async_get_by_criteria(session, custom_field=value)
```

### **Available Repositories**

- **UserRepository**: User account operations
- **PortfolioRepository**: Portfolio and position management
- **AnalysisRepository**: Analysis result caching
- **MarketDataRepository**: Market data caching

## 🗄️ Database Migrations

### **Alembic Setup**

```bash
# Initialize Alembic (already done)
alembic init database/migrations

# Generate migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### **Migration Commands**

```bash
# Check current version
alembic current

# Show migration history
alembic history

# Show SQL for migration
alembic show <revision>

# Upgrade to specific version
alembic upgrade <revision>
```

## 🔍 Model Examples

### **User Model**

```python
from database.models import User

# User with authentication and preferences
user = User(
    email="investor@example.com",
    username="investor",
    hashed_password="...",
    first_name="John",
    last_name="Doe",
    preferences='{"theme": "dark", "notifications": true}'
)
```

### **Portfolio Model**

```python
from database.models import Portfolio, RiskTolerance
from decimal import Decimal

# Portfolio with risk management
portfolio = Portfolio(
    user_id=user.id,
    name="Growth Portfolio",
    description="Long-term growth strategy",
    risk_tolerance=RiskTolerance.AGGRESSIVE,
    target_cash_percentage=Decimal("5.00")
)
```

### **Position Model**

```python
from database.models import Position
from decimal import Decimal

# Stock position with allocation
position = Position(
    portfolio_id=portfolio.id,
    ticker="AAPL",
    shares=Decimal("100.0000"),
    average_cost=Decimal("150.0000"),
    allocation_percentage=Decimal("25.00"),
    notes="Core technology holding"
)
```

### **Analysis Result Model**

```python
from database.models import AnalysisResult, AnalysisType
from datetime import datetime, timedelta

# Analysis result with metadata
analysis = AnalysisResult(
    portfolio_id=portfolio.id,
    ticker="AAPL",
    analysis_type=AnalysisType.VALUE_ANALYSIS,
    strategy="dcf",
    score=Decimal("85.50"),
    metadata={
        "intrinsic_value": 180.00,
        "margin_of_safety": 0.167,
        "assumptions": {
            "growth_rate": 0.05,
            "discount_rate": 0.10
        }
    },
    recommendation="buy",
    confidence_level=Decimal("0.90"),
    expires_at=datetime.utcnow() + timedelta(hours=24)
)
```

## 🔧 Configuration

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `buffetbot` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |
| `DATABASE_URL` | Auto-generated | Full connection URL |
| `DB_ECHO` | `false` | Enable SQL logging |

### **Connection Pool Settings**

```python
# Configured in database/connection.py
pool_config = {
    "pool_size": 20,        # Base connection pool size
    "max_overflow": 30,     # Additional connections allowed
    "pool_pre_ping": True,  # Validate connections
    "pool_recycle": 3600,   # Recycle connections after 1 hour
}
```

## 🧪 Testing

### **Test Database Setup**

```python
import pytest
from database import init_database

@pytest.fixture
async def db_session():
    # Use test database
    db = init_database("postgresql://postgres:postgres@localhost:5432/buffetbot_test")

    # Create tables
    await db.create_tables()

    async with db.get_async_session() as session:
        yield session

    # Cleanup
    await db.drop_tables()
    db.close()
```

### **Repository Testing**

```python
async def test_user_repository(db_session):
    user_repo = UserRepository()

    # Create user
    user = await user_repo.async_create(
        db_session,
        email="test@example.com",
        username="testuser",
        hashed_password="hashed"
    )

    # Test retrieval
    found_user = await user_repo.async_get_by_id(db_session, user.id)
    assert found_user.email == "test@example.com"
```

## 🚀 Performance Tips

### **Query Optimization**

```python
# Use selectinload for relationships
from sqlalchemy.orm import selectinload

query = select(Portfolio).options(
    selectinload(Portfolio.positions)
)

# Use indexes for common queries
# Indexes are automatically created for:
# - Primary keys (id)
# - Foreign keys (user_id, portfolio_id)
# - Common query fields (ticker, analysis_type)
```

### **Batch Operations**

```python
# Batch create positions
positions_data = [
    {"ticker": "AAPL", "shares": 100, "portfolio_id": portfolio_id},
    {"ticker": "MSFT", "shares": 75, "portfolio_id": portfolio_id},
]

async with session.begin():
    for data in positions_data:
        position = Position(**data)
        session.add(position)
```

### **Connection Management**

```python
# Use context managers for automatic cleanup
async with get_async_database_session() as session:
    # Database operations here
    pass  # Session automatically closed and committed/rolled back
```

## 🛡️ Security

### **Data Protection**

- **UUID Primary Keys**: Prevent enumeration attacks
- **Password Hashing**: Never store plaintext passwords
- **Data Isolation**: User-based data separation
- **Input Validation**: SQLAlchemy model validation

### **SQL Injection Prevention**

```python
# ✅ Safe - using SQLAlchemy ORM
user = await user_repo.async_get_by_criteria(session, email=user_email)

# ✅ Safe - using parameterized queries
query = select(User).where(User.email == user_email)

# ❌ Never do this - SQL injection risk
# query = f"SELECT * FROM users WHERE email = '{user_email}'"
```

## 📈 Monitoring

### **Performance Metrics**

- Query execution times
- Connection pool utilization
- Cache hit rates
- Data freshness

### **Logging**

```python
import logging

# Enable SQL logging for debugging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Or set environment variable
# DB_ECHO=true
```

## 🔄 Migration from File Cache

### **Gradual Migration Strategy**

1. **Dual Write**: Write to both file cache and database
2. **Read Preference**: Read from database, fallback to file cache
3. **Data Migration**: Bulk import existing cache data
4. **Cleanup**: Remove file cache dependencies

### **Data Import Script**

```python
async def migrate_cache_data():
    """Migrate existing file cache to database."""
    from buffetbot.utils.cache import Cache

    file_cache = Cache("file")
    market_data_repo = MarketDataRepository()

    async with get_async_database_session() as session:
        # Import cached market data
        for cache_file in cache_dir.glob("*.cache"):
            # Parse and import cache data
            pass
```

## 📚 Additional Resources

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [AsyncPG Documentation](https://magicstack.github.io/asyncpg/)

## 🤝 Contributing

When adding new models or repositories:

1. **Create Model**: Add to appropriate domain module
2. **Add Repository**: Extend BaseRepository with domain methods
3. **Generate Migration**: Use Alembic autogenerate
4. **Update Documentation**: Add to schema documentation
5. **Add Tests**: Include unit and integration tests

## 📄 License

This database layer is part of the BuffetBot project and follows the same MIT license.
