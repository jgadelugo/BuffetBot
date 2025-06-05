# Database Module

This module provides comprehensive database functionality for BuffetBot, including configuration management, migrations, initialization, and data seeding.

## 🏗️ Architecture Overview

The database module follows enterprise patterns with:
- **Environment-specific configuration** with validation
- **Professional migration management** with Alembic
- **Automated initialization and health checks**
- **Development data seeding** for consistent testing
- **Repository pattern** for clean data access
- **Comprehensive testing** with mocks and fixtures

## 📁 Module Structure

```
database/
├── models/                    # SQLAlchemy ORM models
│   ├── user.py               # User management
│   ├── portfolio.py          # Portfolio and positions
│   ├── analysis.py           # Analysis results
│   ├── market_data.py        # Market data cache
│   └── __init__.py           # Model exports
├── repositories/             # Repository pattern implementation
│   ├── base.py              # Base repository with CRUD
│   ├── portfolio_repo.py    # Portfolio data access
│   ├── analysis_repo.py     # Analysis data access
│   └── market_data_repo.py  # Market data operations
├── migrations/              # Alembic migration management
│   ├── alembic.ini         # Alembic configuration
│   ├── env.py              # Migration environment
│   ├── script.py.mako      # Migration template
│   └── versions/           # Migration version files
├── seeds/                  # Development data seeding
│   ├── __init__.py        # Main seeding coordinator
│   ├── sample_portfolios.py  # Sample portfolio data
│   ├── sample_analysis.py    # Sample analysis results
│   └── sample_market_data.py # Sample market data
├── config.py              # Database configuration management
├── connection.py          # Database connection handling
├── initialization.py     # Database lifecycle management
├── cli.py                # Command-line interface
├── exceptions.py         # Custom database exceptions
└── README.md            # This file
```

## ⚙️ Configuration Management

### Environment-Specific Configuration

The database module supports multiple environments with proper validation:

```python
from database.config import DatabaseConfig, DatabaseEnvironment

# Automatic environment detection
config = DatabaseConfig()

# Environment-specific settings
if config.is_development:
    # Development-specific logic
elif config.is_production:
    # Production-specific logic
```

### Environment Variables

Set these environment variables for database configuration:

```bash
# Core database settings
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=buffetbot_dev
DB_PASSWORD=your_password
DB_NAME=buffetbot_development

# Connection pool settings
DB_POOL_SIZE=5
DB_POOL_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Environment and features
ENVIRONMENT=development
DB_ECHO_SQL=true
DB_SSL_MODE=prefer
```

### Configuration Files

Use environment-specific configuration files:
- `config/env.development` - Development settings
- `config/env.testing` - Testing settings
- `config/env.production` - Production template

## 🔄 Migration Management

### Alembic Setup

The module includes professional Alembic configuration with:
- Environment-aware database URL resolution
- Enhanced error handling and logging
- Support for both sync and async operations

### CLI Commands

Use the database CLI for all migration operations:

```bash
# Check database health
python -m database.cli health --detailed

# Create a new migration
python -m database.cli migrate -m "Add new feature"

# Apply migrations
python -m database.cli upgrade

# Rollback migration (development only)
python -m database.cli downgrade

# View migration history
python -m database.cli history

# Show current revision
python -m database.cli current
```

### Migration Workflow

1. **Make model changes** in `database/models/`
2. **Generate migration**: `python -m database.cli migrate -m "Description"`
3. **Review migration file** in `database/migrations/versions/`
4. **Apply migration**: `python -m database.cli upgrade`

## 🚀 Database Initialization

### Automated Setup

The initialization system provides:
- **Health checks** before operations
- **Schema creation** for development
- **Data seeding** with realistic samples
- **Environment safety** (production protection)

```python
from database.initialization import DatabaseInitializer

# Initialize with configuration
initializer = DatabaseInitializer()

# Complete setup
await initializer.initialize_database(
    create_schema=True,
    seed_data=True
)
```

### CLI Initialization

```bash
# Create database schema
python -m database.cli create

# Create and seed with development data
python -m database.cli create --seed

# Reset database (development only)
python -m database.cli reset

# Seed development data
python -m database.cli seed
```

## 🌱 Development Data Seeding

### Sample Data

The seeding system creates realistic development data:
- **Users**: Demo users with different profiles
- **Portfolios**: Various risk tolerance levels
- **Positions**: Realistic stock positions
- **Market Data**: Current and historical data
- **Analysis Results**: Comprehensive analysis samples

### Seeding Usage

```python
from database.seeds import create_sample_data

async with session_manager() as session:
    await create_sample_data(session)
    await session.commit()
```

## 🔌 Database Usage

### Basic Connection

```python
from database.connection import get_async_database_session

async with get_async_database_session() as session:
    # Use session for database operations
    result = await session.execute(select(Portfolio))
    portfolios = result.scalars().all()
```

### Repository Pattern

```python
from database.repositories import PortfolioRepository

async with get_async_database_session() as session:
    portfolio_repo = PortfolioRepository(session)

    # Get portfolio by ID
    portfolio = await portfolio_repo.get_by_id("portfolio-id")

    # Create new portfolio
    new_portfolio = Portfolio(name="My Portfolio", ...)
    saved_portfolio = await portfolio_repo.create(new_portfolio)

    # Update portfolio
    portfolio.name = "Updated Name"
    updated_portfolio = await portfolio_repo.update(portfolio)
```

### Configuration Access

```python
from database.config import get_database_config

config = get_database_config()
print(f"Connected to: {config.host}:{config.port}/{config.database}")
print(f"Environment: {config.environment.value}")
print(f"Pool size: {config.pool_size}")
```

## 🧪 Testing

### Test Configuration

The module includes comprehensive test coverage:

```bash
# Run all database tests
pytest tests/database/

# Run specific test files
pytest tests/database/test_config.py
pytest tests/database/test_initialization.py

# Run with coverage
pytest tests/database/ --cov=database --cov-report=html
```

### Test Database Setup

Tests use isolated test database configuration:

```python
from database.config import get_test_database_config

# Automatically configured for testing
test_config = get_test_database_config()
assert test_config.environment == DatabaseEnvironment.TESTING
assert test_config.database.endswith("_test")
```

## 🔧 Development Workflow

### Daily Development

1. **Start development**: `./run_dashboard.sh` (includes database health check)
2. **Make model changes**: Edit files in `database/models/`
3. **Create migration**: `python -m database.cli migrate -m "Description"`
4. **Apply migration**: `python -m database.cli upgrade`
5. **Test changes**: `pytest tests/database/`

### Database Management

```bash
# Check database status
python -m database.cli config
python -m database.cli health --detailed

# Reset development database
python -m database.cli reset

# Seed fresh development data
python -m database.cli seed --clear
```

## 🚨 Production Considerations

### Safety Features

- **Production protection**: Dangerous operations blocked in production
- **Migration validation**: Comprehensive checks before applying
- **Connection pooling**: Optimized for high-concurrency
- **SSL support**: Configurable SSL/TLS encryption
- **Health monitoring**: Built-in health check endpoints

### Deployment

1. **Set environment variables** for production database
2. **Run migrations**: `python -m database.cli upgrade`
3. **Verify health**: `python -m database.cli health`
4. **Monitor connections**: Use `get_database_info()` for metrics

## 📊 Monitoring and Observability

### Health Checks

```python
from database.initialization import check_database_health

# Simple health check
is_healthy = await check_database_health()

# Detailed database information
initializer = DatabaseInitializer()
info = await initializer.get_database_info()
print(f"Database version: {info['version']}")
print(f"Active connections: {info['active_connections']}")
```

### CLI Monitoring

```bash
# Basic health check
python -m database.cli health

# Detailed information
python -m database.cli health --detailed

# Configuration overview
python -m database.cli config
```

## 🔗 Integration

### With Existing Code

The database module integrates seamlessly with existing BuffetBot components:

```python
# In analysis modules
from database.repositories import AnalysisRepository

async def save_analysis_result(portfolio_id: str, result: AnalysisResult):
    async with get_async_database_session() as session:
        analysis_repo = AnalysisRepository(session)
        return await analysis_repo.create(result)

# In dashboard code
from database.repositories import PortfolioRepository

async def get_user_portfolios(user_id: str):
    async with get_async_database_session() as session:
        portfolio_repo = PortfolioRepository(session)
        return await portfolio_repo.get_by_user_id(user_id)
```

### Environment Setup

Copy the appropriate environment configuration:

```bash
# For development
cp config/env.development .env

# For testing
cp config/env.testing .env.test

# Edit with your database credentials
vim .env
```

## 📚 Additional Resources

- **Schema Design**: See [SCHEMA_DESIGN.md](SCHEMA_DESIGN.md) for detailed schema documentation
- **Model Documentation**: Check individual model files for field descriptions
- **Repository Patterns**: Review `repositories/base.py` for common operations
- **Migration Examples**: Look at `migrations/versions/` for migration patterns

## 🆘 Troubleshooting

### Common Issues

1. **Connection Failed**: Check PostgreSQL is running and credentials are correct
2. **Migration Errors**: Ensure database is accessible and has proper permissions
3. **Seeding Failures**: Verify models are properly imported and relationships exist
4. **Test Failures**: Check test database configuration and isolation

### Debug Commands

```bash
# Test database connection
python -c "from database.initialization import check_database_health; import asyncio; print(asyncio.run(check_database_health()))"

# Verify configuration
python -m database.cli config

# Check migration status
python -m database.cli current
python -m database.cli history
```
