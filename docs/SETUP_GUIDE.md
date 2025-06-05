# BuffetBot Setup Guide

## 🚀 Complete Installation & Configuration Guide

This guide provides step-by-step instructions to set up BuffetBot from scratch, including database configuration, dependencies, and account connections.

## ⚡ Quick Start (TL;DR)

For experienced developers who want to get up and running quickly:

```bash
# 1. Clone and setup
git clone https://github.com/jgadelugo/BuffetBot.git
cd BuffetBot
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio coverage psutil

# 3. Setup PostgreSQL (macOS with Homebrew)
brew install postgresql@14 && brew services start postgresql@14
psql postgres -c "CREATE USER buffetbot_user WITH PASSWORD 'password123';"
psql postgres -c "CREATE DATABASE buffetbot_dev OWNER buffetbot_user;"

# 4. Configure environment
cp env.example .env
# Edit .env with your database credentials:
# DB_USERNAME=buffetbot_user
# DB_PASSWORD=password123
# DB_NAME=buffetbot_dev
# DB_HOST=localhost
# DB_PORT=5432

# 5. Initialize database
python scripts/test_database_init.py

# 6. Run tests to verify setup
python -m pytest tests/database/ -v --tb=short

# 7. Start the application
./scripts/run_dashboard.sh
```

**Having issues?** Check the [troubleshooting section](#-troubleshooting) below.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **PostgreSQL**: 14 or higher
- **Git**: Latest version
- **Operating System**: macOS, Linux, or Windows with WSL

### Required Accounts & API Keys
- **Financial Data Provider**: Alpha Vantage, Yahoo Finance, or similar
- **Database**: PostgreSQL instance (local or cloud)

## 🔧 Step 1: Clone & Initial Setup

```bash
# Clone the repository
git clone https://github.com/jgadelugo/BuffetBot.git
cd BuffetBot

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify you're in the virtual environment
which python  # Should point to venv/bin/python
```

## 🗄️ Step 2: PostgreSQL Database Setup

### Option A: Local PostgreSQL Installation

#### macOS (using Homebrew)
```bash
# Install PostgreSQL
brew install postgresql@14
brew services start postgresql@14

# Create database and user
psql postgres
```

```sql
-- In PostgreSQL shell
CREATE USER buffetbot_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE buffetbot_dev OWNER buffetbot_user;
CREATE DATABASE buffetbot_test OWNER buffetbot_user;
GRANT ALL PRIVILEGES ON DATABASE buffetbot_dev TO buffetbot_user;
GRANT ALL PRIVILEGES ON DATABASE buffetbot_test TO buffetbot_user;
\q
```

#### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
-- In PostgreSQL shell
CREATE USER buffetbot_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE buffetbot_dev OWNER buffetbot_user;
CREATE DATABASE buffetbot_test OWNER buffetbot_user;
GRANT ALL PRIVILEGES ON DATABASE buffetbot_dev TO buffetbot_user;
GRANT ALL PRIVILEGES ON DATABASE buffetbot_test TO buffetbot_user;
\q
```

#### Windows
1. Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Install with default settings
3. Use pgAdmin or command line to create databases as shown above

### Option B: Cloud PostgreSQL (Recommended for Production)

#### AWS RDS
1. Create RDS PostgreSQL instance
2. Note the endpoint, port, and credentials
3. Ensure security groups allow connection from your IP

#### Google Cloud SQL
1. Create Cloud SQL PostgreSQL instance
2. Create database and user
3. Configure authorized networks

#### Heroku Postgres
```bash
# If deploying to Heroku
heroku addons:create heroku-postgresql:hobby-dev
heroku config:get DATABASE_URL
```

## 🔑 Step 3: Environment Configuration

### Create Environment Files

```bash
# Copy example environment file
cp env.example .env

# Create environment-specific files (already exist in config/)
# config/env.development - Development settings
# config/env.testing - Test settings
# config/env.production - Production settings
```

### Configure .env File

Edit `.env` with your database credentials:

```bash
# ===========================================
# BuffetBot Environment Configuration
# ===========================================

# Database Configuration
DB_USERNAME=buffetbot_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=buffetbot_dev

# Environment Setting
ENVIRONMENT=development

# API Keys (obtain from respective providers)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
YAHOO_FINANCE_API_KEY=your_yahoo_finance_key

# Application Settings
DEBUG=true
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Environment-Specific Configuration

#### Development (.env or config/env.development)
```bash
ENVIRONMENT=development
DB_NAME=buffetbot_dev
DEBUG=true
LOG_LEVEL=DEBUG
DB_POOL_SIZE=5
DB_ECHO_SQL=true
```

#### Testing (config/env.testing)
```bash
ENVIRONMENT=testing
DB_NAME=buffetbot_test
DEBUG=false
LOG_LEVEL=INFO
DB_POOL_SIZE=2
DB_ECHO_SQL=false
```

#### Production (config/env.production)
```bash
ENVIRONMENT=production
DB_NAME=buffetbot_prod
DEBUG=false
LOG_LEVEL=WARNING
DB_POOL_SIZE=20
DB_ECHO_SQL=false
DB_SSL_MODE=require
```

## 📦 Step 4: Install Dependencies

```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation
python -c "import streamlit; print('Streamlit installed successfully')"
python -c "import sqlalchemy; print('SQLAlchemy installed successfully')"
```

### Install Additional Testing Dependencies

```bash
# Install testing packages
pip install pytest pytest-asyncio coverage psutil
```

## 🗄️ Step 5: Database Initialization

### Initialize Database Schema

```bash
# Initialize database with our custom CLI tool
python -m database.cli init-db

# Or use the script directly
python scripts/test_database_init.py

# Verify database connection
python scripts/verify_config.py
```

### Run Database Migrations

```bash
# Navigate to database directory
cd database

# Initialize Alembic (if not already done)
python -m alembic init migrations

# Generate initial migration
python -m alembic revision --autogenerate -m "Initial schema"

# Apply migrations
python -m alembic upgrade head

# Return to project root
cd ..
```

### Seed Sample Data (Optional)

```bash
# Load sample data for development
python -c "
from database.seeds.sample_portfolios import create_sample_portfolios
from database.seeds.sample_market_data import create_sample_market_data
import asyncio

async def seed_data():
    await create_sample_portfolios()
    await create_sample_market_data()

asyncio.run(seed_data())
"
```

## 🧪 Step 6: Verify Installation

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run database tests specifically
python -m pytest tests/database/ -v

# Run with coverage
python -m pytest tests/ --cov=database --cov-report=html
```

### Test Database Connection

```bash
# Test database connectivity
python -c "
import asyncio
from database.config import DatabaseConfig
from database.initialization import DatabaseInitializer

async def test_connection():
    config = DatabaseConfig.from_env()
    initializer = DatabaseInitializer(config)

    if await initializer.check_database_health():
        print('✅ Database connection successful!')
    else:
        print('❌ Database connection failed!')

asyncio.run(test_connection())
"
```

## 🚀 Step 7: Run the Application

### Start Streamlit Dashboard

```bash
# Using the provided script
chmod +x scripts/run_dashboard.sh
./scripts/run_dashboard.sh

# Or run directly
streamlit run buffetbot/dashboard/streamlit_app.py --server.port=8501
```

### Access the Application

Open your browser and navigate to:
- **Local Development**: http://localhost:8501
- **Network Access**: http://your-ip-address:8501

## 🔧 Step 8: API Configuration (Optional)

### Financial Data APIs

#### Alpha Vantage Setup (Recommended)
1. **Get API Key**:
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for free account
   - Get your API key (free tier: 5 calls/minute, 500 calls/day)

2. **Add to Environment**:
   ```bash
   # Add to .env file
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   ```

3. **Test API Connection**:
   ```bash
   python -c "
   import os
   from buffetbot.data.fetcher import DataFetcher

   if os.getenv('ALPHA_VANTAGE_API_KEY'):
       fetcher = DataFetcher()
       data = fetcher.get_stock_data('AAPL')
       print('✅ Alpha Vantage API working!' if data else '❌ API call failed')
   else:
       print('❌ ALPHA_VANTAGE_API_KEY not set in environment')
   "
   ```

#### Yahoo Finance Setup (Free Alternative)
```bash
# Yahoo Finance is free but has rate limits
# No API key required for basic usage
pip install yfinance

# Test Yahoo Finance
python -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1d')
print('✅ Yahoo Finance working!' if not data.empty else '❌ Yahoo Finance failed')
"
```

#### Financial Modeling Prep (Premium Option)
1. Visit [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
2. Get API key (free tier: 250 calls/day)
3. Add to `.env`: `FMP_API_KEY=your_fmp_key_here`

#### IEX Cloud (Enterprise Option)
1. Visit [IEX Cloud](https://iexcloud.io/)
2. Get API token
3. Add to `.env`: `IEX_CLOUD_API_KEY=your_iex_key_here`

### Test API Connections

```bash
python -c "
from buffetbot.data.fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.get_stock_data('AAPL')
print('✅ API connection successful!' if data else '❌ API connection failed!')
"
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Database Connection Errors

**Error**: `Field required [type=missing, input_value={}, input_type=dict]`

This error typically occurs when environment variables are not properly loaded or the `.env` file is missing required fields.

**Solution Steps**:

1. **Verify .env file exists and has correct format**:
```bash
# Check if .env file exists
ls -la .env

# Check .env content (ensure it has the required fields)
cat .env
```

2. **Ensure all required database fields are present**:
```bash
# Your .env file MUST contain these fields:
# DB_USERNAME=buffetbot_user
# DB_PASSWORD=your_secure_password
# DB_NAME=buffetbot_dev
# DB_HOST=localhost
# DB_PORT=5432
```

3. **Verify PostgreSQL is running**:
```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432

# Test connection manually
psql -h localhost -U buffetbot_user -d buffetbot_dev -c "SELECT 1;"
```

4. **Test environment loading**:
```bash
# Test if environment variables are loaded correctly
python -c "
import os
from database.config import DatabaseConfig

# Check raw environment variables
print('Raw environment variables:')
for key in ['DB_USERNAME', 'DB_PASSWORD', 'DB_NAME', 'DB_HOST', 'DB_PORT']:
    print(f'{key}: {os.getenv(key, \"NOT SET\")}')

print('\nTesting DatabaseConfig:')
try:
    config = DatabaseConfig.from_env()
    print(f'✅ Config loaded successfully!')
    print(f'Database: {config.database}')
    print(f'Username: {config.username}')
    print(f'Host: {config.host}:{config.port}')
except Exception as e:
    print(f'❌ Config loading failed: {e}')
"
```

5. **Fix .env file if missing or incorrect**:
```bash
# Create .env file from template
cp env.example .env

# Edit .env file with your actual credentials
nano .env  # or your preferred editor
```

**Example working .env file**:
```bash
# Database Configuration - REQUIRED FIELDS
DB_USERNAME=buffetbot_user
DB_PASSWORD=mySecurePassword123
DB_NAME=buffetbot_dev
DB_HOST=localhost
DB_PORT=5432

# Environment
ENVIRONMENT=development

# Optional but recommended
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'database'`

**Solution**:
```bash
# Ensure you're in project root and virtual environment is active
pwd  # Should show BuffetBot directory
which python  # Should show venv/bin/python

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with module flag
python -m buffetbot.dashboard.streamlit_app
```

#### 3. Permission Errors

**Error**: `Permission denied` on PostgreSQL

**Solution**:
```sql
-- In PostgreSQL as superuser
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO buffetbot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO buffetbot_user;
ALTER USER buffetbot_user CREATEDB;
```

#### 4. Port Already in Use

**Error**: `Port 8501 is already in use`

**Solution**:
```bash
# Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run buffetbot/dashboard/streamlit_app.py --server.port=8502
```

#### 5. Virtual Environment Issues

**Error**: `Command not found` or wrong Python version

**Solution**:
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Debugging Commands

```bash
# Check Python environment
python --version
which python
pip list

# Check database status
pg_isready -h localhost -p 5432

# Check environment variables
python -c "import os; print({k:v for k,v in os.environ.items() if 'DB_' in k})"

# Test specific components
python -c "from buffetbot.utils.config import Config; print(Config.DATABASE_URL)"
```

## 🔒 Security Best Practices

### Environment Variables
- Never commit `.env` files to version control
- Use different passwords for each environment
- Enable SSL for production databases
- Use environment-specific API keys

### Database Security
```sql
-- Create read-only user for reporting
CREATE USER buffetbot_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE buffetbot_prod TO buffetbot_readonly;
GRANT USAGE ON SCHEMA public TO buffetbot_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO buffetbot_readonly;
```

### Application Security
```bash
# Set secure permissions on config files
chmod 600 .env
chmod 600 config/env.*

# Use strong passwords
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 🚀 New Database Infrastructure (Phase 1D)

BuffetBot now includes enterprise-grade database infrastructure with comprehensive testing:

### Database Features
- **Async Repository Pattern**: Clean, testable data access layer
- **SQLAlchemy Models**: Type-safe database models with relationships
- **Alembic Migrations**: Version-controlled schema evolution
- **Multi-Environment Config**: Separate settings for dev/test/prod
- **Connection Pooling**: Optimized database connections
- **Health Monitoring**: Database status checking and reporting

### Testing Infrastructure
- **Unit Tests**: Comprehensive repository testing
- **Integration Tests**: Cross-repository workflow validation
- **Performance Tests**: Benchmarking with specific targets
- **Error Handling Tests**: All error scenarios covered
- **Migration Tests**: Schema evolution validation
- **CI/CD Ready**: Automated testing pipeline support

### Quick Database Test
```bash
# Test the new database infrastructure
python -m pytest tests/database/test_initialization.py -v
python -m pytest tests/database/test_repository_pattern.py -v
python -m pytest tests/database/test_performance.py -v
```

### Development Workflow
```bash
# Development branch usage (recommended)
git checkout develop  # Use develop branch for latest features
git checkout main     # Use main branch for stable releases

# Run comprehensive test suite
python tests/database/test_runner.py  # Run all database tests
```

## 📚 Additional Resources

### Documentation
- [Database Schema Design](./database/SCHEMA_DESIGN.md)
- [Repository Pattern Usage](./database/repositories/README.md)
- [Enhanced Database README](./database/README_ENHANCED.md)
- [Phase 1C Completion Summary](./PHASE_1C_COMPLETION_SUMMARY.md)
- [API Documentation](./api/README.md)
- [Testing Guide](./tests/README.md)

### Development Workflow
- [Phase 1C Completion Summary](./PHASE_1C_COMPLETION_SUMMARY.md)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [Git Workflow](./GIT_WORKFLOW.md)

### Support
- **Issues**: [GitHub Issues](https://github.com/jgadelugo/BuffetBot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jgadelugo/BuffetBot/discussions)
- **Documentation**: [Project Wiki](https://github.com/jgadelugo/BuffetBot/wiki)

## ✅ Quick Verification Checklist

- [ ] PostgreSQL installed and running
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Environment variables configured in `.env`
- [ ] Database connection successful
- [ ] Migrations applied
- [ ] Tests passing
- [ ] Streamlit dashboard accessible
- [ ] API keys configured (if using external data)

**🎉 Congratulations! BuffetBot is now ready for use.**

---

*Last updated: December 2024*
*Version: 1.0.0*
*For questions or issues, please refer to the troubleshooting section or create an issue on GitHub.*
