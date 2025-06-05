# BuffetBot GCS Storage Integration

A comprehensive Google Cloud Storage integration for BuffetBot's data persistence layer, designed for cost-efficiency, query performance, and scalability.

## 🏗️ Architecture Overview

The storage system is built with a modular architecture consisting of:

- **GCS Manager**: Core storage operations with intelligent routing and retry logic
- **Schema Management**: Data validation, versioning, and evolution
- **Data Formatting**: Parquet optimization and compression
- **Query Optimization**: Partition pruning and caching
- **Security**: Encryption, access control, and audit logging
- **Monitoring**: Performance metrics and alerting

## 📁 Directory Structure

```
buffetbot/storage/
├── gcs/                    # GCS operations
│   ├── manager.py         # Main storage manager
│   ├── client.py          # GCS client wrapper
│   ├── retry.py           # Retry logic
│   └── connection_pool.py # Connection pooling
├── schemas/               # Data schemas
│   ├── manager.py         # Schema validation
│   ├── market_data.py     # Market data schemas
│   ├── forecasts.py       # Forecast schemas
│   └── options_data.py    # Options data schemas
├── formatters/            # Data formatting
│   ├── parquet_formatter.py
│   ├── compression.py
│   └── metadata.py
├── query/                 # Query optimization
│   ├── optimizer.py
│   ├── cache_manager.py
│   └── partition_analyzer.py
└── utils/                 # Utilities
    ├── config.py          # Configuration management
    ├── monitoring.py      # Metrics and monitoring
    └── security.py        # Security utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements/base.txt

# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

### 2. Configuration

```python
from buffetbot.storage.utils.config import GCSConfig

# Load from environment variables
config = GCSConfig.from_environment()

# Or create manually
config = GCSConfig(
    project_id="your-project-id",
    data_bucket="your-data-bucket",
    archive_bucket="your-archive-bucket",
    backup_bucket="your-backup-bucket",
    temp_bucket="your-temp-bucket",
    service_account_email="your-service-account@project.iam.gserviceaccount.com",
    region="us-central1"
)
```

### 3. Basic Usage

```python
import asyncio
from buffetbot.storage import GCSStorageManager, SchemaManager

async def main():
    # Initialize components
    config = GCSConfig.from_environment()
    storage_manager = GCSStorageManager(config)
    schema_manager = SchemaManager()

    # Sample market data
    market_data = [
        {
            'symbol': 'AAPL',
            'timestamp': datetime.now(timezone.utc),
            'price': 150.25,
            'volume': 1000000,
            'data_source': 'yahoo_finance',
            'created_at': datetime.now(timezone.utc),
            'version': 'v1.2.0'
        }
    ]

    # Validate data
    validation_result = schema_manager.validate_data(
        data=market_data,
        data_type='market_data'
    )

    if validation_result.is_valid:
        # Store data
        result = await storage_manager.store_data(
            data_type='market_data',
            data=market_data
        )
        print(f"Stored {len(market_data)} records to {result.file_path}")
    else:
        print(f"Validation failed: {validation_result.errors}")

asyncio.run(main())
```

## 📊 Data Types and Schemas

### Market Data

```python
# Market data schema includes:
- symbol (string, required)
- timestamp (timestamp, required)
- price (decimal, required)
- volume (int64, optional)
- market_cap (decimal, optional)
- pe_ratio (float64, optional)
- technical indicators (RSI, SMA, volatility, beta)
- metadata (data_source, created_at, version)
```

### Forecast Data

```python
# Forecast schema includes:
- symbol (string, required)
- model_name (string, required)
- model_version (string, required)
- forecast_horizon_days (int32, required)
- predicted_price (decimal, required)
- confidence intervals and scores
- ensemble predictions
- model performance metrics
```

### Options Data

```python
# Options schema includes:
- underlying_symbol (string, required)
- contract_symbol (string, required)
- expiration_date (date, required)
- strike_price (decimal, required)
- option_type (string, required)
- pricing data (bid, ask, last_price, volume)
- Greeks (delta, gamma, theta, vega, rho)
- implied_volatility
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_DATA_BUCKET="your-data-bucket"
export GCS_ARCHIVE_BUCKET="your-archive-bucket"
export GCS_BACKUP_BUCKET="your-backup-bucket"
export GCS_TEMP_BUCKET="your-temp-bucket"
export GCS_SERVICE_ACCOUNT_EMAIL="service-account@project.iam.gserviceaccount.com"

# Optional
export GCS_REGION="us-central1"
export GCS_KMS_KEY_ID="projects/project/locations/region/keyRings/ring/cryptoKeys/key"
export GCS_MAX_CONNECTIONS="50"
export GCS_RETRY_ATTEMPTS="3"
export GCS_TIMEOUT_SECONDS="60"
export GCS_COMPRESSION="snappy"
```

### Terraform Integration

```python
# Load configuration from Terraform outputs
config = GCSConfig.from_terraform_output('terraform_output.json')
```

## 🔍 Query Optimization

### Partition Pruning

The system automatically optimizes queries by:

- **Date-based partitioning**: Queries are pruned to relevant date ranges
- **Symbol filtering**: Only relevant symbol partitions are scanned
- **Metadata indexing**: Fast lookups using partition metadata

### Caching Strategy

- **Hot data cache**: Recent data (last 7 days) in Standard storage
- **Popular symbols**: Top 100 symbols by volume cached
- **Query result cache**: In-memory caching for repeated queries

### Example Query

```python
from buffetbot.storage.query import DataQuery, QueryOptimizer

# Create optimized query
query = DataQuery(
    data_type='market_data',
    filters={'symbol': 'AAPL'},
    date_range={
        'start': '2024-01-01',
        'end': '2024-01-31'
    },
    limit=1000
)

optimizer = QueryOptimizer()
optimized_query = optimizer.optimize(query)

# Execute query
result = await storage_manager.retrieve_data(optimized_query)
```

## 🔒 Security Features

### Data Encryption

```python
from buffetbot.storage.utils.security import SecurityManager

security_manager = SecurityManager()

# Encrypt sensitive data
encrypted = security_manager.encrypt_data("sensitive_data")
decrypted = security_manager.decrypt_data(encrypted)
```

### Access Control

```python
from buffetbot.storage.utils.security import SecurityContext, AccessPolicy

# Define security context
context = SecurityContext(
    user_id="analyst_001",
    roles=["analyst", "user"],
    permissions=["read_market_data"],
    session_id="session_12345"
)

# Check access
has_access = security_manager.check_access(
    context=context,
    resource="market_data/AAPL",
    operation="read"
)
```

### Audit Logging

All access attempts are logged for security auditing:

```python
# Get audit logs
audit_logs = security_manager.get_audit_log(
    since=datetime.now() - timedelta(days=7),
    user_id="analyst_001"
)
```

## 📈 Monitoring and Metrics

### Performance Monitoring

```python
from buffetbot.storage.utils.monitoring import StorageMetrics

metrics = StorageMetrics()

# Record operations
metrics.record_upload(
    data_type="market_data",
    file_size=1024000,
    duration_ms=250,
    success=True
)

metrics.record_query(
    data_type="market_data",
    duration_ms=45,
    records_returned=1000,
    cache_hit=False
)

# Get performance summary
performance = metrics.get_performance_summary()
cache_metrics = metrics.get_cache_metrics()
```

### Key Metrics

- **Upload Performance**: Throughput, latency, success rates
- **Query Performance**: Response times, cache hit rates, partition efficiency
- **Storage Costs**: Usage by bucket and storage class
- **Error Rates**: Failed operations by type and cause

## 🗂️ File Organization

### Bucket Structure

```
gs://buffetbot-data-{env}/
├── raw/                    # Raw ingested data
│   ├── market_data/
│   │   └── year=2024/month=01/day=15/
│   │       ├── AAPL_market_20240115_143052_v1.parquet
│   │       └── MSFT_market_20240115_143052_v1.parquet
│   ├── options_data/
│   └── forecasts/
├── processed/              # Processed data
├── cache/                  # Frequently accessed data
└── metadata/               # System metadata
```

### Naming Conventions

```
{data_category}_{symbol}_{timestamp}_{version}.{format}

Examples:
- AAPL_market_20240115_143052_v1.parquet
- SPY_options_20240115_093000_v1.parquet
- forecast_AAPL_model-lstm_20240115_v2.parquet
```

## 💰 Cost Optimization

### Storage Classes

- **Standard**: Active data (0-30 days)
- **Nearline**: Weekly access (30-90 days)
- **Coldline**: Monthly access (90-365 days)
- **Archive**: Long-term storage (365+ days)

### Lifecycle Policies

Automatic transitions based on data age:

```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
        "condition": {"age": 365}
      }
    ]
  }
}
```

## 🧪 Testing

### Run Example

```bash
# Run the example script
python examples/gcs_storage_example.py
```

### Unit Tests

```bash
# Run storage tests
pytest tests/storage/ -v

# Run with coverage
pytest tests/storage/ --cov=buffetbot.storage --cov-report=html
```

## 🔧 Development

### Adding New Data Types

1. **Define Schema**: Add schema in `schemas/` directory
2. **Update Manager**: Add validation rules in `schemas/manager.py`
3. **Add Formatters**: Implement formatting logic if needed
4. **Update Tests**: Add comprehensive tests

### Schema Evolution

```python
# Add new schema version
schema_manager.evolve_schema(
    data_type='market_data',
    current_version='v1.2.0',
    new_schema=new_schema,
    new_version='v1.3.0'
)
```

## 📚 API Reference

### Core Classes

- **`GCSStorageManager`**: Main storage interface
- **`SchemaManager`**: Data validation and schema management
- **`ParquetFormatter`**: Data formatting and optimization
- **`QueryOptimizer`**: Query optimization and caching
- **`SecurityManager`**: Security and access control
- **`StorageMetrics`**: Performance monitoring

### Configuration

- **`GCSConfig`**: Storage configuration management
- **`ValidationLevel`**: Schema validation strictness
- **`SecurityContext`**: User security context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
