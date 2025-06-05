# BuffetBot Cost Optimization Guide

## 🎯 Overview

This guide provides practical strategies to minimize operational costs while maintaining BuffetBot's performance and reliability. Follow these optimization techniques to reduce your monthly expenses by 30-70%.

## 💡 Quick Wins (Immediate Savings)

### 1. API Call Optimization (Save 60-80%)

**Problem**: Unnecessary API calls are the biggest cost driver
**Solution**: Implement intelligent caching and batching

```python
# buffetbot/utils/optimized_cache.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

class CostOptimizedCache:
    """Smart caching to minimize API costs"""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.call_log: Dict[str, int] = {}

    async def get_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get stock data with intelligent caching"""
        cache_key = f"stock_{symbol}"
        current_time = datetime.now()

        # Check if we have cached data
        if cache_key in self.cache and not force_refresh:
            cached_item = self.cache[cache_key]
            cache_time = datetime.fromisoformat(cached_item['timestamp'])

            # Use cached data if less than 1 hour old during market hours
            # or less than 24 hours old during off-hours
            if self._is_cache_valid(cache_time, current_time):
                return cached_item['data']

        # Make API call and cache result
        data = await self._fetch_from_api(symbol)
        self.cache[cache_key] = {
            'data': data,
            'timestamp': current_time.isoformat()
        }

        # Track API usage
        today = current_time.strftime('%Y-%m-%d')
        self.call_log[today] = self.call_log.get(today, 0) + 1

        return data

    def _is_cache_valid(self, cache_time: datetime, current_time: datetime) -> bool:
        """Determine if cached data is still valid"""
        if self._is_market_hours(current_time):
            # During market hours: cache for 15 minutes
            return current_time - cache_time < timedelta(minutes=15)
        else:
            # Outside market hours: cache for 4 hours
            return current_time - cache_time < timedelta(hours=4)

    def get_daily_api_usage(self) -> Dict[str, int]:
        """Track API usage for cost monitoring"""
        return self.call_log

# Usage Example
cache = CostOptimizedCache()
data = await cache.get_stock_data('AAPL')  # Only calls API when needed
```

**Expected Savings**: $30-150/month on API costs

### 2. Database Query Optimization (Save 30-50%)

```python
# Optimize database queries to reduce computational costs
from sqlalchemy import Index
from database.models import Portfolio, Position

class OptimizedPortfolioRepository:
    """Optimized repository with efficient queries"""

    async def get_portfolio_summary(self, user_id: str) -> Dict:
        """Get portfolio summary with optimized single query"""

        # BAD: Multiple queries (expensive)
        # portfolios = await self.get_user_portfolios(user_id)
        # for portfolio in portfolios:
        #     positions = await self.get_portfolio_positions(portfolio.id)

        # GOOD: Single optimized query (cheap)
        query = """
        SELECT
            p.id,
            p.name,
            p.total_value,
            COUNT(pos.id) as position_count,
            SUM(pos.current_value) as total_positions_value
        FROM portfolios p
        LEFT JOIN positions pos ON p.id = pos.portfolio_id
        WHERE p.user_id = :user_id
        GROUP BY p.id, p.name, p.total_value
        """

        result = await self.session.execute(query, {'user_id': user_id})
        return result.fetchall()

# Add database indexes for faster queries
class OptimizedIndexes:
    """Add these indexes to improve query performance"""

    portfolio_user_idx = Index('idx_portfolio_user', Portfolio.user_id)
    position_portfolio_idx = Index('idx_position_portfolio', Position.portfolio_id)
    position_symbol_idx = Index('idx_position_symbol', Position.symbol)
```

### 3. Smart Resource Scaling (Save 20-40%)

```bash
# Use auto-scaling to minimize hosting costs
# docker-compose.yml for cost-optimized deployment
version: '3.8'
services:
  buffetbot:
    image: buffetbot:latest
    environment:
      - AUTO_SCALE=true
      - MIN_INSTANCES=1
      - MAX_INSTANCES=3
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
```

## 📊 Detailed Optimization Strategies

### Database Cost Optimization

#### 1. Connection Pooling
```python
# database/optimized_config.py
from sqlalchemy.pool import QueuePool

class CostOptimizedDatabaseConfig:
    """Database configuration optimized for cost"""

    def __init__(self):
        self.pool_settings = {
            'pool_size': 5,          # Reduce from default 20
            'max_overflow': 10,      # Reduce from default 30
            'pool_recycle': 3600,    # Recycle connections every hour
            'pool_pre_ping': True,   # Validate connections
            'poolclass': QueuePool
        }
```

#### 2. Query Optimization
```sql
-- Add these indexes to reduce query costs
CREATE INDEX CONCURRENTLY idx_portfolios_user_created
ON portfolios(user_id, created_at);

CREATE INDEX CONCURRENTLY idx_positions_portfolio_symbol
ON positions(portfolio_id, symbol);

CREATE INDEX CONCURRENTLY idx_market_data_symbol_date
ON market_data_cache(symbol, date_cached);

-- Optimize expensive queries
EXPLAIN ANALYZE SELECT * FROM portfolios WHERE user_id = 'user123';
```

#### 3. Data Archival Strategy
```python
# Archive old data to reduce storage costs
class DataArchivalService:
    """Archive old data to reduce database costs"""

    async def archive_old_data(self):
        """Archive data older than 1 year"""

        cutoff_date = datetime.now() - timedelta(days=365)

        # Move old market data to cold storage
        old_data = await self.session.execute(
            "SELECT * FROM market_data_cache WHERE date_cached < :cutoff",
            {'cutoff': cutoff_date}
        )

        # Save to S3 or similar cold storage
        await self._save_to_cold_storage(old_data)

        # Delete from main database
        await self.session.execute(
            "DELETE FROM market_data_cache WHERE date_cached < :cutoff",
            {'cutoff': cutoff_date}
        )
```

### API Cost Optimization

#### 1. Batch API Calls
```python
class BatchAPIClient:
    """Batch multiple API calls to reduce costs"""

    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Get multiple stocks in a single API call"""

        # Instead of: multiple individual calls (expensive)
        # for symbol in symbols:
        #     data = await api.get_stock(symbol)

        # Use: single batch call (cheaper)
        batch_symbols = ','.join(symbols[:100])  # API limit
        batch_data = await self.api_client.get_batch_quotes(batch_symbols)

        return batch_data
```

#### 2. Intelligent Rate Limiting
```python
class CostAwareRateLimiter:
    """Rate limiter that considers API costs"""

    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        self.api_costs = {
            'quote': 0.01,      # $0.01 per quote
            'history': 0.05,    # $0.05 per historical request
            'news': 0.02        # $0.02 per news request
        }

    async def make_api_call(self, call_type: str, **kwargs):
        """Make API call only if within budget"""

        call_cost = self.api_costs.get(call_type, 0.01)

        if self.daily_spend + call_cost > self.daily_budget:
            # Use cached data or return error
            return await self._get_cached_fallback(**kwargs)

        self.daily_spend += call_cost
        return await self._make_actual_call(call_type, **kwargs)
```

### Infrastructure Cost Optimization

#### 1. Environment-Specific Scaling
```yaml
# k8s/cost-optimized-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: buffetbot-cost-optimized
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: buffetbot
        image: buffetbot:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: AUTO_SCALE
          value: "true"
```

#### 2. Cost-Aware Monitoring
```python
# monitoring/cost_monitor.py
class CostMonitor:
    """Monitor and alert on costs"""

    def __init__(self):
        self.daily_budgets = {
            'api_calls': 10.0,
            'database': 5.0,
            'hosting': 15.0
        }
        self.current_spend = {'api_calls': 0, 'database': 0, 'hosting': 0}

    async def check_budget_alerts(self):
        """Send alerts when approaching budget limits"""

        for category, budget in self.daily_budgets.items():
            spend = self.current_spend[category]
            utilization = spend / budget

            if utilization > 0.8:  # 80% of budget used
                await self._send_cost_alert(category, spend, budget)

    async def _send_cost_alert(self, category: str, spend: float, budget: float):
        """Send cost alert notification"""
        message = f"Cost Alert: {category} at ${spend:.2f} / ${budget:.2f} budget"
        # Send to Slack, email, etc.
```

## 🎛️ Cost Monitoring Dashboard

### Cost Tracking Script
```python
# scripts/cost_tracker.py
import asyncio
from datetime import datetime, timedelta
import json

class CostTracker:
    """Track daily costs across all services"""

    async def generate_daily_report(self):
        """Generate daily cost report"""

        today = datetime.now().strftime('%Y-%m-%d')

        costs = {
            'date': today,
            'api_calls': await self._get_api_costs(),
            'database': await self._get_database_costs(),
            'hosting': await self._get_hosting_costs(),
            'storage': await self._get_storage_costs()
        }

        total = sum(costs.values()) - len(costs) + 1  # Subtract non-numeric fields
        costs['total'] = total

        # Save to tracking file
        with open(f'cost_reports/{today}.json', 'w') as f:
            json.dump(costs, f, indent=2)

        return costs

    async def generate_monthly_summary(self):
        """Generate monthly cost summary"""

        # Read all daily reports for current month
        # Calculate trends and projections
        # Generate optimization recommendations
        pass
```

### Cost Dashboard Template
```html
<!-- cost_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>BuffetBot Cost Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="cost-dashboard">
        <h1>BuffetBot Cost Dashboard</h1>

        <div class="cost-overview">
            <div class="cost-card">
                <h3>Today's Spend</h3>
                <span class="cost-amount" id="daily-cost">$0.00</span>
            </div>

            <div class="cost-card">
                <h3>Monthly Projection</h3>
                <span class="cost-amount" id="monthly-projection">$0.00</span>
            </div>

            <div class="cost-card">
                <h3>Budget Remaining</h3>
                <span class="cost-amount" id="budget-remaining">$0.00</span>
            </div>
        </div>

        <canvas id="cost-trend-chart"></canvas>
    </div>
</body>
</html>
```

## 📋 Cost Optimization Checklist

### Daily Optimizations
- [ ] Check API usage against daily limits
- [ ] Review database query performance
- [ ] Monitor resource utilization
- [ ] Validate cache hit rates
- [ ] Check for unused resources

### Weekly Optimizations
- [ ] Analyze cost trends and patterns
- [ ] Review and optimize expensive queries
- [ ] Clean up unused data and resources
- [ ] Update cache strategies based on usage
- [ ] Review API provider pricing changes

### Monthly Optimizations
- [ ] Complete cost analysis and forecasting
- [ ] Optimize resource allocation based on usage
- [ ] Negotiate better rates with providers
- [ ] Archive old data to cold storage
- [ ] Review and update optimization strategies

## 🎯 Environment-Specific Strategies

### Development Environment
```bash
# Minimize costs during development
export ENVIRONMENT=development
export DB_POOL_SIZE=2
export API_RATE_LIMIT=100
export CACHE_TTL=7200  # 2 hours
export LOG_LEVEL=ERROR  # Reduce logging overhead
```

### Staging Environment
```bash
# Balance cost and functionality for staging
export ENVIRONMENT=staging
export DB_POOL_SIZE=5
export API_RATE_LIMIT=500
export CACHE_TTL=3600  # 1 hour
export AUTO_SCALE=true
```

### Production Environment
```bash
# Optimize for performance while controlling costs
export ENVIRONMENT=production
export DB_POOL_SIZE=10
export API_RATE_LIMIT=1000
export CACHE_TTL=900   # 15 minutes
export MONITORING=enhanced
```

## 📊 ROI Measurement

### Cost Tracking Metrics
```python
class ROICalculator:
    """Calculate return on investment for optimizations"""

    def calculate_optimization_savings(self, before_costs: Dict, after_costs: Dict) -> Dict:
        """Calculate savings from optimizations"""

        savings = {}
        for category in before_costs:
            if category in after_costs:
                savings[category] = before_costs[category] - after_costs[category]

        total_savings = sum(savings.values())
        return {
            'category_savings': savings,
            'total_monthly_savings': total_savings,
            'annual_savings': total_savings * 12,
            'optimization_roi': self._calculate_roi(total_savings)
        }
```

## 🔄 Continuous Optimization

### Automated Cost Optimization
```python
# scripts/auto_optimizer.py
class AutoCostOptimizer:
    """Automatically optimize costs based on usage patterns"""

    async def run_daily_optimization(self):
        """Run daily optimization tasks"""

        # 1. Adjust cache TTL based on data freshness needs
        await self._optimize_cache_settings()

        # 2. Scale resources based on actual usage
        await self._optimize_resource_allocation()

        # 3. Batch pending API calls
        await self._optimize_api_calls()

        # 4. Clean up unnecessary data
        await self._cleanup_old_data()
```

This cost optimization guide provides practical, implementable strategies to significantly reduce BuffetBot's operational costs while maintaining functionality and performance. Start with the quick wins and gradually implement the more advanced optimizations based on your specific usage patterns and requirements.

---

*For additional cost optimization support, refer to the main [Cost Analysis](../COST_ANALYSIS.md) document and consider implementing the monitoring tools provided in this guide.*
