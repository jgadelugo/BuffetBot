# BuffetBot Cost Tracking Template

## 📊 Monthly Cost Tracking Spreadsheet

### Template Structure

Copy this template to track your BuffetBot costs:

| Month | Infrastructure | Database | APIs | Monitoring | Storage | Total | Budget | Variance | Notes |
|-------|---------------|----------|------|------------|---------|-------|--------|----------|-------|
| Jan 2024 | $12.00 | $15.00 | $19.00 | $5.00 | $2.00 | $53.00 | $50.00 | +$3.00 | Baseline |
| Feb 2024 | $12.00 | $15.00 | $19.00 | $5.00 | $2.00 | $53.00 | $50.00 | +$3.00 | Stable |
| Mar 2024 | $24.00 | $30.00 | $49.00 | $10.00 | $5.00 | $118.00 | $100.00 | +$18.00 | Scaled up |

### Cost Categories Breakdown

#### Infrastructure Costs
```
Hosting Platform (e.g., DigitalOcean)     : $____
Load Balancer                            : $____
CDN                                      : $____
SSL Certificates                         : $____
Domain Registration                      : $____
Total Infrastructure                     : $____
```

#### Database Costs
```
PostgreSQL Instance                      : $____
Storage (per GB)                        : $____
Backup Storage                          : $____
Connection Pooling                      : $____
Total Database                          : $____
```

#### API Costs
```
Financial Data Provider                  : $____
Market Data Feed                        : $____
News API                                : $____
Additional Data Sources                 : $____
Total API                               : $____
```

#### Monitoring & Tools
```
Logging Service                         : $____
Monitoring (DataDog/New Relic)         : $____
Error Tracking (Sentry)                : $____
Analytics                              : $____
Total Monitoring                       : $____
```

## 🎯 Budget Planning Template

### Annual Budget Projection

| Quarter | Infrastructure | Database | APIs | Monitoring | Total | Growth Factor |
|---------|---------------|----------|------|------------|-------|---------------|
| Q1 2024 | $108 | $135 | $171 | $45 | $459 | Baseline |
| Q2 2024 | $120 | $150 | $190 | $50 | $510 | 11% growth |
| Q3 2024 | $135 | $170 | $215 | $55 | $575 | 13% growth |
| Q4 2024 | $150 | $190 | $240 | $60 | $640 | 11% growth |

### Cost Scenario Planning

#### Conservative Scenario (Low Growth)
```
Monthly Cost: $46
Annual Cost: $552
User Growth: 10% per quarter
API Calls: 50K per month
Database Size: 5GB
```

#### Optimistic Scenario (High Growth)
```
Monthly Cost: $150
Annual Cost: $1,800
User Growth: 50% per quarter
API Calls: 200K per month
Database Size: 20GB
```

#### Pessimistic Scenario (Cost Overrun)
```
Monthly Cost: $300
Annual Cost: $3,600
Unexpected: High API usage, premium services
Mitigation: Implement optimization strategies
```

## 📈 Cost Monitoring Scripts

### Daily Cost Tracker
```python
# scripts/daily_cost_tracker.py
import json
from datetime import datetime
from typing import Dict, Any

class DailyCostTracker:
    """Track daily costs across all services"""

    def __init__(self, budget_file: str = "monthly_budget.json"):
        self.budget_file = budget_file
        self.cost_log = []

    def log_daily_costs(self, costs: Dict[str, float]) -> None:
        """Log daily costs"""
        today = datetime.now().isoformat()[:10]

        entry = {
            "date": today,
            "infrastructure": costs.get("infrastructure", 0),
            "database": costs.get("database", 0),
            "apis": costs.get("apis", 0),
            "monitoring": costs.get("monitoring", 0),
            "storage": costs.get("storage", 0),
            "total": sum(costs.values())
        }

        self.cost_log.append(entry)
        self._save_log()

    def get_monthly_projection(self) -> Dict[str, Any]:
        """Calculate monthly cost projection"""
        if not self.cost_log:
            return {"error": "No cost data available"}

        # Calculate average daily cost
        total_days = len(self.cost_log)
        avg_daily_cost = sum(entry["total"] for entry in self.cost_log) / total_days

        # Project monthly cost
        monthly_projection = avg_daily_cost * 30

        return {
            "days_tracked": total_days,
            "avg_daily_cost": round(avg_daily_cost, 2),
            "monthly_projection": round(monthly_projection, 2),
            "last_30_days": self.cost_log[-30:] if len(self.cost_log) >= 30 else self.cost_log
        }

    def check_budget_status(self, monthly_budget: float) -> Dict[str, Any]:
        """Check current spending against budget"""
        projection = self.get_monthly_projection()

        if "error" in projection:
            return projection

        monthly_projection = projection["monthly_projection"]
        budget_utilization = (monthly_projection / monthly_budget) * 100

        return {
            "monthly_budget": monthly_budget,
            "projected_spend": monthly_projection,
            "budget_utilization": round(budget_utilization, 1),
            "remaining_budget": round(monthly_budget - monthly_projection, 2),
            "status": self._get_budget_status(budget_utilization)
        }

    def _get_budget_status(self, utilization: float) -> str:
        """Get budget status based on utilization"""
        if utilization <= 80:
            return "ON_TRACK"
        elif utilization <= 100:
            return "WARNING"
        else:
            return "OVER_BUDGET"

    def _save_log(self) -> None:
        """Save cost log to file"""
        with open("daily_costs.json", "w") as f:
            json.dump(self.cost_log, f, indent=2)

# Usage Example
tracker = DailyCostTracker()
daily_costs = {
    "infrastructure": 1.50,
    "database": 0.75,
    "apis": 2.25,
    "monitoring": 0.25,
    "storage": 0.10
}
tracker.log_daily_costs(daily_costs)
```

### Weekly Cost Report
```python
# scripts/weekly_cost_report.py
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class WeeklyCostReporter:
    """Generate weekly cost reports and visualizations"""

    def __init__(self, cost_data_file: str = "daily_costs.json"):
        self.cost_data_file = cost_data_file
        self.cost_data = self._load_cost_data()

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly cost report"""
        last_week = self._get_last_week_data()
        previous_week = self._get_previous_week_data()

        report = {
            "report_date": datetime.now().isoformat()[:10],
            "period": {
                "start": last_week[0]["date"] if last_week else None,
                "end": last_week[-1]["date"] if last_week else None
            },
            "weekly_totals": self._calculate_weekly_totals(last_week),
            "daily_breakdown": last_week,
            "week_over_week_change": self._calculate_week_over_week(last_week, previous_week),
            "cost_trends": self._analyze_cost_trends(),
            "recommendations": self._generate_recommendations()
        }

        return report

    def create_cost_visualization(self, output_file: str = "weekly_costs.png") -> None:
        """Create cost visualization chart"""
        if not self.cost_data:
            return

        dates = [entry["date"] for entry in self.cost_data[-14:]]  # Last 2 weeks
        totals = [entry["total"] for entry in self.cost_data[-14:]]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, totals, marker='o', linewidth=2, markersize=6)
        plt.title("BuffetBot Daily Costs - Last 2 Weeks")
        plt.xlabel("Date")
        plt.ylabel("Daily Cost ($)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _load_cost_data(self) -> list:
        """Load cost data from file"""
        try:
            with open(self.cost_data_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _get_last_week_data(self) -> list:
        """Get last week's cost data"""
        if not self.cost_data:
            return []

        today = datetime.now()
        week_ago = today - timedelta(days=7)

        return [
            entry for entry in self.cost_data
            if datetime.fromisoformat(entry["date"]) >= week_ago
        ]

    def _calculate_weekly_totals(self, week_data: list) -> Dict[str, float]:
        """Calculate weekly cost totals by category"""
        if not week_data:
            return {}

        totals = {
            "infrastructure": sum(entry.get("infrastructure", 0) for entry in week_data),
            "database": sum(entry.get("database", 0) for entry in week_data),
            "apis": sum(entry.get("apis", 0) for entry in week_data),
            "monitoring": sum(entry.get("monitoring", 0) for entry in week_data),
            "storage": sum(entry.get("storage", 0) for entry in week_data),
        }

        totals["total"] = sum(totals.values())
        return {k: round(v, 2) for k, v in totals.items()}

# Usage Example
reporter = WeeklyCostReporter()
weekly_report = reporter.generate_weekly_report()
reporter.create_cost_visualization()
```

## 🚨 Cost Alert System

### Budget Alert Configuration
```python
# scripts/cost_alerts.py
import smtplib
from email.mime.text import MimeText
from typing import Dict, List

class CostAlertSystem:
    """Send cost alerts when budgets are exceeded"""

    def __init__(self, alert_config: Dict[str, Any]):
        self.alert_config = alert_config
        self.alert_thresholds = {
            "warning": 80,    # 80% of budget
            "critical": 100,  # 100% of budget
            "emergency": 120  # 120% of budget
        }

    def check_and_send_alerts(self, current_costs: Dict[str, float], budgets: Dict[str, float]) -> None:
        """Check current costs against budgets and send alerts"""

        for category, current_cost in current_costs.items():
            if category not in budgets:
                continue

            budget = budgets[category]
            utilization = (current_cost / budget) * 100

            alert_level = self._get_alert_level(utilization)

            if alert_level:
                self._send_alert(category, current_cost, budget, utilization, alert_level)

    def _get_alert_level(self, utilization: float) -> str:
        """Determine alert level based on budget utilization"""
        if utilization >= self.alert_thresholds["emergency"]:
            return "emergency"
        elif utilization >= self.alert_thresholds["critical"]:
            return "critical"
        elif utilization >= self.alert_thresholds["warning"]:
            return "warning"
        return None

    def _send_alert(self, category: str, cost: float, budget: float, utilization: float, level: str) -> None:
        """Send cost alert notification"""
        subject = f"BuffetBot Cost Alert - {level.upper()}: {category}"

        message = f"""
        Cost Alert for BuffetBot {category}:

        Current Spend: ${cost:.2f}
        Budget: ${budget:.2f}
        Utilization: {utilization:.1f}%
        Alert Level: {level.upper()}

        {"IMMEDIATE ACTION REQUIRED!" if level == "emergency" else "Please review costs."}
        """

        # Send email, Slack notification, etc.
        print(f"ALERT: {subject}")
        print(message)

# Configuration Example
alert_config = {
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "alerts@yourdomain.com",
        "password": "your_password",
        "recipients": ["admin@yourdomain.com"]
    },
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#buffetbot-alerts"
    }
}

# Usage
alert_system = CostAlertSystem(alert_config)
current_costs = {"apis": 25.0, "database": 18.0, "infrastructure": 15.0}
budgets = {"apis": 20.0, "database": 15.0, "infrastructure": 12.0}
alert_system.check_and_send_alerts(current_costs, budgets)
```

## 📋 Cost Review Checklist

### Daily Review (5 minutes)
- [ ] Check daily cost totals
- [ ] Verify API usage is within limits
- [ ] Review any cost anomalies
- [ ] Update cost tracking spreadsheet

### Weekly Review (30 minutes)
- [ ] Generate weekly cost report
- [ ] Analyze cost trends and patterns
- [ ] Review budget utilization
- [ ] Identify optimization opportunities
- [ ] Update monthly projections

### Monthly Review (2 hours)
- [ ] Complete monthly cost analysis
- [ ] Compare actual vs budgeted costs
- [ ] Review vendor invoices and bills
- [ ] Analyze cost efficiency metrics
- [ ] Plan optimization initiatives
- [ ] Update annual budget projections
- [ ] Review and adjust budgets for next month

## 💾 Cost Data Export Templates

### CSV Export Format
```csv
Date,Infrastructure,Database,APIs,Monitoring,Storage,Total,Budget,Variance,Notes
2024-01-01,1.50,0.75,2.25,0.25,0.10,4.85,5.00,-0.15,Normal usage
2024-01-02,1.50,0.75,3.50,0.25,0.10,6.10,5.00,+1.10,High API usage
2024-01-03,1.50,0.75,2.00,0.25,0.10,4.60,5.00,-0.40,Weekend low usage
```

### JSON Export Format
```json
{
  "cost_summary": {
    "period": "2024-01",
    "total_cost": 156.50,
    "budget": 150.00,
    "variance": 6.50,
    "categories": {
      "infrastructure": 45.00,
      "database": 22.50,
      "apis": 78.00,
      "monitoring": 7.50,
      "storage": 3.50
    }
  },
  "daily_breakdown": [
    {
      "date": "2024-01-01",
      "infrastructure": 1.50,
      "database": 0.75,
      "apis": 2.25,
      "monitoring": 0.25,
      "storage": 0.10,
      "total": 4.85
    }
  ]
}
```

## 🎯 Cost Optimization Goals

### Monthly Targets
- [ ] Keep total costs under budget
- [ ] Reduce API costs by 10% through caching
- [ ] Optimize database queries to reduce compute costs
- [ ] Implement auto-scaling to reduce infrastructure waste
- [ ] Archive old data to reduce storage costs

### Quarterly Targets
- [ ] Achieve 15% cost reduction through optimizations
- [ ] Negotiate better rates with service providers
- [ ] Implement cost-efficient alternatives where possible
- [ ] Establish cost monitoring automation
- [ ] Create cost forecasting models

### Annual Targets
- [ ] Maintain cost growth below user growth rate
- [ ] Achieve ROI of 400%+ compared to alternative solutions
- [ ] Establish enterprise-grade cost governance
- [ ] Create cost-aware development practices
- [ ] Build cost optimization into CI/CD pipeline

---

*Use this template alongside the [Cost Analysis](../COST_ANALYSIS.md) and [Cost Optimization Guide](./COST_OPTIMIZATION_GUIDE.md) for comprehensive cost management.*
