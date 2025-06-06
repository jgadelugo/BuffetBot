"""
ML Cost Monitor - Track and alert on ML service costs
Prevents unexpected charges and provides cost visibility
"""
import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class CostAlert:
    """Cost alert configuration"""

    threshold: float
    alert_type: str  # 'daily', 'monthly', 'total'
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class CostEntry:
    """Individual cost tracking entry"""

    timestamp: datetime
    service: str
    operation: str
    cost: float
    details: dict[str, any]


class MLCostMonitor:
    """Monitor and track ML service costs"""

    def __init__(self):
        self.logger = logging.getLogger("ml.cost_monitor")
        self.cost_entries: list[CostEntry] = []
        self.alerts: list[CostAlert] = [
            CostAlert(threshold=25.0, alert_type="daily"),
            CostAlert(threshold=100.0, alert_type="daily"),
            CostAlert(threshold=500.0, alert_type="monthly"),
        ]
        self.monitoring_active = False
        self._monitor_thread = None

    def start_monitoring(self) -> None:
        """Start cost monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.logger.info("✅ Cost monitoring started")

    def stop_monitoring(self) -> None:
        """Stop cost monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("✅ Cost monitoring stopped")

    def track_cost(
        self, service: str, operation: str, cost: float, details: dict[str, any] = None
    ):
        """Track a cost-incurring operation"""
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            service=service,
            operation=operation,
            cost=cost,
            details=details or {},
        )

        self.cost_entries.append(entry)
        self.logger.info(f"Cost tracked: {service}.{operation} = ${cost:.4f}")

        # Check alerts
        self._check_alerts()

    def get_daily_cost(self, date: datetime = None) -> float:
        """Get total cost for a specific day"""
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        return sum(
            entry.cost
            for entry in self.cost_entries
            if start_of_day <= entry.timestamp < end_of_day
        )

    def get_monthly_cost(self, year: int = None, month: int = None) -> float:
        """Get total cost for a specific month"""
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month

        return sum(
            entry.cost
            for entry in self.cost_entries
            if entry.timestamp.year == year and entry.timestamp.month == month
        )

    def get_total_cost(self) -> float:
        """Get total cost across all time"""
        return sum(entry.cost for entry in self.cost_entries)

    def get_service_breakdown(self, days: int = 7) -> dict[str, float]:
        """Get cost breakdown by service for recent days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        service_costs = {}

        for entry in self.cost_entries:
            if entry.timestamp >= cutoff:
                service_costs[entry.service] = (
                    service_costs.get(entry.service, 0) + entry.cost
                )

        return service_costs

    def get_operation_breakdown(
        self, service: str = None, days: int = 7
    ) -> dict[str, float]:
        """Get cost breakdown by operation"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        operation_costs = {}

        for entry in self.cost_entries:
            if entry.timestamp >= cutoff:
                if service is None or entry.service == service:
                    key = f"{entry.service}.{entry.operation}"
                    operation_costs[key] = operation_costs.get(key, 0) + entry.cost

        return operation_costs

    def get_cost_trend(self, days: int = 30) -> list[dict[str, any]]:
        """Get daily cost trend for specified number of days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        daily_costs = {}

        for entry in self.cost_entries:
            if entry.timestamp >= cutoff:
                day_key = entry.timestamp.strftime("%Y-%m-%d")
                daily_costs[day_key] = daily_costs.get(day_key, 0) + entry.cost

        # Fill in missing days with 0
        trend = []
        for i in range(days):
            date = cutoff + timedelta(days=i)
            day_key = date.strftime("%Y-%m-%d")
            trend.append({"date": day_key, "cost": daily_costs.get(day_key, 0.0)})

        return trend

    def add_alert(self, threshold: float, alert_type: str) -> None:
        """Add a new cost alert"""
        alert = CostAlert(threshold=threshold, alert_type=alert_type)
        self.alerts.append(alert)
        self.logger.info(f"Added cost alert: {alert_type} threshold ${threshold:.2f}")

    def remove_alert(self, threshold: float, alert_type: str) -> bool:
        """Remove a cost alert"""
        for i, alert in enumerate(self.alerts):
            if alert.threshold == threshold and alert.alert_type == alert_type:
                del self.alerts[i]
                self.logger.info(
                    f"Removed cost alert: {alert_type} threshold ${threshold:.2f}"
                )
                return True
        return False

    def enable_alert(self, threshold: float, alert_type: str) -> bool:
        """Enable a specific alert"""
        for alert in self.alerts:
            if alert.threshold == threshold and alert.alert_type == alert_type:
                alert.enabled = True
                return True
        return False

    def disable_alert(self, threshold: float, alert_type: str) -> bool:
        """Disable a specific alert"""
        for alert in self.alerts:
            if alert.threshold == threshold and alert.alert_type == alert_type:
                alert.enabled = False
                return True
        return False

    def health_check(self) -> str:
        """Check cost monitor health"""
        try:
            if not self.monitoring_active:
                return "not_active"

            # Check if we can calculate costs
            total_cost = self.get_total_cost()
            daily_cost = self.get_daily_cost()

            return "healthy"
        except Exception as e:
            self.logger.error(f"Cost monitor health check failed: {e}")
            return "error"

    def _check_alerts(self):
        """Check if any cost thresholds have been exceeded"""
        for alert in self.alerts:
            if not alert.enabled:
                continue

            current_cost = 0.0
            if alert.alert_type == "daily":
                current_cost = self.get_daily_cost()
            elif alert.alert_type == "monthly":
                current_cost = self.get_monthly_cost()
            elif alert.alert_type == "total":
                current_cost = self.get_total_cost()

            if current_cost >= alert.threshold:
                self._trigger_alert(alert, current_cost)

    def _trigger_alert(self, alert: CostAlert, current_cost: float):
        """Trigger a cost alert"""
        # Avoid spam - only alert once per hour
        if (
            alert.last_triggered
            and datetime.utcnow() - alert.last_triggered < timedelta(hours=1)
        ):
            return

        self.logger.warning(
            f"🚨 ML COST ALERT: {alert.alert_type} spending ${current_cost:.2f} "
            f"exceeds threshold ${alert.threshold:.2f}"
        )

        alert.last_triggered = datetime.utcnow()

    def get_cost_summary(self) -> dict[str, any]:
        """Get comprehensive cost summary"""
        return {
            "monitoring_active": self.monitoring_active,
            "total_cost": self.get_total_cost(),
            "daily_cost": self.get_daily_cost(),
            "monthly_cost": self.get_monthly_cost(),
            "service_breakdown": self.get_service_breakdown(),
            "operation_breakdown": self.get_operation_breakdown(),
            "total_operations": len(self.cost_entries),
            "active_alerts": len([a for a in self.alerts if a.enabled]),
            "cost_trend_7_days": self.get_cost_trend(7),
        }

    def reset_costs(self) -> None:
        """Reset all cost tracking (use with caution)"""
        self.cost_entries.clear()
        self.logger.warning("⚠️ All cost tracking data has been reset")

    def export_costs(
        self, start_date: datetime = None, end_date: datetime = None
    ) -> list[dict[str, any]]:
        """Export cost data for analysis"""
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        filtered_entries = [
            entry
            for entry in self.cost_entries
            if start_date <= entry.timestamp <= end_date
        ]

        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "service": entry.service,
                "operation": entry.operation,
                "cost": entry.cost,
                "details": entry.details,
            }
            for entry in filtered_entries
        ]
