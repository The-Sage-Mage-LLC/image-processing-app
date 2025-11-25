#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitoring Dashboard
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Real-time performance monitoring dashboard with metrics collection,
visualization, and alerting capabilities for enterprise-grade monitoring.

Features:
- Real-time metrics collection and aggregation
- Performance trend analysis and alerting
- Resource utilization monitoring
- Custom metric definitions and thresholds
- Export capabilities for external monitoring systems
"""

import asyncio
import time
import threading
import json
import sqlite3
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import psutil
import logging

# Import our structured logging system
from ..utils.structured_logging import CorrelationLogger, with_correlation


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Current value at point in time
    HISTOGRAM = "histogram"       # Distribution of values
    TIMING = "timing"            # Duration measurements
    RATE = "rate"               # Events per unit time


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric measurement point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    """Performance alert definition and state."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 80", "< 0.1"
    threshold: float
    severity: AlertSeverity
    description: str
    enabled: bool = True
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot at a point in time."""
    timestamp: datetime
    system_metrics: Dict[str, float]
    application_metrics: Dict[str, float]
    custom_metrics: Dict[str, float]
    active_alerts: List[str]


class MetricsCollector:
    """
    High-performance metrics collection system.
    
    This class efficiently collects, aggregates, and stores performance
    metrics with minimal overhead to the application being monitored.
    
    Features:
    - Lock-free metric collection using queues
    - Batch processing for database efficiency
    - Configurable collection intervals
    - Memory-bounded metric storage
    - Thread-safe operation
    """
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database for metric storage
        """
        self.db_path = Path(db_path)
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.running = False
        self.collector_thread: Optional[threading.Thread] = None
        self.batch_size = 100
        self.flush_interval = 5.0  # seconds
        self.logger = CorrelationLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Metric aggregation windows (in seconds)
        self.aggregation_windows = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
        
        self.logger.info("Metrics collector initialized", db_path=str(self.db_path))
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metric storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    window_seconds INTEGER NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    avg_value REAL,
                    count INTEGER,
                    sum_value REAL,
                    percentile_50 REAL,
                    percentile_95 REAL,
                    percentile_99 REAL,
                    window_start REAL NOT NULL,
                    window_end REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregates_name_window 
                ON metric_aggregates(name, window_seconds, window_start)
            """)
    
    def start(self) -> None:
        """Start the metrics collection background thread."""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        self.collector_thread.start()
        self.logger.info("Metrics collection started")
    
    def stop(self) -> None:
        """Stop metrics collection and flush remaining metrics."""
        if not self.running:
            return
        
        self.running = False
        
        if self.collector_thread:
            self.collector_thread.join(timeout=10.0)
        
        # Flush any remaining metrics
        self._flush_metrics()
        self.logger.info("Metrics collection stopped")
    
    def record_metric(self, 
                     name: str, 
                     value: float,
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name (e.g., 'app.processing_time')
            value: Metric value
            metric_type: Type of metric
            tags: Additional tags for the metric
        """
        try:
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics_queue.put_nowait(metric)
            
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping metric", 
                               metric_name=name)
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        last_flush = time.time()
        batch = []
        
        while self.running:
            try:
                # Collect metrics with timeout
                try:
                    metric = self.metrics_queue.get(timeout=1.0)
                    batch.append(metric)
                except queue.Empty:
                    pass
                
                # Check if we should flush
                now = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (now - last_flush) >= self.flush_interval)
                )
                
                if should_flush:
                    self._store_batch(batch)
                    batch.clear()
                    last_flush = now
                    
            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))
                time.sleep(1.0)
        
        # Final flush
        if batch:
            self._store_batch(batch)
    
    def _store_batch(self, batch: List[MetricPoint]) -> None:
        """Store a batch of metrics to database."""
        if not batch:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """INSERT INTO metrics 
                       (name, value, timestamp, metric_type, tags) 
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (
                            m.name,
                            m.value,
                            m.timestamp.timestamp(),
                            m.metric_type.value,
                            json.dumps(m.tags)
                        )
                        for m in batch
                    ]
                )
                
            self.logger.debug(f"Stored {len(batch)} metrics to database")
            
        except Exception as e:
            self.logger.error("Error storing metrics batch", 
                             error=str(e), batch_size=len(batch))
    
    def _flush_metrics(self) -> None:
        """Flush any remaining metrics in the queue."""
        batch = []
        try:
            while True:
                metric = self.metrics_queue.get_nowait()
                batch.append(metric)
        except queue.Empty:
            pass
        
        if batch:
            self._store_batch(batch)
            self.logger.info(f"Flushed {len(batch)} remaining metrics")
    
    def get_recent_metrics(self, 
                          metric_name: str,
                          duration_minutes: int = 60) -> List[MetricPoint]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT name, value, timestamp, metric_type, tags
                   FROM metrics 
                   WHERE name = ? AND timestamp >= ?
                   ORDER BY timestamp DESC""",
                (metric_name, cutoff_time.timestamp())
            )
            
            results = []
            for row in cursor.fetchall():
                results.append(MetricPoint(
                    name=row[0],
                    value=row[1],
                    timestamp=datetime.fromtimestamp(row[2]),
                    tags=json.loads(row[4]) if row[4] else {},
                    metric_type=MetricType(row[3])
                ))
            
            return results
    
    def get_metric_summary(self, 
                          metric_name: str,
                          duration_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary for a metric over a time period."""
        metrics = self.get_recent_metrics(metric_name, duration_minutes)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[0],  # Most recent value
            "oldest": values[-1]   # Oldest value in range
        }


class SystemMetricsCollector:
    """Collector for system-level performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize system metrics collector."""
        self.metrics_collector = metrics_collector
        self.running = False
        self.collection_interval = 10.0  # seconds
        self.collector_thread: Optional[threading.Thread] = None
        self.logger = CorrelationLogger(__name__)
    
    def start(self) -> None:
        """Start system metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(
            target=self._collection_loop,
            name="SystemMetricsCollector",
            daemon=True
        )
        self.collector_thread.start()
        self.logger.info("System metrics collection started")
    
    def stop(self) -> None:
        """Stop system metrics collection."""
        if not self.running:
            return
        
        self.running = False
        
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        
        self.logger.info("System metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main system metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error("Error collecting system metrics", error=str(e))
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.metrics_collector.record_metric(
                "system.cpu.percent", cpu_percent, MetricType.GAUGE
            )
            
            cpu_count = psutil.cpu_count()
            self.metrics_collector.record_metric(
                "system.cpu.count", cpu_count, MetricType.GAUGE
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(
                "system.memory.percent", memory.percent, MetricType.GAUGE
            )
            self.metrics_collector.record_metric(
                "system.memory.total_gb", memory.total / (1024**3), MetricType.GAUGE
            )
            self.metrics_collector.record_metric(
                "system.memory.available_gb", memory.available / (1024**3), MetricType.GAUGE
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric(
                "system.disk.percent", disk_percent, MetricType.GAUGE
            )
            self.metrics_collector.record_metric(
                "system.disk.free_gb", disk.free / (1024**3), MetricType.GAUGE
            )
            
            # Network metrics
            network = psutil.net_io_counters()
            if network:
                self.metrics_collector.record_metric(
                    "system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER
                )
                self.metrics_collector.record_metric(
                    "system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER
                )
            
            # Process metrics
            current_process = psutil.Process()
            self.metrics_collector.record_metric(
                "process.memory_mb", 
                current_process.memory_info().rss / (1024**2), 
                MetricType.GAUGE
            )
            self.metrics_collector.record_metric(
                "process.cpu_percent", 
                current_process.cpu_percent(), 
                MetricType.GAUGE
            )
            self.metrics_collector.record_metric(
                "process.threads", 
                current_process.num_threads(), 
                MetricType.GAUGE
            )
            
        except Exception as e:
            self.logger.error("Error in system metrics collection", error=str(e))


class AlertManager:
    """
    Performance alert management system.
    
    Monitors metrics against defined thresholds and triggers alerts
    when conditions are met. Supports multiple severity levels and
    alert state management.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager."""
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.running = False
        self.check_interval = 30.0  # seconds
        self.alert_thread: Optional[threading.Thread] = None
        self.logger = CorrelationLogger(__name__)
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Setup default performance alerts."""
        default_alerts = [
            Alert(
                name="high_cpu_usage",
                metric_name="system.cpu.percent",
                condition="> 80",
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                description="CPU usage above 80%"
            ),
            Alert(
                name="critical_cpu_usage",
                metric_name="system.cpu.percent",
                condition="> 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="CPU usage above 95%"
            ),
            Alert(
                name="high_memory_usage",
                metric_name="system.memory.percent",
                condition="> 85",
                threshold=85.0,
                severity=AlertSeverity.HIGH,
                description="Memory usage above 85%"
            ),
            Alert(
                name="critical_disk_usage",
                metric_name="system.disk.percent",
                condition="> 90",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                description="Disk usage above 90%"
            )
        ]
        
        for alert in default_alerts:
            self.add_alert(alert)
    
    def add_alert(self, alert: Alert) -> None:
        """Add an alert definition."""
        self.alerts[alert.name] = alert
        self.logger.info(f"Added alert: {alert.name}", 
                        metric=alert.metric_name, 
                        threshold=alert.threshold)
    
    def remove_alert(self, alert_name: str) -> None:
        """Remove an alert definition."""
        if alert_name in self.alerts:
            del self.alerts[alert_name]
            self.logger.info(f"Removed alert: {alert_name}")
    
    def start(self) -> None:
        """Start alert monitoring."""
        if self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AlertManager",
            daemon=True
        )
        self.alert_thread.start()
        self.logger.info("Alert monitoring started")
    
    def stop(self) -> None:
        """Stop alert monitoring."""
        if not self.running:
            return
        
        self.running = False
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        self.logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error("Error in alert monitoring loop", error=str(e))
                time.sleep(self.check_interval)
    
    def _check_alerts(self) -> None:
        """Check all alerts against current metrics."""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            try:
                self._check_single_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Error checking alert {alert.name}", error=str(e))
    
    def _check_single_alert(self, alert: Alert) -> None:
        """Check a single alert against current metrics."""
        # Get recent metric values
        recent_metrics = self.metrics_collector.get_recent_metrics(
            alert.metric_name, duration_minutes=5
        )
        
        if not recent_metrics:
            return
        
        # Get latest value
        latest_value = recent_metrics[0].value
        
        # Check condition
        is_triggered = self._evaluate_condition(
            latest_value, alert.condition, alert.threshold
        )
        
        # Manage alert state
        was_triggered = alert.triggered_at is not None and alert.resolved_at is None
        
        if is_triggered and not was_triggered:
            # New alert triggered
            alert.triggered_at = datetime.now()
            alert.resolved_at = None
            
            self._fire_alert(alert, latest_value)
            
        elif not is_triggered and was_triggered:
            # Alert resolved
            alert.resolved_at = datetime.now()
            
            self._resolve_alert(alert, latest_value)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition against current value."""
        condition = condition.strip()
        
        if condition.startswith('>'):
            return value > threshold
        elif condition.startswith('<'):
            return value < threshold
        elif condition.startswith('>='):
            return value >= threshold
        elif condition.startswith('<='):
            return value <= threshold
        elif condition.startswith('=='):
            return abs(value - threshold) < 0.001  # Float comparison
        elif condition.startswith('!='):
            return abs(value - threshold) >= 0.001
        
        return False
    
    def _fire_alert(self, alert: Alert, current_value: float) -> None:
        """Fire an alert notification."""
        alert_data = {
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "current_value": current_value,
            "threshold": alert.threshold,
            "severity": alert.severity.value,
            "description": alert.description,
            "triggered_at": alert.triggered_at.isoformat(),
            "action": "triggered"
        }
        
        self.alert_history.append(alert_data)
        
        self.logger.warning(
            f"ALERT TRIGGERED: {alert.name}",
            **alert_data
        )
        
        # Record alert metric
        self.metrics_collector.record_metric(
            f"alerts.triggered.{alert.name}",
            1.0,
            MetricType.COUNTER,
            tags={"severity": alert.severity.value}
        )
    
    def _resolve_alert(self, alert: Alert, current_value: float) -> None:
        """Resolve an alert notification."""
        duration = (alert.resolved_at - alert.triggered_at).total_seconds()
        
        alert_data = {
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "current_value": current_value,
            "threshold": alert.threshold,
            "severity": alert.severity.value,
            "description": alert.description,
            "resolved_at": alert.resolved_at.isoformat(),
            "duration_seconds": duration,
            "action": "resolved"
        }
        
        self.alert_history.append(alert_data)
        
        self.logger.info(
            f"ALERT RESOLVED: {alert.name}",
            **alert_data
        )
        
        # Record alert resolution metric
        self.metrics_collector.record_metric(
            f"alerts.resolved.{alert.name}",
            1.0,
            MetricType.COUNTER,
            tags={"severity": alert.severity.value}
        )
        
        self.metrics_collector.record_metric(
            f"alerts.duration.{alert.name}",
            duration,
            MetricType.TIMING,
            tags={"severity": alert.severity.value}
        )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        active = []
        
        for alert in self.alerts.values():
            if alert.triggered_at and not alert.resolved_at:
                active.append({
                    "name": alert.name,
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "description": alert.description,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "duration_seconds": (datetime.now() - alert.triggered_at).total_seconds()
                })
        
        return active


class PerformanceDashboard:
    """
    Main performance monitoring dashboard interface.
    
    Provides a unified interface for metrics collection, alerting,
    and performance data visualization with real-time updates.
    """
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        """Initialize performance dashboard."""
        self.metrics_collector = MetricsCollector(db_path)
        self.system_collector = SystemMetricsCollector(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.logger = CorrelationLogger(__name__)
        
        self.logger.info("Performance dashboard initialized")
    
    def start(self) -> None:
        """Start all monitoring components."""
        self.metrics_collector.start()
        self.system_collector.start()
        self.alert_manager.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop all monitoring components."""
        self.alert_manager.stop()
        self.system_collector.stop()
        self.metrics_collector.stop()
        
        self.logger.info("Performance monitoring stopped")
    
    @with_correlation("dashboard.record_metric")
    def record_metric(self, 
                     name: str, 
                     value: float,
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom application metric."""
        self.metrics_collector.record_metric(name, value, metric_type, tags)
    
    @with_correlation("dashboard.record_timing")
    def record_timing(self, operation_name: str, duration_ms: float) -> None:
        """Record operation timing metric."""
        self.record_metric(
            f"app.timing.{operation_name}",
            duration_ms,
            MetricType.TIMING,
            tags={"operation": operation_name}
        )
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        # System metrics summary
        system_metrics = {}
        for metric_name in [
            "system.cpu.percent",
            "system.memory.percent", 
            "system.disk.percent",
            "process.memory_mb",
            "process.cpu_percent"
        ]:
            summary = self.metrics_collector.get_metric_summary(metric_name, 5)
            if summary:
                system_metrics[metric_name] = summary.get("latest", 0.0)
        
        # Application metrics summary
        app_metrics = {}
        recent_app_metrics = self.metrics_collector.get_recent_metrics("app.*", 15)
        for metric in recent_app_metrics[:10]:  # Last 10 app metrics
            app_metrics[metric.name] = metric.value
        
        # Active alerts
        active_alerts = [alert["name"] for alert in self.alert_manager.get_active_alerts()]
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            system_metrics=system_metrics,
            application_metrics=app_metrics,
            custom_metrics={},
            active_alerts=active_alerts
        )
    
    def get_dashboard_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive dashboard data for visualization."""
        return {
            "snapshot": asdict(self.get_performance_snapshot()),
            "alerts": {
                "active": self.alert_manager.get_active_alerts(),
                "recent": self.alert_manager.alert_history[-50:]  # Last 50 alerts
            },
            "metrics_summary": {
                "cpu": self.metrics_collector.get_metric_summary("system.cpu.percent", hours * 60),
                "memory": self.metrics_collector.get_metric_summary("system.memory.percent", hours * 60),
                "disk": self.metrics_collector.get_metric_summary("system.disk.percent", hours * 60),
                "process_memory": self.metrics_collector.get_metric_summary("process.memory_mb", hours * 60)
            },
            "health_status": {
                "collectors_running": all([
                    self.metrics_collector.running,
                    self.system_collector.running,
                    self.alert_manager.running
                ]),
                "metrics_count": len(self.metrics_collector.get_recent_metrics("*", hours * 60)),
                "active_alert_count": len(self.alert_manager.get_active_alerts())
            }
        }


# Example usage and demo
async def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    dashboard = PerformanceDashboard()
    
    try:
        print("?? Starting Performance Monitoring Dashboard...")
        dashboard.start()
        
        # Simulate some application metrics
        print("?? Recording sample metrics...")
        for i in range(10):
            # Simulate processing time
            dashboard.record_timing("image_processing", 50 + i * 10)
            
            # Simulate request count
            dashboard.record_metric(
                "app.requests.total",
                i + 1,
                MetricType.COUNTER,
                tags={"endpoint": "process_image"}
            )
            
            # Simulate error rate
            dashboard.record_metric(
                "app.errors.rate",
                0.02 if i % 5 == 0 else 0.0,
                MetricType.GAUGE
            )
            
            await asyncio.sleep(1)
        
        print("?? Getting performance snapshot...")
        snapshot = dashboard.get_performance_snapshot()
        
        print(f"System Metrics:")
        for name, value in snapshot.system_metrics.items():
            print(f"  {name}: {value:.2f}")
        
        print(f"\nActive Alerts: {len(snapshot.active_alerts)}")
        for alert in snapshot.active_alerts:
            print(f"  - {alert}")
        
        print("\n? Performance monitoring demonstration complete")
        
    finally:
        print("?? Stopping performance monitoring...")
        dashboard.stop()


if __name__ == "__main__":
    asyncio.run(demo_performance_monitoring())