#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Metrics Collection System
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive application metrics collection with automated monitoring.
"""

import time
import psutil
import threading
import json
import sqlite3
import socket
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import functools
import subprocess
import sys
import argparse


@dataclass
class PerformanceMetric:
    """Structure for performance measurements."""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    

@dataclass
class SystemMetric:
    """Structure for system metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_percent: float
    disk_used: int
    disk_total: int
    network_sent: int
    network_recv: int


class MetricsCollector:
    """Advanced application metrics collection system."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.project_root = Path(__file__).parent
        self.metrics_dir = self.project_root / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Database for persistent storage
        self.db_path = db_path or str(self.metrics_dir / "app_metrics.db")
        self.init_database()
        
        # In-memory metrics for real-time access
        self.current_metrics: Dict[str, Any] = {}
        
        # Collection control
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_interval = 5.0  # seconds
        
        # Performance tracking
        self.function_metrics: Dict[str, List[float]] = {}
        
        # Process information
        self.process = psutil.Process()
        self.start_time = datetime.now()
        
    def init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used INTEGER,
                    memory_total INTEGER,
                    disk_percent REAL,
                    disk_used INTEGER,
                    disk_total INTEGER,
                    network_sent INTEGER,
                    network_recv INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Application events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    severity TEXT DEFAULT 'info',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sys_timestamp ON system_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON app_events(timestamp)")
            
            conn.commit()
        
        print(f"? Metrics database initialized: {self.db_path}")
    
    def record_performance_metric(self, name: str, value: float, unit: str = "", 
                                tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics 
                (timestamp, metric_name, value, unit, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.timestamp,
                metric.metric_name,
                metric.value,
                metric.unit,
                json.dumps(metric.tags)
            ))
            conn.commit()
        
        # Update in-memory metrics
        self.current_metrics[name] = metric
    
    def collect_system_metrics(self) -> SystemMetric:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network usage
        network = psutil.net_io_counters()
        
        metric = SystemMetric(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_percent=disk.percent if disk else 0,
            disk_used=disk.used if disk else 0,
            disk_total=disk.total if disk else 0,
            network_sent=network.bytes_sent if network else 0,
            network_recv=network.bytes_recv if network else 0
        )
        
        return metric
    
    def store_system_metrics(self, metric: SystemMetric) -> None:
        """Store system metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used, memory_total,
                 disk_percent, disk_used, disk_total, network_sent, network_recv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp,
                metric.cpu_percent,
                metric.memory_percent,
                metric.memory_used,
                metric.memory_total,
                metric.disk_percent,
                metric.disk_used,
                metric.disk_total,
                metric.network_sent,
                metric.network_recv
            ))
            conn.commit()
    
    def log_application_event(self, event_type: str, event_data: Dict[str, Any], 
                             severity: str = "info") -> None:
        """Log application events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO app_events (timestamp, event_type, event_data, severity)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                event_type,
                json.dumps(event_data),
                severity
            ))
            conn.commit()
    
    @contextmanager
    def measure_performance(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation performance."""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record metrics
            self.record_performance_metric(
                f"{operation_name}_execution_time",
                execution_time,
                "seconds",
                tags
            )
            
            self.record_performance_metric(
                f"{operation_name}_memory_delta",
                memory_delta,
                "bytes",
                tags
            )
            
            # Log event
            self.log_application_event(
                "performance_measurement",
                {
                    "operation": operation_name,
                    "execution_time": execution_time,
                    "memory_delta": memory_delta,
                    "tags": tags or {}
                }
            )
    
    def performance_monitor(self, operation_name: str = None, tags: Optional[Dict[str, str]] = None):
        """Decorator for automatic performance monitoring."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.measure_performance(op_name, tags):
                    result = func(*args, **kwargs)
                
                # Track function call frequency
                if op_name not in self.function_metrics:
                    self.function_metrics[op_name] = []
                
                return result
            
            return wrapper
        return decorator
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous system metrics collection."""
        if self.collection_active:
            print("?? Continuous monitoring already active")
            return
        
        self.collection_active = True
        
        def collection_loop():
            while self.collection_active:
                try:
                    # Collect and store system metrics
                    sys_metrics = self.collect_system_metrics()
                    self.store_system_metrics(sys_metrics)
                    
                    # Sleep for the specified interval
                    time.sleep(self.collection_interval)
                    
                except Exception as e:
                    self.log_application_event(
                        "metrics_collection_error",
                        {"error": str(e)},
                        "error"
                    )
                    time.sleep(self.collection_interval)
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
        
        print(f"? Started continuous monitoring (interval: {self.collection_interval}s)")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        print("? Stopped continuous monitoring")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        summary = {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "performance_metrics": {},
            "system_metrics": {},
            "application_events": {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics summary
            cursor.execute("""
                SELECT metric_name, COUNT(*), AVG(value), MIN(value), MAX(value)
                FROM performance_metrics 
                WHERE timestamp > ?
                GROUP BY metric_name
            """, (cutoff_time,))
            
            perf_data = cursor.fetchall()
            for metric_name, count, avg_val, min_val, max_val in perf_data:
                summary["performance_metrics"][metric_name] = {
                    "count": count,
                    "average": round(avg_val, 4) if avg_val else 0,
                    "minimum": round(min_val, 4) if min_val else 0,
                    "maximum": round(max_val, 4) if max_val else 0
                }
            
            # System metrics summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as samples,
                    AVG(cpu_percent) as avg_cpu,
                    MAX(cpu_percent) as max_cpu,
                    AVG(memory_percent) as avg_memory,
                    MAX(memory_percent) as max_memory,
                    AVG(disk_percent) as avg_disk,
                    MAX(disk_percent) as max_disk
                FROM system_metrics 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            sys_data = cursor.fetchone()
            if sys_data and sys_data[0] > 0:
                summary["system_metrics"] = {
                    "samples": sys_data[0],
                    "cpu": {
                        "average": round(sys_data[1], 2) if sys_data[1] else 0,
                        "peak": round(sys_data[2], 2) if sys_data[2] else 0
                    },
                    "memory": {
                        "average": round(sys_data[3], 2) if sys_data[3] else 0,
                        "peak": round(sys_data[4], 2) if sys_data[4] else 0
                    },
                    "disk": {
                        "average": round(sys_data[5], 2) if sys_data[5] else 0,
                        "peak": round(sys_data[6], 2) if sys_data[6] else 0
                    }
                }
            
            # Application events summary
            cursor.execute("""
                SELECT event_type, severity, COUNT(*)
                FROM app_events 
                WHERE timestamp > ?
                GROUP BY event_type, severity
            """, (cutoff_time,))
            
            events_data = cursor.fetchall()
            for event_type, severity, count in events_data:
                if event_type not in summary["application_events"]:
                    summary["application_events"][event_type] = {}
                summary["application_events"][event_type][severity] = count
        
        return summary
    
    def export_metrics(self, output_format: str = "json", hours: int = 24) -> str:
        """Export metrics data in specified format."""
        summary = self.get_metrics_summary(hours)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format.lower() == "json":
            output_file = self.metrics_dir / f"metrics_export_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
        
        elif output_format.lower() == "csv":
            import csv
            output_file = self.metrics_dir / f"metrics_export_{timestamp}.csv"
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write performance metrics
                writer.writerow(["Type", "Metric", "Count", "Average", "Minimum", "Maximum"])
                for metric_name, data in summary["performance_metrics"].items():
                    writer.writerow([
                        "Performance", metric_name, data["count"],
                        data["average"], data["minimum"], data["maximum"]
                    ])
                
                # Write system metrics
                if summary["system_metrics"]:
                    sys_metrics = summary["system_metrics"]
                    writer.writerow(["System", "CPU Average", "", sys_metrics["cpu"]["average"], "", ""])
                    writer.writerow(["System", "CPU Peak", "", sys_metrics["cpu"]["peak"], "", ""])
                    writer.writerow(["System", "Memory Average", "", sys_metrics["memory"]["average"], "", ""])
                    writer.writerow(["System", "Memory Peak", "", sys_metrics["memory"]["peak"], "", ""])
        
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        print(f"?? Metrics exported to: {output_file}")
        return str(output_file)
    
    def generate_metrics_report(self) -> str:
        """Generate comprehensive metrics report."""
        summary = self.get_metrics_summary(24)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_file = self.metrics_dir / f"metrics_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Application Metrics Report\n\n")
            f.write(f"**Generated:** {summary['generated_at']}\n")
            f.write(f"**Period:** Last {summary['period_hours']} hours\n\n")
            
            # System Health Overview
            f.write("## System Health Overview\n\n")
            if summary["system_metrics"]:
                sys_metrics = summary["system_metrics"]
                f.write(f"- **Data Points Collected:** {sys_metrics['samples']}\n")
                f.write(f"- **CPU Usage:** {sys_metrics['cpu']['average']:.1f}% avg, {sys_metrics['cpu']['peak']:.1f}% peak\n")
                f.write(f"- **Memory Usage:** {sys_metrics['memory']['average']:.1f}% avg, {sys_metrics['memory']['peak']:.1f}% peak\n")
                f.write(f"- **Disk Usage:** {sys_metrics['disk']['average']:.1f}% avg, {sys_metrics['disk']['peak']:.1f}% peak\n\n")
                
                # Health assessment
                if sys_metrics['cpu']['peak'] > 80:
                    f.write("?? **High CPU usage detected** - consider optimization\n")
                if sys_metrics['memory']['peak'] > 80:
                    f.write("?? **High memory usage detected** - check for memory leaks\n")
                if sys_metrics['disk']['peak'] > 90:
                    f.write("?? **High disk usage detected** - consider cleanup\n")
            else:
                f.write("No system metrics data available for this period.\n")
            
            f.write("\n")
            
            # Performance Metrics
            f.write("## Performance Metrics\n\n")
            if summary["performance_metrics"]:
                f.write("| Operation | Calls | Avg Time | Min Time | Max Time |\n")
                f.write("|-----------|-------|----------|----------|----------|\n")
                
                for metric_name, data in summary["performance_metrics"].items():
                    if "execution_time" in metric_name:
                        operation = metric_name.replace("_execution_time", "")
                        f.write(f"| {operation} | {data['count']} | {data['average']:.4f}s | {data['minimum']:.4f}s | {data['maximum']:.4f}s |\n")
            else:
                f.write("No performance metrics data available for this period.\n")
            
            f.write("\n")
            
            # Application Events
            f.write("## Application Events\n\n")
            if summary["application_events"]:
                for event_type, severities in summary["application_events"].items():
                    f.write(f"### {event_type.title()}\n")
                    for severity, count in severities.items():
                        f.write(f"- **{severity.upper()}:** {count} events\n")
                    f.write("\n")
            else:
                f.write("No application events recorded for this period.\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = self._generate_recommendations(summary)
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("? No specific recommendations - system performance is healthy.\n")
        
        print(f"?? Metrics report generated: {report_file}")
        return str(report_file)
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        # System metrics recommendations
        if summary["system_metrics"]:
            sys_metrics = summary["system_metrics"]
            
            if sys_metrics["cpu"]["peak"] > 90:
                recommendations.append("Investigate high CPU usage - consider code optimization or scaling")
            elif sys_metrics["cpu"]["average"] > 70:
                recommendations.append("Monitor CPU usage trends - may need optimization")
            
            if sys_metrics["memory"]["peak"] > 90:
                recommendations.append("Critical memory usage - check for memory leaks and optimize")
            elif sys_metrics["memory"]["average"] > 80:
                recommendations.append("High memory usage - consider memory optimization")
            
            if sys_metrics["disk"]["peak"] > 95:
                recommendations.append("Critical disk space - immediate cleanup required")
            elif sys_metrics["disk"]["average"] > 85:
                recommendations.append("Monitor disk usage - plan for storage expansion")
        
        # Performance metrics recommendations
        if summary["performance_metrics"]:
            slow_operations = []
            for metric_name, data in summary["performance_metrics"].items():
                if "execution_time" in metric_name and data["average"] > 1.0:
                    operation = metric_name.replace("_execution_time", "")
                    slow_operations.append(f"{operation} ({data['average']:.2f}s avg)")
            
            if slow_operations:
                recommendations.append(f"Optimize slow operations: {', '.join(slow_operations)}")
        
        # Events recommendations
        if summary["application_events"]:
            error_count = 0
            for event_type, severities in summary["application_events"].items():
                error_count += severities.get("error", 0)
            
            if error_count > 10:
                recommendations.append(f"High error count ({error_count}) - investigate and fix recurring issues")
            elif error_count > 0:
                recommendations.append(f"Monitor application errors ({error_count} recorded)")
        
        return recommendations
    
    def setup_prometheus_integration(self) -> None:
        """Setup Prometheus metrics integration."""
        try:
            import prometheus_client
            from prometheus_client import start_http_server, Gauge, Counter, Histogram
            
            # Create Prometheus metrics
            self.prom_cpu_gauge = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
            self.prom_memory_gauge = Gauge('app_memory_usage_percent', 'Memory usage percentage')
            self.prom_requests_counter = Counter('app_requests_total', 'Total requests', ['operation'])
            self.prom_response_time = Histogram('app_response_time_seconds', 'Response time', ['operation'])
            
            # Start Prometheus HTTP server
            start_http_server(8001)
            print("? Prometheus metrics server started on port 8001")
            
            # Update Prometheus metrics from our data
            def update_prometheus_metrics():
                while self.collection_active:
                    try:
                        sys_metrics = self.collect_system_metrics()
                        self.prom_cpu_gauge.set(sys_metrics.cpu_percent)
                        self.prom_memory_gauge.set(sys_metrics.memory_percent)
                        time.sleep(10)  # Update every 10 seconds
                    except Exception as e:
                        print(f"Error updating Prometheus metrics: {e}")
                        time.sleep(10)
            
            prometheus_thread = threading.Thread(target=update_prometheus_metrics, daemon=True)
            prometheus_thread.start()
            
        except ImportError:
            print("?? Prometheus client not available. Install with: pip install prometheus-client")


def main():
    """Main entry point for metrics collection."""
    parser = argparse.ArgumentParser(description="Application metrics collection system")
    parser.add_argument("--start", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring")
    parser.add_argument("--report", action="store_true", help="Generate metrics report")
    parser.add_argument("--export", choices=["json", "csv"], help="Export metrics data")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to include")
    parser.add_argument("--prometheus", action="store_true", help="Enable Prometheus integration")
    
    args = parser.parse_args()
    
    collector = MetricsCollector()
    
    if args.prometheus:
        collector.setup_prometheus_integration()
    
    if args.start:
        collector.start_continuous_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            collector.stop_continuous_monitoring()
    
    elif args.stop:
        collector.stop_continuous_monitoring()
    
    elif args.report:
        collector.generate_metrics_report()
    
    elif args.export:
        collector.export_metrics(args.export, args.hours)
    
    elif len(sys.argv) == 1:
        # Default: show current metrics summary
        summary = collector.get_metrics_summary(args.hours)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()