#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Observability and Monitoring Framework
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive observability framework with OpenTelemetry, Prometheus metrics,
distributed tracing, and modern monitoring dashboards.

Features:
- OpenTelemetry tracing and metrics
- Prometheus metrics export
- Structured logging with correlation
- Health checks and readiness probes
- Performance monitoring and alerting
- Distributed tracing across services
- Custom metrics and dashboards
- Service level objectives (SLOs)
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
from contextlib import contextmanager, asynccontextmanager
import functools

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

# Prometheus client
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, CollectorRegistry
import prometheus_client

# Health check framework
import psutil
import aiohttp
from aiofiles import os as aio_os

from ..config.modern_settings import get_settings
from ..utils.structured_logging import CorrelationLogger


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_func: Callable[[], bool]
    description: str
    timeout: float = 5.0
    critical: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Custom metric definition."""
    name: str
    description: str
    metric_type: str  # counter, gauge, histogram, summary
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class ServiceLevelObjective:
    """Service Level Objective definition."""
    name: str
    description: str
    target_percentage: float  # e.g., 99.9
    measurement_window: timedelta
    metric_query: str
    alert_threshold: float = 0.95  # Alert when SLO achievement drops below this


class OpenTelemetryManager:
    """
    OpenTelemetry configuration and management.
    
    Provides centralized setup and configuration for distributed tracing
    and metrics collection using OpenTelemetry standards.
    """
    
    def __init__(self, service_name: str = "image-processing-app"):
        """Initialize OpenTelemetry manager."""
        self.service_name = service_name
        self.settings = get_settings()
        self.logger = CorrelationLogger(__name__)
        
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer = None
        self.meter = None
        
        self._setup_tracing()
        self._setup_metrics()
        self._setup_instrumentation()
    
    def _setup_tracing(self) -> None:
        """Setup distributed tracing."""
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource={
                "service.name": self.service_name,
                "service.version": self.settings.app_version,
                "service.environment": self.settings.environment
            }
        )
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        self.logger.info("OpenTelemetry tracing configured")
    
    def _setup_metrics(self) -> None:
        """Setup metrics collection."""
        # Create Prometheus metric reader
        prometheus_reader = PrometheusMetricReader()
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            metric_readers=[prometheus_reader],
            resource={
                "service.name": self.service_name,
                "service.version": self.settings.app_version,
            }
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(__name__)
        
        self.logger.info("OpenTelemetry metrics configured")
    
    def _setup_instrumentation(self) -> None:
        """Setup automatic instrumentation."""
        # Instrument common libraries
        RequestsInstrumentor().instrument()
        AsyncioInstrumentor().instrument()
        SQLite3Instrumentor().instrument()
        
        self.logger.info("OpenTelemetry auto-instrumentation configured")
    
    def create_span(self, name: str, **attributes):
        """Create a new span."""
        return self.tracer.start_span(name, attributes=attributes)
    
    def create_counter(self, name: str, description: str, unit: str = "1"):
        """Create a counter metric."""
        return self.meter.create_counter(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_histogram(self, name: str, description: str, unit: str = "ms"):
        """Create a histogram metric."""
        return self.meter.create_histogram(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_gauge(self, name: str, description: str, unit: str = "1"):
        """Create a gauge metric."""
        return self.meter.create_gauge(
            name=name,
            description=description,
            unit=unit
        )


class PrometheusMetricsCollector:
    """
    Prometheus metrics collector with custom metrics.
    
    Provides high-performance metrics collection and export
    compatible with Prometheus monitoring systems.
    """
    
    def __init__(self):
        """Initialize Prometheus metrics collector."""
        self.settings = get_settings()
        self.logger = CorrelationLogger(__name__)
        
        # Custom registry
        self.registry = CollectorRegistry()
        
        # Standard metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Application-specific metrics
        self.images_processed = Counter(
            'images_processed_total',
            'Total number of images processed',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.image_processing_duration = Histogram(
            'image_processing_duration_seconds',
            'Image processing duration in seconds',
            ['operation_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'processing_queue_size',
            'Number of items in processing queue',
            ['queue_type'],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'error_rate',
            'Current error rate',
            ['service', 'error_type'],
            registry=self.registry
        )
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self) -> None:
        """Start background system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.set(memory.percent)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.system_disk_usage.set(disk_percent)
                    
                    time.sleep(10)  # Collect every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_image_processing(self, operation_type: str, duration: float, success: bool):
        """Record image processing metrics."""
        status = "success" if success else "error"
        
        self.images_processed.labels(
            operation_type=operation_type,
            status=status
        ).inc()
        
        self.image_processing_duration.labels(
            operation_type=operation_type
        ).observe(duration)
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size metric."""
        self.queue_size.labels(queue_type=queue_type).set(size)
    
    def update_error_rate(self, service: str, error_type: str, rate: float):
        """Update error rate metric."""
        self.error_rate.labels(service=service, error_type=error_type).set(rate)
    
    def start_metrics_server(self, port: int = 8090):
        """Start Prometheus metrics HTTP server."""
        start_http_server(port, registry=self.registry)
        self.logger.info(f"Prometheus metrics server started on port {port}")


class HealthCheckManager:
    """
    Comprehensive health check management.
    
    Provides configurable health checks with different criticality levels
    and aggregated health status reporting.
    """
    
    def __init__(self):
        """Initialize health check manager."""
        self.settings = get_settings()
        self.logger = CorrelationLogger(__name__)
        
        self.checks: Dict[str, HealthCheck] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
        self._check_interval = 30.0  # seconds
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        
        def check_disk_space() -> bool:
            """Check available disk space."""
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 90  # Fail if >90% full
        
        def check_memory_usage() -> bool:
            """Check memory usage."""
            memory = psutil.virtual_memory()
            return memory.percent < 95  # Fail if >95% used
        
        def check_cpu_usage() -> bool:
            """Check CPU usage."""
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90  # Fail if >90% used
        
        # Register checks
        self.register_check(HealthCheck(
            name="disk_space",
            check_func=check_disk_space,
            description="Check available disk space",
            critical=True,
            tags={"category": "system"}
        ))
        
        self.register_check(HealthCheck(
            name="memory_usage",
            check_func=check_memory_usage,
            description="Check memory usage",
            critical=True,
            tags={"category": "system"}
        ))
        
        self.register_check(HealthCheck(
            name="cpu_usage",
            check_func=check_cpu_usage,
            description="Check CPU usage",
            critical=False,
            tags={"category": "system"}
        ))
    
    def register_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.checks[health_check.name] = health_check
        self.logger.debug(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            if name in self.last_check_results:
                del self.last_check_results[name]
            self.logger.debug(f"Unregistered health check: {name}")
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {
                "name": name,
                "status": ServiceStatus.UNKNOWN.value,
                "error": f"Health check '{name}' not found",
                "timestamp": datetime.now().isoformat()
            }
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(check.check_func),
                timeout=check.timeout
            )
            
            duration = time.time() - start_time
            
            check_result = {
                "name": name,
                "status": ServiceStatus.HEALTHY.value if result else ServiceStatus.UNHEALTHY.value,
                "description": check.description,
                "duration": duration,
                "critical": check.critical,
                "tags": check.tags,
                "timestamp": datetime.now().isoformat()
            }
            
            if not result:
                check_result["error"] = "Health check failed"
            
            return check_result
            
        except asyncio.TimeoutError:
            return {
                "name": name,
                "status": ServiceStatus.UNHEALTHY.value,
                "error": f"Health check timed out after {check.timeout}s",
                "duration": time.time() - start_time,
                "critical": check.critical,
                "tags": check.tags,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "name": name,
                "status": ServiceStatus.UNHEALTHY.value,
                "error": f"Health check failed: {str(e)}",
                "duration": time.time() - start_time,
                "critical": check.critical,
                "tags": check.tags,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = [self.run_check(name) for name in self.checks.keys()]
        check_results = await asyncio.gather(*tasks)
        
        # Organize results
        for result in check_results:
            results[result["name"]] = result
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(results)
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": self._generate_summary(results)
        }
    
    def _calculate_overall_status(self, results: Dict[str, Any]) -> str:
        """Calculate overall service status."""
        critical_failures = []
        non_critical_failures = []
        
        for result in results.values():
            if result["status"] != ServiceStatus.HEALTHY.value:
                if result.get("critical", True):
                    critical_failures.append(result["name"])
                else:
                    non_critical_failures.append(result["name"])
        
        if critical_failures:
            return ServiceStatus.UNHEALTHY.value
        elif non_critical_failures:
            return ServiceStatus.DEGRADED.value
        else:
            return ServiceStatus.HEALTHY.value
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health check summary."""
        total = len(results)
        healthy = sum(1 for r in results.values() if r["status"] == ServiceStatus.HEALTHY.value)
        unhealthy = sum(1 for r in results.values() if r["status"] == ServiceStatus.UNHEALTHY.value)
        
        return {
            "total_checks": total,
            "healthy_checks": healthy,
            "unhealthy_checks": unhealthy,
            "health_percentage": (healthy / total * 100) if total > 0 else 0
        }
    
    def start_background_checks(self) -> None:
        """Start background health check monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._background_check_loop, daemon=True)
        self._check_thread.start()
        
        self.logger.info("Background health checks started")
    
    def stop_background_checks(self) -> None:
        """Stop background health check monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        
        self.logger.info("Background health checks stopped")
    
    def _background_check_loop(self) -> None:
        """Background check execution loop."""
        while self._running:
            try:
                # Run all checks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self.run_all_checks())
                loop.close()
                
                # Store results
                self.last_check_results = results
                
                # Log any failures
                for check_name, result in results.get("checks", {}).items():
                    if result["status"] != ServiceStatus.HEALTHY.value:
                        self.logger.warning(
                            f"Health check failed: {check_name}",
                            status=result["status"],
                            error=result.get("error"),
                            critical=result.get("critical", True)
                        )
                
                time.sleep(self._check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background health check loop: {e}")
                time.sleep(self._check_interval)
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Get Kubernetes-style readiness probe status."""
        if not self.last_check_results:
            return {"status": ServiceStatus.UNKNOWN.value, "ready": False}
        
        # Service is ready if overall status is healthy or degraded
        overall_status = self.last_check_results.get("status", ServiceStatus.UNKNOWN.value)
        ready = overall_status in [ServiceStatus.HEALTHY.value, ServiceStatus.DEGRADED.value]
        
        return {
            "status": overall_status,
            "ready": ready,
            "timestamp": self.last_check_results.get("timestamp"),
            "summary": self.last_check_results.get("summary", {})
        }
    
    def get_liveness_status(self) -> Dict[str, Any]:
        """Get Kubernetes-style liveness probe status."""
        if not self.last_check_results:
            return {"status": ServiceStatus.UNKNOWN.value, "alive": False}
        
        # Service is alive if not completely unhealthy
        overall_status = self.last_check_results.get("status", ServiceStatus.UNKNOWN.value)
        alive = overall_status != ServiceStatus.UNHEALTHY.value
        
        return {
            "status": overall_status,
            "alive": alive,
            "timestamp": self.last_check_results.get("timestamp")
        }


class DistributedTraceManager:
    """
    Distributed tracing with correlation ID propagation.
    
    Provides enhanced tracing capabilities with automatic correlation
    ID propagation and context management.
    """
    
    def __init__(self, otel_manager: OpenTelemetryManager):
        """Initialize distributed trace manager."""
        self.otel_manager = otel_manager
        self.logger = CorrelationLogger(__name__)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Trace operation with automatic span management."""
        with self.otel_manager.create_span(operation_name, **attributes) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                span.end()
    
    @asynccontextmanager
    async def async_trace_operation(self, operation_name: str, **attributes):
        """Async trace operation with automatic span management."""
        with self.otel_manager.create_span(operation_name, **attributes) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                span.end()
    
    def trace_function(self, operation_name: Optional[str] = None):
        """Decorator to automatically trace function calls."""
        def decorator(func):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__qualname__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.async_trace_operation(operation_name) as span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    with self.trace_operation(operation_name) as span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        return func(*args, **kwargs)
                return wrapper
        
        return decorator


class ObservabilityManager:
    """
    Main observability manager orchestrating all monitoring components.
    
    Provides unified interface for tracing, metrics, logging, and health checks
    with modern observability patterns.
    """
    
    def __init__(self, service_name: str = "image-processing-app"):
        """Initialize observability manager."""
        self.service_name = service_name
        self.settings = get_settings()
        self.logger = CorrelationLogger(__name__)
        
        # Initialize components
        self.otel_manager = OpenTelemetryManager(service_name)
        self.metrics_collector = PrometheusMetricsCollector()
        self.health_manager = HealthCheckManager()
        self.trace_manager = DistributedTraceManager(self.otel_manager)
        
        # Service Level Objectives
        self.slos: Dict[str, ServiceLevelObjective] = {}
        
        self._setup_default_slos()
        
        self.logger.info("Observability manager initialized")
    
    def _setup_default_slos(self) -> None:
        """Setup default Service Level Objectives."""
        
        # API Response Time SLO
        self.slos["api_response_time"] = ServiceLevelObjective(
            name="API Response Time",
            description="95% of API requests should complete within 1 second",
            target_percentage=95.0,
            measurement_window=timedelta(minutes=5),
            metric_query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) < 1.0"
        )
        
        # System Availability SLO
        self.slos["system_availability"] = ServiceLevelObjective(
            name="System Availability",
            description="99.9% system uptime",
            target_percentage=99.9,
            measurement_window=timedelta(hours=24),
            metric_query="rate(up[24h]) > 0.999"
        )
        
        # Image Processing Success Rate SLO
        self.slos["processing_success_rate"] = ServiceLevelObjective(
            name="Image Processing Success Rate",
            description="99% of image processing operations should succeed",
            target_percentage=99.0,
            measurement_window=timedelta(hours=1),
            metric_query="rate(images_processed_total{status='success'}[1h]) / rate(images_processed_total[1h]) > 0.99"
        )
    
    def start_monitoring(self, metrics_port: int = 8090) -> None:
        """Start all monitoring components."""
        # Start metrics server
        self.metrics_collector.start_metrics_server(metrics_port)
        
        # Start health checks
        self.health_manager.start_background_checks()
        
        self.logger.info(f"Monitoring started - metrics on port {metrics_port}")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.health_manager.stop_background_checks()
        self.logger.info("Monitoring stopped")
    
    def register_custom_health_check(self, health_check: HealthCheck) -> None:
        """Register custom health check."""
        self.health_manager.register_check(health_check)
    
    def add_slo(self, slo: ServiceLevelObjective) -> None:
        """Add Service Level Objective."""
        self.slos[slo.name] = slo
        self.logger.info(f"Added SLO: {slo.name}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        health_status = await self.health_manager.run_all_checks()
        
        return {
            "service": self.service_name,
            "version": self.settings.app_version,
            "environment": self.settings.environment,
            "timestamp": datetime.now().isoformat(),
            "health": health_status,
            "readiness": self.health_manager.get_readiness_status(),
            "liveness": self.health_manager.get_liveness_status(),
            "slo_status": self._get_slo_status()
        }
    
    def _get_slo_status(self) -> Dict[str, Any]:
        """Get SLO compliance status."""
        # In a real implementation, this would query metrics to check SLO compliance
        # For now, return placeholder status
        slo_status = {}
        for name, slo in self.slos.items():
            slo_status[name] = {
                "target": slo.target_percentage,
                "current": 99.5,  # Placeholder
                "compliant": True,  # Placeholder
                "description": slo.description
            }
        return slo_status
    
    # Convenience methods for common operations
    def trace_operation(self, operation_name: str, **attributes):
        """Create traced operation context."""
        return self.trace_manager.trace_operation(operation_name, **attributes)
    
    def trace_function(self, operation_name: Optional[str] = None):
        """Decorator for function tracing."""
        return self.trace_manager.trace_function(operation_name)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.metrics_collector.record_request(method, endpoint, status_code, duration)
    
    def record_image_processing(self, operation_type: str, duration: float, success: bool):
        """Record image processing metrics."""
        self.metrics_collector.record_image_processing(operation_type, duration, success)


# Global observability instance
_observability: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get global observability manager instance."""
    global _observability
    if _observability is None:
        _observability = ObservabilityManager()
    return _observability


def initialize_observability(service_name: str = "image-processing-app") -> ObservabilityManager:
    """Initialize global observability manager."""
    global _observability
    _observability = ObservabilityManager(service_name)
    return _observability


# Utility decorators
def traced(operation_name: Optional[str] = None):
    """Decorator to automatically trace function execution."""
    def decorator(func):
        obs = get_observability()
        return obs.trace_function(operation_name)(func)
    return decorator


def timed_metric(metric_name: str, operation_type: str):
    """Decorator to record operation timing metrics."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obs = get_observability()
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                obs.record_image_processing(operation_type, duration, success)
        
        return wrapper
    return decorator


# Example usage and demonstration
async def demo_observability():
    """Demonstrate observability features."""
    print("?? Modern Observability Framework Demo")
    print("=" * 50)
    
    # Initialize observability
    obs = initialize_observability("demo-app")
    
    try:
        # Start monitoring
        obs.start_monitoring(metrics_port=8091)
        
        # Demo traced operation
        with obs.trace_operation("demo_operation") as span:
            span.set_attribute("demo.version", "1.0")
            await asyncio.sleep(0.1)
            print("? Traced operation completed")
        
        # Demo metrics recording
        obs.record_request("GET", "/api/status", 200, 0.05)
        obs.record_image_processing("resize", 0.15, True)
        print("? Metrics recorded")
        
        # Demo health checks
        health_status = await obs.get_service_status()
        print(f"? Service status: {health_status['health']['status']}")
        print(f"   Health checks: {health_status['health']['summary']['healthy_checks']}/{health_status['health']['summary']['total_checks']}")
        
        # Demo custom health check
        def custom_check():
            return True  # Always healthy for demo
        
        custom_health = HealthCheck(
            name="demo_check",
            check_func=custom_check,
            description="Demo health check",
            critical=False
        )
        obs.register_custom_health_check(custom_health)
        print("? Custom health check registered")
        
        print(f"\n?? Metrics server running on http://localhost:8091/metrics")
        print(f"?? Health endpoint available at /health")
        
    finally:
        obs.stop_monitoring()
    
    print("? Observability demonstration complete")


if __name__ == "__main__":
    asyncio.run(demo_observability())