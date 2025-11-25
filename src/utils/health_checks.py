#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Check Endpoints Implementation
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive health check system with multiple endpoint types for
enterprise-grade monitoring, alerting, and automated recovery.

Features:
- Basic health checks for load balancers
- Detailed component health monitoring
- Readiness and liveness probes for Kubernetes
- Diagnostic endpoints for troubleshooting
- Performance metrics integration
- Dependency health monitoring
- Graceful degradation support
"""

import asyncio
import time
import json
import sqlite3
import threading
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Import our structured logging system
from .structured_logging import CorrelationLogger, with_correlation, CorrelationContext


class HealthStatus(Enum):
    """Health status enumeration with clear semantics."""
    HEALTHY = "healthy"        # Component is fully operational
    DEGRADED = "degraded"      # Component is operational with reduced functionality  
    UNHEALTHY = "unhealthy"    # Component is not operational
    UNKNOWN = "unknown"        # Component status cannot be determined


@dataclass
class ComponentHealth:
    """
    Health information for a single system component.
    
    This class standardizes health reporting across all system components,
    enabling consistent monitoring and alerting capabilities.
    
    Attributes:
        name: Component identifier (e.g., "database", "file_system")
        status: Current health status
        message: Human-readable status description
        details: Component-specific diagnostic information
        last_checked: Timestamp of last health check
        response_time_ms: Health check execution time
        dependencies: List of dependent component names
        tags: Additional metadata tags
    """
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_checked: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    dependencies: Optional[List[str]] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class SystemHealth:
    """
    Overall system health aggregation.
    
    This class provides a comprehensive view of system health by
    aggregating individual component health status and computing
    overall system health metrics.
    """
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str
    environment: str
    components: Dict[str, ComponentHealth]
    metrics: Optional[Dict[str, Any]] = None
    dependencies_healthy: Optional[bool] = None
    degraded_components: Optional[List[str]] = None
    unhealthy_components: Optional[List[str]] = None


class HealthChecker(ABC):
    """
    Abstract base class for health checkers.
    
    This interface defines the contract for all health checking implementations,
    ensuring consistency and enabling pluggable health monitoring.
    
    Implementation Guidelines:
    - Health checks should complete within timeout_seconds
    - Checks should be idempotent and safe to run frequently
    - Expensive operations should be cached with appropriate TTL
    - Failures should be logged with correlation context
    - Dependencies should be clearly documented
    """
    
    @abstractmethod
    async def check_health(self) -> ComponentHealth:
        """
        Perform health check and return component status.
        
        Returns:
            ComponentHealth instance with current status
            
        Raises:
            Exception: If health check cannot be performed
        """
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return the unique component name."""
        pass
    
    @property
    def timeout_seconds(self) -> float:
        """Return the health check timeout (default: 5 seconds)."""
        return 5.0
    
    @property
    def dependencies(self) -> List[str]:
        """Return list of component dependencies."""
        return []
    
    @property
    def tags(self) -> Dict[str, str]:
        """Return component metadata tags."""
        return {}


class DatabaseHealthChecker(HealthChecker):
    """
    Health checker for SQLite database connectivity and performance.
    
    This checker validates:
    - Database file accessibility
    - Connection establishment
    - Basic query execution
    - Database schema integrity
    - Performance metrics
    
    Performance Thresholds:
    - Response time < 100ms: Healthy
    - Response time < 500ms: Degraded
    - Response time >= 500ms: Unhealthy
    """
    
    def __init__(self, db_path: str, schema_tables: Optional[List[str]] = None):
        """
        Initialize database health checker.
        
        Args:
            db_path: Path to SQLite database file
            schema_tables: Expected tables for schema validation
        """
        self.db_path = Path(db_path)
        self.schema_tables = schema_tables or []
        self.logger = CorrelationLogger(__name__)
    
    @property
    def component_name(self) -> str:
        return "database"
    
    @property
    def timeout_seconds(self) -> float:
        return 10.0  # Database checks may take longer
    
    @property
    def tags(self) -> Dict[str, str]:
        return {
            "database_type": "sqlite",
            "database_path": str(self.db_path),
            "schema_validation": str(bool(self.schema_tables))
        }
    
    @with_correlation("health_check.database")
    async def check_health(self) -> ComponentHealth:
        """Perform comprehensive database health check."""
        start_time = time.perf_counter()
        
        try:
            # Check file accessibility
            if not self.db_path.exists():
                return ComponentHealth(
                    name=self.component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Database file not found: {self.db_path}",
                    last_checked=datetime.now(),
                    tags=self.tags
                )
            
            # Check file permissions
            if not os.access(self.db_path, os.R_OK | os.W_OK):
                return ComponentHealth(
                    name=self.component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Database file not accessible: {self.db_path}",
                    last_checked=datetime.now(),
                    tags=self.tags
                )
            
            # Test database connectivity and basic operations
            health_details = await self._check_database_operations()
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Determine status based on response time
            if response_time < 100:
                status = HealthStatus.HEALTHY
                message = "Database is responsive"
            elif response_time < 500:
                status = HealthStatus.DEGRADED
                message = f"Database is slow (response time: {response_time:.1f}ms)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Database is very slow (response time: {response_time:.1f}ms)"
            
            # Override status if operations failed
            if not health_details.get("connection_test", False):
                status = HealthStatus.UNHEALTHY
                message = "Database connection failed"
            
            self.logger.info(f"Database health check completed",
                           metrics={
                               "response_time_ms": response_time,
                               "status": status.value
                           })
            
            return ComponentHealth(
                name=self.component_name,
                status=status,
                message=message,
                details=health_details,
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Database health check failed: {str(e)}")
            
            return ComponentHealth(
                name=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check error: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
    
    async def _check_database_operations(self) -> Dict[str, Any]:
        """Perform detailed database operations check."""
        details = {
            "connection_test": False,
            "query_test": False,
            "schema_validation": False,
            "database_size_mb": 0,
            "table_count": 0,
            "error": None
        }
        
        try:
            # Test basic connectivity
            with sqlite3.connect(str(self.db_path), timeout=5.0) as conn:
                details["connection_test"] = True
                
                # Test basic query
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                details["query_test"] = result is not None and result[0] == 1
                
                # Get database metadata
                details["database_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Get table count
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                details["table_count"] = cursor.fetchone()[0]
                
                # Validate schema if tables specified
                if self.schema_tables:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    existing_tables = {row[0] for row in cursor.fetchall()}
                    
                    missing_tables = set(self.schema_tables) - existing_tables
                    details["schema_validation"] = len(missing_tables) == 0
                    
                    if missing_tables:
                        details["missing_tables"] = list(missing_tables)
                else:
                    details["schema_validation"] = True
                
        except Exception as e:
            details["error"] = str(e)
            self.logger.error(f"Database operation failed: {str(e)}")
        
        return details


class SystemResourcesHealthChecker(HealthChecker):
    """
    Health checker for system resource utilization.
    
    Monitors:
    - CPU utilization percentage
    - Memory utilization percentage
    - Disk space utilization
    - Process count and limits
    - Network interface status
    
    Thresholds:
    - CPU/Memory < 70%: Healthy
    - CPU/Memory 70-85%: Degraded
    - CPU/Memory > 85%: Unhealthy
    - Disk space > 90%: Unhealthy regardless of other metrics
    """
    
    def __init__(self, 
                 cpu_threshold_degraded: float = 70.0,
                 cpu_threshold_unhealthy: float = 85.0,
                 memory_threshold_degraded: float = 70.0,
                 memory_threshold_unhealthy: float = 85.0,
                 disk_threshold_unhealthy: float = 90.0):
        """
        Initialize system resources health checker.
        
        Args:
            cpu_threshold_degraded: CPU usage % for degraded status
            cpu_threshold_unhealthy: CPU usage % for unhealthy status
            memory_threshold_degraded: Memory usage % for degraded status
            memory_threshold_unhealthy: Memory usage % for unhealthy status
            disk_threshold_unhealthy: Disk usage % for unhealthy status
        """
        self.cpu_threshold_degraded = cpu_threshold_degraded
        self.cpu_threshold_unhealthy = cpu_threshold_unhealthy
        self.memory_threshold_degraded = memory_threshold_degraded
        self.memory_threshold_unhealthy = memory_threshold_unhealthy
        self.disk_threshold_unhealthy = disk_threshold_unhealthy
        self.logger = CorrelationLogger(__name__)
    
    @property
    def component_name(self) -> str:
        return "system_resources"
    
    @property
    def timeout_seconds(self) -> float:
        return 5.0
    
    @property
    def tags(self) -> Dict[str, str]:
        return {
            "monitor_type": "system_resources",
            "cpu_threshold_degraded": str(self.cpu_threshold_degraded),
            "memory_threshold_degraded": str(self.memory_threshold_degraded),
            "disk_threshold_unhealthy": str(self.disk_threshold_unhealthy)
        }
    
    @with_correlation("health_check.system_resources")
    async def check_health(self) -> ComponentHealth:
        """Perform comprehensive system resources health check."""
        start_time = time.perf_counter()
        
        try:
            # Collect system metrics
            metrics = await self._collect_system_metrics()
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Determine overall status
            status, issues = self._evaluate_system_status(metrics)
            
            # Generate status message
            if status == HealthStatus.HEALTHY:
                message = "System resources are operating normally"
            elif status == HealthStatus.DEGRADED:
                message = f"System resources are degraded: {'; '.join(issues)}"
            else:
                message = f"System resources are unhealthy: {'; '.join(issues)}"
            
            self.logger.info(f"System resources health check completed",
                           metrics={
                               "response_time_ms": response_time,
                               "status": status.value,
                               "cpu_percent": metrics.get("cpu_percent"),
                               "memory_percent": metrics.get("memory_percent"),
                               "disk_percent": metrics.get("disk_percent")
                           })
            
            return ComponentHealth(
                name=self.component_name,
                status=status,
                message=message,
                details=metrics,
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"System resources health check failed: {str(e)}")
            
            return ComponentHealth(
                name=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics.update({
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                "cpu_freq_max": cpu_freq.max if cpu_freq else None
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.update({
                "memory_percent": memory.percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "swap_percent": swap.percent,
                "swap_total_gb": swap.total / (1024**3)
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.update({
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_gb": disk.used / (1024**3)
            })
            
            if disk_io:
                metrics.update({
                    "disk_read_bytes": disk_io.read_bytes,
                    "disk_write_bytes": disk_io.write_bytes,
                    "disk_read_count": disk_io.read_count,
                    "disk_write_count": disk_io.write_count
                })
            
            # Process metrics
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            metrics.update({
                "process_count": process_count,
                "current_process_memory_mb": current_process.memory_info().rss / (1024**2),
                "current_process_cpu_percent": current_process.cpu_percent(),
                "current_process_threads": current_process.num_threads()
            })
            
            # Network metrics (basic)
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.update({
                    "network_bytes_sent": network_io.bytes_sent,
                    "network_bytes_recv": network_io.bytes_recv,
                    "network_packets_sent": network_io.packets_sent,
                    "network_packets_recv": network_io.packets_recv
                })
            
        except Exception as e:
            metrics["collection_error"] = str(e)
            self.logger.error(f"Error collecting system metrics: {str(e)}")
        
        return metrics
    
    def _evaluate_system_status(self, metrics: Dict[str, Any]) -> tuple[HealthStatus, List[str]]:
        """Evaluate system status based on metrics."""
        status = HealthStatus.HEALTHY
        issues = []
        
        # Check CPU usage
        cpu_percent = metrics.get("cpu_percent", 0)
        if cpu_percent >= self.cpu_threshold_unhealthy:
            status = HealthStatus.UNHEALTHY
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        elif cpu_percent >= self.cpu_threshold_degraded:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
        
        # Check memory usage
        memory_percent = metrics.get("memory_percent", 0)
        if memory_percent >= self.memory_threshold_unhealthy:
            status = HealthStatus.UNHEALTHY
            issues.append(f"High memory usage: {memory_percent:.1f}%")
        elif memory_percent >= self.memory_threshold_degraded:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
        
        # Check disk usage (critical threshold)
        disk_percent = metrics.get("disk_percent", 0)
        if disk_percent >= self.disk_threshold_unhealthy:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Critical disk usage: {disk_percent:.1f}%")
        
        # Check swap usage (warning indicator)
        swap_percent = metrics.get("swap_percent", 0)
        if swap_percent > 50:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"High swap usage: {swap_percent:.1f}%")
        
        return status, issues


class FileSystemHealthChecker(HealthChecker):
    """
    Health checker for file system accessibility and permissions.
    
    Validates:
    - Critical directory existence
    - Read/write permissions
    - Available disk space
    - File system integrity
    """
    
    def __init__(self, critical_paths: List[str]):
        """
        Initialize file system health checker.
        
        Args:
            critical_paths: List of critical paths to monitor
        """
        self.critical_paths = [Path(p) for p in critical_paths]
        self.logger = CorrelationLogger(__name__)
    
    @property
    def component_name(self) -> str:
        return "file_system"
    
    @property
    def tags(self) -> Dict[str, str]:
        return {
            "monitor_type": "file_system",
            "critical_paths_count": str(len(self.critical_paths))
        }
    
    @with_correlation("health_check.file_system")
    async def check_health(self) -> ComponentHealth:
        """Check file system health."""
        start_time = time.perf_counter()
        
        try:
            status = HealthStatus.HEALTHY
            issues = []
            path_details = {}
            
            for path in self.critical_paths:
                path_status = self._check_path(path)
                path_details[str(path)] = path_status
                
                if not path_status["accessible"]:
                    status = HealthStatus.UNHEALTHY
                    issues.append(f"Path not accessible: {path}")
                elif not path_status.get("writable", True):
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.DEGRADED
                    issues.append(f"Path not writable: {path}")
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            message = "File system is accessible" if not issues else f"File system issues: {'; '.join(issues)}"
            
            return ComponentHealth(
                name=self.component_name,
                status=status,
                message=message,
                details={"paths": path_details},
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"File system health check failed: {str(e)}")
            
            return ComponentHealth(
                name=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"File system check error: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=response_time,
                tags=self.tags
            )
    
    def _check_path(self, path: Path) -> Dict[str, Any]:
        """Check individual path status."""
        try:
            details = {
                "exists": path.exists(),
                "accessible": False,
                "readable": False,
                "writable": False,
                "is_directory": False,
                "size_bytes": None
            }
            
            if details["exists"]:
                details["is_directory"] = path.is_dir()
                details["readable"] = os.access(path, os.R_OK)
                details["writable"] = os.access(path, os.W_OK)
                details["accessible"] = details["readable"]
                
                if not details["is_directory"]:
                    details["size_bytes"] = path.stat().st_size
            
            return details
            
        except Exception as e:
            return {"error": str(e), "accessible": False}


class HealthManager:
    """
    Central health management system with caching and coordination.
    
    This class orchestrates health checks across all system components,
    provides caching for performance optimization, and generates
    comprehensive system health reports.
    
    Features:
    - Concurrent health check execution
    - Response caching with TTL
    - Dependency-aware health evaluation
    - Performance metrics collection
    - Graceful error handling
    - Configurable timeouts and thresholds
    """
    
    def __init__(self, cache_ttl_seconds: int = 30):
        """
        Initialize health manager.
        
        Args:
            cache_ttl_seconds: Cache TTL for health check results
        """
        self.checkers: Dict[str, HealthChecker] = {}
        self.cache: Dict[str, ComponentHealth] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.start_time = time.time()
        self.version = "1.0.0"
        self.environment = "production"
        self.logger = CorrelationLogger(__name__)
        
        # Threading for concurrent health checks
        self.executor = None
    
    def register_checker(self, checker: HealthChecker) -> None:
        """
        Register a health checker.
        
        Args:
            checker: HealthChecker instance to register
        """
        self.checkers[checker.component_name] = checker
        self.logger.info(f"Registered health checker: {checker.component_name}",
                        tags={"component": checker.component_name})
    
    def unregister_checker(self, component_name: str) -> None:
        """Unregister a health checker."""
        if component_name in self.checkers:
            del self.checkers[component_name]
            # Clean up cache
            self.cache.pop(component_name, None)
            self.cache_timestamps.pop(component_name, None)
            self.logger.info(f"Unregistered health checker: {component_name}")
    
    @with_correlation("health_check.basic")
    async def get_basic_health(self) -> Dict[str, Any]:
        """
        Get basic health status for load balancer checks.
        
        This endpoint is optimized for speed and provides minimal
        information suitable for high-frequency load balancer probes.
        
        Returns:
            Basic health status dictionary
        """
        try:
            overall_status = HealthStatus.HEALTHY
            
            # Use cached results for speed
            for component_name in self.checkers.keys():
                cached_health = self._get_cached_health(component_name)
                if cached_health:
                    if cached_health.status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                        break
                    elif cached_health.status == HealthStatus.DEGRADED:
                        overall_status = HealthStatus.DEGRADED
            
            return {
                "status": overall_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "version": self.version
            }
            
        except Exception as e:
            self.logger.error(f"Basic health check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    @with_correlation("health_check.detailed")
    async def get_detailed_health(self) -> SystemHealth:
        """
        Get detailed health status with component information.
        
        This endpoint provides comprehensive health information including
        individual component status, performance metrics, and diagnostic data.
        
        Returns:
            SystemHealth instance with detailed information
        """
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run health checks concurrently
        tasks = []
        for component_name, checker in self.checkers.items():
            task = asyncio.create_task(
                self._run_health_check_with_timeout(component_name, checker)
            )
            tasks.append((component_name, task))
        
        # Collect results
        degraded_components = []
        unhealthy_components = []
        
        for component_name, task in tasks:
            try:
                component_health = await task
                components[component_name] = component_health
                
                # Update cache
                self._update_cache(component_name, component_health)
                
                # Update overall status
                if component_health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                    unhealthy_components.append(component_name)
                elif component_health.status == HealthStatus.DEGRADED:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                    degraded_components.append(component_name)
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {str(e)}")
                
                # Create error health status
                error_health = ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check error: {str(e)}",
                    last_checked=datetime.now()
                )
                components[component_name] = error_health
                overall_status = HealthStatus.UNHEALTHY
                unhealthy_components.append(component_name)
        
        # Check dependencies
        dependencies_healthy = self._check_dependencies_health(components)
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
            environment=self.environment,
            components=components,
            dependencies_healthy=dependencies_healthy,
            degraded_components=degraded_components if degraded_components else None,
            unhealthy_components=unhealthy_components if unhealthy_components else None
        )
    
    async def _run_health_check_with_timeout(self, 
                                           component_name: str, 
                                           checker: HealthChecker) -> ComponentHealth:
        """Run health check with timeout protection."""
        try:
            return await asyncio.wait_for(
                checker.check_health(),
                timeout=checker.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {component_name}")
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout ({checker.timeout_seconds}s)",
                last_checked=datetime.now()
            )
    
    def _get_cached_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get cached health status if not expired."""
        if component_name not in self.cache:
            return None
        
        timestamp = self.cache_timestamps.get(component_name)
        if timestamp and datetime.now() - timestamp < self.cache_ttl:
            return self.cache[component_name]
        
        return None
    
    def _update_cache(self, component_name: str, health: ComponentHealth) -> None:
        """Update health check cache."""
        self.cache[component_name] = health
        self.cache_timestamps[component_name] = datetime.now()
    
    def _check_dependencies_health(self, components: Dict[str, ComponentHealth]) -> bool:
        """Check if all dependencies are healthy."""
        for component_name, checker in self.checkers.items():
            if checker.dependencies:
                for dep_name in checker.dependencies:
                    dep_health = components.get(dep_name)
                    if not dep_health or dep_health.status == HealthStatus.UNHEALTHY:
                        return False
        return True
    
    @with_correlation("health_check.readiness")
    async def get_readiness_status(self) -> Dict[str, Any]:
        """
        Kubernetes readiness probe endpoint.
        
        Determines if the application is ready to receive traffic.
        
        Returns:
            Readiness status dictionary
        """
        try:
            health_data = await self.get_detailed_health()
            
            # Define critical components for readiness
            critical_components = ["database", "file_system"]
            
            ready = True
            for component_name in critical_components:
                component_health = health_data.components.get(component_name)
                if not component_health or component_health.status == HealthStatus.UNHEALTHY:
                    ready = False
                    break
            
            return {
                "ready": ready,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "critical_components": {
                    name: health_data.components.get(name, {}).status.value 
                    if health_data.components.get(name) else "unknown"
                    for name in critical_components
                }
            }
            
        except Exception as e:
            self.logger.error(f"Readiness check failed: {str(e)}")
            return {
                "ready": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    @with_correlation("health_check.liveness")
    async def get_liveness_status(self) -> Dict[str, Any]:
        """
        Kubernetes liveness probe endpoint.
        
        Determines if the application is still alive and should not be restarted.
        
        Returns:
            Liveness status dictionary
        """
        try:
            return {
                "alive": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "version": self.version
            }
        except Exception as e:
            self.logger.error(f"Liveness check failed: {str(e)}")
            return {
                "alive": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }


# Example usage and setup function
async def setup_health_monitoring(app_config: Dict[str, Any]) -> HealthManager:
    """
    Setup comprehensive health monitoring for the application.
    
    Args:
        app_config: Application configuration dictionary
        
    Returns:
        Configured HealthManager instance
    """
    health_manager = HealthManager()
    
    # Register database health checker
    if app_config.get("database", {}).get("enabled", True):
        db_path = app_config["database"]["path"]
        schema_tables = app_config["database"].get("schema_tables", [])
        health_manager.register_checker(DatabaseHealthChecker(db_path, schema_tables))
    
    # Register system resources checker
    health_manager.register_checker(SystemResourcesHealthChecker())
    
    # Register file system checker
    critical_paths = app_config.get("critical_paths", ["./logs", "./data", "./config"])
    health_manager.register_checker(FileSystemHealthChecker(critical_paths))
    
    logger = CorrelationLogger(__name__)
    logger.info("Health monitoring setup complete",
               tags={"registered_checkers": list(health_manager.checkers.keys())})
    
    return health_manager


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Setup health monitoring
        config = {
            "database": {
                "enabled": True,
                "path": "metrics.db",
                "schema_tables": ["performance_metrics", "system_metrics"]
            },
            "critical_paths": ["./logs", "./data"]
        }
        
        health_manager = await setup_health_monitoring(config)
        
        # Test health checks
        print("=== Basic Health ===")
        basic_health = await health_manager.get_basic_health()
        print(json.dumps(basic_health, indent=2))
        
        print("\n=== Detailed Health ===")
        detailed_health = await health_manager.get_detailed_health()
        print(json.dumps(asdict(detailed_health), indent=2, default=str))
        
        print("\n=== Readiness Check ===")
        readiness = await health_manager.get_readiness_status()
        print(json.dumps(readiness, indent=2))
        
        print("\n=== Liveness Check ===")
        liveness = await health_manager.get_liveness_status()
        print(json.dumps(liveness, indent=2))
    
    asyncio.run(main())