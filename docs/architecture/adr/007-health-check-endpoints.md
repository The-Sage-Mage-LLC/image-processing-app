# ADR-007: Health Check Endpoints Design

## Status

Accepted

## Context

The Image Processing Application requires comprehensive health monitoring for:

- **Production Monitoring**: Real-time application health status
- **Load Balancer Integration**: Health checks for high-availability deployments
- **Diagnostic Information**: Detailed system status for troubleshooting
- **Dependency Monitoring**: Check external service dependencies
- **Performance Metrics**: System performance indicators
- **Automated Alerts**: Integration with monitoring and alerting systems

Requirements:
- HTTP-based health check endpoints
- Multiple levels of health information (basic, detailed, diagnostic)
- Fast response times for basic health checks (<100ms)
- Comprehensive dependency checking
- Security considerations for sensitive information
- Integration with existing monitoring infrastructure

## Decision

We will implement **comprehensive health check endpoints** with multiple tiers of health information:

1. **Basic Health Endpoint** (`/health`) - Fast, minimal health status
2. **Detailed Health Endpoint** (`/health/detailed`) - Component-level status
3. **Diagnostic Endpoint** (`/health/diagnostic`) - Full system diagnostic
4. **Readiness Endpoint** (`/health/ready`) - Application readiness status
5. **Liveness Endpoint** (`/health/live`) - Basic liveness check

## Consequences

### Positive

- **Production Monitoring**: Real-time visibility into application health
- **Automated Recovery**: Support for automated restart/recovery systems
- **Performance Tracking**: Built-in performance metrics collection
- **Dependency Visibility**: Clear status of all system dependencies
- **Troubleshooting**: Rich diagnostic information for issue resolution
- **Standards Compliance**: Industry-standard health check patterns
- **Integration Ready**: Easy integration with monitoring tools

### Negative

- **Implementation Overhead**: Additional code and testing required
- **Security Risk**: Potential information disclosure in diagnostic endpoints
- **Performance Impact**: Health checks consume system resources
- **Maintenance Burden**: Need to keep health checks updated with system changes

### Neutral

- **HTTP Server Dependency**: Requires HTTP server for non-web applications
- **Monitoring Complexity**: Additional endpoints to monitor and maintain

## Implementation Notes

### Core Health Check Framework

#### Health Status Enumeration
```python
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """Health information for a system component."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_checked: Optional[datetime] = None
    response_time_ms: Optional[float] = None

@dataclass
class SystemHealth:
    """Overall system health information."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str
    components: Dict[str, ComponentHealth]
    metrics: Optional[Dict[str, Any]] = None
```

#### Health Check Interface
```python
from abc import ABC, abstractmethod

class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    @abstractmethod
    async def check_health(self) -> ComponentHealth:
        """Perform health check and return status."""
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return the component name."""
        pass
    
    @property
    def timeout_seconds(self) -> float:
        """Return the health check timeout."""
        return 5.0
```

#### Health Manager
```python
import asyncio
import time
from typing import Dict, List
from datetime import datetime, timedelta

class HealthManager:
    """Manages system health checks and status."""
    
    def __init__(self):
        self._checkers: Dict[str, HealthChecker] = {}
        self._cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl = timedelta(seconds=30)
        self._start_time = time.time()
        self._version = "1.0.0"
    
    def register_checker(self, checker: HealthChecker) -> None:
        """Register a health checker."""
        self._checkers[checker.component_name] = checker
    
    async def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status (fast check)."""
        overall_status = HealthStatus.HEALTHY
        
        # Check cached status for quick response
        for component_name, checker in self._checkers.items():
            cached_health = self._get_cached_health(component_name)
            if cached_health and cached_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif cached_health and cached_health.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - self._start_time
        }
    
    async def get_detailed_health(self) -> SystemHealth:
        """Get detailed health status with component checks."""
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all health checks
        tasks = []
        for component_name, checker in self._checkers.items():
            task = asyncio.create_task(
                self._run_health_check(component_name, checker)
            )
            tasks.append((component_name, task))
        
        # Collect results
        for component_name, task in tasks:
            try:
                component_health = await asyncio.wait_for(
                    task, timeout=checker.timeout_seconds
                )
                components[component_name] = component_health
                self._cache[component_name] = component_health
                
                # Update overall status
                if component_health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (component_health.status == HealthStatus.DEGRADED and 
                      overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED
                    
            except asyncio.TimeoutError:
                component_health = ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message="Health check timeout",
                    last_checked=datetime.utcnow()
                )
                components[component_name] = component_health
                overall_status = HealthStatus.UNHEALTHY
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - self._start_time,
            version=self._version,
            components=components
        )
```

### Specific Health Checkers

#### Database Health Checker
```python
class DatabaseHealthChecker(HealthChecker):
    """Health checker for SQLite metrics database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    @property
    def component_name(self) -> str:
        return "database"
    
    async def check_health(self) -> ComponentHealth:
        start_time = time.perf_counter()
        
        try:
            import sqlite3
            
            # Test database connection
            with sqlite3.connect(self.db_path, timeout=1.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            if result[0] == 1:
                return ComponentHealth(
                    name=self.component_name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    response_time_ms=response_time,
                    last_checked=datetime.utcnow(),
                    details={"db_path": self.db_path}
                )
            else:
                return ComponentHealth(
                    name=self.component_name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database query failed",
                    last_checked=datetime.utcnow()
                )
                
        except Exception as e:
            return ComponentHealth(
                name=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}",
                last_checked=datetime.utcnow()
            )
```

#### System Resources Health Checker
```python
import psutil

class SystemResourcesHealthChecker(HealthChecker):
    """Health checker for system resources."""
    
    def __init__(self, cpu_threshold=80.0, memory_threshold=85.0, disk_threshold=90.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    @property
    def component_name(self) -> str:
        return "system_resources"
    
    async def check_health(self) -> ComponentHealth:
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > self.memory_threshold:
                status = HealthStatus.DEGRADED
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > self.disk_threshold:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources normal"
            
            return ComponentHealth(
                name=self.component_name,
                status=status,
                message=message,
                last_checked=datetime.utcnow(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                last_checked=datetime.utcnow()
            )
```

### HTTP Endpoints Implementation

#### FastAPI Health Endpoints
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()
health_manager = HealthManager()

@app.get("/health")
async def basic_health():
    """Basic health check endpoint."""
    try:
        health_data = await health_manager.get_basic_health()
        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(content=health_data, status_code=status_code)
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with component status."""
    try:
        health_data = await health_manager.get_detailed_health()
        status_code = 200 if health_data.status == HealthStatus.HEALTHY else 503
        
        return JSONResponse(
            content={
                "status": health_data.status.value,
                "timestamp": health_data.timestamp.isoformat(),
                "uptime_seconds": health_data.uptime_seconds,
                "version": health_data.version,
                "components": {
                    name: {
                        "status": comp.status.value,
                        "message": comp.message,
                        "last_checked": comp.last_checked.isoformat() if comp.last_checked else None,
                        "response_time_ms": comp.response_time_ms,
                        "details": comp.details
                    }
                    for name, comp in health_data.components.items()
                }
            },
            status_code=status_code
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    health_data = await health_manager.get_detailed_health()
    
    # Application is ready if all critical components are healthy
    critical_components = ["database", "file_system"]
    ready = all(
        health_data.components.get(comp, ComponentHealth("", HealthStatus.UNHEALTHY)).status 
        in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        for comp in critical_components
    )
    
    status_code = 200 if ready else 503
    return JSONResponse(
        content={
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=status_code
    )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    # Simple check that the application is running
    return JSONResponse(
        content={
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - health_manager._start_time
        },
        status_code=200
    )
```

### Integration with Main Application

#### Setup Health Checks
```python
def setup_health_checks(app_config: Dict[str, Any]) -> HealthManager:
    """Setup health check system."""
    health_manager = HealthManager()
    
    # Register database health checker
    if app_config.get("database", {}).get("enabled", True):
        db_path = app_config["database"]["path"]
        health_manager.register_checker(DatabaseHealthChecker(db_path))
    
    # Register system resources checker
    health_manager.register_checker(SystemResourcesHealthChecker())
    
    # Register file system checker
    health_manager.register_checker(FileSystemHealthChecker())
    
    # Register processing engine checker
    health_manager.register_checker(ProcessingEngineHealthChecker())
    
    return health_manager
```

### Monitoring Integration

#### Prometheus Metrics
```python
from prometheus_client import Gauge, Counter

# Health check metrics
health_status_gauge = Gauge(
    'app_health_status',
    'Application health status (1=healthy, 0.5=degraded, 0=unhealthy)',
    ['component']
)

health_check_duration_gauge = Gauge(
    'app_health_check_duration_seconds',
    'Health check duration in seconds',
    ['component']
)

health_check_total = Counter(
    'app_health_checks_total',
    'Total health checks performed',
    ['component', 'status']
)

async def update_health_metrics():
    """Update Prometheus metrics with health status."""
    health_data = await health_manager.get_detailed_health()
    
    for name, component in health_data.components.items():
        # Update status gauge
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: 0.0
        }[component.status]
        
        health_status_gauge.labels(component=name).set(status_value)
        
        # Update duration gauge
        if component.response_time_ms:
            health_check_duration_gauge.labels(component=name).set(
                component.response_time_ms / 1000
            )
        
        # Increment counter
        health_check_total.labels(
            component=name, 
            status=component.status.value
        ).inc()
```

## Related ADRs

- ADR-003: SQLite for Metrics Storage (database health checks)
- ADR-006: Correlation ID Logging Strategy (health check logging)
- ADR-008: Asynchronous Processing Model (async health checks)

## References

- [Health Check Response Format](https://tools.ietf.org/id/draft-inadarei-api-health-check-06.html)
- [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Microservices Health Check Patterns](https://microservices.io/patterns/observability/health-check-api.html)
- [FastAPI Health Checks](https://fastapi.tiangolo.com/advanced/custom-response/)