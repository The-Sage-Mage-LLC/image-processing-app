# ADR-006: Correlation ID Logging Strategy

## Status

Accepted

## Context

The Image Processing Application requires comprehensive logging for debugging, monitoring, and audit purposes. With multiple concurrent processing operations, batch jobs, and user interactions, it becomes critical to track related log entries across different components and threads.

Key requirements:
- **Traceability**: Track operations across multiple components
- **Debugging**: Correlate errors with specific user actions or batch jobs
- **Performance Monitoring**: Associate performance metrics with specific operations
- **Audit Trail**: Maintain audit logs for enterprise compliance
- **Concurrent Operations**: Handle multiple simultaneous image processing tasks

Challenges:
- Multi-threaded processing makes log correlation difficult
- Batch operations span multiple files and processing stages
- GUI interactions trigger cascading backend operations
- Need to maintain correlation across process boundaries

## Decision

We will implement **structured logging with correlation IDs** throughout the application using a centralized logging framework that:

1. **Generates unique correlation IDs** for each operation or user session
2. **Propagates correlation IDs** through all related function calls and threads
3. **Includes correlation IDs** in all log entries as structured metadata
4. **Supports nested operations** with parent-child correlation relationships
5. **Integrates with performance metrics** for complete traceability

## Consequences

### Positive

- **Complete Traceability**: Track operations from user action to completion
- **Enhanced Debugging**: Quickly identify all logs related to a specific issue
- **Performance Analysis**: Correlate timing data across operation lifecycle
- **Audit Compliance**: Comprehensive audit trails for enterprise requirements
- **Operations Support**: Simplified troubleshooting and monitoring
- **Thread Safety**: Correlation works across multiple threads and processes
- **Structured Data**: Machine-readable log format for automated analysis

### Negative

- **Implementation Complexity**: Additional code required for correlation propagation
- **Performance Overhead**: Small overhead for correlation ID generation and propagation
- **Memory Usage**: Additional memory for storing correlation context
- **Code Changes**: Requires updates to existing functions to accept correlation context
- **Testing Complexity**: Need to test correlation ID propagation paths

### Neutral

- **Log Volume**: Structured format may increase log verbosity
- **Storage Requirements**: Additional metadata increases log storage needs
- **Integration Effort**: Need to integrate with existing monitoring tools

## Implementation Notes

### Core Components

#### Correlation Context Manager
```python
import uuid
import contextvars
from typing import Optional

class CorrelationContext:
    """Manages correlation IDs across operations."""
    
    _correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'correlation_id', default=None
    )
    _operation_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
        'operation_stack', default_factory=list
    )
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the current correlation ID."""
        cls._correlation_id.set(correlation_id)
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get the current correlation ID."""
        return cls._correlation_id.get()
```

#### Structured Logger
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured logger with correlation ID support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_formatter()
    
    def _get_log_context(self) -> Dict[str, Any]:
        """Get current log context including correlation ID."""
        return {
            'correlation_id': CorrelationContext.get_correlation_id(),
            'timestamp': datetime.utcnow().isoformat(),
            'thread_id': threading.get_ident(),
            'process_id': os.getpid(),
            'operation_stack': CorrelationContext.get_operation_stack(),
        }
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with correlation context."""
        context = self._get_log_context()
        context.update(kwargs)
        self.logger.info(json.dumps({
            'message': message,
            'level': 'INFO',
            **context
        }))
```

#### Operation Decorator
```python
import functools
from typing import Callable, Any

def with_correlation(operation_name: str = None):
    """Decorator to add correlation tracking to operations."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate correlation ID if not present
            correlation_id = CorrelationContext.get_correlation_id()
            if not correlation_id:
                correlation_id = CorrelationContext.generate_id()
                CorrelationContext.set_correlation_id(correlation_id)
            
            # Track operation
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            CorrelationContext.push_operation(op_name)
            
            logger = StructuredLogger(__name__)
            logger.info(f"Starting operation: {op_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed operation: {op_name}")
                return result
            except Exception as e:
                logger.error(f"Failed operation: {op_name}", error=str(e))
                raise
            finally:
                CorrelationContext.pop_operation()
        
        return wrapper
    return decorator
```

### Integration Points

#### Image Processing Operations
```python
@with_correlation("image_processing")
def process_image(self, input_path: str, output_path: str) -> ProcessingResult:
    """Process image with correlation tracking."""
    logger = StructuredLogger(__name__)
    
    logger.info("Processing image", 
                input_path=input_path, 
                output_path=output_path)
    
    # Processing implementation
    pass
```

#### Batch Processing
```python
@with_correlation("batch_processing")
def process_batch(self, file_list: List[str]) -> BatchResult:
    """Process batch with correlation tracking."""
    batch_id = CorrelationContext.generate_id()
    
    for i, file_path in enumerate(file_list):
        # Create child correlation for each file
        file_correlation_id = f"{batch_id}-file-{i:04d}"
        
        with CorrelationContext.child_context(file_correlation_id):
            self.process_single_file(file_path)
```

#### GUI Event Handling
```python
class MainWindow(QMainWindow):
    def handle_process_button_clicked(self):
        """Handle process button with correlation."""
        correlation_id = CorrelationContext.generate_id()
        
        with CorrelationContext.context(correlation_id):
            logger = StructuredLogger(__name__)
            logger.info("User initiated processing", 
                       action="process_button_clicked")
            
            self.start_processing_operation()
```

### Configuration

#### Logging Configuration
```yaml
# logging_config.yaml
version: 1
formatters:
  structured:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: structured
    
  correlation_file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/correlation.log
    maxBytes: 10485760
    backupCount: 10
    formatter: structured

loggers:
  correlation:
    level: INFO
    handlers: [correlation_file]
    propagate: false
```

### Performance Considerations

- **Minimal Overhead**: Correlation ID generation adds <1ms per operation
- **Context Variables**: Use Python's `contextvars` for thread-safe context
- **Lazy Evaluation**: Generate correlation data only when logging occurs
- **Batch Optimization**: Group related log entries for efficient I/O

### Monitoring Integration

#### Metrics Collection
```python
class CorrelationMetrics:
    """Collect metrics with correlation context."""
    
    @staticmethod
    def record_operation_duration(operation: str, duration: float):
        correlation_id = CorrelationContext.get_correlation_id()
        
        metrics_logger = StructuredLogger('metrics')
        metrics_logger.info("Operation duration", 
                           operation=operation,
                           duration_ms=duration * 1000,
                           metric_type="duration")
```

## Related ADRs

- ADR-002: Modular Processing Architecture (logging integration points)
- ADR-007: Health Check Endpoints (correlation in health checks)
- ADR-008: Asynchronous Processing Model (correlation across async operations)

## References

- [Python contextvars Documentation](https://docs.python.org/3/library/contextvars.html)
- [Structured Logging Best Practices](https://www.structlog.org/)
- [Correlation IDs for Microservices](https://blog.rapid7.com/2016/12/23/the-value-of-correlation-ids/)
- [Python JSON Logger](https://github.com/madzak/python-json-logger)