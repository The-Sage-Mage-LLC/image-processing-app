#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Async/Await Patterns and Concurrency Framework
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Modern asynchronous programming patterns with async/await, concurrent processing,
and advanced task management for high-performance image processing.

Features:
- Async/await patterns for I/O operations
- Concurrent image processing with asyncio
- Task queues and job scheduling
- Resource pooling and management
- Error handling and retry mechanisms
- Performance monitoring and metrics
"""

import asyncio
import aiofiles
import aiohttp
from aiofiles import os as aio_os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, List, 
    Optional, Set, TypeVar, Union, Generic
)
import logging
import time
import uuid
import weakref
from queue import Queue
import threading

import numpy as np
from PIL import Image
import cv2

from ..config.modern_settings import get_settings
from ..utils.structured_logging import CorrelationLogger, with_correlation


T = TypeVar('T')
R = TypeVar('R')


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult(Generic[T]):
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED and self.error is None
    
    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILED
    
    @property
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        if self.execution_time:
            return self.execution_time
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class AsyncTask(Generic[T]):
    """Asynchronous task definition."""
    task_id: str
    name: str
    coro: Awaitable[T]
    priority: Priority = Priority.NORMAL
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ResourcePool(Generic[T]):
    """
    Asynchronous resource pool for managing shared resources.
    
    Provides efficient resource pooling with async context managers
    and automatic cleanup.
    """
    
    def __init__(self, 
                 resource_factory: Callable[[], Awaitable[T]],
                 max_size: int = 10,
                 timeout: float = 30.0,
                 cleanup_func: Optional[Callable[[T], Awaitable[None]]] = None):
        """
        Initialize resource pool.
        
        Args:
            resource_factory: Async function to create new resources
            max_size: Maximum number of resources in pool
            timeout: Timeout for acquiring resource
            cleanup_func: Optional cleanup function for resources
        """
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.timeout = timeout
        self.cleanup_func = cleanup_func
        
        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self.logger = CorrelationLogger(__name__)
    
    async def _create_resource(self) -> T:
        """Create new resource."""
        resource = await self.resource_factory()
        self._created_count += 1
        self.logger.debug(f"Created new resource, total: {self._created_count}")
        return resource
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[T, None]:
        """
        Acquire resource from pool.
        
        Yields:
            Resource instance from pool
        """
        if self._closed:
            raise RuntimeError("Resource pool is closed")
        
        resource = None
        try:
            # Try to get from pool first
            try:
                resource = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                # Create new resource if pool is empty and under limit
                async with self._lock:
                    if self._created_count < self.max_size:
                        resource = await self._create_resource()
                    else:
                        # Wait for resource from pool
                        resource = await asyncio.wait_for(
                            self._pool.get(), timeout=self.timeout
                        )
            
            yield resource
            
        except asyncio.TimeoutError:
            self.logger.warning("Timeout acquiring resource from pool")
            raise
        except Exception as e:
            self.logger.error(f"Error acquiring resource: {e}")
            if resource and self.cleanup_func:
                try:
                    await self.cleanup_func(resource)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up resource: {cleanup_error}")
            raise
        finally:
            # Return resource to pool if acquisition was successful
            if resource:
                try:
                    self._pool.put_nowait(resource)
                except asyncio.QueueFull:
                    # Pool is full, cleanup resource
                    if self.cleanup_func:
                        await self.cleanup_func(resource)
                    self._created_count -= 1
    
    async def close(self) -> None:
        """Close pool and cleanup all resources."""
        self._closed = True
        
        if self.cleanup_func:
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    await self.cleanup_func(resource)
                    self._created_count -= 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    self.logger.error(f"Error cleaning up resource during close: {e}")
        
        self.logger.info(f"Resource pool closed, {self._created_count} resources remaining")


class AsyncTaskQueue:
    """
    Advanced asynchronous task queue with priority handling and concurrency control.
    
    Provides high-performance task execution with priority queuing,
    dependency management, and comprehensive error handling.
    """
    
    def __init__(self, max_concurrent: int = 10, max_retries: int = 3):
        """
        Initialize async task queue.
        
        Args:
            max_concurrent: Maximum concurrent tasks
            max_retries: Default maximum retries for tasks
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        
        # Task queues by priority
        self._queues = {
            Priority.URGENT: asyncio.Queue(),
            Priority.HIGH: asyncio.Queue(),
            Priority.NORMAL: asyncio.Queue(),
            Priority.LOW: asyncio.Queue()
        }
        
        # Task tracking
        self._running_tasks: Set[str] = set()
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown = False
        
        # Metrics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        
        self.logger = CorrelationLogger(__name__)
        
        # Start worker coroutine
        self._worker_task = asyncio.create_task(self._worker())
    
    async def submit(self, 
                    name: str,
                    coro: Awaitable[T],
                    priority: Priority = Priority.NORMAL,
                    max_retries: Optional[int] = None,
                    timeout: Optional[float] = None,
                    dependencies: Optional[List[str]] = None,
                    **metadata) -> str:
        """
        Submit task to queue.
        
        Args:
            name: Task name
            coro: Coroutine to execute
            priority: Task priority
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            dependencies: List of task IDs this task depends on
            **metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = f"{name}_{uuid.uuid4().hex[:8]}"
        
        task = AsyncTask(
            task_id=task_id,
            name=name,
            coro=coro,
            priority=priority,
            max_retries=max_retries or self.max_retries,
            timeout=timeout,
            dependencies=dependencies or [],
            metadata=metadata
        )
        
        # Track dependencies
        if task.dependencies:
            self._task_dependencies[task_id] = set(task.dependencies)
        
        # Add to appropriate priority queue
        await self._queues[priority].put(task)
        self._total_submitted += 1
        
        self.logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    async def _worker(self) -> None:
        """Main worker coroutine."""
        while not self._shutdown:
            try:
                # Get next task from highest priority queue
                task = await self._get_next_task()
                if task:
                    # Check if dependencies are satisfied
                    if await self._dependencies_satisfied(task.task_id):
                        # Execute task
                        asyncio.create_task(self._execute_task(task))
                    else:
                        # Re-queue task to wait for dependencies
                        await self._queues[task.priority].put(task)
                        await asyncio.sleep(0.1)  # Brief delay to prevent busy waiting
                else:
                    # No tasks available, brief sleep
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in task queue worker: {e}")
                await asyncio.sleep(1.0)
    
    async def _get_next_task(self) -> Optional[AsyncTask]:
        """Get next task from priority queues."""
        # Check queues in priority order
        for priority in [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            try:
                return self._queues[priority].get_nowait()
            except asyncio.QueueEmpty:
                continue
        return None
    
    async def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        if task_id not in self._task_dependencies:
            return True
        
        dependencies = self._task_dependencies[task_id]
        for dep_id in dependencies:
            if dep_id in self._running_tasks:
                return False
            if dep_id not in self._completed_tasks:
                return False
            if not self._completed_tasks[dep_id].is_successful:
                return False
        
        return True
    
    async def _execute_task(self, task: AsyncTask[T]) -> None:
        """Execute individual task."""
        async with self._semaphore:
            self._running_tasks.add(task.task_id)
            
            result = TaskResult[T](
                task_id=task.task_id,
                status=TaskStatus.RUNNING,
                start_time=datetime.now()
            )
            
            try:
                self.logger.info(f"Starting task {task.task_id}")
                
                # Execute with timeout if specified
                if task.timeout:
                    task_result = await asyncio.wait_for(task.coro, timeout=task.timeout)
                else:
                    task_result = await task.coro
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.end_time = datetime.now()
                result.execution_time = (result.end_time - result.start_time).total_seconds()
                
                self._total_completed += 1
                self.logger.info(f"Completed task {task.task_id} in {result.execution_time:.2f}s")
                
            except asyncio.TimeoutError:
                result.status = TaskStatus.FAILED
                result.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
                self.logger.warning(f"Task {task.task_id} timed out")
                self._total_failed += 1
                
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = e
                result.end_time = datetime.now()
                result.execution_time = (result.end_time - result.start_time).total_seconds()
                
                # Retry logic
                if result.retry_count < task.max_retries:
                    result.retry_count += 1
                    result.status = TaskStatus.RETRYING
                    
                    # Re-submit with exponential backoff
                    delay = min(2 ** result.retry_count, 60)  # Max 60s delay
                    await asyncio.sleep(delay)
                    
                    self.logger.warning(f"Retrying task {task.task_id} (attempt {result.retry_count + 1})")
                    await self._queues[task.priority].put(task)
                else:
                    self.logger.error(f"Task {task.task_id} failed after {task.max_retries} retries: {e}")
                    self._total_failed += 1
            
            finally:
                self._running_tasks.discard(task.task_id)
                self._completed_tasks[task.task_id] = result
                
                # Clean up dependencies tracking
                if task.task_id in self._task_dependencies:
                    del self._task_dependencies[task.task_id]
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult[T]:
        """
        Wait for specific task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        while task_id not in self._completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
        
        return self._completed_tasks[task_id]
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of task."""
        if task_id in self._running_tasks:
            return TaskStatus.RUNNING
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        queue_sizes = {
            priority.name: self._queues[priority].qsize() 
            for priority in Priority
        }
        
        return {
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "currently_running": len(self._running_tasks),
            "queue_sizes": queue_sizes,
            "success_rate": (self._total_completed / max(self._total_submitted, 1)) * 100
        }
    
    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown task queue."""
        self._shutdown = True
        
        if wait:
            # Wait for running tasks to complete
            while self._running_tasks:
                await asyncio.sleep(0.1)
        
        # Cancel worker
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        
        self.logger.info("Task queue shutdown complete")


class AsyncImageProcessor:
    """
    Asynchronous image processing with concurrent operations.
    
    Provides high-performance image processing using async patterns,
    resource pooling, and concurrent execution.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async image processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.settings = get_settings()
        self.logger = CorrelationLogger(__name__)
        
        # Thread pools for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers//2)
        
        # Task queue for managing operations
        self.task_queue = AsyncTaskQueue(max_concurrent=max_workers)
        
        # Resource pools
        self._cv2_pool: Optional[ResourcePool] = None
        self._pil_pool: Optional[ResourcePool] = None
        
        self._setup_resource_pools()
    
    def _setup_resource_pools(self):
        """Setup resource pools for image processing libraries."""
        
        async def create_cv2_context():
            """Create OpenCV context (placeholder for actual resource)."""
            return {"type": "cv2", "created_at": datetime.now()}
        
        async def create_pil_context():
            """Create PIL context (placeholder for actual resource)."""
            return {"type": "pil", "created_at": datetime.now()}
        
        async def cleanup_context(ctx):
            """Cleanup context (placeholder)."""
            pass
        
        self._cv2_pool = ResourcePool(
            create_cv2_context, max_size=self.max_workers, cleanup_func=cleanup_context
        )
        self._pil_pool = ResourcePool(
            create_pil_context, max_size=self.max_workers, cleanup_func=cleanup_context
        )
    
    async def load_image_async(self, file_path: Path) -> np.ndarray:
        """
        Load image asynchronously.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Loaded image as numpy array
        """
        async with aiofiles.open(file_path, 'rb') as f:
            image_data = await f.read()
        
        # Use thread pool for CPU-intensive decoding
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            self._thread_pool,
            self._decode_image,
            image_data
        )
        
        return image
    
    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image data to numpy array."""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    async def save_image_async(self, image: np.ndarray, file_path: Path, quality: int = 85) -> None:
        """
        Save image asynchronously.
        
        Args:
            image: Image as numpy array
            file_path: Output file path
            quality: JPEG quality (1-100)
        """
        # Use thread pool for CPU-intensive encoding
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(
            self._thread_pool,
            self._encode_image,
            image,
            file_path.suffix,
            quality
        )
        
        # Save asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_data)
    
    def _encode_image(self, image: np.ndarray, format_ext: str, quality: int) -> bytes:
        """Encode numpy array to image bytes."""
        if format_ext.lower() in ['.jpg', '.jpeg']:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif format_ext.lower() == '.png':
            _, buffer = cv2.imencode('.png', image)
        else:
            # Default to JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return buffer.tobytes()
    
    async def resize_image_async(self, 
                               image: np.ndarray, 
                               width: int, 
                               height: int,
                               interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        """
        Resize image asynchronously.
        
        Args:
            image: Input image
            width: Target width
            height: Target height
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        async with self._cv2_pool.acquire():
            loop = asyncio.get_event_loop()
            resized = await loop.run_in_executor(
                self._thread_pool,
                cv2.resize,
                image,
                (width, height),
                None,
                0,
                0,
                interpolation
            )
            return resized
    
    async def apply_filter_async(self,
                               image: np.ndarray,
                               filter_type: str,
                               **params) -> np.ndarray:
        """
        Apply image filter asynchronously.
        
        Args:
            image: Input image
            filter_type: Type of filter to apply
            **params: Filter parameters
            
        Returns:
            Filtered image
        """
        async with self._cv2_pool.acquire():
            loop = asyncio.get_event_loop()
            filtered = await loop.run_in_executor(
                self._thread_pool,
                self._apply_filter,
                image,
                filter_type,
                params
            )
            return filtered
    
    def _apply_filter(self, image: np.ndarray, filter_type: str, params: Dict[str, Any]) -> np.ndarray:
        """Apply filter to image (CPU-intensive operation)."""
        if filter_type == "blur":
            kernel_size = params.get("kernel_size", 5)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        
        elif filter_type == "edge_detect":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        else:
            return image
    
    async def process_batch_async(self,
                                input_paths: List[Path],
                                output_dir: Path,
                                operations: List[Dict[str, Any]],
                                progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None) -> List[TaskResult]:
        """
        Process batch of images asynchronously.
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            operations: List of operations to apply
            progress_callback: Optional progress callback
            
        Returns:
            List of task results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Submit all tasks
        task_ids = []
        for i, input_path in enumerate(input_paths):
            output_path = output_dir / f"processed_{input_path.name}"
            
            task_id = await self.task_queue.submit(
                name=f"process_image_{i}",
                coro=self._process_single_image(input_path, output_path, operations),
                priority=Priority.NORMAL,
                metadata={"input_path": str(input_path), "output_path": str(output_path)}
            )
            task_ids.append(task_id)
        
        # Wait for completion and track progress
        results = []
        completed = 0
        
        for task_id in task_ids:
            result = await self.task_queue.wait_for_task(task_id)
            results.append(result)
            completed += 1
            
            if progress_callback:
                await progress_callback(completed, len(task_ids))
        
        return results
    
    async def _process_single_image(self,
                                  input_path: Path,
                                  output_path: Path,
                                  operations: List[Dict[str, Any]]) -> Path:
        """Process single image with operations."""
        # Load image
        image = await self.load_image_async(input_path)
        
        # Apply operations
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "resize":
                width = params.get("width", image.shape[1])
                height = params.get("height", image.shape[0])
                image = await self.resize_image_async(image, width, height)
            
            elif op_type == "filter":
                filter_type = params.get("filter_type", "blur")
                image = await self.apply_filter_async(image, filter_type, **params)
        
        # Save result
        quality = operations[-1].get("params", {}).get("quality", 85) if operations else 85
        await self.save_image_async(image, output_path, quality)
        
        return output_path
    
    async def cleanup(self) -> None:
        """Cleanup async image processor."""
        await self.task_queue.shutdown()
        
        if self._cv2_pool:
            await self._cv2_pool.close()
        if self._pil_pool:
            await self._pil_pool.close()
        
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)
        
        self.logger.info("Async image processor cleanup complete")


# Utility decorators and context managers

def with_timeout(timeout: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


@asynccontextmanager
async def async_timer() -> AsyncGenerator[Dict[str, Any], None]:
    """Async context manager for timing operations."""
    start_time = time.time()
    timing_data = {"start_time": start_time}
    
    try:
        yield timing_data
    finally:
        end_time = time.time()
        timing_data.update({
            "end_time": end_time,
            "duration": end_time - start_time
        })


# Example usage and demonstration
async def demo_async_patterns():
    """Demonstrate modern async patterns."""
    print("?? Modern Async/Await Patterns Demo")
    print("=" * 50)
    
    # Initialize async image processor
    processor = AsyncImageProcessor(max_workers=4)
    
    try:
        # Create demo images
        demo_dir = Path("demo_async")
        demo_dir.mkdir(exist_ok=True)
        
        # Create sample image data
        sample_images = []
        for i in range(5):
            # Create simple test image
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image_path = demo_dir / f"test_image_{i}.jpg"
            
            # Save using traditional method for demo setup
            cv2.imwrite(str(image_path), image)
            sample_images.append(image_path)
        
        print(f"Created {len(sample_images)} test images")
        
        # Define processing operations
        operations = [
            {"type": "resize", "params": {"width": 150, "height": 150}},
            {"type": "filter", "params": {"filter_type": "blur", "kernel_size": 3}},
        ]
        
        # Process batch asynchronously with progress tracking
        output_dir = demo_dir / "output"
        
        async def progress_callback(completed: int, total: int):
            progress = (completed / total) * 100
            print(f"Progress: {completed}/{total} ({progress:.1f}%)")
        
        async with async_timer() as timer:
            results = await processor.process_batch_async(
                sample_images,
                output_dir,
                operations,
                progress_callback
            )
        
        # Display results
        successful = sum(1 for r in results if r.is_successful)
        failed = sum(1 for r in results if r.is_failed)
        
        print(f"\nBatch processing completed in {timer['duration']:.2f}s")
        print(f"Successful: {successful}, Failed: {failed}")
        
        # Show task queue metrics
        metrics = processor.task_queue.get_metrics()
        print(f"Queue metrics: {metrics}")
        
        # Cleanup demo files
        import shutil
        shutil.rmtree(demo_dir, ignore_errors=True)
        
    finally:
        await processor.cleanup()
    
    print("? Async patterns demonstration complete")


if __name__ == "__main__":
    asyncio.run(demo_async_patterns())