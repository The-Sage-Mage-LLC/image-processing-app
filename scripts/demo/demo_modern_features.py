#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Features Integration Demo
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive demonstration of all modern features working together:
- Modern configuration with Pydantic
- Async/await patterns and concurrency
- Comprehensive testing framework
- Modern observability and monitoring

Features:
- End-to-end modern workflow demonstration
- Integration testing of all components
- Performance benchmarking
- Configuration validation
- Monitoring and observability showcase
"""

import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import json
import sys
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import modern features
from src.config.modern_settings import get_settings, reload_settings, AppSettings
from src.async_processing.modern_concurrency import AsyncImageProcessor, AsyncTaskQueue, Priority
from src.observability.modern_monitoring import initialize_observability, get_observability

# Import testing framework
from tests.conftest import TestImageFactory, PerformanceMonitor

# Standard imports
import numpy as np
import cv2


class ModernFeaturesDemo:
    """
    Comprehensive demonstration of modern application features.
    
    Showcases all modernized components working together in a realistic
    workflow with proper observability, configuration, and testing.
    """
    
    def __init__(self):
        """Initialize modern features demo."""
        self.demo_dir = Path(tempfile.mkdtemp(prefix="modern_features_demo_"))
        self.settings: Optional[AppSettings] = None
        self.processor: Optional[AsyncImageProcessor] = None
        self.observability = None
        
        print(f"?? Modern Features Demo initialized")
        print(f"?? Demo directory: {self.demo_dir}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all modern features."""
        try:
            print("\n" + "="*80)
            print("?? MODERN IMAGE PROCESSING APPLICATION DEMO")
            print("="*80)
            
            # 1. Configuration Management Demo
            await self._demo_configuration_management()
            
            # 2. Observability Setup Demo
            await self._demo_observability_setup()
            
            # 3. Async Processing Demo
            await self._demo_async_processing()
            
            # 4. Performance Monitoring Demo
            await self._demo_performance_monitoring()
            
            # 5. Integration Testing Demo
            await self._demo_integration_testing()
            
            # 6. Error Handling Demo
            await self._demo_error_handling()
            
            # 7. Comprehensive Workflow Demo
            await self._demo_comprehensive_workflow()
            
            print("\n" + "="*80)
            print("? MODERN FEATURES DEMO COMPLETED SUCCESSFULLY!")
            print("?? Your application now has enterprise-grade modern features!")
            print("="*80)
            
        except Exception as e:
            print(f"\n? Demo failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _demo_configuration_management(self):
        """Demonstrate modern configuration management with Pydantic."""
        print("\n1??  Modern Configuration Management Demo")
        print("-" * 50)
        
        # Load default configuration
        self.settings = get_settings()
        print(f"? Loaded configuration for environment: {self.settings.environment}")
        print(f"   App Name: {self.settings.app_name}")
        print(f"   Debug Mode: {self.settings.debug}")
        print(f"   Database Type: {self.settings.database.type}")
        print(f"   Max Workers: {self.settings.processing.max_concurrent_jobs}")
        
        # Demonstrate nested configuration access
        print(f"   API Settings: {self.settings.api.host}:{self.settings.api.port}")
        print(f"   Log Level: {self.settings.logging.level}")
        
        # Demonstrate configuration validation
        try:
            # Try to create invalid configuration
            with patch.dict('os.environ', {
                'ENVIRONMENT': 'production',
                'DEBUG': 'true',  # Invalid: debug in production
                'SECURITY_SECRET_KEY': 'change-me-in-production'  # Invalid: default secret
            }):
                invalid_settings = reload_settings()
        except Exception as e:
            print(f"? Configuration validation working: {type(e).__name__}")
        
        # Create .env template for users
        env_template_path = self.demo_dir / ".env.template"
        from src.config.modern_settings import create_env_template
        create_env_template(env_template_path)
        print(f"? Environment template created: {env_template_path}")
        
        # Export current settings (safe)
        export_path = self.demo_dir / "current_config.json"
        from src.config.modern_settings import export_settings
        export_settings(self.settings, export_path)
        print(f"? Configuration exported: {export_path}")
    
    async def _demo_observability_setup(self):
        """Demonstrate observability and monitoring setup."""
        print("\n2??  Modern Observability & Monitoring Demo")
        print("-" * 50)
        
        # Initialize observability
        self.observability = initialize_observability("demo-image-app")
        
        # Start monitoring (using alternative port to avoid conflicts)
        try:
            self.observability.start_monitoring(metrics_port=8092)
            print("? Observability system started")
            print("   ?? Prometheus metrics: http://localhost:8092/metrics")
            print("   ?? Health checks: Background monitoring active")
            print("   ?? Distributed tracing: OpenTelemetry configured")
        except Exception as e:
            print(f"??  Observability startup issue (continuing demo): {e}")
        
        # Demonstrate custom health check
        def demo_health_check():
            # Simulate health check logic
            return self.demo_dir.exists()
        
        from src.observability.modern_monitoring import HealthCheck
        custom_check = HealthCheck(
            name="demo_resources",
            check_func=demo_health_check,
            description="Demo resources availability",
            critical=False
        )
        self.observability.register_custom_health_check(custom_check)
        print("? Custom health check registered")
        
        # Get service status
        try:
            status = await self.observability.get_service_status()
            print(f"? Service status: {status['health']['status']}")
            print(f"   Readiness: {status['readiness']['ready']}")
            print(f"   Liveness: {status['liveness']['alive']}")
            print(f"   Health checks: {status['health']['summary']['healthy_checks']}/{status['health']['summary']['total_checks']} passing")
        except Exception as e:
            print(f"??  Status check issue (continuing): {e}")
    
    async def _demo_async_processing(self):
        """Demonstrate modern async/await processing patterns."""
        print("\n3??  Modern Async/Await Processing Demo")
        print("-" * 50)
        
        # Initialize async processor
        max_workers = self.settings.processing.max_concurrent_jobs
        self.processor = AsyncImageProcessor(max_workers=max_workers)
        print(f"? Async image processor initialized with {max_workers} workers")
        
        # Create test images
        test_images = []
        image_factory = TestImageFactory()
        
        for i in range(5):
            image_data = image_factory.create_gradient_image(100 + i*20, 100 + i*20)
            image_path = self.demo_dir / f"test_image_{i}.jpg"
            cv2.imwrite(str(image_path), image_data)
            test_images.append(image_path)
        
        print(f"? Created {len(test_images)} test images")
        
        # Define processing operations
        operations = [
            {"type": "resize", "params": {"width": 200, "height": 200}},
            {"type": "filter", "params": {"filter_type": "blur", "kernel_size": 3}},
        ]
        
        # Demonstrate async batch processing with monitoring
        output_dir = self.demo_dir / "async_output"
        progress_updates = []
        
        async def progress_callback(completed: int, total: int):
            progress = (completed / total) * 100
            progress_updates.append((completed, total, progress))
            print(f"   Processing: {completed}/{total} ({progress:.1f}%)")
        
        # Process with tracing
        if self.observability:
            with self.observability.trace_operation("batch_image_processing", batch_size=len(test_images)):
                start_time = time.time()
                results = await self.processor.process_batch_async(
                    test_images,
                    output_dir, 
                    operations,
                    progress_callback
                )
                processing_time = time.time() - start_time
        else:
            start_time = time.time()
            results = await self.processor.process_batch_async(
                test_images,
                output_dir, 
                operations,
                progress_callback
            )
            processing_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r.is_successful)
        failed = sum(1 for r in results if r.is_failed)
        throughput = len(test_images) / processing_time
        
        print(f"? Async batch processing completed:")
        print(f"   Total time: {processing_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} images/second")
        print(f"   Success: {successful}/{len(results)} images")
        print(f"   Failed: {failed} images")
        
        # Record metrics
        if self.observability:
            for result in results:
                self.observability.record_image_processing(
                    "batch_resize_blur", 
                    result.duration_seconds, 
                    result.is_successful
                )
        
        # Demonstrate task queue with priorities
        task_queue = AsyncTaskQueue(max_concurrent=3)
        
        async def priority_task(name: str, priority: Priority, delay: float):
            await asyncio.sleep(delay)
            return f"Task {name} completed"
        
        # Submit tasks with different priorities
        task_ids = []
        for i, priority in enumerate([Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.URGENT]):
            task_id = await task_queue.submit(
                f"priority_test_{i}",
                priority_task(f"P{i}", priority, 0.1),
                priority=priority
            )
            task_ids.append(task_id)
        
        # Wait for completion
        for task_id in task_ids:
            await task_queue.wait_for_task(task_id)
        
        # Check metrics
        queue_metrics = task_queue.get_metrics()
        print(f"? Task queue demonstration:")
        print(f"   Processed: {queue_metrics['total_completed']} tasks")
        print(f"   Success rate: {queue_metrics['success_rate']:.1f}%")
        
        await task_queue.shutdown()
    
    async def _demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        print("\n4??  Performance Monitoring Demo")
        print("-" * 50)
        
        # Create performance monitor
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Simulate various operations with monitoring
        operations_count = 20
        
        for i in range(operations_count):
            operation_start = time.time()
            
            # Simulate image processing operation
            if self.processor:
                test_image = TestImageFactory().create_noise_image(50, 50)
                test_path = self.demo_dir / f"perf_test_{i}.jpg"
                cv2.imwrite(str(test_path), test_image)
                
                # Process with monitoring
                output_path = self.demo_dir / f"perf_output_{i}.jpg"
                await self.processor.save_image_async(test_image, output_path, quality=85)
            
            operation_duration = time.time() - operation_start
            
            # Record metrics
            if self.observability:
                self.observability.record_image_processing(
                    "performance_test", 
                    operation_duration, 
                    True
                )
            
            # Brief delay to simulate realistic load
            await asyncio.sleep(0.02)
        
        monitor.stop()
        
        print(f"? Performance monitoring completed:")
        print(f"   Operations: {operations_count}")
        print(f"   Duration: {monitor.duration:.2f}s")
        print(f"   Memory delta: {monitor.memory_delta_mb:.1f} MB")
        print(f"   Avg CPU: {monitor.avg_cpu_percent:.1f}%")
        
        # Demonstrate load testing
        from tests.conftest import LoadTestRunner
        load_runner = LoadTestRunner()
        
        async def simple_task():
            await asyncio.sleep(0.01)  # 10ms simulated work
            return "completed"
        
        # Run async load test
        load_results = await load_runner.run_async_load_test(
            simple_task,
            concurrent_users=5,
            duration_seconds=2
        )
        
        print(f"? Load testing results:")
        print(f"   Requests: {load_results['total_requests']}")
        print(f"   Success rate: {load_results['success_rate']:.1f}%")
        print(f"   RPS: {load_results['requests_per_second']:.1f}")
        print(f"   Avg duration: {load_results['avg_duration']:.3f}s")
    
    async def _demo_integration_testing(self):
        """Demonstrate integration testing capabilities."""
        print("\n5??  Integration Testing Demo")
        print("-" * 50)
        
        # Import and run key tests programmatically
        try:
            import pytest
            
            # Create test configuration
            test_results = {
                "config_validation": False,
                "async_processing": False,
                "observability": False,
                "error_handling": False
            }
            
            # Test 1: Configuration validation
            try:
                settings = get_settings()
                assert settings.app_name is not None
                assert settings.environment in ["development", "testing", "production", "staging"]
                test_results["config_validation"] = True
                print("? Configuration validation tests passed")
            except Exception as e:
                print(f"? Configuration validation failed: {e}")
            
            # Test 2: Async processing
            try:
                if self.processor:
                    # Test image loading/saving
                    test_image = TestImageFactory().create_solid_color_image(50, 50, (255, 128, 0))
                    test_path = self.demo_dir / "integration_test.jpg"
                    await self.processor.save_image_async(test_image, test_path)
                    
                    loaded_image = await self.processor.load_image_async(test_path)
                    assert loaded_image.shape[:2] == (50, 50)
                    
                    test_results["async_processing"] = True
                    print("? Async processing tests passed")
            except Exception as e:
                print(f"? Async processing tests failed: {e}")
            
            # Test 3: Observability
            try:
                if self.observability:
                    # Test health checks
                    status = await self.observability.get_service_status()
                    assert "health" in status
                    assert "readiness" in status
                    
                    test_results["observability"] = True
                    print("? Observability tests passed")
            except Exception as e:
                print(f"? Observability tests failed: {e}")
            
            # Test 4: Error handling
            try:
                # Test graceful error handling
                if self.processor:
                    invalid_path = Path("/nonexistent/file.jpg")
                    try:
                        await self.processor.load_image_async(invalid_path)
                        assert False, "Should have raised exception"
                    except FileNotFoundError:
                        test_results["error_handling"] = True
                        print("? Error handling tests passed")
            except Exception as e:
                print(f"? Error handling tests failed: {e}")
            
            # Summary
            passed = sum(test_results.values())
            total = len(test_results)
            success_rate = (passed / total) * 100
            
            print(f"? Integration test summary:")
            print(f"   Tests passed: {passed}/{total} ({success_rate:.1f}%)")
            
            for test_name, result in test_results.items():
                status = "? PASS" if result else "? FAIL"
                print(f"   {test_name}: {status}")
        
        except ImportError:
            print("??  pytest not available, skipping programmatic testing")
    
    async def _demo_error_handling(self):
        """Demonstrate robust error handling."""
        print("\n6??  Error Handling & Resilience Demo")
        print("-" * 50)
        
        error_scenarios = []
        
        # Scenario 1: Invalid file handling
        try:
            if self.processor:
                invalid_file = self.demo_dir / "invalid.jpg"
                invalid_file.write_text("This is not an image file")
                
                await self.processor.load_image_async(invalid_file)
        except Exception as e:
            error_scenarios.append(f"Invalid file: {type(e).__name__}")
            print(f"? Handled invalid file error: {type(e).__name__}")
        
        # Scenario 2: Missing file handling
        try:
            if self.processor:
                missing_file = Path("/absolutely/nonexistent/file.jpg")
                await self.processor.load_image_async(missing_file)
        except Exception as e:
            error_scenarios.append(f"Missing file: {type(e).__name__}")
            print(f"? Handled missing file error: {type(e).__name__}")
        
        # Scenario 3: Resource exhaustion simulation
        try:
            # Create task queue with very limited capacity
            limited_queue = AsyncTaskQueue(max_concurrent=1)
            
            async def resource_intensive_task():
                await asyncio.sleep(2.0)  # Long running task
                return "completed"
            
            # Submit many tasks quickly
            task_ids = []
            for i in range(3):
                task_id = await limited_queue.submit(
                    f"resource_test_{i}",
                    resource_intensive_task(),
                    timeout=1.0  # Shorter timeout to trigger timeout errors
                )
                task_ids.append(task_id)
            
            # Wait and expect some timeouts
            timeout_count = 0
            for task_id in task_ids:
                try:
                    result = await limited_queue.wait_for_task(task_id, timeout=3.0)
                    if result.is_failed and "timeout" in str(result.error).lower():
                        timeout_count += 1
                except:
                    timeout_count += 1
            
            await limited_queue.shutdown()
            
            if timeout_count > 0:
                error_scenarios.append(f"Timeouts: {timeout_count} handled")
                print(f"? Handled {timeout_count} timeout scenarios")
            
        except Exception as e:
            error_scenarios.append(f"Resource exhaustion: {type(e).__name__}")
            print(f"? Handled resource exhaustion: {type(e).__name__}")
        
        # Scenario 4: Configuration error handling
        try:
            from src.config.modern_settings import AppSettings
            with patch.dict('os.environ', {'PROCESSING_MAX_CONCURRENT_JOBS': 'invalid_number'}):
                settings = reload_settings()
        except Exception as e:
            error_scenarios.append(f"Configuration: {type(e).__name__}")
            print(f"? Handled configuration error: {type(e).__name__}")
        
        print(f"? Error handling demonstration complete:")
        print(f"   Scenarios tested: {len(error_scenarios)}")
        for scenario in error_scenarios:
            print(f"   - {scenario}")
    
    async def _demo_comprehensive_workflow(self):
        """Demonstrate end-to-end workflow with all modern features."""
        print("\n7??  Comprehensive Modern Workflow Demo")
        print("-" * 50)
        
        workflow_start_time = time.time()
        
        # Step 1: Configuration-driven setup
        max_images = self.settings.processing.batch_size // 10  # Scale down for demo
        quality = self.settings.processing.default_quality
        output_format = self.settings.processing.default_output_format
        
        print(f"? Workflow configured:")
        print(f"   Images to process: {max_images}")
        print(f"   Quality setting: {quality}")
        print(f"   Output format: {output_format}")
        
        # Step 2: Create diverse test images
        workflow_images = []
        image_factory = TestImageFactory()
        
        for i in range(max_images):
            if i % 3 == 0:
                image = image_factory.create_gradient_image(80 + i*5, 80 + i*5)
            elif i % 3 == 1:
                image = image_factory.create_noise_image(90 + i*5, 90 + i*5)
            else:
                image = image_factory.create_test_pattern_image(70 + i*5, 70 + i*5)
            
            image_path = self.demo_dir / f"workflow_input_{i}.jpg"
            cv2.imwrite(str(image_path), image)
            workflow_images.append(image_path)
        
        print(f"? Created {len(workflow_images)} diverse test images")
        
        # Step 3: Multi-stage processing pipeline
        operations = [
            {"type": "resize", "params": {"width": 150, "height": 150}},
            {"type": "filter", "params": {"filter_type": "sharpen"}},
            {"type": "filter", "params": {"filter_type": "blur", "kernel_size": 2}},
        ]
        
        # Step 4: Process with full observability
        workflow_output_dir = self.demo_dir / "workflow_output"
        
        if self.observability:
            # Record workflow start
            self.observability.metrics_collector.update_queue_size("workflow", len(workflow_images))
        
        # Process with progress tracking and error handling
        successful_results = []
        failed_results = []
        
        async def workflow_progress(completed: int, total: int):
            progress = (completed / total) * 100
            if self.observability:
                # Update queue size as processing progresses
                remaining = total - completed
                self.observability.metrics_collector.update_queue_size("workflow", remaining)
            
            print(f"   Workflow progress: {completed}/{total} ({progress:.1f}%)")
        
        try:
            if self.observability:
                with self.observability.trace_operation(
                    "comprehensive_workflow",
                    workflow_id="demo_workflow_001",
                    image_count=len(workflow_images)
                ):
                    results = await self.processor.process_batch_async(
                        workflow_images,
                        workflow_output_dir,
                        operations,
                        workflow_progress
                    )
            else:
                results = await self.processor.process_batch_async(
                    workflow_images,
                    workflow_output_dir,
                    operations,
                    workflow_progress
                )
            
            # Categorize results
            for result in results:
                if result.is_successful:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                
                # Record individual operation metrics
                if self.observability:
                    self.observability.record_image_processing(
                        "comprehensive_workflow",
                        result.duration_seconds,
                        result.is_successful
                    )
        
        except Exception as e:
            print(f"? Workflow error: {e}")
            failed_results.append({"error": str(e)})
        
        # Step 5: Results analysis and reporting
        workflow_duration = time.time() - workflow_start_time
        total_processing_time = sum(r.duration_seconds for r in successful_results)
        avg_processing_time = total_processing_time / len(successful_results) if successful_results else 0
        
        print(f"? Comprehensive workflow completed:")
        print(f"   Total time: {workflow_duration:.2f}s")
        print(f"   Successful: {len(successful_results)}/{len(workflow_images)}")
        print(f"   Failed: {len(failed_results)}")
        print(f"   Avg processing: {avg_processing_time:.3f}s per image")
        print(f"   Throughput: {len(successful_results)/workflow_duration:.1f} images/sec")
        
        # Step 6: Output verification
        output_files = list(workflow_output_dir.glob("*.jpg")) if workflow_output_dir.exists() else []
        print(f"   Output files: {len(output_files)}")
        
        # Step 7: Health and status check
        if self.observability:
            try:
                final_status = await self.observability.get_service_status()
                print(f"   Final service status: {final_status['health']['status']}")
                
                # Reset queue size
                self.observability.metrics_collector.update_queue_size("workflow", 0)
            except Exception as e:
                print(f"   Status check error: {e}")
        
        return {
            "workflow_duration": workflow_duration,
            "successful_count": len(successful_results),
            "failed_count": len(failed_results),
            "throughput": len(successful_results)/workflow_duration,
            "output_files": len(output_files)
        }
    
    async def _cleanup(self):
        """Clean up demo resources."""
        print(f"\n?? Cleaning up demo resources...")
        
        try:
            # Cleanup async processor
            if self.processor:
                await self.processor.cleanup()
                print("? Async processor cleaned up")
            
            # Stop observability
            if self.observability:
                self.observability.stop_monitoring()
                print("? Observability stopped")
            
            # Remove demo directory
            if self.demo_dir.exists():
                shutil.rmtree(self.demo_dir)
                print(f"? Demo directory removed: {self.demo_dir}")
        
        except Exception as e:
            print(f"??  Cleanup warning: {e}")


async def main():
    """Run the comprehensive modern features demonstration."""
    print("?? Starting Modern Features Integration Demo...")
    
    demo = ModernFeaturesDemo()
    
    try:
        await demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n??  Demo interrupted by user")
    except Exception as e:
        print(f"\n? Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n?? Demo finished")


if __name__ == "__main__":
    # Required imports with fallback
    try:
        from unittest.mock import patch
    except ImportError:
        print("??  unittest.mock not available, some features may be limited")
        patch = None
    
    # Run the demo
    asyncio.run(main())