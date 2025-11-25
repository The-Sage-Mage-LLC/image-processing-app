#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Configuration and Utilities
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

pytest configuration and testing utilities for comprehensive test coverage
including async tests, GUI tests, API tests, and performance tests.

Features:
- pytest configuration and custom markers
- Test fixtures and data factories
- GUI testing utilities with PyQt6
- API testing with FastAPI
- Performance testing utilities
- Mock and stub utilities
- Test reporting and coverage
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from typing import Generator, AsyncGenerator, List, Dict, Any
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# PyQt6 testing imports
import pytest_qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QTimer

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Image processing imports
import numpy as np
from PIL import Image
import cv2

# Performance testing
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gui: marks tests that require GUI"
    )
    config.addinivalue_line(
        "markers", "api: marks tests for API functionality"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests for security functionality"
    )
    config.addinivalue_line(
        "markers", "functional: marks tests as functional tests"
    )
    config.addinivalue_line(
        "markers", "ui: marks tests for user interface"
    )
    config.addinivalue_line(
        "markers", "usability: marks tests for usability"
    )
    config.addinivalue_line(
        "markers", "load: marks tests for load testing"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests for stress testing"
    )
    config.addinivalue_line(
        "markers", "compatibility: marks tests for compatibility"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on file path
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "test_gui" in str(item.fspath):
            item.add_marker(pytest.mark.gui)
        elif "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        
        # Add markers based on test names
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        if "security" in item.name:
            item.add_marker(pytest.mark.security)
        if "load" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.performance)


# Base Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for GUI tests."""
    if QApplication.instance() is None:
        app = QApplication([])
    else:
        app = QApplication.instance()
    yield app
    if app:
        app.quit()


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_data_dir() -> Path:
    """Get test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Image Testing Fixtures
@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create sample RGB image for testing."""
    # Create 100x100 RGB gradient image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = [
                int((i / 100) * 255),  # Red gradient
                int((j / 100) * 255),  # Green gradient  
                int(((i + j) / 200) * 255)  # Blue gradient
            ]
    return image


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create sample grayscale image for testing."""
    # Create 100x100 grayscale gradient
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = int(((i + j) / 200) * 255)
    return image


@pytest.fixture
def sample_image_files(temp_directory: Path) -> List[Path]:
    """Create multiple sample image files."""
    images = []
    formats = ['.jpg', '.png', '.bmp']
    
    for i, fmt in enumerate(formats):
        # Create simple test image
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        file_path = temp_directory / f"test_image_{i}{fmt}"
        
        if fmt == '.jpg':
            cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        elif fmt == '.png':
            cv2.imwrite(str(file_path), image)
        elif fmt == '.bmp':
            cv2.imwrite(str(file_path), image)
        
        images.append(file_path)
    
    return images


# Performance Testing Utilities
class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.start_cpu = None
        self.end_cpu = None
        self.process = psutil.Process()
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.start_cpu = self.process.cpu_percent()
    
    def stop(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss
        self.end_cpu = self.process.cpu_percent()
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def memory_delta_mb(self) -> float:
        """Get memory usage delta in MB."""
        if self.start_memory and self.end_memory:
            return (self.end_memory - self.start_memory) / 1024 / 1024
        return 0.0
    
    @property
    def avg_cpu_percent(self) -> float:
        """Get average CPU usage."""
        if self.start_cpu is not None and self.end_cpu is not None:
            return (self.start_cpu + self.end_cpu) / 2
        return 0.0


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    monitor.start()
    yield monitor
    monitor.stop()


# Load Testing Utilities
class LoadTestRunner:
    """Utility for running load tests."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results = []
        self.errors = []
    
    async def run_async_load_test(self, 
                                  test_func,
                                  concurrent_users: int,
                                  duration_seconds: int,
                                  *args, **kwargs):
        """Run async load test."""
        start_time = time.time()
        tasks = []
        
        async def user_session():
            """Simulate user session."""
            session_start = time.time()
            try:
                while time.time() - start_time < duration_seconds:
                    result = await test_func(*args, **kwargs)
                    self.results.append({
                        'timestamp': time.time(),
                        'duration': time.time() - session_start,
                        'result': result,
                        'success': True
                    })
                    await asyncio.sleep(0.1)  # Brief pause between requests
            except Exception as e:
                self.errors.append({
                    'timestamp': time.time(),
                    'error': str(e),
                    'type': type(e).__name__
                })
        
        # Start concurrent user sessions
        for _ in range(concurrent_users):
            task = asyncio.create_task(user_session())
            tasks.append(task)
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'total_requests': len(self.results),
            'total_errors': len(self.errors),
            'success_rate': len(self.results) / (len(self.results) + len(self.errors)) * 100,
            'avg_duration': sum(r['duration'] for r in self.results) / len(self.results) if self.results else 0,
            'requests_per_second': len(self.results) / duration_seconds
        }
    
    def run_thread_load_test(self,
                            test_func, 
                            concurrent_users: int,
                            duration_seconds: int,
                            *args, **kwargs):
        """Run threaded load test."""
        start_time = time.time()
        
        def user_session():
            """Simulate user session."""
            session_start = time.time()
            try:
                while time.time() - start_time < duration_seconds:
                    result = test_func(*args, **kwargs)
                    self.results.append({
                        'timestamp': time.time(),
                        'duration': time.time() - session_start,
                        'result': result,
                        'success': True
                    })
                    time.sleep(0.1)  # Brief pause
            except Exception as e:
                self.errors.append({
                    'timestamp': time.time(),
                    'error': str(e),
                    'type': type(e).__name__
                })
        
        # Start threads
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(user_session)
                for _ in range(concurrent_users)
            ]
            
            # Wait for completion
            for future in futures:
                future.result()
        
        return {
            'total_requests': len(self.results),
            'total_errors': len(self.errors),
            'success_rate': len(self.results) / (len(self.results) + len(self.errors)) * 100 if (len(self.results) + len(self.errors)) > 0 else 0,
            'avg_duration': sum(r['duration'] for r in self.results) / len(self.results) if self.results else 0,
            'requests_per_second': len(self.results) / duration_seconds
        }


@pytest.fixture
def load_test_runner() -> LoadTestRunner:
    """Load test runner fixture."""
    return LoadTestRunner()


# GUI Testing Utilities
class GUITestHelper:
    """Helper utilities for GUI testing."""
    
    def __init__(self, qtbot):
        self.qtbot = qtbot
    
    def click_button_safe(self, button):
        """Safely click button with wait."""
        if button.isEnabled():
            self.qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
            QTest.qWait(100)  # Brief wait for UI update
    
    def enter_text_safe(self, widget, text: str):
        """Safely enter text in widget."""
        widget.clear()
        self.qtbot.keyClicks(widget, text)
        QTest.qWait(50)
    
    def wait_for_signal(self, signal, timeout: int = 5000):
        """Wait for signal with timeout."""
        with self.qtbot.waitSignal(signal, timeout=timeout) as blocker:
            return blocker
    
    def wait_for_condition(self, condition_func, timeout: int = 5000):
        """Wait for condition to be true."""
        start_time = time.time()
        while time.time() - start_time < timeout / 1000:
            if condition_func():
                return True
            QTest.qWait(100)
        return False
    
    def simulate_file_dialog(self, file_paths: List[str]):
        """Mock file dialog to return specific paths."""
        return patch('PyQt6.QtWidgets.QFileDialog.getOpenFileNames', 
                    return_value=(file_paths, ''))


@pytest.fixture
def gui_test_helper(qtbot):
    """GUI test helper fixture."""
    return GUITestHelper(qtbot)


# API Testing Utilities
class APITestClient:
    """Extended test client for API testing."""
    
    def __init__(self, app):
        self.client = TestClient(app)
        self.async_client = None
    
    async def get_async_client(self):
        """Get async client for testing."""
        if not self.async_client:
            self.async_client = AsyncClient(app=self.client.app, base_url="http://testserver")
        return self.async_client
    
    def test_endpoint_performance(self, method: str, url: str, expected_max_time: float = 1.0, **kwargs):
        """Test endpoint performance."""
        start_time = time.time()
        
        if method.upper() == 'GET':
            response = self.client.get(url, **kwargs)
        elif method.upper() == 'POST':
            response = self.client.post(url, **kwargs)
        elif method.upper() == 'PUT':
            response = self.client.put(url, **kwargs)
        elif method.upper() == 'DELETE':
            response = self.client.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        duration = time.time() - start_time
        
        assert response.status_code < 500, f"Server error: {response.status_code}"
        assert duration < expected_max_time, f"Endpoint too slow: {duration:.2f}s > {expected_max_time}s"
        
        return response, duration
    
    def test_endpoint_load(self, method: str, url: str, concurrent_requests: int = 10, **kwargs):
        """Test endpoint under load."""
        results = []
        errors = []
        
        def make_request():
            try:
                start_time = time.time()
                if method.upper() == 'GET':
                    response = self.client.get(url, **kwargs)
                elif method.upper() == 'POST':
                    response = self.client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                duration = time.time() - start_time
                results.append({'status_code': response.status_code, 'duration': duration})
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            for future in futures:
                future.result()
        
        # Analyze results
        successful_requests = len([r for r in results if r['status_code'] < 500])
        avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
        
        return {
            'total_requests': concurrent_requests,
            'successful_requests': successful_requests,
            'error_count': len(errors),
            'success_rate': (successful_requests / concurrent_requests) * 100,
            'avg_duration': avg_duration,
            'max_duration': max((r['duration'] for r in results), default=0),
            'min_duration': min((r['duration'] for r in results), default=0)
        }


# Mock and Stub Utilities
class MockImageProcessor:
    """Mock image processor for testing."""
    
    def __init__(self):
        self.processed_images = []
        self.processing_time = 0.1
    
    async def process_image_async(self, input_path: str, output_path: str, operations: List[Dict]):
        """Mock async image processing."""
        await asyncio.sleep(self.processing_time)
        self.processed_images.append({
            'input': input_path,
            'output': output_path,
            'operations': operations,
            'timestamp': time.time()
        })
        return output_path
    
    def set_processing_time(self, seconds: float):
        """Set mock processing time."""
        self.processing_time = seconds
    
    def get_processing_history(self):
        """Get processing history."""
        return self.processed_images.copy()


@pytest.fixture
def mock_image_processor():
    """Mock image processor fixture."""
    return MockImageProcessor()


# Test Data Factories
class TestImageFactory:
    """Factory for creating test images."""
    
    @staticmethod
    def create_solid_color_image(width: int = 100, height: int = 100, color: tuple = (255, 0, 0)) -> np.ndarray:
        """Create solid color image."""
        image = np.full((height, width, 3), color, dtype=np.uint8)
        return image
    
    @staticmethod
    def create_noise_image(width: int = 100, height: int = 100) -> np.ndarray:
        """Create random noise image."""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    @staticmethod
    def create_gradient_image(width: int = 100, height: int = 100) -> np.ndarray:
        """Create gradient image."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = [
                    int((i / height) * 255),
                    int((j / width) * 255),
                    int(((i + j) / (height + width)) * 255)
                ]
        return image
    
    @staticmethod
    def create_test_pattern_image(width: int = 100, height: int = 100) -> np.ndarray:
        """Create test pattern image."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create checkerboard pattern
        for i in range(height):
            for j in range(width):
                if (i // 10 + j // 10) % 2 == 0:
                    image[i, j] = [255, 255, 255]  # White
                else:
                    image[i, j] = [0, 0, 0]  # Black
        
        return image


@pytest.fixture
def test_image_factory():
    """Test image factory fixture."""
    return TestImageFactory()


# Regression Test Utilities
class RegressionTestManager:
    """Manager for regression testing."""
    
    def __init__(self, baseline_dir: Path):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(exist_ok=True)
    
    def save_baseline(self, test_name: str, result: Any):
        """Save baseline result for regression testing."""
        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result, np.ndarray):
            result = result.tolist()
        elif isinstance(result, dict):
            result = self._convert_numpy_to_lists(result)
        
        with open(baseline_file, 'w') as f:
            import json
            json.dump(result, f, indent=2)
    
    def compare_with_baseline(self, test_name: str, current_result: Any, tolerance: float = 1e-6) -> bool:
        """Compare current result with baseline."""
        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"
        
        if not baseline_file.exists():
            # Save as new baseline
            self.save_baseline(test_name, current_result)
            return True
        
        with open(baseline_file, 'r') as f:
            import json
            baseline_result = json.load(f)
        
        return self._compare_results(baseline_result, current_result, tolerance)
    
    def _convert_numpy_to_lists(self, obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        else:
            return obj
    
    def _compare_results(self, baseline, current, tolerance):
        """Compare two results with tolerance."""
        if type(baseline) != type(current):
            return False
        
        if isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
            return abs(baseline - current) <= tolerance
        elif isinstance(baseline, list) and isinstance(current, list):
            if len(baseline) != len(current):
                return False
            return all(self._compare_results(b, c, tolerance) for b, c in zip(baseline, current))
        elif isinstance(baseline, dict) and isinstance(current, dict):
            if set(baseline.keys()) != set(current.keys()):
                return False
            return all(self._compare_results(baseline[k], current[k], tolerance) for k in baseline.keys())
        else:
            return baseline == current


@pytest.fixture
def regression_test_manager(temp_directory: Path):
    """Regression test manager fixture."""
    baseline_dir = temp_directory / "baselines"
    return RegressionTestManager(baseline_dir)


# Test Reporting Utilities
def generate_test_report(test_results: Dict[str, Any], output_file: Path):
    """Generate comprehensive test report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": test_results,
        "details": {
            "total_tests": test_results.get("total", 0),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "errors": test_results.get("errors", 0)
        }
    }
    
    with open(output_file, 'w') as f:
        import json
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Run configuration tests
    print("?? Testing Framework Configuration")
    print("=" * 50)
    
    # Test basic imports
    try:
        import pytest
        print("? pytest imported successfully")
    except ImportError as e:
        print(f"? pytest import failed: {e}")
    
    try:
        import pytest_asyncio
        print("? pytest-asyncio imported successfully")
    except ImportError as e:
        print(f"? pytest-asyncio import failed: {e}")
    
    try:
        import pytest_qt
        print("? pytest-qt imported successfully")
    except ImportError as e:
        print(f"? pytest-qt import failed: {e}")
    
    # Test fixture creation
    try:
        factory = TestImageFactory()
        test_image = factory.create_gradient_image(50, 50)
        assert test_image.shape == (50, 50, 3)
        print("? Test image factory working")
    except Exception as e:
        print(f"? Test image factory failed: {e}")
    
    print("\n? Test framework configuration complete")