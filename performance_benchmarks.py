#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Benchmarking Suite
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Industry-standard performance benchmarks and profiling for image processing operations.
"""

import time
import psutil
import gc
import os
import sys
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable, Tuple
import tempfile
from contextlib import contextmanager
import argparse

import numpy as np
from PIL import Image
import pytest


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.benchmarks_dir = self.project_root / "benchmark_reports"
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.baseline_file = self.benchmarks_dir / "performance_baseline.json"
        self.process = psutil.Process(os.getpid())
        
    @contextmanager
    def monitor_performance(self, operation_name: str):
        """Context manager for monitoring performance metrics."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Initial measurements
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        start_cpu_times = self.process.cpu_times()
        
        # CPU usage tracking
        cpu_percent_start = self.process.cpu_percent()
        
        try:
            yield
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            end_cpu_times = self.process.cpu_times()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + \
                          (end_cpu_times.system - start_cpu_times.system)
            cpu_percent_end = self.process.cpu_percent()
            
            # Store results
            self.last_measurement = {
                "operation": operation_name,
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "cpu_time": cpu_time_used,
                "cpu_percent": max(cpu_percent_start, cpu_percent_end),
                "timestamp": datetime.now().isoformat()
            }
    
    def create_test_images(self, temp_dir: Path) -> List[Path]:
        """Create test images of various sizes for benchmarking."""
        test_images = []
        
        image_configs = [
            ("small", (256, 256)),
            ("medium", (1024, 1024)), 
            ("large", (2048, 2048)),
            ("very_large", (4096, 4096)),
            ("ultra_wide", (4096, 1024)),
            ("portrait", (1024, 4096)),
        ]
        
        for name, size in image_configs:
            # Create RGB image with gradient pattern
            width, height = size
            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create gradient pattern for more realistic processing
            for x in range(width):
                for y in range(height):
                    image_array[y, x, 0] = int((x / width) * 255)  # Red gradient
                    image_array[y, x, 1] = int((y / height) * 255)  # Green gradient
                    image_array[y, x, 2] = int(((x + y) / (width + height)) * 255)  # Blue gradient
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, image_array.shape, dtype=np.int16)
            image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            image = Image.fromarray(image_array)
            image_path = temp_dir / f"benchmark_{name}.jpg"
            image.save(image_path, "JPEG", quality=95)
            test_images.append(image_path)
            
            print(f"   Created {name} image: {width}x{height} ({image_path.stat().st_size // 1024} KB)")
        
        return test_images
    
    def benchmark_image_loading(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Benchmark image loading performance."""
        print("?? Benchmarking image loading...")
        
        results = {"operation": "image_loading", "metrics": []}
        
        for image_path in image_paths:
            # Test PIL loading
            with self.monitor_performance(f"pil_load_{image_path.stem}"):
                image = Image.open(image_path)
                image.load()  # Force loading into memory
                width, height = image.size
            
            pil_metrics = self.last_measurement.copy()
            pil_metrics["library"] = "PIL"
            pil_metrics["image_size"] = f"{width}x{height}"
            pil_metrics["file_size"] = image_path.stat().st_size
            
            # Test OpenCV loading if available
            try:
                import cv2
                with self.monitor_performance(f"opencv_load_{image_path.stem}"):
                    cv_image = cv2.imread(str(image_path))
                    height, width = cv_image.shape[:2]
                
                opencv_metrics = self.last_measurement.copy()
                opencv_metrics["library"] = "OpenCV"
                opencv_metrics["image_size"] = f"{width}x{height}"
                opencv_metrics["file_size"] = image_path.stat().st_size
                results["metrics"].append(opencv_metrics)
                
            except ImportError:
                print("   OpenCV not available for loading benchmark")
            
            results["metrics"].append(pil_metrics)
        
        return results
    
    def benchmark_image_transformations(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Benchmark common image transformation operations."""
        print("?? Benchmarking image transformations...")
        
        results = {"operation": "image_transformations", "metrics": []}
        
        transformations = [
            ("grayscale", lambda img: img.convert("L")),
            ("resize_half", lambda img: img.resize((img.width // 2, img.height // 2))),
            ("resize_double", lambda img: img.resize((img.width * 2, img.height * 2))),
            ("rotate_90", lambda img: img.rotate(90, expand=True)),
            ("flip_horizontal", lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
            ("flip_vertical", lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
        ]
        
        # Test on medium-sized image for consistency
        test_image_path = next(p for p in image_paths if "medium" in p.stem)
        base_image = Image.open(test_image_path)
        
        for transform_name, transform_func in transformations:
            with self.monitor_performance(f"transform_{transform_name}"):
                result_image = transform_func(base_image.copy())
                # Force processing by accessing pixel data
                _ = result_image.size
            
            transform_metrics = self.last_measurement.copy()
            transform_metrics["transformation"] = transform_name
            transform_metrics["input_size"] = f"{base_image.width}x{base_image.height}"
            transform_metrics["output_size"] = f"{result_image.width}x{result_image.height}"
            results["metrics"].append(transform_metrics)
        
        return results
    
    def benchmark_file_operations(self, temp_dir: Path) -> Dict[str, Any]:
        """Benchmark file I/O operations."""
        print("?? Benchmarking file operations...")
        
        results = {"operation": "file_operations", "metrics": []}
        
        # Create test image
        test_image_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image_array)
        
        # Test different save formats and quality settings
        save_tests = [
            ("jpeg_high", {"format": "JPEG", "quality": 95}),
            ("jpeg_medium", {"format": "JPEG", "quality": 75}),
            ("jpeg_low", {"format": "JPEG", "quality": 50}),
            ("png", {"format": "PNG"}),
            ("bmp", {"format": "BMP"}),
        ]
        
        for test_name, save_kwargs in save_tests:
            output_path = temp_dir / f"benchmark_save_{test_name}.{save_kwargs['format'].lower()}"
            
            # Benchmark save operation
            with self.monitor_performance(f"save_{test_name}"):
                test_image.save(output_path, **save_kwargs)
            
            save_metrics = self.last_measurement.copy()
            save_metrics["test_type"] = "save"
            save_metrics["format"] = save_kwargs["format"]
            save_metrics["quality"] = save_kwargs.get("quality", "N/A")
            save_metrics["file_size"] = output_path.stat().st_size if output_path.exists() else 0
            results["metrics"].append(save_metrics)
            
            # Benchmark load operation
            if output_path.exists():
                with self.monitor_performance(f"load_{test_name}"):
                    loaded_image = Image.open(output_path)
                    loaded_image.load()
                
                load_metrics = self.last_measurement.copy()
                load_metrics["test_type"] = "load"
                load_metrics["format"] = save_kwargs["format"]
                load_metrics["file_size"] = output_path.stat().st_size
                results["metrics"].append(load_metrics)
        
        return results
    
    def benchmark_batch_processing(self, image_paths: List[Path], temp_dir: Path) -> Dict[str, Any]:
        """Benchmark batch processing operations."""
        print("?? Benchmarking batch processing...")
        
        results = {"operation": "batch_processing", "metrics": []}
        
        # Test sequential vs parallel processing simulation
        def process_image_sequential(image_path: Path, output_dir: Path) -> None:
            """Process single image sequentially."""
            image = Image.open(image_path)
            # Simulate processing: grayscale + resize + save
            processed = image.convert("L").resize((512, 512))
            output_path = output_dir / f"processed_{image_path.stem}.jpg"
            processed.save(output_path, "JPEG", quality=85)
        
        # Sequential processing
        sequential_dir = temp_dir / "sequential"
        sequential_dir.mkdir(exist_ok=True)
        
        with self.monitor_performance("batch_sequential"):
            for image_path in image_paths[:4]:  # Process first 4 images
                process_image_sequential(image_path, sequential_dir)
        
        sequential_metrics = self.last_measurement.copy()
        sequential_metrics["method"] = "sequential"
        sequential_metrics["image_count"] = 4
        results["metrics"].append(sequential_metrics)
        
        # Simulate parallel processing with threading
        import concurrent.futures
        import threading
        
        parallel_dir = temp_dir / "parallel"
        parallel_dir.mkdir(exist_ok=True)
        
        with self.monitor_performance("batch_parallel"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_image_sequential, image_path, parallel_dir)
                    for image_path in image_paths[:4]
                ]
                concurrent.futures.wait(futures)
        
        parallel_metrics = self.last_measurement.copy()
        parallel_metrics["method"] = "parallel"
        parallel_metrics["image_count"] = 4
        parallel_metrics["worker_count"] = 4
        results["metrics"].append(parallel_metrics)
        
        return results
    
    def benchmark_memory_usage(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("?? Benchmarking memory usage...")
        
        results = {"operation": "memory_usage", "metrics": []}
        
        # Test memory usage with different image sizes
        for image_path in image_paths:
            gc.collect()  # Clean slate
            initial_memory = self.process.memory_info().rss
            
            # Load image and measure memory
            with self.monitor_performance(f"memory_load_{image_path.stem}"):
                image = Image.open(image_path)
                image.load()
                
                # Create multiple copies to stress memory
                copies = [image.copy() for _ in range(5)]
                
                # Measure peak memory
                peak_memory = self.process.memory_info().rss
                
                # Clean up
                del copies, image
                gc.collect()
            
            memory_metrics = self.last_measurement.copy()
            memory_metrics["initial_memory"] = initial_memory
            memory_metrics["peak_memory"] = peak_memory
            memory_metrics["memory_increase"] = peak_memory - initial_memory
            memory_metrics["file_size"] = image_path.stat().st_size
            results["metrics"].append(memory_metrics)
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite."""
        print("?? COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("Project ID: Image Processing App 20251119")
        print("=" * 60)
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "benchmarks": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            print("?? Creating test images...")
            test_images = self.create_test_images(temp_path)
            
            # Run benchmark suites
            benchmark_suites = [
                ("image_loading", self.benchmark_image_loading, test_images),
                ("transformations", self.benchmark_image_transformations, test_images),
                ("file_operations", self.benchmark_file_operations, temp_path),
                ("batch_processing", self.benchmark_batch_processing, test_images, temp_path),
                ("memory_usage", self.benchmark_memory_usage, test_images),
            ]
            
            for suite_name, benchmark_func, *args in benchmark_suites:
                try:
                    print(f"\n?? Running {suite_name} benchmark...")
                    results = benchmark_func(*args)
                    benchmark_results["benchmarks"][suite_name] = results
                    print(f"   ? {suite_name} benchmark completed")
                except Exception as e:
                    print(f"   ? {suite_name} benchmark failed: {e}")
                    benchmark_results["benchmarks"][suite_name] = {"error": str(e)}
        
        # Calculate performance scores
        benchmark_results["performance_score"] = self.calculate_performance_score(benchmark_results["benchmarks"])
        
        # Save results
        report_file = self.benchmarks_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Generate summary report
        self.generate_benchmark_report(benchmark_results)
        
        # Compare with baseline if available
        if self.baseline_file.exists():
            self.compare_with_baseline(benchmark_results)
        else:
            self.save_baseline(benchmark_results)
        
        return benchmark_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        import platform
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }
    
    def calculate_performance_score(self, benchmarks: Dict[str, Any]) -> int:
        """Calculate overall performance score (0-100)."""
        scores = []
        
        for suite_name, suite_results in benchmarks.items():
            if "error" in suite_results:
                continue
            
            metrics = suite_results.get("metrics", [])
            if not metrics:
                continue
            
            # Calculate average execution time for this suite
            execution_times = [m.get("execution_time", 0) for m in metrics]
            avg_time = statistics.mean(execution_times) if execution_times else 0
            
            # Score based on execution time (lower is better)
            # Adjust thresholds based on operation type
            if "loading" in suite_name:
                # Image loading should be < 0.1s for good performance
                suite_score = max(0, 100 - (avg_time / 0.1 * 100))
            elif "transformations" in suite_name:
                # Transformations should be < 0.5s
                suite_score = max(0, 100 - (avg_time / 0.5 * 100))
            elif "file_operations" in suite_name:
                # File operations should be < 0.2s
                suite_score = max(0, 100 - (avg_time / 0.2 * 100))
            elif "batch_processing" in suite_name:
                # Batch processing should show parallel improvement
                sequential_time = next((m["execution_time"] for m in metrics if m.get("method") == "sequential"), 0)
                parallel_time = next((m["execution_time"] for m in metrics if m.get("method") == "parallel"), 0)
                if sequential_time > 0 and parallel_time > 0:
                    speedup = sequential_time / parallel_time
                    suite_score = min(100, speedup * 25)  # 4x speedup = 100 points
                else:
                    suite_score = 50
            else:
                # General performance scoring
                suite_score = max(0, 100 - (avg_time * 100))
            
            scores.append(min(100, suite_score))
        
        return int(statistics.mean(scores)) if scores else 0
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Generate human-readable benchmark report."""
        report_file = self.benchmarks_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Performance Score:** {results['performance_score']}/100\n\n")
            
            # System Information
            f.write("## System Information\n\n")
            system_info = results['system_info']
            f.write(f"- **Platform:** {system_info['platform']}\n")
            f.write(f"- **Processor:** {system_info['processor']}\n")
            f.write(f"- **CPU Cores:** {system_info['cpu_count']} logical, {system_info['cpu_count_physical']} physical\n")
            f.write(f"- **Memory:** {system_info['memory_total'] // (1024**3)} GB total, {system_info['memory_available'] // (1024**3)} GB available\n")
            f.write(f"- **Python Version:** {system_info['python_version']}\n\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            
            for suite_name, suite_results in results['benchmarks'].items():
                if "error" in suite_results:
                    f.write(f"### {suite_name.title()} - ? Error\n")
                    f.write(f"Error: {suite_results['error']}\n\n")
                    continue
                
                metrics = suite_results.get("metrics", [])
                if not metrics:
                    continue
                
                f.write(f"### {suite_name.title()}\n\n")
                
                # Calculate summary statistics
                execution_times = [m.get("execution_time", 0) for m in metrics]
                memory_deltas = [m.get("memory_delta", 0) for m in metrics if m.get("memory_delta")]
                
                if execution_times:
                    f.write(f"- **Average Execution Time:** {statistics.mean(execution_times):.3f}s\n")
                    f.write(f"- **Fastest:** {min(execution_times):.3f}s\n")
                    f.write(f"- **Slowest:** {max(execution_times):.3f}s\n")
                
                if memory_deltas:
                    avg_memory = statistics.mean(memory_deltas)
                    f.write(f"- **Average Memory Delta:** {avg_memory / (1024*1024):.1f} MB\n")
                
                f.write(f"- **Tests Run:** {len(metrics)}\n\n")
            
            # Performance Grade
            score = results['performance_score']
            if score >= 90:
                grade = "?? Excellent"
            elif score >= 75:
                grade = "?? Good"
            elif score >= 60:
                grade = "?? Fair"
            else:
                grade = "?? Needs Improvement"
            
            f.write(f"## Overall Assessment: {grade}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if score < 75:
                f.write("1. **Optimize image loading** - Consider caching and lazy loading\n")
                f.write("2. **Implement parallel processing** - Use multiprocessing for CPU-intensive tasks\n")
                f.write("3. **Memory management** - Monitor and optimize memory usage patterns\n")
            if score >= 75:
                f.write("? Performance is within acceptable ranges\n")
            
        print(f"?? Benchmark report saved to: {report_file}")
    
    def save_baseline(self, results: Dict[str, Any]) -> None:
        """Save current results as performance baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"?? Performance baseline saved to: {self.baseline_file}")
    
    def compare_with_baseline(self, current_results: Dict[str, Any]) -> None:
        """Compare current results with baseline."""
        try:
            with open(self.baseline_file, 'r') as f:
                baseline_results = json.load(f)
            
            print("\n?? BASELINE COMPARISON")
            print("-" * 40)
            
            current_score = current_results['performance_score']
            baseline_score = baseline_results['performance_score']
            
            score_diff = current_score - baseline_score
            if score_diff > 5:
                print(f"?? Performance IMPROVED: {current_score} vs {baseline_score} (+{score_diff})")
            elif score_diff < -5:
                print(f"?? Performance DEGRADED: {current_score} vs {baseline_score} ({score_diff})")
            else:
                print(f"?? Performance STABLE: {current_score} vs {baseline_score} ({score_diff:+})")
            
            # Update baseline with current results if better
            if current_score > baseline_score:
                self.save_baseline(current_results)
                print("? New performance baseline established!")
            
        except Exception as e:
            print(f"?? Could not compare with baseline: {e}")


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Pytest integration for performance benchmarks."""
    
    def test_image_loading_benchmark(self, benchmark):
        """Benchmark image loading with pytest-benchmark."""
        from PIL import Image
        import numpy as np
        
        # Create test image
        image_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        test_image = Image.fromarray(image_array)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            test_image.save(tmp.name, "JPEG", quality=95)
            
            # Benchmark the loading operation
            def load_image():
                img = Image.open(tmp.name)
                img.load()
                return img.size
            
            result = benchmark(load_image)
            assert result == (1024, 1024)
    
    def test_image_transformation_benchmark(self, benchmark):
        """Benchmark image transformation operations."""
        from PIL import Image
        import numpy as np
        
        # Create test image
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_image = Image.fromarray(image_array)
        
        def transform_image():
            # Chain multiple transformations
            processed = test_image.copy()
            processed = processed.convert("L")  # Grayscale
            processed = processed.resize((256, 256))  # Resize
            processed = processed.rotate(90)  # Rotate
            return processed.size
        
        result = benchmark(transform_image)
        assert result == (256, 256)
    
    def test_batch_processing_benchmark(self, benchmark, tmp_path):
        """Benchmark batch processing operations."""
        from PIL import Image
        import numpy as np
        
        # Create multiple test images
        test_images = []
        for i in range(5):
            image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            image_path = tmp_path / f"test_image_{i}.jpg"
            image.save(image_path, "JPEG")
            test_images.append(image_path)
        
        def process_batch():
            results = []
            for image_path in test_images:
                img = Image.open(image_path)
                processed = img.convert("L").resize((128, 128))
                results.append(processed.size)
            return len(results)
        
        result = benchmark(process_batch)
        assert result == 5


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_benchmarks.py --run         # Run complete benchmark suite
  python performance_benchmarks.py --compare     # Compare with baseline
  python performance_benchmarks.py --baseline    # Set new baseline
  python performance_benchmarks.py --pytest     # Run pytest benchmarks
"""
    )
    
    parser.add_argument("--run", action="store_true", help="Run complete benchmark suite")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline only")
    parser.add_argument("--baseline", action="store_true", help="Set new performance baseline")
    parser.add_argument("--pytest", action="store_true", help="Run pytest benchmarks")
    
    args = parser.parse_args()
    
    benchmarker = PerformanceBenchmark()
    
    if args.pytest:
        # Run pytest benchmarks
        pytest.main([__file__, "-v", "-m", "benchmark", "--benchmark-enable"])
    elif args.run or len(sys.argv) == 1:
        benchmarker.run_comprehensive_benchmark()
    elif args.compare:
        if benchmarker.baseline_file.exists():
            # Run current benchmark and compare
            current_results = benchmarker.run_comprehensive_benchmark()
        else:
            print("? No baseline found. Run with --baseline first.")
    elif args.baseline:
        results = benchmarker.run_comprehensive_benchmark()
        benchmarker.save_baseline(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()