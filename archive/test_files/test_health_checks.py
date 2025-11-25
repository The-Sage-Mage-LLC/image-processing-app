#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Health Check Test
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Simplified health check test to verify basic functionality.
"""

import asyncio
import json
import time
import psutil
from datetime import datetime
from typing import Dict, Any


class SimpleHealthChecker:
    """Simple health checker for basic system monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
    
    async def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status."""
        try:
            # Basic system checks
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "version": "1.0.0"
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health with system metrics."""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine overall status
            status = "healthy"
            if cpu_percent > 80 or memory.percent > 80:
                status = "degraded"
            if cpu_percent > 90 or memory.percent > 90:
                status = "unhealthy"
            
            return {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "components": {
                    "system_resources": {
                        "status": status,
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": (disk.used / disk.total) * 100,
                        "details": {
                            "cpu_count": psutil.cpu_count(),
                            "memory_total_gb": memory.total / (1024**3),
                            "disk_total_gb": disk.total / (1024**3)
                        }
                    }
                }
            }
        except Exception as e:
            return {
                "status": "unknown",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }


async def test_health_checks():
    """Test basic health check functionality."""
    print("Testing Health Check System")
    print("=" * 50)
    
    # Create health checker
    health_checker = SimpleHealthChecker()
    
    # Test basic health
    print("?? Running basic health check...")
    basic_health = await health_checker.get_basic_health()
    print("?? Basic Health:")
    print(json.dumps(basic_health, indent=2))
    
    print("\n?? Running detailed health check...")
    detailed_health = await health_checker.get_detailed_health()
    print("?? Detailed Health:")
    print(json.dumps(detailed_health, indent=2))
    
    print("\n? Health check system test completed")


if __name__ == "__main__":
    asyncio.run(test_health_checks())