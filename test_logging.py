#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Structured Logging Test
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Simplified structured logging test to verify basic functionality.
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any


class SimpleCorrelationLogger:
    """Simple structured logger with basic correlation support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with correlation context."""
        correlation_id = str(uuid.uuid4())
        log_data = {
            "message": message,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.info(json.dumps(log_data))


def test_structured_logging():
    """Test basic structured logging functionality."""
    print("Testing Structured Logging System")
    print("=" * 50)
    
    # Create logger
    logger = SimpleCorrelationLogger("test_logger")
    
    # Test basic logging
    logger.info("Application started", component="startup")
    logger.info("Processing request", user_id="user123", action="process_image")
    logger.info("Operation completed", duration_ms=150.5, status="success")
    
    print("? Structured logging test completed")
    print("?? Check console output above for JSON-formatted logs")


if __name__ == "__main__":
    test_structured_logging()