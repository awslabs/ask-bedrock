#!/usr/bin/env python
"""
Test runner for ask-bedrock.

This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest
import time
import signal


class TimeoutException(Exception):
    """Exception raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutException("Test execution timed out")


def run_tests():
    """Discover and run all tests."""
    # Add the parent directory to the path so that imports work correctly
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    # Register the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Get individual test modules
    test_modules = []
    for root, dirs, files in os.walk("tests"):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                module_path = os.path.join(root, file)
                test_modules.append(module_path)
    
    print(f"Found {len(test_modules)} test modules: {test_modules}")
    
    # Run each test module separately
    failures = 0
    for module in test_modules:
        print(f"\n{'=' * 70}\nRunning tests in {module}\n{'=' * 70}")
        try:
            # Set a timeout for each test module
            signal.alarm(10)  # 10 second timeout
            
            # Load the tests from the module
            module_name = os.path.splitext(module)[0].replace("/", ".")
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run the tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Reset the alarm
            signal.alarm(0)
            
            if not result.wasSuccessful():
                failures += 1
                
        except TimeoutException:
            print(f"ERROR: Tests in {module} timed out after 10 seconds")
            failures += 1
        except Exception as e:
            print(f"ERROR: Failed to run tests in {module}: {e}")
            failures += 1
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())