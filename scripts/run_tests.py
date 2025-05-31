#!/usr/bin/env python
"""Simple test runner script for BuffetBot.

This script runs all tests and provides a summary of results.
"""

import subprocess
import sys
import os


def run_tests():
    """Run all tests using pytest."""
    print("🧪 Running BuffetBot Tests...")
    print("=" * 50)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it with:")
        print("   pip install pytest pytest-cov")
        return 1
    
    # Run tests with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",  # verbose
        "--tb=short",  # short traceback format
    ]
    
    # Add coverage if pytest-cov is available
    try:
        import pytest_cov
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    except ImportError:
        print("ℹ️  pytest-cov not installed. Running without coverage report.")
        print("   Install with: pip install pytest-cov")
        print()
    
    # Run the tests
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("\n" + "=" * 50)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests()) 