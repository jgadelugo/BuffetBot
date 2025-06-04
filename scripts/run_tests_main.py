#!/usr/bin/env python3
"""
Test runner for the modular dashboard application.

This script provides a convenient way to run different types of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the output."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the modular dashboard")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--module",
        help="Run tests for a specific module (e.g., formatters, data_utils)",
    )

    args = parser.parse_args()

    # Base pytest command - use the virtual environment's Python
    cmd = ["python", "-m", "pytest"]

    # Add test directories based on type
    if args.type == "unit":
        cmd.append("tests/unit/")
    elif args.type == "integration":
        cmd.append("tests/integration/")
    else:  # all
        cmd.append("tests/")

    # Add specific module if specified
    if args.module:
        if args.type == "unit":
            cmd[-1] = f"tests/unit/test_{args.module}.py"
        else:
            print(f"Module filtering only supported for unit tests")
            return False

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add parallel execution
    if args.parallel > 1:
        cmd.extend(["-n", str(args.parallel)])

    # Add coverage if requested
    if args.coverage:
        cmd.extend(
            [
                "--cov=buffetbot",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ]
        )

    # Add other useful options
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "-ra",  # Show extra test summary info
            "--strict-markers",  # Strict marker checking
        ]
    )

    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")

    if success:
        print(f"\n✅ {args.type.title()} tests completed successfully!")

        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
    else:
        print(f"\n❌ {args.type.title()} tests failed!")
        return False

    return True


def run_style_checks():
    """Run code style and quality checks."""
    checks = [
        (
            ["python", "-m", "black", "--check", "buffetbot/", "tests/"],
            "Black code formatting",
        ),
        (
            ["python", "-m", "isort", "--check-only", "buffetbot/", "tests/"],
            "Import sorting",
        ),
        (["python", "-m", "flake8", "buffetbot/", "tests/"], "Flake8 linting"),
        (["python", "-m", "mypy", "buffetbot/"], "Type checking"),
    ]

    print("\n🔍 Running code quality checks...")
    all_passed = True

    for cmd, description in checks:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {description}: PASSED")
        except subprocess.CalledProcessError:
            print(f"❌ {description}: FAILED")
            all_passed = False
        except FileNotFoundError:
            print(f"⚠️  {description}: SKIPPED (tool not installed)")

    return all_passed


def run_security_checks():
    """Run security checks."""
    checks = [
        (["python", "-m", "bandit", "-r", "buffetbot/"], "Security scanning (Bandit)"),
        (["python", "-m", "safety", "check"], "Dependency vulnerability check"),
    ]

    print("\n🔒 Running security checks...")
    all_passed = True

    for cmd, description in checks:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {description}: PASSED")
        except subprocess.CalledProcessError:
            print(f"❌ {description}: FAILED")
            all_passed = False
        except FileNotFoundError:
            print(f"⚠️  {description}: SKIPPED (tool not installed)")

    return all_passed


if __name__ == "__main__":
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent.absolute()
    if not (project_root / "buffetbot").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Change to project root
    import os

    os.chdir(project_root)

    # Check for pytest installation
    try:
        subprocess.run(
            ["python", "-m", "pytest", "--version"], check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: pytest is not installed. Please install it with:")
        print("   pip install pytest pytest-cov pytest-xdist")
        sys.exit(1)

    success = main()

    # Run additional checks if main tests passed
    if success and len(sys.argv) == 1:  # Only if no specific args provided
        print("\n" + "=" * 60)
        print("Running additional checks...")
        print("=" * 60)

        style_passed = run_style_checks()
        security_passed = run_security_checks()

        if style_passed and security_passed:
            print("\n🎉 All checks passed!")
        else:
            print("\n⚠️  Some checks failed. See output above for details.")
            sys.exit(1)
    elif not success:
        sys.exit(1)
