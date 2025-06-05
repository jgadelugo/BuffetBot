#!/usr/bin/env python3
"""
Phase 2 Readiness Validation Script

This script validates that Phase 1 is properly implemented and the environment
is ready for Phase 2 BigQuery integration and real-time streaming development.

Usage:
    python scripts/validate_phase2_readiness.py

Exit Codes:
    0: All validations passed - ready for Phase 2
    1: Critical issues found - resolve before Phase 2
    2: Warnings found - review before Phase 2
"""

import asyncio
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    details: Optional[str] = None


class Phase2ReadinessValidator:
    """Comprehensive Phase 2 readiness validation."""

    def __init__(self):
        self.results: list[ValidationResult] = []
        self.project_root = project_root

    def add_result(self, name: str, status: str, message: str, details: str = None):
        """Add a validation result."""
        self.results.append(ValidationResult(name, status, message, details))

    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")

    def print_result(self, result: ValidationResult):
        """Print a validation result with appropriate emoji."""
        emoji_map = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
        emoji = emoji_map.get(result.status, "❓")
        print(f"{emoji} {result.name}: {result.message}")
        if result.details:
            print(f"   Details: {result.details}")

    async def validate_phase1_foundation(self):
        """Validate Phase 1 implementation is complete and working."""
        self.print_header("Phase 1 Foundation Validation")

        # Check Phase 1 directory structure
        phase1_dirs = [
            "buffetbot/storage/gcs",
            "buffetbot/storage/schemas",
            "buffetbot/storage/formatters",
            "buffetbot/storage/query",
            "buffetbot/storage/utils",
        ]

        for dir_path in phase1_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.add_result(
                    f"Phase 1 Directory: {dir_path}", "PASS", "Directory exists"
                )
            else:
                self.add_result(
                    f"Phase 1 Directory: {dir_path}",
                    "FAIL",
                    "Directory missing - Phase 1 incomplete",
                )

        # Check critical Phase 1 files
        critical_files = [
            "buffetbot/storage/gcs/manager.py",
            "buffetbot/storage/schemas/manager.py",
            "buffetbot/storage/formatters/parquet_formatter.py",
            "buffetbot/utils/cache.py",  # Critical unified cache
        ]

        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.add_result(f"Phase 1 File: {file_path}", "PASS", "File exists")
            else:
                self.add_result(
                    f"Phase 1 File: {file_path}",
                    "FAIL",
                    "Critical Phase 1 file missing",
                )

    def validate_python_imports(self):
        """Validate Phase 1 Python modules can be imported."""
        self.print_header("Phase 1 Import Validation")

        critical_imports = [
            ("buffetbot.storage.gcs.manager", "GCSStorageManager"),
            ("buffetbot.storage.schemas.manager", "SchemaManager"),
            ("buffetbot.utils.cache", "Cache"),
            ("buffetbot.storage.formatters.parquet_formatter", "ParquetFormatter"),
        ]

        for module_name, class_name in critical_imports:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    self.add_result(
                        f"Import: {module_name}.{class_name}",
                        "PASS",
                        "Successfully imported",
                    )
                else:
                    self.add_result(
                        f"Import: {module_name}.{class_name}",
                        "FAIL",
                        f"Class {class_name} not found in module",
                    )
            except ImportError as e:
                self.add_result(f"Import: {module_name}", "FAIL", f"Import failed: {e}")

    def validate_gcp_prerequisites(self):
        """Validate GCP setup for Phase 2."""
        self.print_header("GCP Prerequisites Validation")

        # Check for gcloud CLI
        try:
            result = subprocess.run(
                ["gcloud", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.add_result(
                    "gcloud CLI", "PASS", "gcloud CLI installed and accessible"
                )
            else:
                self.add_result("gcloud CLI", "FAIL", "gcloud CLI not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_result(
                "gcloud CLI", "FAIL", "gcloud CLI not installed or not in PATH"
            )

        # Check for authentication
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                auth_data = json.loads(result.stdout)
                active_accounts = [
                    acc for acc in auth_data if acc.get("status") == "ACTIVE"
                ]
                if active_accounts:
                    self.add_result(
                        "GCP Authentication",
                        "PASS",
                        f"Active account: {active_accounts[0]['account']}",
                    )
                else:
                    self.add_result(
                        "GCP Authentication",
                        "FAIL",
                        "No active GCP authentication found",
                    )
            else:
                self.add_result(
                    "GCP Authentication",
                    "FAIL",
                    "Cannot check GCP authentication status",
                )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            self.add_result(
                "GCP Authentication", "WARN", "Cannot verify GCP authentication status"
            )

        # Check current project
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                project_id = result.stdout.strip()
                self.add_result("GCP Project", "PASS", f"Current project: {project_id}")
            else:
                self.add_result("GCP Project", "FAIL", "No GCP project configured")
        except subprocess.TimeoutExpired:
            self.add_result(
                "GCP Project", "WARN", "Cannot verify GCP project configuration"
            )

    def validate_required_apis(self):
        """Validate required GCP APIs are enabled."""
        self.print_header("GCP APIs Validation")

        required_apis = [
            "bigquery.googleapis.com",
            "pubsub.googleapis.com",
            "storage-api.googleapis.com",
            "dataflow.googleapis.com",
            "secretmanager.googleapis.com",
        ]

        for api in required_apis:
            try:
                result = subprocess.run(
                    [
                        "gcloud",
                        "services",
                        "list",
                        "--enabled",
                        f"--filter=name:{api}",
                        "--format=value(name)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0 and api in result.stdout:
                    self.add_result(f"GCP API: {api}", "PASS", "API is enabled")
                else:
                    self.add_result(
                        f"GCP API: {api}",
                        "WARN",
                        f"API may not be enabled. Enable with: gcloud services enable {api}",
                    )
            except subprocess.TimeoutExpired:
                self.add_result(
                    f"GCP API: {api}", "WARN", "Cannot verify API status (timeout)"
                )

    def validate_python_environment(self):
        """Validate Python environment and dependencies."""
        self.print_header("Python Environment Validation")

        # Check Python version
        self.add_result(
            "Python Version",
            "PASS",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

        # Check Phase 1 dependencies are installed
        phase1_deps = [
            "google-cloud-storage",
            "pyarrow",
            "pandas",
            "pytest",
            "aiofiles",
        ]

        for dep in phase1_deps:
            try:
                importlib.import_module(dep.replace("-", "_"))
                self.add_result(
                    f"Phase 1 Dependency: {dep}", "PASS", "Installed and importable"
                )
            except ImportError:
                self.add_result(
                    f"Phase 1 Dependency: {dep}",
                    "FAIL",
                    "Not installed or not importable",
                )

    def validate_project_structure(self):
        """Validate overall project structure."""
        self.print_header("Project Structure Validation")

        required_files = [
            "requirements.txt",
            "pyproject.toml",
            "buffetbot/__init__.py",
            "tests/",
            "scripts/",
        ]

        for item in required_files:
            path = self.project_root / item
            if path.exists():
                self.add_result(f"Project File: {item}", "PASS", "Exists")
            else:
                self.add_result(
                    f"Project File: {item}",
                    "WARN",
                    "Missing - may be needed for Phase 2",
                )

    def validate_phase2_readiness(self):
        """Check if environment is ready for Phase 2 directories."""
        self.print_header("Phase 2 Readiness Check")

        # Check if Phase 2 directories already exist (shouldn't yet)
        phase2_dirs = ["buffetbot/analytics", "buffetbot/streaming"]

        for dir_path in phase2_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.add_result(
                    f"Phase 2 Directory: {dir_path}",
                    "WARN",
                    "Already exists - ensure it doesn't conflict with new implementation",
                )
            else:
                self.add_result(
                    f"Phase 2 Directory: {dir_path}",
                    "PASS",
                    "Ready to create (doesn't exist yet)",
                )

        # Check if Phase 2 requirements file exists
        phase2_req_path = self.project_root / "requirements" / "phase2.txt"
        if phase2_req_path.exists():
            self.add_result(
                "Phase 2 Requirements", "PASS", "requirements/phase2.txt exists"
            )
        else:
            self.add_result(
                "Phase 2 Requirements",
                "WARN",
                "requirements/phase2.txt not found - create it first",
            )

    async def run_all_validations(self):
        """Run all validation checks."""
        print("🚀 BuffetBot Phase 2 Readiness Validation")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Project Root: {self.project_root}")

        # Run all validation checks
        await self.validate_phase1_foundation()
        self.validate_python_imports()
        self.validate_gcp_prerequisites()
        self.validate_required_apis()
        self.validate_python_environment()
        self.validate_project_structure()
        self.validate_phase2_readiness()

        # Print all results
        print("\n" + "=" * 60)
        print("📊 VALIDATION RESULTS")
        print("=" * 60)

        for result in self.results:
            self.print_result(result)

        # Summary
        pass_count = len([r for r in self.results if r.status == "PASS"])
        fail_count = len([r for r in self.results if r.status == "FAIL"])
        warn_count = len([r for r in self.results if r.status == "WARN"])

        print(f"\n{'='*60}")
        print("📈 SUMMARY")
        print(f"{'='*60}")
        print(f"✅ PASSED: {pass_count}")
        print(f"❌ FAILED: {fail_count}")
        print(f"⚠️  WARNINGS: {warn_count}")
        print(f"📊 TOTAL: {len(self.results)}")

        # Determine exit code and recommendations
        if fail_count > 0:
            print(f"\n❌ PHASE 2 NOT READY")
            print(
                "Critical issues found. Resolve the following before starting Phase 2:"
            )
            for result in self.results:
                if result.status == "FAIL":
                    print(f"  • {result.name}: {result.message}")
            return 1
        elif warn_count > 0:
            print(f"\n⚠️  PHASE 2 READY WITH WARNINGS")
            print("Review the following warnings before starting Phase 2:")
            for result in self.results:
                if result.status == "WARN":
                    print(f"  • {result.name}: {result.message}")
            return 2
        else:
            print(f"\n✅ PHASE 2 READY!")
            print(
                "All validations passed. You can proceed with Phase 2 implementation."
            )
            print("\nNext steps:")
            print(
                "1. Install Phase 2 dependencies: pip install -r requirements/phase2.txt"
            )
            print("2. Follow the Phase 2 setup guide: plan/PHASE2_SETUP_GUIDE.md")
            print(
                "3. Use the application code prompt: plan/PHASE2_APPLICATION_CODE_PROMPT.md"
            )
            return 0


async def main():
    """Main execution function."""
    validator = Phase2ReadinessValidator()
    exit_code = await validator.run_all_validations()
    sys.exit(exit_code)


if __name__ == "__main__":
    # Run the async validation
    asyncio.run(main())
