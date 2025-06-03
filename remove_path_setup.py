#!/usr/bin/env python3
"""
Script to remove repetitive path setup code from all Python files.
This script will clean up the project by removing the manual sys.path manipulation
that is no longer needed after proper package installation.
"""

import os
import re
from pathlib import Path
from typing import List


def find_python_files_with_path_setup(root_dir: Path) -> list[Path]:
    """Find all Python files that contain path setup code."""
    python_files = []

    # Patterns to look for
    patterns = [
        r"project_root\s*=.*Path\(__file__\)",
        r"sys\.path\.insert\(0",
        r'os\.environ\["PYTHONPATH"\]',
        r"from pathlib import Path.*sys",
    ]

    for py_file in root_dir.rglob("*.py"):
        # Skip files in venv, build, dist, etc.
        if any(
            part in str(py_file)
            for part in ["venv", "build", "dist", ".git", "__pycache__"]
        ):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            # Check if any pattern matches
            for pattern in patterns:
                if re.search(pattern, content):
                    python_files.append(py_file)
                    break

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return python_files


def remove_path_setup_from_file(file_path: Path) -> bool:
    """Remove path setup code from a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip path setup related lines
            if any(
                pattern in line
                for pattern in [
                    "project_root = ",
                    "sys.path.insert(0",
                    'os.environ["PYTHONPATH"]',
                    "# Path setup",
                    "# Add project root to Python path",
                    "# Set PYTHONPATH environment variable",
                ]
            ):
                # Skip this line and potentially the next few related lines
                while i < len(lines) and (
                    "project_root" in lines[i]
                    or "sys.path.insert" in lines[i]
                    or "PYTHONPATH" in lines[i]
                    or (
                        lines[i].strip() == ""
                        and i > 0
                        and any(
                            x in lines[i - 1]
                            for x in ["project_root", "sys.path", "PYTHONPATH"]
                        )
                    )
                ):
                    i += 1
                continue

            new_lines.append(lines[i])
            i += 1

        # Remove excessive blank lines at the top
        while new_lines and new_lines[0].strip() == "":
            new_lines.pop(0)

        # Write back the cleaned content
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def update_imports_in_file(file_path: Path) -> bool:
    """Update import statements to use proper package imports."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Common import replacements
        replacements = {
            "from utils.": "from buffetbot.utils.",
            "from data.": "from buffetbot.data.",
            "from dashboard.": "from buffetbot.dashboard.",
            "from analysis.": "from buffetbot.analysis.",
            "from recommend.": "from buffetbot.recommend.",
            "import utils.": "import buffetbot.utils.",
            "import data.": "import buffetbot.data.",
            "import dashboard.": "import buffetbot.dashboard.",
            "import analysis.": "import buffetbot.analysis.",
            "import recommend.": "import buffetbot.recommend.",
        }

        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(old, new)

        # Special case for dashboard utils rename
        content = content.replace(
            "from buffetbot.dashboard.utils.",
            "from buffetbot.dashboard.dashboard_utils.",
        )
        content = content.replace(
            "import buffetbot.dashboard.utils.",
            "import buffetbot.dashboard.dashboard_utils.",
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"Error updating imports in {file_path}: {e}")
        return False


def main():
    """Main function to clean up the project."""
    root_dir = Path(__file__).parent

    print("🔍 Finding Python files with path setup code...")
    files_with_path_setup = find_python_files_with_path_setup(root_dir)

    print(f"📁 Found {len(files_with_path_setup)} files with path setup code:")
    for file_path in files_with_path_setup:
        print(f"  - {file_path.relative_to(root_dir)}")

    print("\n🧹 Removing path setup code...")
    success_count = 0
    for file_path in files_with_path_setup:
        if remove_path_setup_from_file(file_path):
            print(f"  ✅ Cleaned: {file_path.relative_to(root_dir)}")
            success_count += 1
        else:
            print(f"  ❌ Failed: {file_path.relative_to(root_dir)}")

    print(f"\n🔄 Updating import statements...")
    all_python_files = list(root_dir.rglob("*.py"))
    import_success_count = 0

    for file_path in all_python_files:
        # Skip files in venv, build, dist, etc.
        if any(
            part in str(file_path)
            for part in ["venv", "build", "dist", ".git", "__pycache__"]
        ):
            continue

        if update_imports_in_file(file_path):
            import_success_count += 1

    print(f"  ✅ Updated imports in {import_success_count} files")

    print(f"\n🎉 Cleanup complete!")
    print(
        f"  - Removed path setup from {success_count}/{len(files_with_path_setup)} files"
    )
    print(f"  - Updated imports in {import_success_count} files")
    print(f"\n💡 The project now uses proper Python packaging!")
    print(f"   You can import modules using: from buffetbot.module import ...")


if __name__ == "__main__":
    main()
