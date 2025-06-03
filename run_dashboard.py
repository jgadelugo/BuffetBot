#!/usr/bin/env python
"""
Run script for BuffetBot Dashboard.
This script ensures proper module paths are set before running the Streamlit app.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable - this is crucial!
# Include both the project root and any existing PYTHONPATH
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if existing_pythonpath:
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
else:
    os.environ["PYTHONPATH"] = str(project_root)

# Run the streamlit app
if __name__ == "__main__":
    # Construct the command
    streamlit_cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "streamlit",
        "run",
        str(project_root / "dashboard" / "app.py"),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
    ]

    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        streamlit_cmd.extend(sys.argv[1:])

    print(f"Starting BuffetBot Dashboard...")
    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

    try:
        # Run streamlit with the modified environment
        # Use the current environment with our modifications
        env = os.environ.copy()
        subprocess.run(streamlit_cmd, env=env)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)
