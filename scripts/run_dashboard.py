#!/usr/bin/env python
"""
Run script for BuffetBot Dashboard.
This script runs the Streamlit app using the properly installed package.
"""

import subprocess
import sys
from pathlib import Path

# Run the streamlit app
if __name__ == "__main__":
    # Get the project root for the main.py file
    project_root = Path(__file__).parent.parent.absolute()

    # Construct the command
    streamlit_cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "streamlit",
        "run",
        str(project_root / "main.py"),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
    ]

    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        streamlit_cmd.extend(sys.argv[1:])

    print(f"Starting BuffetBot Dashboard...")
    print(f"Using main.py from: {project_root / 'main.py'}")
    print(f"Python executable: {sys.executable}")

    try:
        # Run streamlit
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)
