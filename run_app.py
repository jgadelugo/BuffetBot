#!/usr/bin/env python
"""
Entry point for BuffetBot Dashboard that properly sets up Python paths.
"""

import os
import signal
import sys
from pathlib import Path

# Add project root to Python path BEFORE any imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable
os.environ["PYTHONPATH"] = str(project_root)


# Signal handler for clean shutdown
def signal_handler(sig, frame):
    print("\n\nShutting down BuffetBot Dashboard gracefully...")
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Now we can import and run the dashboard
if __name__ == "__main__":
    # Import streamlit and run the app
    import streamlit.web.cli as stcli

    # Construct the streamlit command - use streamlit_app.py wrapper
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "dashboard" / "streamlit_app.py"),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
    ]

    print("Starting BuffetBot Dashboard on http://localhost:8501")
    print("Press Ctrl+C to stop the server gracefully\n")

    # Run streamlit
    sys.exit(stcli.main())
