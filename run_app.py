#!/usr/bin/env python
"""
Entry point for BuffetBot Dashboard using the properly installed package.
"""

import signal
import sys
from pathlib import Path


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

    # Use the main.py file which has proper imports
    project_root = Path(__file__).parent.absolute()

    # Construct the streamlit command - use main.py which has clean imports
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "main.py"),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
    ]

    print("Starting BuffetBot Dashboard on http://localhost:8501")
    print("Press Ctrl+C to stop the server gracefully\n")

    # Run streamlit
    sys.exit(stcli.main())
