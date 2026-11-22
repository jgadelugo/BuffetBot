"""Console-script entry point that starts the dashboard under Streamlit.

`buffetbot.dashboard.app.main` renders the page and only works inside a
Streamlit run, so `buffetbot-dashboard` points here instead. Extra arguments
are passed through to `streamlit run`.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run `streamlit run` on the dashboard script (local, analytics off)."""
    os.environ.setdefault("STREAMLIT_ENV", "development")
    script = Path(__file__).with_name("streamlit_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
