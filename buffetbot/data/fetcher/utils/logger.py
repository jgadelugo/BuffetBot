"""
Logging utilities for the data fetcher.
"""

# Path setup to ensure proper imports
import sys
from pathlib import Path

# Ensure project root is in path for absolute imports
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from buffetbot.utils.logger import JSONFormatter, get_logger, setup_logger

__all__ = ["setup_logger", "get_logger", "JSONFormatter"]
