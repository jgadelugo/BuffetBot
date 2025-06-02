"""Streamlit app wrapper that ensures proper imports."""
import os
import sys
from pathlib import Path

# Add parent directory (project root) to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root)

# Execute the main app file
app_file = Path(__file__).parent / "app.py"

# Read and execute the app file with proper globals
with open(app_file, 'r') as f:
    app_code = f.read()

# Create a proper globals dict for execution
app_globals = {
    '__name__': '__main__',
    '__file__': str(app_file),
}

# Execute the app code
exec(app_code, app_globals)
