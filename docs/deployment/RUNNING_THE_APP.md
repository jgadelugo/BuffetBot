# Running the BuffetBot App

## Quick Start

```bash
./run_dashboard.sh
```

The application will start on `http://localhost:8501`

## Alternative Methods

1. **Using Python directly:**
   ```bash
   python scripts/run_dashboard.py
   ```

2. **Using the shell script from scripts directory:**
   ```bash
   ./scripts/run_dashboard.sh
   ```

## Troubleshooting

### Port Already in Use

The `scripts/run_dashboard.sh` script automatically handles this, but if you need to manually clear the port:

```bash
# Find and kill processes on port 8501
lsof -ti:8501 | xargs kill -9

# Or use the stop script
./scripts/stop_dashboard.sh
```

### Virtual Environment

If you haven't activated your virtual environment:

```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### Dependencies

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
```

## Development Mode

For development with auto-reload:

```bash
streamlit run main.py --server.runOnSave=true
```

## Features
- **Automatic port cleanup**: The run script checks for and clears port 8501 before starting
- **Graceful shutdown**: Handles Ctrl+C properly for clean termination
- **Virtual environment**: Automatically activates the venv if present
