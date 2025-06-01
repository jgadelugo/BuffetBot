# Running BuffetBot Dashboard

## Quick Start

### Starting the Dashboard
```bash
./run_dashboard.sh
```
This script will:
- Automatically check for and clear any processes using port 8501
- Activate the virtual environment
- Start the dashboard on http://localhost:8501

### Stopping the Dashboard
**Option 1: Clean shutdown (recommended)**
```bash
./stop_dashboard.sh
```

**Option 2: While the dashboard is running**
- Press `Ctrl+C` in the terminal where it's running

## Troubleshooting

### Port 8501 is already in use
The `run_dashboard.sh` script automatically handles this, but if you need to manually clear the port:
```bash
./stop_dashboard.sh
```

### Manual port cleanup
If the scripts don't work, you can manually clear port 8501:
```bash
# Find processes using port 8501
lsof -i :8501

# Kill a specific process
kill <PID>

# Force kill if needed
kill -9 <PID>
```

## Features
- **Automatic port cleanup**: The run script checks for and clears port 8501 before starting
- **Graceful shutdown**: Handles Ctrl+C properly for clean termination
- **Virtual environment**: Automatically activates the venv if present 