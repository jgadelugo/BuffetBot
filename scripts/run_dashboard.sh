#!/bin/bash
# Run BuffetBot Dashboard with virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to kill processes on port 8501
cleanup_port() {
    echo "Checking for existing processes on port 8501..."
    # Get PIDs of processes using port 8501
    PIDS=$(lsof -ti:8501 2>/dev/null)

    if [ ! -z "$PIDS" ]; then
        echo "Found existing process(es) on port 8501: $PIDS"
        echo "Terminating..."
        # Try graceful kill first
        kill $PIDS 2>/dev/null
        sleep 2

        # Check if still running and force kill if necessary
        PIDS=$(lsof -ti:8501 2>/dev/null)
        if [ ! -z "$PIDS" ]; then
            echo "Force killing stubborn process(es)..."
            kill -9 $PIDS 2>/dev/null
        fi
        echo "Port 8501 cleared!"
    else
        echo "Port 8501 is available."
    fi
}

# Cleanup port before starting
cleanup_port

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "Warning: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create a virtual environment first with: python -m venv venv"
fi

# Check database health before starting dashboard
echo "Checking database connection..."
python -m database.cli health --detailed

# if [ $? -ne 0 ]; then
#     echo "Database health check failed. Attempting to initialize database..."
#     python -m database.cli create --seed

#     if [ $? -ne 0 ]; then
#         echo "❌ Failed to initialize database. Please check your database configuration."
#         echo "💡 Make sure PostgreSQL is running and your .env file is configured correctly."
#         exit 1
#     fi
# fi

# Run the dashboard using the new structure
echo "Starting BuffetBot Dashboard..."
echo "Starting BuffetBot Dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop the server gracefully"
echo ""

# Use the run_dashboard.py script from the scripts directory
python "$SCRIPT_DIR/scripts/run_dashboard.py" "$@"
