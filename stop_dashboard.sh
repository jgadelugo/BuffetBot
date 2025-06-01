#!/bin/bash
# Stop BuffetBot Dashboard

echo "Stopping BuffetBot Dashboard..."

# Get PIDs of processes using port 8501
PIDS=$(lsof -ti:8501 2>/dev/null)

if [ ! -z "$PIDS" ]; then
    echo "Found dashboard process(es): $PIDS"
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
    
    echo "Dashboard stopped!"
else
    echo "No dashboard process found on port 8501."
fi

# Also kill any streamlit processes just to be sure
STREAMLIT_PIDS=$(pgrep -f "streamlit run" 2>/dev/null)
if [ ! -z "$STREAMLIT_PIDS" ]; then
    echo "Found additional streamlit processes: $STREAMLIT_PIDS"
    kill $STREAMLIT_PIDS 2>/dev/null
fi 