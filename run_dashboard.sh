#!/bin/bash
# Convenience wrapper: run the dashboard from the project root.
exec "$(dirname "${BASH_SOURCE[0]}")/scripts/run_dashboard.sh" "$@"
