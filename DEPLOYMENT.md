# BuffetBot Dashboard - Deployment Guide

## Streamlit Cloud Deployment

### Main Entry Points

The application can be deployed using any of these entry points:

1. **`main.py`** - Recommended for Streamlit Cloud deployment
2. **`dashboard/app.py`** - Direct app file with embedded path setup
3. **`dashboard/streamlit_app.py`** - Wrapper for local development

### Deployment Configuration

When deploying to Streamlit Cloud:

1. **Main file**: Use `main.py` as the entry point
2. **Python version**: 3.10 or higher (specified in `pyproject.toml`)
3. **Dependencies**: All listed in `requirements.txt`

### Path Setup Fix

The main issue that was resolved:
- **Problem**: Module import errors (`ModuleNotFoundError`) when deploying to cloud platforms
- **Cause**: Python path not properly configured for absolute imports from project root
- **Solution**: Added path setup code at the beginning of `dashboard/app.py` to ensure the project root is always in `sys.path`

### Files Modified for Deployment Compatibility

1. **`dashboard/app.py`**: Added path setup code at the beginning
2. **`main.py`**: Created as a clean entry point for cloud deployment
3. **`.streamlit/config.toml`**: Configuration for Streamlit settings
4. **`packages.txt`**: System dependencies for cloud deployment
5. **`dashboard/streamlit_app.py`**: Simplified wrapper for local development

### Local Development

For local development, continue using:
```bash
./run_dashboard.sh
```

This will use the existing setup with proper virtual environment activation.

### Troubleshooting

If you encounter import errors:
1. Ensure the main entry point includes path setup code
2. Verify all directories have `__init__.py` files
3. Check that `requirements.txt` includes all dependencies
4. Use absolute imports from the project root (e.g., `from utils.logger import get_logger`)

### Testing Deployment Locally

To test the deployment setup locally:
```bash
streamlit run main.py
```

This mimics how Streamlit Cloud will run the application.
