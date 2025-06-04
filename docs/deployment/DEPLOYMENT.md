# BuffetBot Deployment Guide

This document covers deployment options for the BuffetBot dashboard application.

## Local Development

### Quick Start
```bash
./run_dashboard.sh
```

This script:
- Checks for and clears any processes using port 8501
- Activates the virtual environment (if present)
- Starts the Streamlit dashboard

### Alternative Methods
```bash
# Using Python directly
python scripts/run_dashboard.py

# Using Streamlit directly
streamlit run main.py

# From scripts directory
./scripts/run_dashboard.sh
```

### Stopping the Application
```bash
# Clean shutdown (recommended)
./scripts/stop_dashboard.sh

# Or press Ctrl+C in the terminal
```

## Streamlit Community Cloud

The application is configured for one-click deployment to Streamlit Community Cloud.

### Setup Steps

1. **Fork/Clone the Repository**
   ```bash
   git clone https://github.com/your-username/BuffetBot.git
   ```

2. **Connect to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository

3. **Configure Deployment**
   - **Main file path**: `main.py`
   - **Python version**: `3.12` (or as specified in `.python-version`)
   - **Requirements file**: `requirements-streamlit.txt`

4. **Environment Variables** (if needed)
   - Add any required API keys or configuration in Streamlit Cloud secrets
   - Reference the `env.example` file for available options

### Deployment Flow

| **Environment** | **Entry Point** | **Flow** |
|-----------------|-----------------|----------|
| **Local Development** | `./scripts/run_dashboard.sh` | `scripts/run_dashboard.py` → `main.py` |
| **Streamlit Cloud** | `main.py` | Direct execution |

## Docker Deployment (Optional)

### Using Docker Compose
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop
docker-compose down
```

### Manual Docker
```bash
# Build image
docker build -t buffetbot .

# Run container
docker run -p 8501:8501 buffetbot

# With environment file
docker run -p 8501:8501 --env-file .env buffetbot
```

## Production Considerations

### Performance
- Consider using a reverse proxy (nginx) for production
- Enable caching for better performance
- Monitor memory usage with large datasets

### Security
- Use environment variables for sensitive configuration
- Consider authentication for production deployments
- Regularly update dependencies

### Monitoring
- Set up logging and monitoring
- Use Streamlit's built-in metrics
- Monitor resource usage and performance

## Environment Configuration

### Required Files
- `requirements-streamlit.txt` - Streamlit-specific dependencies
- `main.py` - Entry point for cloud deployment
- `.streamlit/config.toml` - Streamlit configuration (optional)

### Optional Configuration
- `.env` - Environment variables (use `env.example` as template)
- `Dockerfile` - For containerized deployment
- `docker-compose.yml` - For multi-service deployment

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
./scripts/stop_dashboard.sh
# or manually:
lsof -ti:8501 | xargs kill -9
```

**Dependencies not found:**
```bash
pip install -r requirements-streamlit.txt
```

**Virtual environment issues:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements-streamlit.txt
```

### Logs and Debugging

**Local logs:**
- Check terminal output
- Dashboard logs in `logs/` directory

**Streamlit Cloud logs:**
- Available in the Streamlit Cloud dashboard
- Click on "Manage app" → "Logs"

## Support

For deployment issues:
1. Check the logs first
2. Verify all requirements are installed
3. Ensure environment variables are set correctly
4. Check the GitHub issues for known problems
