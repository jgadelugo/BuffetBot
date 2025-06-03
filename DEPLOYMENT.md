# BuffetBot Dashboard - Deployment Guide

## 🚀 Streamlit Cloud Deployment (Post-Refactoring)

### Quick Setup for Streamlit Cloud

1. **Repository**: `https://github.com/jgadelugo/BuffetBot`
2. **Branch**: `main`
3. **Main file path**: `main.py`
4. **Python version**: 3.10+

### Entry Points (Updated Structure)

After the refactoring, use **only** these entry points:

1. **`main.py`** ✅ **RECOMMENDED** - Clean entry point for Streamlit Cloud
2. **`buffetbot/dashboard/streamlit_app.py`** - Alternative wrapper (if needed)

⚠️ **DEPRECATED**: Old `dashboard/app.py` path no longer exists

### Deployment Files

Ensure these files are properly configured:

#### **`main.py`** (Entry Point)
```python
from buffetbot.dashboard.app import main

if __name__ == "__main__":
    main()
```

#### **`requirements.txt`** (Python Dependencies)
All dependencies are specified and automatically installed

#### **`packages.txt`** (System Dependencies)
```
build-essential
```

#### **`.streamlit/config.toml`** (Streamlit Configuration)
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"  # Dark theme
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[server]
port = 8501
maxUploadSize = 200
```

### 🎯 Streamlit Cloud Setup Steps

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Connect Repository**: `jgadelugo/BuffetBot`
3. **Branch**: `main`
4. **Main file path**: `main.py`
5. **Click**: Deploy

### ✅ Key Benefits of New Structure

- **✅ No Path Setup**: Eliminated manual `sys.path` manipulation
- **✅ Proper Packaging**: Uses `pip install -e .` for clean imports
- **✅ Standard Imports**: All imports use `buffetbot.*` namespace
- **✅ Cloud Compatible**: Works seamlessly with Streamlit Cloud
- **✅ Zero Configuration**: No special environment setup needed

### 🧪 Testing Deployment Locally

```bash
# Method 1: Test exactly like Streamlit Cloud runs it
streamlit run main.py

# Method 2: Use the development script
./run_dashboard.sh

# Method 3: Use the Python wrapper
python run_dashboard.py
```

### 🔧 Local Development vs Deployment

| Environment | Command | Entry Point |
|-------------|---------|-------------|
| **Streamlit Cloud** | `streamlit run main.py` | `main.py` |
| **Local Development** | `./run_dashboard.sh` | `run_dashboard.py` → `main.py` |
| **Manual Local** | `streamlit run main.py` | `main.py` |

### 🚨 Troubleshooting

#### If deployment fails:

1. **Check Entry Point**: Must be `main.py` (not old paths)
2. **Verify Branch**: Must be `main` branch with latest commits
3. **Python Version**: Ensure 3.10+ is selected
4. **Requirements**: All dependencies in `requirements.txt`

#### Common Issues:

- ❌ **Using old `dashboard/app.py` path** → ✅ Use `main.py`
- ❌ **Missing dependencies** → ✅ Check `requirements.txt`
- ❌ **Import errors** → ✅ Ensure package installed properly

### 🎉 Deployment Verification

Once deployed, your app should:
- ✅ Load with dark theme
- ✅ Show "Stock Analysis Dashboard" title
- ✅ Have working sidebar with ticker input
- ✅ Display all tabs properly
- ✅ No import or path errors

### 📧 Support

If you encounter issues, check:
1. Streamlit Cloud deployment logs
2. GitHub repository latest commits
3. Package installation status
