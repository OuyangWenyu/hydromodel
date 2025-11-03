# Installation

This guide covers all methods to install hydromodel on different platforms.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Disk Space**: ~500 MB for package + data storage for datasets

## Quick Installation

The fastest way to install hydromodel:

```bash
pip install hydromodel
```

This installs the latest stable release from PyPI.

## Recommended Installation Methods

### Method 1: Using pip (Standard)

For most users, pip is the recommended installation method:

```bash
# Create a virtual environment (recommended)
python -m venv hydromodel-env

# Activate virtual environment
# On Windows:
hydromodel-env\Scripts\activate
# On macOS/Linux:
source hydromodel-env/bin/activate

# Install hydromodel
pip install hydromodel

# Install hydrodataset for data access
pip install hydrodataset
```

### Method 2: Using uv (Faster)

[uv](https://github.com/astral-sh/uv) is a faster package manager:

```bash
# Install uv
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install packages
uv pip install hydromodel hydrodataset
```

Installation with uv is typically 10-100x faster than pip.

### Method 3: Using conda

If you use conda/mamba:

```bash
# Create environment
conda create -n hydromodel python=3.11

# Activate environment
conda activate hydromodel

# Install from conda-forge (if available)
conda install -c conda-forge hydromodel

# Or use pip within conda
pip install hydromodel hydrodataset
```

## Installation from Source

For developers or to get the latest development version:

### Option A: Direct from GitHub

```bash
pip install git+https://github.com/OuyangWenyu/hydromodel.git
```

### Option B: Clone and Install

```bash
# Clone repository
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel

# Install in editable mode (for development)
pip install -e .

# Or install with all development dependencies
pip install -e ".[dev,docs]"
```

### Option C: Using uv (Recommended for Developers)

```bash
# Clone repository
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel

# Create environment and install dependencies
uv sync --all-extras

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

## Optional Dependencies

### For Data Access

```bash
# CAMELS and other public datasets
pip install hydrodataset
```

### For Visualization

```bash
# Plotting and analysis
pip install matplotlib seaborn plotly
```

### For Development

```bash
# Testing, linting, documentation
pip install pytest black flake8 mkdocs mkdocs-material
```

### All Optional Dependencies

```bash
# Install everything
pip install "hydromodel[all]"
```

## Platform-Specific Instructions

### Windows

1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Install hydromodel**:
   ```cmd
   pip install hydromodel hydrodataset
   ```

3. **Verify installation**:
   ```cmd
   python -c "import hydromodel; print(hydromodel.__version__)"
   ```

### macOS

1. **Install Python** (using Homebrew):
   ```bash
   brew install python@3.11
   ```

2. **Install hydromodel**:
   ```bash
   pip3 install hydromodel hydrodataset
   ```

3. **Verify installation**:
   ```bash
   python3 -c "import hydromodel; print(hydromodel.__version__)"
   ```

### Linux (Ubuntu/Debian)

1. **Install Python**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Install hydromodel**:
   ```bash
   pip3 install hydromodel hydrodataset
   ```

3. **Verify installation**:
   ```bash
   python3 -c "import hydromodel; print(hydromodel.__version__)"
   ```

## Verifying Installation

After installation, verify everything works:

```python
# Test import
import hydromodel
import hydrodataset

# Check versions
print(f"hydromodel version: {hydromodel.__version__}")

# Test basic functionality
from hydromodel.models.model_factory import model_factory

# Create a model
model = model_factory(model_name="xaj_mz")
print(f"✓ Successfully created {model.__class__.__name__}")
print(f"✓ Model has {model.param_limits.shape[0]} parameters")
print("✓ Installation verified!")
```

Expected output:
```
hydromodel version: 0.1.0
✓ Successfully created XajMz
✓ Model has 15 parameters
✓ Installation verified!
```

## Configuration Setup

After installation, configure data paths:

### Step 1: Create Configuration File

Create `hydro_setting.yml` in your home directory:

**Windows:** `C:\Users\YourUsername\hydro_setting.yml`
**macOS/Linux:** `~/hydro_setting.yml`

```yaml
local_data_path:
  root: '/path/to/data'
  datasets-origin: '/path/to/data/datasets'
  cache: '/path/to/data/.cache'
```

### Step 2: Verify Configuration

```python
from hydromodel import SETTING

print("Configuration loaded:")
print(f"Root: {SETTING.get('local_data_path', {}).get('root')}")
print(f"Datasets: {SETTING.get('local_data_path', {}).get('datasets-origin')}")
```

## Troubleshooting

### Common Issues

#### Issue 1: "No module named 'hydromodel'"

**Solution**: Make sure the virtual environment is activated:
```bash
# Check Python path
which python  # macOS/Linux
where python  # Windows

# Should point to virtual environment, not system Python
```

#### Issue 2: "Permission denied" during installation

**Solution**: Use `--user` flag or virtual environment:
```bash
pip install --user hydromodel
```

#### Issue 3: Dependency conflicts

**Solution**: Use a fresh virtual environment:
```bash
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows
pip install hydromodel
```

#### Issue 4: Slow pip installation

**Solution**: Use uv for faster installation:
```bash
pip install uv
uv pip install hydromodel
```

#### Issue 5: "Microsoft Visual C++ required" (Windows)

**Solution**: Install Visual C++ Build Tools:
- Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or install via Anaconda which includes pre-compiled packages

### Getting Help

If you encounter issues:

1. **Check documentation**: Browse [docs](https://OuyangWenyu.github.io/hydromodel)
2. **Search issues**: [GitHub Issues](https://github.com/OuyangWenyu/hydromodel/issues)
3. **Ask questions**: Open a new issue with:
   - Your OS and Python version
   - Full error message
   - Installation command used

## Updating hydromodel

### Update to Latest Stable Version

```bash
pip install --upgrade hydromodel
```

### Update to Development Version

```bash
pip install --upgrade git+https://github.com/OuyangWenyu/hydromodel.git
```

### Check Current Version

```python
import hydromodel
print(hydromodel.__version__)
```

## Uninstallation

To remove hydromodel:

```bash
pip uninstall hydromodel
```

To remove everything including dependencies:

```bash
# List installed packages
pip list | grep hydro

# Uninstall
pip uninstall hydromodel hydrodataset
```

To remove the virtual environment:

```bash
# Deactivate first
deactivate

# Remove directory
rm -rf hydromodel-env  # Linux/macOS
rmdir /s hydromodel-env  # Windows
```

## Docker Installation (Advanced)

For reproducible environments:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir hydromodel hydrodataset

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Run your script
CMD ["python", "your_script.py"]
```

Build and run:

```bash
docker build -t hydromodel-app .
docker run -v $(pwd)/data:/app/data hydromodel-app
```

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Quick Start Guide](quickstart.md)
2. **Configuration**: Set up [data paths and settings](usage.md#configuration)
3. **Tutorial**: Try the [usage examples](usage.md)
4. **API Documentation**: Browse the [API reference](api.md)

## Support

- **Documentation**: https://OuyangWenyu.github.io/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues
- **Discussions**: https://github.com/OuyangWenyu/hydromodel/discussions
