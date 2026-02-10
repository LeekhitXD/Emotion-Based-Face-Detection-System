# Installation Guide

## Quick Installation

Open your terminal and run these commands:

### Option 1: Install all dependencies at once

```bash
cd /Users/leekhitpatwal/Desktop/face_detection_ntcc
pip3 install -r requirements.txt
```

### Option 2: Install packages individually

If the above doesn't work, try installing each package separately:

```bash
pip3 install opencv-python
pip3 install tensorflow
pip3 install numpy
pip3 install keras
pip3 install Pillow
pip3 install imutils
```

### Option 3: Use Python's built-in pip module

```bash
python3 -m pip install -r requirements.txt
```

### Option 4: Use a virtual environment (Recommended)

If you're having permission issues, use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Then when you run the application:

```bash
# Make sure virtual environment is activated
source venv/bin/activate
python main.py
```

## Troubleshooting

### Permission Denied Error

If you get a "Permission denied" error, try:

1. **Use `--user` flag:**
   ```bash
   pip3 install --user -r requirements.txt
   ```

2. **Use `sudo` (not recommended, but works):**
   ```bash
   sudo pip3 install -r requirements.txt
   ```

3. **Use a virtual environment** (best practice):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Module Not Found After Installation

If you still get "ModuleNotFoundError" after installation:

1. Make sure you're using the same Python interpreter:
   ```bash
   which python3
   python3 --version
   ```

2. Check if packages are installed:
   ```bash
   python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

3. If using a virtual environment, make sure it's activated before running the script.

## Verify Installation

After installing, verify everything works:

```bash
python3 -c "import cv2; import tensorflow as tf; import numpy as np; print('All packages imported successfully!')"
```

If this command runs without errors, you're ready to go!

## Next Steps

Once installation is complete, run:

```bash
python3 main.py
```
