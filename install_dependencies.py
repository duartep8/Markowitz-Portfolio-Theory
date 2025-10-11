import subprocess
import sys

# List of required packages
packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "yfinance"
]

# Install each package
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("All dependencies installed successfully!")
