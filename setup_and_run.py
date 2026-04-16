"""
Quick setup and run script for Stock Price Prediction app.
Run this with: python setup_and_run.py
"""
import subprocess
import sys
import os

def install_packages():
    print("Installing required packages...")
    packages = [
        "flask",
        "yfinance",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "plotly",
        "ta",
        "requests",
    ]
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("All packages installed!\n")

def run_app():
    print("=" * 50)
    print("  Stock Price Prediction App Starting...")
    print("=" * 50)
    print("\n  Open browser at: http://localhost:5000\n")
    print("  Press Ctrl+C to stop\n")
    print("=" * 50)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    try:
        import flask, yfinance, plotly, sklearn
        print("Dependencies already installed.\n")
    except ImportError:
        install_packages()
    run_app()
