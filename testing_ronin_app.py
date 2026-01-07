# Usage:
# Install dependencies and run the Streamlit app from a terminal:
#   pip install streamlit pandas joblib scikit-learn plotly
#   streamlit run app.py

# Optional: run the two commands programmatically from Python by executing this file.
# Note: Automatically installing packages from scripts can be surprising; use with care.
if __name__ == "__main__":
    import subprocess
    import sys

    print("Installing required packages (if missing) and launching Streamlit app...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "pandas", "joblib", "scikit-learn", "plotly"])
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "ronin_app.py"])