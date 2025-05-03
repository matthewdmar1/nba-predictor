#!/usr/bin/env python
"""
Launch script for NBA Parlay Predictor
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_environment():
    """Create necessary directories and check requirements"""
    print("Setting up NBA Parlay Predictor environment...")
    
    # Check if we're in the correct directory
    if not Path('nba-parlay-predictor').exists():
        print("Creating project directory structure...")
        # Create directory structure
        dirs = [
            'nba-parlay-predictor/data/raw',
            'nba-parlay-predictor/data/processed',
            'nba-parlay-predictor/notebooks',
            'nba-parlay-predictor/results/models',
            'nba-parlay-predictor/results/figures'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Change to project directory if not already in it
    if not Path('main.py').exists() and Path('nba-parlay-predictor/main.py').exists():
        os.chdir('nba-parlay-predictor')
        print("Changed to project directory:", os.getcwd())
    
    # Check for required files
    required_files = ['main.py', 'config.py', 'requirements.txt']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Warning: Missing required files: {', '.join(missing_files)}")
        return False
    
    return True

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error installing requirements. Please check your Python environment.")
        return False

def generate_sample_data():
    """Generate synthetic data for testing"""
    print("Generating sample data...")
    try:
        if Path('generate_sample_data.py').exists():
            subprocess.run([sys.executable, "generate_sample_data.py"], check=True)
        else:
            print("Sample data generator not found. Skipping this step.")
        return True
    except subprocess.CalledProcessError:
        print("Error generating sample data.")
        return False

def run_pipeline():
    """Run the NBA Parlay Prediction pipeline"""
    print("Running NBA Parlay Prediction pipeline...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("Pipeline completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error running the pipeline.")
        return False

def launch_notebook():
    """Launch Jupyter notebook"""
    print("Launching Jupyter notebook...")
    try:
        subprocess.Popen(["jupyter", "notebook", "--notebook-dir=."])
        print("Jupyter notebook server started. You can access it through your browser.")
        return True
    except FileNotFoundError:
        print("Jupyter not found. Make sure it's installed.")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NBA Parlay Predictor Launch Script')
    parser.add_argument('--setup', action='store_true', help='Set up the environment')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    parser.add_argument('--generate-data', action='store_true', help='Generate sample data')
    parser.add_argument('--run', action='store_true', help='Run the pipeline')
    parser.add_argument('--notebook', action='store_true', help='Launch Jupyter notebook')
    parser.add_argument('--all', action='store_true', help='Perform all setup steps and run the pipeline')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    if args.all or args.setup:
        if not setup_environment():
            print("Environment setup failed. Exiting.")
            sys.exit(1)
    
    if args.all or args.install:
        if not install_requirements():
            print("Requirements installation failed. Exiting.")
            sys.exit(1)
    
    if args.all or args.generate_data:
        generate_sample_data()
    
    if args.all or args.run:
        if not run_pipeline():
            print("Pipeline execution failed. Exiting.")
            sys.exit(1)
    
    if args.all or args.notebook:
        launch_notebook()

if __name__ == "__main__":
    main()