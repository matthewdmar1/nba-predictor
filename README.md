# NBA Parlay Predictor

A machine learning system for predicting NBA game outcomes and generating optimal betting parlays.

## Overview

This project uses historical NBA game data and betting odds to train a machine learning model that predicts the outcomes of NBA games. The model is used to identify high-confidence predictions and generate parlay betting recommendations designed to maximize ROI.

## Features

- Data fetching from NBA stats API
- Automated feature engineering for game statistics
- Machine learning model for predicting game outcomes
- Parlay generation and ROI simulation
- Visualization of model performance and betting results
- Jupyter notebooks for interactive exploration

## Prerequisites

- Python 3.8+ 
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

1. Clone the repository (or download and extract the ZIP file):

```bash
git clone https://github.com/yourusername/nba-parlay-predictor.git
cd nba-parlay-predictor
```

2. Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
nba-parlay-predictor/
├── config.py                 # Configuration settings
├── main.py                   # Main script to run the pipeline
├── fetch_data.py             # Alternative data fetching module
├── run.py                    # Helper script for setting up and running
├── integrate_real_data.py    # Script for using real NBA data
├── nba_data_demo.py          # Demo for NBA data retrieval
├── requirements.txt          # Required Python packages
├── data/                     # Data storage directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation.ipynb
├── results/                  # Output directory
│   ├── models/               # Trained models
│   └── figures/              # Generated visualizations
└── src/                      # Source code
    ├── data/                 # Data processing modules
    ├── features/             # Feature engineering
    ├── models/               # Model training and evaluation
    └── visualization/        # Visualization utilities
```

## Usage

### Quick Start

Run the main pipeline with default settings:

```bash
python main.py
```

This will:
1. Fetch or generate NBA game data
2. Preprocess the data and engineer features
3. Train a prediction model
4. Generate parlay recommendations
5. Simulate ROI for the recommended parlays
6. Create visualizations of the results

### Running with Options

For more control, use the `run.py` script with various options:

```bash
# Set up environment and install requirements
python run.py --setup --install

# Generate sample data and run the pipeline
python run.py --generate-data --run

# Launch Jupyter notebook for interactive exploration
python run.py --notebook

# Run everything (setup, install, generate data, run pipeline, launch notebook)
python run.py --all
```

### Using Real NBA Data

To use real NBA data instead of synthetic data:

```bash
python integrate_real_data.py
```

Note that this requires an internet connection and may be subject to API rate limits.

### Exploring in Jupyter Notebooks

Launch Jupyter to explore the data and models interactively:

```bash
jupyter notebook notebooks/
```

The provided notebooks include:
- `01_data_exploration.ipynb`: Explore and visualize the NBA game data
- `02_model_evaluation.ipynb`: Evaluate different models and parlay strategies

## Configuration

You can customize the behavior of the system by editing `config.py`. Key settings include:

- `NBA_SEASON`: NBA season to analyze (format: 'YYYY-YY')
- `TEST_SIZE`: Proportion of data to use for testing (default: 0.3)
- `MIN_CONFIDENCE`: Minimum prediction confidence for parlay selection (default: 0.65)
- `MAX_PARLAY_SIZE`: Maximum number of games in a parlay (default: 3)

## How It Works

1. **Data Collection**: The system fetches NBA game data and betting odds either from APIs or generates synthetic data.

2. **Feature Engineering**: Raw data is processed into predictive features, including team performance metrics, statistical differences, and betting market information.

3. **Model Training**: A logistic regression model is trained to predict home team wins based on the engineered features.

4. **Parlay Generation**: The system identifies high-confidence predictions and combines them into parlays designed to maximize expected return.

5. **ROI Simulation**: The performance of the recommended parlays is simulated to estimate potential returns.

## Extending the Project

### Adding New Features

To add new predictive features, edit `src/features/build_features.py` and implement your feature engineering logic.

### Testing Different Models

The project uses logistic regression by default, but you can implement and test different models in `src/models/train_model.py`.

### Adding Real-time Data

To use real-time data for upcoming games, extend the data fetching modules to retrieve the latest game information and odds.