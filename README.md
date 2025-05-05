# NBA Parlay Predictor

A machine learning system for predicting NBA game outcomes and generating optimal betting parlays.

## Overview

This project uses historical NBA game data and betting odds to train a machine learning model that predicts the outcomes of NBA games. The model is used to identify high-confidence predictions and generate parlay betting recommendations designed to maximize ROI.

## Features

- **Data Collection:** Fetch NBA game stats via the NBA API
- **Machine Learning:** Predict game outcomes with logistic regression
- **Parlay Generation:** Automatically create optimal parlays based on predictions
- **ROI Analysis:** Simulate and track betting performance
- **Web Interface:** Interactive frontend for real-time predictions

## Project Structure

```
nba-parlay-predictor/
├── config.py                 # Configuration settings
├── main.py                   # Main script to run the pipeline
├── data/                     # Data storage directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── results/                  # Output directory
│   ├── models/               # Trained models
│   └── figures/              # Generated visualizations
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   ├── features/             # Feature engineering
│   ├── models/               # Model training and evaluation
│   ├── visualization/        # Visualization utilities
│   └── evaluation/           # Evaluation tools
└── frontend/                 # Streamlit web interface
    └── frontend.py           # Web app code
```

## Prerequisites

- Python 3.8+ 
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nba-parlay-predictor.git
cd nba-parlay-predictor
```

### Step 2: Create and Activate Virtual Environment

For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 4: Create Directory Structure

Make sure all required directories exist:

```bash
mkdir -p data/raw data/processed results/models results/figures
```

## Usage

### Run the Main Pipeline

To train the model and generate parlay recommendations:

```bash
python main.py
```

This will:
1. Fetch or generate NBA game data
2. Preprocess the data and engineer features
3. Train a prediction model
4. Generate parlay recommendations
5. Simulate ROI for the recommended parlays
6. Create visualizations

### Using the Helper Script

For more control, use the `run.py` script:

```bash
# Set up environment and install requirements
python run.py --setup --install

# Generate sample data and run the pipeline
python run.py --generate-data --run

# Run everything (setup, install, generate data, run pipeline)
python run.py --all
```

### Web Interface

For a user-friendly interactive experience, run the Streamlit web app:

```bash
cd frontend
streamlit run frontend.py
```

This will launch a browser-based interface where you can:
- View live betting odds
- Get predictions for upcoming games
- Build custom parlays
- Analyze expected ROI

## Data Sources

The system can use data from several sources:

1. **NBA API:** Real NBA game statistics (requires an internet connection)
2. **Betting APIs:** Real-time odds from various sportsbooks
3. **Synthetic Data:** For testing when real data is unavailable

## Model Training

The default model is a logistic regression classifier with the following features:
- Team performance metrics (FG%, rebounds, assists, etc.)
- Statistical differences between home and away teams
- Betting market information (odds, spreads)

## Configuration

You can customize the behavior of the system by editing `config.py`. Key settings include:

- `NBA_SEASON`: NBA season to analyze (format: 'YYYY-YY')
- `TEST_SIZE`: Proportion of data to use for testing (default: 0.3)
- `MIN_CONFIDENCE`: Minimum prediction confidence for parlay selection (default: 0.65)
- `MAX_PARLAY_SIZE`: Maximum number of games in a parlay (default: 3)

## Evaluation

To evaluate model performance, use the comprehensive evaluation script:

```bash
python run_evaluation.py comprehensive
```

Or run specific evaluation tools:

```bash
# Run backtesting simulation
python run_evaluation.py backtest --confidence 0.7 --max-games 3

# Compare different models
python run_evaluation.py compare

# Analyze feature importance
python run_evaluation.py features --top-n 15
```

## Integration with Real Data

To use real NBA data instead of synthetic data:

```bash
python integrate_real_data.py
```

Note that this requires an internet connection and may be subject to API rate limits.

## How It Works

1. **Data Collection**: The system fetches NBA game data and betting odds either from APIs or generates synthetic data.

2. **Feature Engineering**: Raw data is processed into predictive features, including team performance metrics, statistical differences, and betting market information.

3. **Model Training**: A logistic regression model is trained to predict home team wins based on the engineered features.

4. **Parlay Generation**: The system identifies high-confidence predictions and combines them into parlays designed to maximize expected return.

5. **ROI Simulation**: The performance of the recommended parlays is simulated to estimate potential returns.

6. **Web Interface**: Users can interact with predictions and build custom parlays through a user-friendly web app.

## Extending the Project

### Adding New Features

To add new predictive features, edit `src/features/build_features.py` and implement your feature engineering logic.

### Testing Different Models

The project uses logistic regression by default, but you can implement and test different models in `src/models/train_model.py`.

### Adding Real-time Data

To use real-time data for upcoming games, extend the data fetching modules to retrieve the latest game information and odds.

## Disclaimer

This software is for educational and entertainment purposes only. Sports betting involves financial risk, and no betting system can guarantee profits. Always gamble responsibly and within your means.

## Acknowledgments

- NBA API for providing access to game statistics
- Streamlit for the interactive web framework
- scikit-learn for machine learning tools
