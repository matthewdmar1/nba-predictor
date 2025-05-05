"""
Configuration for NBA Parlay Prediction project
"""

# Paths
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'results/models'
FIGURES_DIR = 'results/figures'

# Data parameters
NBA_SEASON = '2023-24'  # NBA season to analyze
TEST_SIZE = 0.3         # Proportion of data for testing

# Model parameters
RANDOM_STATE = 42       # For reproducibility
CV_FOLDS = 5            # Cross-validation folds

# Parlay parameters
MIN_CONFIDENCE = 0.65   # Minimum confidence for parlay selection
MAX_PARLAY_SIZE = 3     # Maximum number of games in a parlay

# Evaluation parameters
ENABLE_TEMPORAL_VALIDATION = True  # Enable/disable temporal validation
EVALUATION_DIR = 'results/evaluation'  # Directory for evaluation results
TRACK_PARLAYS = True  # Enable/disable parlay tracking