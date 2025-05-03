"""
Integration Script for NBA Parlay Predictor with Real NBA Data
This script connects real NBA data retrieval with the parlay prediction system.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import NBA data functions (from the same directory)
from nba_data_demo import get_team_data, get_games_data, get_team_stats, create_betting_features

# Import project modules
try:
    from src.data.preprocess import preprocess_data
    from src.features.build_features import engineer_features
    from src.models.train_model import train_model
    from src.models.evaluate import evaluate_model, generate_parlays, simulate_roi
    from src.visualization.visualize import plot_feature_importance, plot_roi, plot_win_probability_distribution
    
    # Import configuration
    import config
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    print("Current directory:", os.getcwd())
    sys.exit(1)

def fetch_and_prepare_data(season='2023-24', use_cached=True):
    """
    Fetch real NBA data and prepare it for the prediction model
    
    Parameters:
    -----------
    season : str
        NBA season in format 'YYYY-YY'
    use_cached : bool
        Whether to use cached data if available
        
    Returns:
    --------
    games_df : pandas DataFrame
        NBA game data
    odds_df : pandas DataFrame
        Betting odds data
    """
    # Define cache file paths
    games_cache = f"data/raw/nba_games_{season.replace('-', '_')}.csv"
    odds_cache = f"data/processed/betting_odds.csv"
    team_stats_cache = f"data/raw/nba_team_stats_{season.replace('-', '_')}.csv"
    
    # Check if we can use cached data
    if use_cached and os.path.exists(games_cache) and os.path.exists(odds_cache):
        print(f"Loading cached data for {season} season...")
        games_df = pd.read_csv(games_cache)
        odds_df = pd.read_csv(odds_cache)
        return games_df, odds_df
    
    print(f"Fetching fresh NBA data for {season} season...")
    
    # Fetch team data (mostly for reference)
    get_team_data()
    
    # Fetch game data
    games_df = get_games_data(season)
    if games_df is None:
        print("Failed to retrieve game data, using fallback...")
        # Try to load from cache if it exists
        if os.path.exists(games_cache):
            games_df = pd.read_csv(games_cache)
        else:
            # Fall back to synthetic data
            print("No cached game data found. Using synthetic data.")
            from src.data.fetch_data import generate_synthetic_data
            games_df, _ = generate_synthetic_data()
    
    # Fetch team stats
    team_stats_df = get_team_stats(season)
    if team_stats_df is None and os.path.exists(team_stats_cache):
        print("Failed to retrieve team stats. Loading from cache...")
        team_stats_df = pd.read_csv(team_stats_cache)
    
    # Generate betting features based on team performance
    # (In a production system, you'd fetch real odds from a betting API)
    if team_stats_df is not None:
        odds_df = create_betting_features(games_df, team_stats_df)
    else:
        # Fall back to synthetic data if stats fetching failed
        print("Warning: Could not fetch team stats. Using fully synthetic odds data.")
        from src.data.fetch_data import generate_synthetic_data
        _, odds_df = generate_synthetic_data()
    
    if odds_df is None:
        print("Failed to create betting features, using synthetic data...")
        from src.data.fetch_data import generate_synthetic_data
        _, odds_df = generate_synthetic_data()
    
    return games_df, odds_df

def run_full_pipeline_with_real_data(season='2023-24', use_cached=True):
    """
    Run the full NBA Parlay Prediction pipeline with real NBA data
    
    Parameters:
    -----------
    season : str
        NBA season in format 'YYYY-YY'
    use_cached : bool
        Whether to use cached data if available
    """
    print("=" * 80)
    print(f"NBA Parlay Prediction System with Real Data - Season: {season}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # 1. Fetch and prepare real NBA data
    print("\n1. Fetching and preparing real NBA data...")
    games_df, odds_df = fetch_and_prepare_data(season, use_cached)
    
    if games_df is None or odds_df is None:
        print("Error: Failed to obtain required data. Pipeline aborted.")
        return None, None
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    try:
        processed_df = preprocess_data(games_df, odds_df)
        processed_file = f"{config.DATA_PROCESSED_DIR}/processed_data.csv"
        processed_df.to_csv(processed_file, index=False)
        print(f"   Saved processed data to {processed_file}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None
    
    # 3. Engineer features
    print("\n3. Engineering features...")
    try:
        feature_df = engineer_features(processed_df)
        feature_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
        feature_df.to_csv(feature_file, index=False)
        print(f"   Saved features to {feature_file}")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None, None
    
    # 4. Split data
    print("\n4. Splitting data into training and testing sets...")
    from sklearn.model_selection import train_test_split
    
    # Drop non-feature columns for model training
    non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
    feature_cols = [col for col in feature_df.columns if col not in non_feature_cols]
    
    X = feature_df[feature_cols]
    y = feature_df['HOME_WIN']
    
    # Save game info for reference
    teams_info = feature_df[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, teams_info_train, teams_info_test = train_test_split(
        X, y, teams_info, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"   Training set size: {X_train.shape[0]} games")
    print(f"   Testing set size: {X_test.shape[0]} games")
    
    # 5. Train model
    print("\n5. Training prediction model...")
    try:
        model = train_model(X_train, y_train, cv_folds=config.CV_FOLDS)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None
    
    # Save model
    from joblib import dump
    model_file = f"{config.MODELS_DIR}/model.pkl"
    dump(model, model_file)
    print(f"   Saved model to {model_file}")
    
    # 6. Evaluate model
    print("\n6. Evaluating model performance...")
    try:
        results_df = evaluate_model(model, X_test, y_test, teams_info_test)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None, None
    
    # 7. Generate parlay recommendations
    print("\n7. Generating parlay recommendations...")
    try:
        parlays = generate_parlays(
            results_df, 
            min_prob=config.MIN_CONFIDENCE, 
            max_games=config.MAX_PARLAY_SIZE
        )
    except Exception as e:
        print(f"Error generating parlays: {e}")
        return None, None
    
    # 8. Display results
    print("\n" + "=" * 40)
    print("PARLAY RECOMMENDATIONS")
    print("=" * 40)
    
    for i, parlay in enumerate(parlays):
        print(f"\nParlay #{i+1}:")
        print(f"Combined probability: {parlay['combined_probability']:.4f}")
        print(f"Number of games: {parlay['parlay_size']}")
        print("\nGames in parlay:")
        for game in parlay['games']:
            prediction = "HOME WIN" if game['predicted_home_win'] else "AWAY WIN"
            confidence = game['home_win_probability'] if game['predicted_home_win'] else (1 - game['home_win_probability'])
            
            print(f"  {game['HOME_TEAM']} vs {game['AWAY_TEAM']} - " 
                  f"Prediction: {prediction} (Confidence: {confidence:.4f})")
    
    # 9. Simulate ROI
    print("\n8. Simulating ROI for parlay recommendations...")
    try:
        roi_results = simulate_roi(parlays, results_df)
    except Exception as e:
        print(f"Error simulating ROI: {e}")
        roi_results = None
    
    # 10. Create visualizations
    print("\n9. Generating visualizations...")
    
    try:
        # Feature importance plot
        feature_importance_file = f"{config.FIGURES_DIR}/feature_importance.png"
        plot_feature_importance(model, X.columns, save_path=feature_importance_file)
        print(f"   Saved feature importance plot to {feature_importance_file}")
        
        # ROI simulation plot
        if roi_results:
            roi_file = f"{config.FIGURES_DIR}/roi_simulation.png"
            plot_roi(roi_results, save_path=roi_file)
            print(f"   Saved ROI simulation plot to {roi_file}")
        
        # Win probability distribution
        win_prob_file = f"{config.FIGURES_DIR}/win_probability_distribution.png"
        plot_win_probability_distribution(results_df, save_path=win_prob_file)
        print(f"   Saved win probability distribution plot to {win_prob_file}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print("\n" + "=" * 80)
    print("NBA Parlay Prediction System complete!")
    print("=" * 80)
    
    return parlays, roi_results

if __name__ == "__main__":
    # You can specify different seasons or use cached data
    run_full_pipeline_with_real_data(season='2023-24', use_cached=True)