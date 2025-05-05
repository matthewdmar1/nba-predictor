"""
Main script for NBA Parlay Prediction system
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import configuration
import config

# Import project modules
from src.data.fetch_data import load_nba_data, load_odds_data
from src.data.preprocess import preprocess_data
from src.features.build_features import engineer_features
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model, generate_parlays, simulate_roi
from src.visualization.visualize import plot_feature_importance, plot_roi
from src.evaluation.parlay_tracker import ParlayTracker
from src.evaluation.backtesting import BacktestEngine

def main():
    """
    Run the NBA Parlay Prediction pipeline
    """
    print("Starting NBA Parlay Prediction System...")
    
    # Create directories if they don't exist
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    try:
        games_df = load_nba_data(season=config.NBA_SEASON)
        odds_df = load_odds_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        from src.data.fetch_data import generate_synthetic_data
        games_df, odds_df = generate_synthetic_data()
    
    # Check if data was loaded successfully
    if games_df.empty:
        print("Error: No game data available. Exiting.")
        return    
    
    # 2. Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_data(games_df, odds_df)
    processed_df.to_csv(f"{config.DATA_PROCESSED_DIR}/processed_data.csv", index=False)
    
    # Check if preprocessing was successful
    if processed_df.empty:
        print("Error: Preprocessing failed. Using synthetic data for demonstration.")
        # Generate some synthetic processed data
        import numpy as np
        np.random.seed(42)
        n_games = 100
        processed_df = pd.DataFrame({
            'GAME_DATE': pd.date_range(start='2023-01-01', periods=n_games),
            'HOME_TEAM': [f'Team_{i%15+1}' for i in range(n_games)],
            'AWAY_TEAM': [f'Team_{(i+7)%15+1}' for i in range(n_games)],
            'HOME_PTS': np.random.randint(85, 125, n_games),
            'AWAY_PTS': np.random.randint(85, 125, n_games),
            'HOME_FG_PCT': np.random.uniform(0.4, 0.55, n_games),
            'AWAY_FG_PCT': np.random.uniform(0.4, 0.55, n_games),
            'HOME_FT_PCT': np.random.uniform(0.7, 0.9, n_games),
            'AWAY_FT_PCT': np.random.uniform(0.7, 0.9, n_games),
            'HOME_REB': np.random.randint(30, 55, n_games),
            'AWAY_REB': np.random.randint(30, 55, n_games),
            'HOME_AST': np.random.randint(15, 35, n_games),
            'AWAY_AST': np.random.randint(15, 35, n_games),
            'HOME_STL': np.random.randint(5, 15, n_games),
            'AWAY_STL': np.random.randint(5, 15, n_games),
            'HOME_BLK': np.random.randint(2, 10, n_games),
            'AWAY_BLK': np.random.randint(2, 10, n_games),
            'HOME_TOV': np.random.randint(8, 20, n_games),
            'AWAY_TOV': np.random.randint(8, 20, n_games), 
            'HOME_PLUS_MINUS': np.random.randint(-20, 20, n_games),
            'AWAY_PLUS_MINUS': np.random.randint(-20, 20, n_games),
            'HOME_WIN': np.random.randint(0, 2, n_games)
        })
    
    processed_df.to_csv(f"{config.DATA_PROCESSED_DIR}/processed_data.csv", index=False)

    # 3. Feature engineering
    print("Engineering features...")
    feature_df = engineer_features(processed_df)
    feature_df.to_csv(f"{config.DATA_PROCESSED_DIR}/features.csv", index=False)
    
    # 4. Split data
    print("Splitting data into training and testing sets...")
    X = feature_df.drop(['HOME_WIN', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS'], axis=1)
    y = feature_df['HOME_WIN']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Store game info for reference
    teams_info = feature_df[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
    teams_info_test = teams_info.iloc[X_test.index]
    
    # 5. Train model
    print("Training model...")
    model = train_model(X_train, y_train, cv_folds=config.CV_FOLDS)
    
    # Save model
    from joblib import dump
    dump(model, f"{config.MODELS_DIR}/model.pkl")
    
    # 6. Evaluate model
    print("Evaluating model performance...")
    results_df = evaluate_model(model, X_test, y_test, teams_info_test)
    
    # 7. Generate parlay recommendations
    print("Generating parlay recommendations...")
    parlays = generate_parlays(
        results_df, 
        min_prob=config.MIN_CONFIDENCE, 
        max_games=config.MAX_PARLAY_SIZE
    )
    
    # 8. Display results
    print("\n----- PARLAY RECOMMENDATIONS -----")
    for i, parlay in enumerate(parlays):
        print(f"\nParlay #{i+1}:")
        print(f"Combined probability: {parlay['combined_probability']:.4f}")
        print(f"Number of games: {parlay['parlay_size']}")
        print("\nGames in parlay:")
        for game in parlay['games']:
            prediction = "HOME WIN" if game['predicted_home_win'] else "AWAY WIN"
            print(f"  {game['HOME_TEAM']} vs {game['AWAY_TEAM']} - " 
                  f"Prediction: {prediction} (Confidence: {game['home_win_probability']:.4f})")
    
    # 9. Simulate ROI
    print("\n----- ROI SIMULATION -----")
    roi_results = simulate_roi(parlays, results_df)
    
    # 10. Create visualizations
    print("\nGenerating visualizations...")
    plot_feature_importance(model, X.columns, save_path=f"{config.FIGURES_DIR}/feature_importance.png")
    plot_roi(roi_results, save_path=f"{config.FIGURES_DIR}/roi_simulation.png")
    
    print("\nNBA Parlay Prediction System complete!")

def evaluate_parlay_strategy(parlays, results_df, model):
    """
    Evaluate parlay prediction strategy and track results
    """
    print("\n----- PARLAY STRATEGY EVALUATION -----")
    
    # Initialize tracker
    tracker = ParlayTracker()
    
    # Record predictions
    for i, parlay in enumerate(parlays):
        prediction_id = tracker.record_prediction(
            parlay['games'],
            parlay['combined_probability'],
            model_version=f"v{config.NBA_SEASON.replace('-', '_')}",
            confidence=parlay['combined_probability']
        )
        
        print(f"Recorded prediction #{prediction_id}")
    
    # Get recent performance
    performance = tracker.get_recent_performance(n_days=30)
    
    print("\nRecent parlay performance (last 30 days):")
    if performance['sample_size'] > 0:
        print(f"  Accuracy: {performance['accuracy']:.2%}")
        print(f"  Average Confidence: {performance['average_confidence']:.2%}")
        print(f"  Calibration Error: {performance['calibration_error']:.2%}")
        print(f"  Total Profit: ${performance['profit']:.2f}")
        print(f"  ROI: {performance['roi']:.2f}%")
        print(f"  Sample Size: {performance['sample_size']} parlays")
    else:
        print("  No recent parlays recorded")
    
    # Run a quick backtest if we have enough data
    if 'GAME_DATE' in results_df.columns:
        print("\nRunning quick backtest on test data...")
        
        # Get feature columns
        non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 
                          'HOME_PTS', 'AWAY_PTS', 'HOME_WIN', 
                          'predicted_home_win', 'home_win_probability', 'correct_prediction']
        feature_cols = [col for col in results_df.columns if col not in non_feature_cols]
        
        if len(feature_cols) > 0:
            backtest_engine = BacktestEngine(
                results_df,
                target_column='HOME_WIN',
                feature_columns=feature_cols,
                date_column='GAME_DATE',
                min_confidence=config.MIN_CONFIDENCE,
                max_games=config.MAX_PARLAY_SIZE
            )
            
            backtest_results = backtest_engine.run_backtest()
            
            if backtest_results['total_bets'] > 0:
                print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
                print(f"  Total Profit: ${backtest_results['total_profit']:.2f}")
                print(f"  ROI: {backtest_results['total_roi']:.2f}%")
                print(f"  Total Bets: {backtest_results['total_bets']}")
            else:
                print("  Not enough data for backtest")
    
    return performance
if __name__ == "__main__":
    main()