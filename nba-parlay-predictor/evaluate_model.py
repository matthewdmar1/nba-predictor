import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns 

# Import project modules
from src.data.preprocess import preprocess_data
from src.features.build_features import engineer_features
from src.models.train_model import train_model

# Import evaluation modules
from src.evaluation.parlay_tracker import ParlayTracker
from src.evaluation.temporal_validation import time_based_validation, plot_temporal_validation_results
from src.evaluation.backtesting import BacktestEngine
from src.evaluation.model_comparison import compare_models, plot_model_comparison
from src.evaluation.feature_analysis import analyze_features, plot_feature_analysis

def load_and_prepare_data(config):
    """Load and prepare data for evaluation"""
    print("Loading and preparing data...")
    
    # Load processed data if available
    processed_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
    
    if os.path.exists(processed_file):
        print(f"Loading processed data from {processed_file}")
        feature_df = pd.read_csv(processed_file)
        
        # Convert date to datetime
        if 'GAME_DATE' in feature_df.columns:
            feature_df['GAME_DATE'] = pd.to_datetime(feature_df['GAME_DATE'])
    else:
        # Load and process raw data
        print("Processed data not found. Loading raw data...")
        
        from src.data.fetch_data import load_nba_data, load_odds_data
        
        try:
            games_df = load_nba_data(season=config.NBA_SEASON)
            odds_df = load_odds_data()
            
            # Preprocess data
            processed_df = preprocess_data(games_df, odds_df)
            
            # Engineer features
            feature_df = engineer_features(processed_df)
            
            # Save processed data
            os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
            feature_df.to_csv(processed_file, index=False)
            
        except Exception as e:
            print(f"Error loading and processing data: {e}")
            print("Using synthetic data for testing...")
            
            from src.data.fetch_data import generate_synthetic_data
            games_df, odds_df = generate_synthetic_data()
            processed_df = preprocess_data(games_df, odds_df)
            feature_df = engineer_features(processed_df)
    
    return feature_df

def run_comprehensive_evaluation(data, config, output_dir):
    """Run comprehensive model evaluation"""
    print("\nRunning comprehensive evaluation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare features and target
    target_column = 'HOME_WIN'
    non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
    feature_columns = [col for col in data.columns if col not in non_feature_cols]
    
    # 1. Temporal validation
    print("\n1. Running temporal validation...")
    
    temporal_results = time_based_validation(
        data, 
        target_column=target_column,
        feature_columns=feature_columns,
        time_column='GAME_DATE',
        min_train_size=0.5,
        n_splits=5
    )
    
    # Plot and save results
    temporal_fig = plot_temporal_validation_results(temporal_results)
    temporal_fig.savefig(f"{output_dir}/temporal_validation.png", dpi=300, bbox_inches='tight')
    
    # Save metrics
    temporal_metrics = temporal_results['results_by_period']
    temporal_metrics.to_csv(f"{output_dir}/temporal_validation_metrics.csv", index=False)
    
    # 2. Backtesting
    print("\n2. Running backtesting simulation...")
    
    backtest_engine = BacktestEngine(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        date_column='GAME_DATE',
        min_confidence=config.MIN_CONFIDENCE,
        max_games=config.MAX_PARLAY_SIZE,
        stake=100,
        window_size=60  # 60-day rolling window
    )
    
    # Run backtest
    backtest_results = backtest_engine.run_backtest()
    
    # Plot and save results
    backtest_fig = backtest_engine.plot_results(backtest_results)
    if isinstance(backtest_fig, plt.Figure):
        backtest_fig.savefig(f"{output_dir}/backtest_results.png", dpi=300, bbox_inches='tight')
    
    # Save detailed results
    if len(backtest_results['bets']) > 0:
        backtest_results['bets'].to_csv(f"{output_dir}/backtest_bets.csv", index=False)
    
    if len(backtest_results['daily_results']) > 0:
        backtest_results['daily_results'].to_csv(f"{output_dir}/backtest_daily.csv", index=False)
    
    # 3. Model comparison
    print("\n3. Comparing different model architectures...")
    
    model_comparison = compare_models(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=0.3,
        random_state=config.RANDOM_STATE
    )
    
    # Plot and save results
    model_figs = plot_model_comparison(model_comparison)
    for i, fig in enumerate(model_figs):
        fig.savefig(f"{output_dir}/model_comparison_{i+1}.png", dpi=300, bbox_inches='tight')
    
    # Save metrics
    model_comparison['model_metrics'].to_csv(f"{output_dir}/model_metrics.csv", index=False)
    
    if len(model_comparison['parlay_metrics']) > 0:
        model_comparison['parlay_metrics'].to_csv(f"{output_dir}/model_parlay_metrics.csv", index=False)
    
    # 4. Feature analysis
    print("\n4. Analyzing feature importance...")
    
    # Get best model from comparison
    best_model_name = model_comparison['model_metrics'].sort_values('roc_auc', ascending=False)['model'].iloc[0]
    best_model = model_comparison['models'][best_model_name]
    
    # Analyze features
    feature_analysis = analyze_features(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        model=best_model,
        top_n=15
    )
    
    # Plot and save results
    feature_figs = plot_feature_analysis(feature_analysis, top_n=15)
    for i, fig in enumerate(feature_figs):
        fig.savefig(f"{output_dir}/feature_analysis_{i+1}.png", dpi=300, bbox_inches='tight')
    
    # Save feature scores
    feature_analysis['feature_scores'].to_csv(f"{output_dir}/feature_scores.csv", index=False)
    
    # 5. Parlay tracker analysis
    print("\n5. Analyzing parlay tracking data...")
    
    tracker = ParlayTracker()
    performance = tracker.get_recent_performance(n_days=30)
    
    print(f"Recent parlay performance (last 30 days):")
    for metric, value in performance.items():
        print(f"  {metric}: {value}")
    
    # Analyze trends if enough data
    trends = tracker.analyze_trends()
    if isinstance(trends, pd.DataFrame) and len(trends) > 0:
        # Plot trends
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        sns.lineplot(data=trends, x='date', y='rolling_accuracy', ax=ax1)
        ax1.set_title('Rolling Accuracy (7-day window)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Accuracy')
        
        sns.lineplot(data=trends, x='date', y='rolling_profit', ax=ax2)
        ax2.set_title('Rolling Profit (7-day window)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Profit ($)')
        
        plt.tight_layout()
        fig.savefig(f"{output_dir}/parlay_trends.png", dpi=300, bbox_inches='tight')
        
        # Save trends data
        trends.to_csv(f"{output_dir}/parlay_trends.csv", index=False)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    return {
        'temporal_validation': temporal_results,
        'backtesting': backtest_results,
        'model_comparison': model_comparison,
        'feature_analysis': feature_analysis,
        'parlay_performance': performance
    }

def main():
    """Main entry point for evaluation"""
    parser = argparse.ArgumentParser(description='NBA Parlay Prediction Model Evaluation')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Import config
    import config
    
    # Create timestamp for output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/{timestamp}"
    
    # Load data
    data = load_and_prepare_data(config)
    
    # Run evaluation
    results = run_comprehensive_evaluation(data, config, output_dir)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()