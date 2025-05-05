import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import config

def main(output_dir=None, season=None):
    """
    Run comprehensive evaluation of NBA prediction model
    
    Parameters:
    -----------
    output_dir : str
        Directory to save evaluation results
    season : str
        NBA season to evaluate
    """
    # Use default values if not provided
    if output_dir is None:
        output_dir = config.EVALUATION_DIR
        
    if season is None:
        season = config.NBA_SEASON
    
    # Create timestamp for output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running comprehensive evaluation for season {season}")
    print(f"Results will be saved to {output_dir}")
    
    # Load data
    from src.data.fetch_data import load_nba_data, load_odds_data
    from src.data.preprocess import preprocess_data
    from src.features.build_features import engineer_features
    
    try:
        # Try to load processed data first
        processed_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
        if os.path.exists(processed_file):
            print(f"Loading processed data from {processed_file}")
            feature_df = pd.read_csv(processed_file)
            
            # Convert date to datetime
            if 'GAME_DATE' in feature_df.columns:
                feature_df['GAME_DATE'] = pd.to_datetime(feature_df['GAME_DATE'])
        else:
            # Load and process raw data
            print(f"Loading raw data for season {season}")
            games_df = load_nba_data(season=season)
            odds_df = load_odds_data()
            
            print("Preprocessing data...")
            processed_df = preprocess_data(games_df, odds_df)
            
            print("Engineering features...")
            feature_df = engineer_features(processed_df)
        
        # Define feature and target columns
        non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
        feature_columns = [col for col in feature_df.columns if col not in non_feature_cols]
        
        # Run temporal validation
        print("\n1. Running temporal validation...")
        from src.evaluation.temporal_validation import time_based_validation, plot_temporal_validation_results
        
        temporal_results = time_based_validation(
            feature_df, 
            target_column='HOME_WIN',
            feature_columns=feature_columns,
            time_column='GAME_DATE',
            min_train_size=0.5,
            n_splits=3  # Fewer splits for quicker execution
        )
        
        # Plot and save results
        temporal_fig = plot_temporal_validation_results(temporal_results)
        temporal_fig.savefig(f"{output_dir}/temporal_validation.png", dpi=300, bbox_inches='tight')
        
        # Run backtesting
        print("\n2. Running backtesting simulation...")
        from src.evaluation.backtesting import BacktestEngine
        
        backtest_engine = BacktestEngine(
            feature_df,
            target_column='HOME_WIN',
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
        
        # Run model comparison
        print("\n3. Comparing different model architectures...")
        from src.evaluation.model_comparison import compare_models, plot_model_comparison
        
        model_comparison = compare_models(
            feature_df,
            target_column='HOME_WIN',
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
        
        # Run feature analysis
        print("\n4. Analyzing feature importance...")
        from src.evaluation.feature_analysis import analyze_features, plot_feature_analysis
        
        # Get best model from comparison
        best_model_name = model_comparison['model_metrics'].sort_values('roc_auc', ascending=False)['model'].iloc[0]
        best_model = model_comparison['models'][best_model_name]
        
        # Analyze features
        feature_analysis = analyze_features(
            feature_df,
            target_column='HOME_WIN',
            feature_columns=feature_columns,
            model=best_model,
            top_n=15
        )
        
        # Plot and save results
        feature_figs = plot_feature_analysis(feature_analysis, top_n=15)
        for i, fig in enumerate(feature_figs):
            fig.savefig(f"{output_dir}/feature_analysis_{i+1}.png", dpi=300, bbox_inches='tight')
        
        # Check parlay performance
        print("\n5. Analyzing parlay performance...")
        from src.evaluation.parlay_tracker import ParlayTracker
        
        tracker = ParlayTracker()
        performance = tracker.get_recent_performance(n_days=30)
        
        print("\nSummary of evaluation results:")
        print(f"  Temporal validation average accuracy: {temporal_results['avg_accuracy']:.4f}")
        print(f"  Temporal validation average ROI: {temporal_results['avg_roi']:.2f}%")
        print(f"  Best model: {best_model_name}")
        print(f"  Recent parlay accuracy: {performance['accuracy'] if performance['accuracy'] is not None else 'N/A'}")
        
        print(f"\nEvaluation complete! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    return output_dir

if __name__ == "__main__":
    main()