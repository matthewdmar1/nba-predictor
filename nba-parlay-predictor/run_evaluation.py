import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import config
import matplotlib.pyplot as plt

from src.evaluation.comprehensive import main as run_comprehensive

def main():
    """Command-line interface for model evaluation"""
    parser = argparse.ArgumentParser(description='NBA Parlay Prediction Model Evaluation')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Evaluation command')
    
    # Comprehensive evaluation command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive evaluation')
    comprehensive_parser.add_argument('--output-dir', type=str, default=config.EVALUATION_DIR,
                                    help='Directory to save evaluation results')
    comprehensive_parser.add_argument('--season', type=str, default=config.NBA_SEASON,
                                    help='NBA season to evaluate')
    
    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting simulation')
    backtest_parser.add_argument('--output-dir', type=str, default=config.EVALUATION_DIR,
                               help='Directory to save evaluation results')
    backtest_parser.add_argument('--confidence', type=float, default=config.MIN_CONFIDENCE,
                               help='Minimum confidence threshold')
    backtest_parser.add_argument('--max-games', type=int, default=config.MAX_PARLAY_SIZE,
                               help='Maximum number of games per parlay')
    backtest_parser.add_argument('--window', type=int, default=60,
                               help='Window size in days for backtesting')
    
    # Model comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare different models')
    compare_parser.add_argument('--output-dir', type=str, default=config.EVALUATION_DIR,
                              help='Directory to save evaluation results')
    
    # Feature analysis command
    feature_parser = subparsers.add_parser('features', help='Analyze feature importance')
    feature_parser.add_argument('--output-dir', type=str, default=config.EVALUATION_DIR,
                              help='Directory to save evaluation results')
    feature_parser.add_argument('--top-n', type=int, default=15,
                              help='Number of top features to show')
    
    # Track parlays command
    track_parser = subparsers.add_parser('track', help='Track parlay predictions and outcomes')
    track_parser.add_argument('--action', choices=['predict', 'record', 'analyze'], required=True,
                            help='Track action to perform')
    track_parser.add_argument('--game-date', type=str,
                            help='Game date for tracking (YYYY-MM-DD)')
    track_parser.add_argument('--parlay-id', type=int,
                            help='ID of the parlay to record')
    track_parser.add_argument('--outcome', type=str, choices=['win', 'loss'],
                            help='Outcome of the parlay')
    track_parser.add_argument('--stake', type=float, default=100.0,
                            help='Stake amount for the parlay')
    track_parser.add_argument('--payout', type=float,
                            help='Payout amount for the parlay')
    
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Handle different commands
    if args.command == 'comprehensive':
        print("Running comprehensive evaluation...")
        run_comprehensive()
        
    elif args.command == 'backtest':
        print("Running backtesting simulation...")
        output_dir = f"{args.output_dir}/backtest_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        from src.data.fetch_data import load_nba_data, load_odds_data
        from src.data.preprocess import preprocess_data
        from src.features.build_features import engineer_features
        from src.evaluation.backtesting import BacktestEngine
        
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
                games_df = load_nba_data(season=config.NBA_SEASON)
                odds_df = load_odds_data()
                processed_df = preprocess_data(games_df, odds_df)
                feature_df = engineer_features(processed_df)
            
            # Define features
            non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
            feature_columns = [col for col in feature_df.columns if col not in non_feature_cols]
            
            # Run backtest
            backtest_engine = BacktestEngine(
                feature_df,
                target_column='HOME_WIN',
                feature_columns=feature_columns,
                date_column='GAME_DATE',
                min_confidence=args.confidence,
                max_games=args.max_games,
                stake=100,
                window_size=args.window
            )
            
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
                
            print(f"Backtesting complete! Results saved to {output_dir}")
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            sys.exit(1)
            
    elif args.command == 'compare':
        print("Running model comparison...")
        output_dir = f"{args.output_dir}/models_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        from src.data.fetch_data import load_nba_data, load_odds_data
        from src.data.preprocess import preprocess_data
        from src.features.build_features import engineer_features
        from src.evaluation.model_comparison import compare_models, plot_model_comparison
        
        try:
            # Try to load processed data first
            processed_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
            if os.path.exists(processed_file):
                print(f"Loading processed data from {processed_file}")
                feature_df = pd.read_csv(processed_file)
            else:
                # Load and process raw data
                games_df = load_nba_data(season=config.NBA_SEASON)
                odds_df = load_odds_data()
                processed_df = preprocess_data(games_df, odds_df)
                feature_df = engineer_features(processed_df)
            
            # Define features
            non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
            feature_columns = [col for col in feature_df.columns if col not in non_feature_cols]
            
            # Run model comparison
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
            
            if len(model_comparison['parlay_metrics']) > 0:
                model_comparison['parlay_metrics'].to_csv(f"{output_dir}/model_parlay_metrics.csv", index=False)
                
            print(f"Model comparison complete! Results saved to {output_dir}")
            
        except Exception as e:
            print(f"Error running model comparison: {e}")
            sys.exit(1)
            
    elif args.command == 'features':
        print("Running feature analysis...")
        output_dir = f"{args.output_dir}/features_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        from src.data.fetch_data import load_nba_data, load_odds_data
        from src.data.preprocess import preprocess_data
        from src.features.build_features import engineer_features
        from src.models.train_model import train_model
        from src.evaluation.feature_analysis import analyze_features, plot_feature_analysis
        
        try:
            # Try to load processed data first
            processed_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
            if os.path.exists(processed_file):
                print(f"Loading processed data from {processed_file}")
                feature_df = pd.read_csv(processed_file)
            else:
                # Load and process raw data
                games_df = load_nba_data(season=config.NBA_SEASON)
                odds_df = load_odds_data()
                processed_df = preprocess_data(games_df, odds_df)
                feature_df = engineer_features(processed_df)
            
            # Define features
            non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
            feature_columns = [col for col in feature_df.columns if col not in non_feature_cols]
            
            # Train a model for feature importance
            from sklearn.model_selection import train_test_split
            
            X = feature_df[feature_columns]
            y = feature_df['HOME_WIN']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=config.RANDOM_STATE
            )
            
            model = train_model(X_train, y_train)
            
            # Analyze features
            feature_analysis = analyze_features(
                feature_df,
                target_column='HOME_WIN',
                feature_columns=feature_columns,
                model=model,
                top_n=args.top_n
            )
            
            # Plot and save results
            feature_figs = plot_feature_analysis(feature_analysis, top_n=args.top_n)
            for i, fig in enumerate(feature_figs):
                fig.savefig(f"{output_dir}/feature_analysis_{i+1}.png", dpi=300, bbox_inches='tight')
            
            # Save feature scores
            feature_analysis['feature_scores'].to_csv(f"{output_dir}/feature_scores.csv", index=False)
                
            print(f"Feature analysis complete! Results saved to {output_dir}")
            
        except Exception as e:
            print(f"Error running feature analysis: {e}")
            sys.exit(1)
            
    elif args.command == 'track':
        from src.evaluation.parlay_tracker import ParlayTracker
        
        tracker = ParlayTracker()
        
        if args.action == 'predict':
            print("To record a prediction, you need to run the main prediction pipeline.")
            print("Use: python main.py")
            
        elif args.action == 'record':
            if not args.parlay_id:
                print("Error: --parlay-id is required for recording outcomes")
                sys.exit(1)
                
            if not args.outcome:
                print("Error: --outcome is required for recording outcomes")
                sys.exit(1)
                
            if not args.payout:
                print("Error: --payout is required for recording outcomes")
                sys.exit(1)
                
            # Get prediction details
            from src.evaluation.parlay_tracker import ParlayTracker
            
            tracker = ParlayTracker()
            
            # Find prediction
            if len(tracker.predictions) > 0:
                prediction = tracker.predictions[tracker.predictions['prediction_id'] == args.parlay_id]
                
                if len(prediction) > 0:
                    # Record outcome
                    actual_outcome = args.outcome == 'win'
                    
                    tracker.record_outcome(
                        args.parlay_id,
                        prediction['games'].iloc[0],
                        prediction['predicted_probability'].iloc[0],
                        actual_outcome,
                        args.stake,
                        args.payout if actual_outcome else 0
                    )
                    
                    print(f"Recorded outcome for parlay #{args.parlay_id}: {'WIN' if actual_outcome else 'LOSS'}")
                else:
                    print(f"Error: Prediction with ID {args.parlay_id} not found")
            else:
                print("No predictions found in tracker")
            
        elif args.action == 'analyze':
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
            
            # Analyze trends
            trends = tracker.analyze_trends()
            if isinstance(trends, pd.DataFrame) and len(trends) > 0:
                print("\nTrend analysis:")
                print("  Last 7-day rolling accuracy:", trends['rolling_accuracy'].iloc[-1])
                print("  Last 7-day rolling profit:", trends['rolling_profit'].iloc[-1])
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()