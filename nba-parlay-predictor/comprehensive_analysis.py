import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import config

# Import project modules
from src.data.fetch_data import load_nba_data, load_odds_data, generate_synthetic_data
from src.data.preprocess import preprocess_data
from src.features.build_features import engineer_features
from src.models.train_model import train_model

# Import evaluation modules
from src.evaluation.parlay_tracker import ParlayTracker
from src.evaluation.temporal_validation import time_based_validation, plot_temporal_validation_results
from src.evaluation.backtesting import BacktestEngine
from src.evaluation.model_comparison import compare_models, plot_model_comparison
from src.evaluation.feature_analysis import analyze_features, plot_feature_analysis

def create_html_report(results, output_dir):
    """
    Create HTML report of evaluation results
    
    Parameters:
    -----------
    results : dict
        Dictionary with evaluation results
    output_dir : str
        Directory to save the report
    """
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NBA Parlay Prediction Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .metric { font-weight: bold; }
            .good { color: green; }
            .bad { color: red; }
            .image-container { margin: 20px 0; }
            img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>NBA Parlay Prediction Evaluation Report</h1>
        <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        
        <div class="section">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
    """
    
    # Add temporal validation summary
    if 'temporal_validation' in results:
        tv = results['temporal_validation']
        html += f"""
                <tr>
                    <td class="metric">Temporal Validation Accuracy</td>
                    <td>{tv['avg_accuracy']:.4f}</td>
                </tr>
                <tr>
                    <td class="metric">Temporal Validation ROI</td>
                    <td class="{'good' if tv['avg_roi'] > 0 else 'bad'}">{tv['avg_roi']:.2f}%</td>
                </tr>
        """
    
    # Add backtesting summary
    if 'backtesting' in results:
        bt = results['backtesting']
        html += f"""
                <tr>
                    <td class="metric">Backtest Win Rate</td>
                    <td>{bt['win_rate']:.2%}</td>
                </tr>
                <tr>
                    <td class="metric">Backtest ROI</td>
                    <td class="{'good' if bt['total_roi'] > 0 else 'bad'}">{bt['total_roi']:.2f}%</td>
                </tr>
                <tr>
                    <td class="metric">Backtest Total Bets</td>
                    <td>{bt['total_bets']}</td>
                </tr>
        """
    
    # Add model comparison summary
    if 'model_comparison' in results and len(results['model_comparison']['model_metrics']) > 0:
        mc = results['model_comparison']
        best_model = mc['model_metrics'].sort_values('roc_auc', ascending=False).iloc[0]
        html += f"""
                <tr>
                    <td class="metric">Best Model</td>
                    <td>{best_model['model']}</td>
                </tr>
                <tr>
                    <td class="metric">Best Model ROC AUC</td>
                    <td>{best_model['roc_auc']:.4f}</td>
                </tr>
        """
    
    # Add parlay performance summary
    if 'parlay_performance' in results:
        pp = results['parlay_performance']
        html += f"""
                <tr>
                    <td class="metric">Recent Parlay Accuracy</td>
                    <td>{pp['accuracy'] if pp['accuracy'] is not None else 'N/A'}</td>
                </tr>
                <tr>
                    <td class="metric">Recent Parlay ROI</td>
                    <td class="{'good' if pp['roi'] > 0 else 'bad'}">{pp['roi']:.2f}%</td>
                </tr>
                <tr>
                    <td class="metric">Recent Parlay Count</td>
                    <td>{pp['sample_size']}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    # Add temporal validation section
    if 'temporal_validation' in results:
        html += """
        <div class="section">
            <h2>Temporal Validation</h2>
            <p>Results of validating the model across different time periods.</p>
            
            <div class="image-container">
                <img src="temporal_validation.png" alt="Temporal Validation Results">
            </div>
            
            <h3>Metrics by Time Period</h3>
            <table>
                <tr>
                    <th>Period Start</th>
                    <th>Period End</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>ROI</th>
                </tr>
        """
        
        for _, row in results['temporal_validation']['results_by_period'].iterrows():
            html += f"""
                <tr>
                    <td>{row['test_start_date']}</td>
                    <td>{row['test_end_date']}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['precision']:.4f}</td>
                    <td>{row['recall']:.4f}</td>
                    <td>{row['f1']:.4f}</td>
                    <td class="{'good' if row['roi'] > 0 else 'bad'}">{row['roi']:.2f}%</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add backtesting section
    if 'backtesting' in results:
        html += """
        <div class="section">
            <h2>Backtesting Results</h2>
            <p>Results of simulating the parlay strategy on historical data.</p>
            
            <div class="image-container">
                <img src="backtest_results.png" alt="Backtest Results">
            </div>
        """
        
        if len(results['backtesting']['daily_results']) > 0:
            html += """
            <h3>Top Performing Days</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Parlays</th>
                    <th>Profit</th>
                    <th>ROI</th>
                </tr>
            """
            
            # Show top 5 most profitable days
            top_days = results['backtesting']['daily_results'].sort_values('profit', ascending=False).head(5)
            for _, row in top_days.iterrows():
                html += f"""
                    <tr>
                        <td>{row['date']}</td>
                        <td>{row['num_parlays']}</td>
                        <td class="good">${row['profit']:.2f}</td>
                        <td class="good">{row['roi']:.2f}%</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
    
    # Add model comparison section
    if 'model_comparison' in results:
        html += """
        <div class="section">
            <h2>Model Comparison</h2>
            <p>Comparison of different model architectures.</p>
            
            <div class="image-container">
                <img src="model_comparison_1.png" alt="Model Metrics Comparison">
            </div>
            
            <div class="image-container">
                <img src="model_comparison_2.png" alt="ROC Curves">
            </div>
        """
        
        if len(results['model_comparison']['model_metrics']) > 0:
            html += """
            <h3>Model Performance Metrics</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>ROC AUC</th>
                </tr>
            """
            
            for _, row in results['model_comparison']['model_metrics'].iterrows():
                html += f"""
                    <tr>
                        <td>{row['model']}</td>
                        <td>{row['accuracy']:.4f}</td>
                        <td>{row['precision']:.4f}</td>
                        <td>{row['recall']:.4f}</td>
                        <td>{row['f1']:.4f}</td>
                        <td>{row['roc_auc']:.4f}</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        if len(results['model_comparison']['parlay_metrics']) > 0:
            html += """
            <h3>Model Parlay Performance</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Parlays</th>
                    <th>Avg Size</th>
                    <th>Win Rate</th>
                    <th>Profit</th>
                    <th>ROI</th>
                </tr>
            """
            
            for _, row in results['model_comparison']['parlay_metrics'].iterrows():
                html += f"""
                    <tr>
                        <td>{row['model']}</td>
                        <td>{row['num_parlays']}</td>
                        <td>{row['avg_parlay_size']:.2f}</td>
                        <td>{row['win_rate']:.2%}</td>
                        <td class="{'good' if row['total_profit'] > 0 else 'bad'}">${row['total_profit']:.2f}</td>
                        <td class="{'good' if row['roi'] > 0 else 'bad'}">{row['roi']:.2f}%</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
    
    # Add feature analysis section
    if 'feature_analysis' in results:
        html += """
        <div class="section">
            <h2>Feature Analysis</h2>
            <p>Analysis of feature importance and relationships.</p>
            
            <div class="image-container">
                <img src="feature_analysis_1.png" alt="Feature Importance">
            </div>
            
            <div class="image-container">
                <img src="feature_analysis_2.png" alt="Feature Correlation">
            </div>
        """
        
        if 'feature_scores' in results['feature_analysis']:
            html += """
            <h3>Top Feature Scores</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Correlation</th>
                    <th>Mutual Info</th>
            """
            
            if 'permutation_importance' in results['feature_analysis']['feature_scores'].columns:
                html += """
                    <th>Permutation Importance</th>
                """
            
            html += """
                </tr>
            """
            
            # Show top 10 features by mutual info
            top_features = results['feature_analysis']['feature_scores'].sort_values('mutual_info', ascending=False).head(10)
            for _, row in top_features.iterrows():
                html += f"""
                    <tr>
                        <td>{row['feature']}</td>
                        <td>{row['correlation']:.4f}</td>
                        <td>{row['mutual_info']:.4f}</td>
                """
                
                if 'permutation_importance' in results['feature_analysis']['feature_scores'].columns:
                    html += f"""
                        <td>{row['permutation_importance']:.4f}</td>
                    """
                
                html += """
                    </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
    
    # Add recent parlay section
    if 'parlay_performance' in results and results['parlay_performance']['sample_size'] > 0:
        html += """
        <div class="section">
            <h2>Recent Parlay Performance</h2>
            <p>Analysis of recent parlay predictions.</p>
            
            <div class="image-container">
                <img src="parlay_trends.png" alt="Parlay Trends">
            </div>
            
            <h3>Performance Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        pp = results['parlay_performance']
        metrics = [
            ('Accuracy', f"{pp['accuracy']:.2%}" if pp['accuracy'] is not None else 'N/A'),
            ('Average Confidence', f"{pp['average_confidence']:.2%}" if pp['average_confidence'] is not None else 'N/A'),
            ('Calibration Error', f"{pp['calibration_error']:.2%}" if pp['calibration_error'] is not None else 'N/A'),
            ('Total Profit', f"${pp['profit']:.2f}"),
            ('ROI', f"{pp['roi']:.2f}%"),
            ('Sample Size', str(pp['sample_size']))
        ]
        
        for metric, value in metrics:
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{value}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Finish HTML
    html += """
    </body>
    </html>
    """
    
    # Write to file
    with open(f"{output_dir}/evaluation_report.html", 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_dir}/evaluation_report.html")

def main():
    """Main entry point for comprehensive evaluation"""
    parser = argparse.ArgumentParser(description='NBA Parlay Prediction Comprehensive Evaluation')
    parser.add_argument('--output-dir', type=str, default=config.EVALUATION_DIR,
                      help='Directory to save evaluation results')
    parser.add_argument('--season', type=str, default=config.NBA_SEASON,
                      help='NBA season to evaluate')
    args = parser.parse_args()
    
    # Create timestamp for output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    try:
        games_df = load_nba_data(season=args.season)
        odds_df = load_odds_data()
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = preprocess_data(games_df, odds_df)
        
        # Engineer features
        print("Engineering features...")
        feature_df = engineer_features(processed_df)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using previously processed data if available...")
        
        processed_file = f"{config.DATA_PROCESSED_DIR}/features.csv"
        if os.path.exists(processed_file):
            feature_df = pd.read_csv(processed_file)
            
            # Convert date to datetime
            if 'GAME_DATE' in feature_df.columns:
                feature_df['GAME_DATE'] = pd.to_datetime(feature_df['GAME_DATE'])
        else:
            print("No processed data found. Using synthetic data...")
            games_df, odds_df = generate_synthetic_data()
            processed_df = preprocess_data(games_df, odds_df)
            feature_df = engineer_features(processed_df)
    
    # Define feature and target columns
    target_column = 'HOME_WIN'
    non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']
    feature_columns = [col for col in feature_df.columns if col not in non_feature_cols]
    
    # Results container
    evaluation_results = {}
    
    # 1. Temporal validation
    print("\n1. Running temporal validation...")
    temporal_results = time_based_validation(
        feature_df, 
        target_column=target_column,
        feature_columns=feature_columns,
        time_column='GAME_DATE',
        min_train_size=0.5,
        n_splits=5
    )
    
    # Plot and save results
    temporal_fig = plot_temporal_validation_results(temporal_results)
    temporal_fig.savefig(f"{output_dir}/temporal_validation.png", dpi=300, bbox_inches='tight')
    
    # Save results
    evaluation_results['temporal_validation'] = temporal_results
    
    # 2. Backtesting
    print("\n2. Running backtesting simulation...")
    backtest_engine = BacktestEngine(
        feature_df,
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
    
    # Save results
    evaluation_results['backtesting'] = backtest_results
    
    # 3. Model comparison
    print("\n3. Comparing different model architectures...")
    model_comparison = compare_models(
        feature_df,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=0.3,
        random_state=config.RANDOM_STATE
    )
    
    # Plot and save results
    model_figs = plot_model_comparison(model_comparison)
    for i, fig in enumerate(model_figs):
        fig.savefig(f"{output_dir}/model_comparison_{i+1}.png", dpi=300, bbox_inches='tight')
    
    # Save results
    evaluation_results['model_comparison'] = model_comparison
    
    # 4. Feature analysis
    print("\n4. Analyzing feature importance...")
    
    # Get best model from comparison
    best_model_name = model_comparison['model_metrics'].sort_values('roc_auc', ascending=False)['model'].iloc[0]
    best_model = model_comparison['models'][best_model_name]
    
    # Analyze features
    feature_analysis = analyze_features(
        feature_df,
        target_column=target_column,
        feature_columns=feature_columns,
        model=best_model,
        top_n=15
    )
    
    # Plot and save results
    feature_figs = plot_feature_analysis(feature_analysis, top_n=15)
    for i, fig in enumerate(feature_figs):
        fig.savefig(f"{output_dir}/feature_analysis_{i+1}.png", dpi=300, bbox_inches='tight')
    
    # Save results
    evaluation_results['feature_analysis'] = feature_analysis
    
    # 5. Parlay tracker analysis
    print("\n5. Analyzing parlay tracking data...")
    tracker = ParlayTracker()
    performance = tracker.get_recent_performance(n_days=30)
    
    # Save results
    evaluation_results['parlay_performance'] = performance
    
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
    
    # Create HTML report
    print("\n6. Creating evaluation report...")
    create_html_report(evaluation_results, output_dir)
    
    print(f"\nComprehensive evaluation complete! Results saved to {output_dir}")
    
    return evaluation_results

if __name__ == "__main__":
    main()