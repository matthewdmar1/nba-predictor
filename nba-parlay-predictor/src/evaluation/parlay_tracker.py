import pandas as pd
import numpy as np
import os
import datetime

class ParlayTracker:
    """Track parlays and their outcomes over time"""
    
    def __init__(self, data_dir="data/evaluation"):
        """Initialize tracker with data directory"""
        self.data_dir = data_dir
        self.history_file = f"{data_dir}/parlay_history.csv"
        self.predictions_file = f"{data_dir}/parlay_predictions.csv"
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing history if available
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
        else:
            self.history = pd.DataFrame(columns=[
                'parlay_id', 'date_created', 'games', 'predicted_probability', 
                'actual_outcome', 'stake', 'payout', 'profit'
            ])
            
        # Load existing predictions if available
        if os.path.exists(self.predictions_file):
            self.predictions = pd.read_csv(self.predictions_file)
        else:
            self.predictions = pd.DataFrame(columns=[
                'prediction_id', 'date_created', 'games', 'predicted_probability',
                'model_version', 'confidence'
            ])
    
    def record_prediction(self, games, probability, model_version, confidence):
        """Record a new parlay prediction"""
        prediction_id = len(self.predictions) + 1
        date_created = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Convert games to standardized format
        games_str = '; '.join([
            f"{game['HOME_TEAM']} vs {game['AWAY_TEAM']}: {'HOME' if game['predicted_home_win'] else 'AWAY'}" 
            for game in games
        ])
        
        # Add new prediction
        new_prediction = pd.DataFrame([{
            'prediction_id': prediction_id,
            'date_created': date_created,
            'games': games_str,
            'predicted_probability': probability,
            'model_version': model_version,
            'confidence': confidence
        }])
        
        self.predictions = pd.concat([self.predictions, new_prediction], ignore_index=True)
        self.predictions.to_csv(self.predictions_file, index=False)
        
        return prediction_id
    
    def record_outcome(self, parlay_id, games, predicted_probability, 
                     actual_outcome, stake, payout):
        """Record the outcome of a parlay bet"""
        date_created = datetime.datetime.now().strftime('%Y-%m-%d')
        profit = payout - stake if actual_outcome else -stake
        
        # Convert games to standardized format if not already
        if isinstance(games, list):
            games_str = '; '.join([
                f"{game['HOME_TEAM']} vs {game['AWAY_TEAM']}: {'HOME' if game['predicted_home_win'] else 'AWAY'}" 
                for game in games
            ])
        else:
            games_str = games
            
        # Add new outcome
        new_outcome = pd.DataFrame([{
            'parlay_id': parlay_id,
            'date_created': date_created,
            'games': games_str,
            'predicted_probability': predicted_probability,
            'actual_outcome': actual_outcome,
            'stake': stake,
            'payout': payout,
            'profit': profit
        }])
        
        self.history = pd.concat([self.history, new_outcome], ignore_index=True)
        self.history.to_csv(self.history_file, index=False)
        
        return parlay_id
    
    def calculate_prediction_accuracy(self, lookback_days=None):
        """Calculate accuracy of predictions compared to actual outcomes"""
        # Join predictions with outcomes on games
        merged = pd.merge(
            self.predictions, 
            self.history, 
            left_on='games', 
            right_on='games', 
            suffixes=('_pred', '_actual')
        )
        
        # Filter by lookback period if specified
        if lookback_days:
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            merged = merged[merged['date_created_actual'] >= cutoff_date]
            
        # Calculate metrics
        if len(merged) == 0:
            return {
                'accuracy': None,
                'average_confidence': None,
                'calibration_error': None,
                'profit': 0,
                'roi': 0,
                'sample_size': 0
            }
            
        accuracy = (merged['actual_outcome'] == True).mean()
        avg_confidence = merged['predicted_probability'].mean()
        calibration_error = abs(accuracy - avg_confidence)
        total_profit = merged['profit'].sum()
        roi = (total_profit / merged['stake'].sum()) * 100 if merged['stake'].sum() > 0 else 0
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'calibration_error': calibration_error,
            'profit': total_profit,
            'roi': roi,
            'sample_size': len(merged)
        }
    
    def get_recent_performance(self, n_days=30):
        """Get performance metrics over recent time period"""
        return self.calculate_prediction_accuracy(lookback_days=n_days)
    
    def analyze_trends(self, window_size=7):
        """Analyze trends in prediction performance over time"""
        if len(self.history) < window_size:
            return "Not enough data for trend analysis"
            
        # Convert date strings to datetime
        self.history['date_created'] = pd.to_datetime(self.history['date_created'])
        
        # Sort by date
        sorted_history = self.history.sort_values('date_created')
        
        # Create rolling window metrics
        rolling_accuracy = sorted_history['actual_outcome'].rolling(window=window_size).mean()
        rolling_profit = sorted_history['profit'].rolling(window=window_size).sum()
        
        # Return as dataframe for plotting
        return pd.DataFrame({
            'date': sorted_history['date_created'],
            'rolling_accuracy': rolling_accuracy,
            'rolling_profit': rolling_profit
        }).dropna()